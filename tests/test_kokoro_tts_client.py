"""Tests unitarios para KokoroTTSClient.

Verifica síntesis de voz local con Kokoro-ONNX, lazy-load del modelo,
conversión float32→int16, manejo de errores, y que no se filtren
secretos en logs.

Todos los tests son ``@pytest.mark.unit`` — sin red, sin disco,
sin modelo real. Se mockea ``kokoro_onnx.Kokoro`` con ``unittest.mock``.
"""

from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────
# Stub de kokoro_onnx — registrado ANTES de importar el handler.
#
# kokoro_onnx NO está instalado en el entorno de tests (mismo problema
# que piper con Python 3.14). El handler hace ``from kokoro_onnx import
# Kokoro`` top-level, así que si no stubbeamos el módulo, el import
# rompe la colección de pytest.
#
# Mismo patrón que ``tests/test_local_integration.py`` para piper y
# faster_whisper.
# ──────────────────────────────────────────────────────────────────

if "kokoro_onnx" not in sys.modules:
    kokoro_stub = types.ModuleType("kokoro_onnx")
    kokoro_stub.Kokoro = MagicMock(name="Kokoro")
    sys.modules["kokoro_onnx"] = kokoro_stub

# Ahora el import del handler resuelve sin tocar disco ni red
from handlers.kokoro_tts_client import KokoroTTSClient  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def kokoro_settings() -> dict:
    """Settings sintéticos con la sección ``local.kokoro``."""
    return {
        "local": {
            "kokoro": {
                "model_path": "models/kokoro/kokoro-v1.0.onnx",
                "voices_path": "models/kokoro/voices-v1.0.bin",
                "voice": "em_alex",
                "lang": "es",
                "speed": 1.0,
            }
        }
    }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _fake_create(text, voice, speed, lang, trim=None):
    """Retorna (samples float32, sample_rate 24000) como Kokoro real.

    Acepta ``trim`` para mantener compatibilidad con la nueva firma de
    ``Kokoro.create()`` tras el fix de phonemizer mismatch (kokoro-onnx
    ahora recibe ``trim=False`` desde KokoroTTSClient.synthesize).
    """
    # 100 samples float32 en [-1, 1]
    samples = np.array([0.5, -0.5, 0.0] * 33 + [0.25], dtype=np.float32)
    return samples, 24000


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestKokoroTTSClient:
    """Suite: KokoroTTSClient — síntesis, lazy-load, errores, conversión."""

    def test_synthesize_success(self, kokoro_settings):
        """Mock Kokoro.create retorna (np.array float32, 24000) → PCM crudo int16.

        Verificar que ``len(result) == len(samples) * 2`` (2 bytes por sample int16).
        """
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                client = KokoroTTSClient(kokoro_settings)
                result = client.synthesize("Hola mundo")

                # 100 samples float32 → 100 samples int16 → 200 bytes
                assert len(result) == 200
                assert isinstance(result, bytes)

    def test_synthesize_lazy_load(self, kokoro_settings):
        """El modelo NO se carga en __init__, sí en la 1ra llamada a synthesize().

        Mock Kokoro class, verificar que el constructor NO se llama en
        __init__, SÍ en la 1ra synthesize().
        """
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                # __init__ no debe cargar el modelo
                client = KokoroTTSClient(kokoro_settings)
                MockKokoro.assert_not_called()

                # 1ra synthesize() sí debe cargar el modelo
                client.synthesize("Hola")
                MockKokoro.assert_called_once()

    def test_synthesize_model_not_found(self, kokoro_settings):
        """Mock Path.exists retorna False para model_path → RuntimeError.

        Debe contener "Modelo Kokoro no encontrado" y la URL de descarga.
        """
        def fake_exists(self):
            return "kokoro-v1.0.onnx" not in str(self)

        with patch("handlers.kokoro_tts_client.Path.exists", autospec=True, side_effect=fake_exists):
            client = KokoroTTSClient(kokoro_settings)
            with pytest.raises(RuntimeError, match="Modelo Kokoro no encontrado"):
                client.synthesize("Hola")

    def test_synthesize_voices_not_found(self, kokoro_settings):
        """Mock Path.exists retorna True para model, False para voices → RuntimeError.

        Debe contener "Voces Kokoro no encontradas".
        """
        def fake_exists(self):
            return "kokoro-v1.0.onnx" in str(self)

        with patch("handlers.kokoro_tts_client.Path.exists", autospec=True, side_effect=fake_exists):
            client = KokoroTTSClient(kokoro_settings)
            with pytest.raises(RuntimeError, match="Voces Kokoro no encontradas"):
                client.synthesize("Hola")

    def test_synthesize_failure(self, kokoro_settings):
        """Mock Kokoro.create lanza excepción → RuntimeError("Kokoro TTS falló")."""
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create.side_effect = ValueError("ONNX inference error")

                client = KokoroTTSClient(kokoro_settings)
                with pytest.raises(RuntimeError, match="Kokoro TTS falló"):
                    client.synthesize("Hola")

    def test_synthesize_stream_chunks(self, kokoro_settings):
        """synthesize_stream() yields chunks de hasta 4096 bytes PCM."""
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                client = KokoroTTSClient(kokoro_settings)
                chunks = list(client.synthesize_stream("Hola mundo"))

                # 200 bytes total → 1 chunk (menor que 4096)
                assert len(chunks) == 1
                assert len(chunks[0]) == 200
                # Todos los chunks deben ser <= 4096
                for chunk in chunks:
                    assert len(chunk) <= 4096

    def test_style_hint_ignored(self, kokoro_settings):
        """synthesize(text, "cheerful") produce el mismo resultado que synthesize(text, "")."""
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                client = KokoroTTSClient(kokoro_settings)
                result_cheerful = client.synthesize("Hola", "cheerful")
                result_empty = client.synthesize("Hola", "")

                assert result_cheerful == result_empty

    def test_returns_pcm_not_wav(self, kokoro_settings):
        """El resultado NO empieza con "RIFF" (cabecera WAV), es PCM crudo."""
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                client = KokoroTTSClient(kokoro_settings)
                result = client.synthesize("Hola")

                assert not result.startswith(b"RIFF")

    def test_float32_to_int16_conversion(self, kokoro_settings):
        """Mock Kokoro.create retorna array con valores fuera de [-1,1] → se clipa.

        Verificar que los samples resultantes están en rango int16 [-32768, 32767].
        """
        def fake_create_out_of_range(text, voice, speed, lang, trim=None):
            samples = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
            return samples, 24000

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = fake_create_out_of_range

                client = KokoroTTSClient(kokoro_settings)
                result = client.synthesize("Hola")

                # 4 samples int16 → 8 bytes
                assert len(result) == 8
                # Reconstruir samples int16
                samples_int16 = np.frombuffer(result, dtype=np.int16)
                # 2.0 → clip 1.0 → 32767, -2.0 → clip -1.0 → -32767
                # 0.5 → 16383.5 → trunc int16 → 16383, -0.5 → -16383
                # (numpy .astype(np.int16) trunca hacia cero, no redondea)
                assert samples_int16[0] == 32767
                assert samples_int16[1] == -32767
                assert samples_int16[2] == 16383
                assert samples_int16[3] == -16383
                # Todos en rango int16
                assert np.all(samples_int16 >= -32768)
                assert np.all(samples_int16 <= 32767)

    def test_no_secrets_logged(self, kokoro_settings, caplog):
        """Verificar que paths sensibles NO aparecen en logs.

        Setear model_path con un path "sensible" y verificar con caplog
        que "SECRET_USER_DO_NOT_LEAK_999" NO aparece en logs.
        """
        kokoro_settings["local"]["kokoro"]["model_path"] = (
            r"C:\Users\SECRET_USER_DO_NOT_LEAK_999\models\kokoro\kokoro-v1.0.onnx"
        )

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = _fake_create

                with caplog.at_level(logging.DEBUG, logger="handlers.kokoro_tts_client"):
                    client = KokoroTTSClient(kokoro_settings)
                    client.synthesize("Hola")

                # Verificar que el path sensible NO aparece en ningún log
                log_text = caplog.text
                assert "SECRET_USER_DO_NOT_LEAK_999" not in log_text


# ──────────────────────────────────────────────────────────────────
# Tests de normalización de whitespace (fix kokoro phonemizer mismatch)
# ──────────────────────────────────────────────────────────────────
#
# Estos tests cubren el fix de la spec ``specs/bug_kokoro_phonemizer_mismatch.md``.
# El handler ahora aplica ``re.sub(r"\s+", " ", text).strip()`` antes de
# invocar ``Kokoro.create()`` y pasa ``trim=False``. Los tests verifican
# que esa normalización es correcta y NO destruye caracteres relevantes
# (em-dash, números con dos puntos, ñ/acentos).


@pytest.mark.unit
class TestKokoroNormalization:
    """Suite: normalización de whitespace en KokoroTTSClient.synthesize."""

    _PROD_MULTILINE = (
        "Mañana lunes 29 de junio tenés:\n"
        "18:00 a 20:00 — Estudio Teclab\n"
        "20:00 a 21:00 — Ejercicio / Entrenamiento\n"
        "El resto del día"
    )

    def test_a_multiline_newlines_collapsed(self, kokoro_settings):
        """Test A — Texto multilinea del log de prod: newlines colapsados a espacios.

        Tras el fix de chunking, ``Kokoro.create`` se llama UNA VEZ POR
        CHUNK (``re.split`` produce varios chunks para este texto por los
        ``:`` y ``—``). Verificamos que TODOS los chunks recibidos están
        normalizados (sin ``\\n``/tabs, sin espacios múltiples) y que su
        concatenación reconstruye el texto normalizado original.
        """
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                # MagicMock con side_effect preserva call_args / call_args_list
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(self._PROD_MULTILINE)

                # Ahora hay N llamadas (una por chunk). Verificamos que cada
                # chunk recibido está normalizado.
                assert mock_instance.create.call_count >= 1
                received_chunks = [c.args[0] for c in mock_instance.create.call_args_list]

                # Sin newlines, sin tabs en NINGÚN chunk
                for chunk in received_chunks:
                    assert "\n" not in chunk
                    assert "\t" not in chunk
                    assert "\r" not in chunk
                    # Sin secuencias de 2+ espacios consecutivos
                    assert "  " not in chunk

                # Sanity: la concatenación de los chunks (con espacios entre
                # ellos, ya que el split por puntuación consume el whitespace
                # post-signo) reconstruye el texto normalizado.
                expected_normalized = re.sub(r"\s+", " ", self._PROD_MULTILINE).strip()
                reconstructed = " ".join(received_chunks)
                # Normalizar el reconstructed (el split puede comerse espacios)
                reconstructed_normalized = re.sub(r"\s+", " ", reconstructed).strip()
                assert reconstructed_normalized == expected_normalized

                # Además, cada chunk se llamó con trim=False (no True ni None)
                for call in mock_instance.create.call_args_list:
                    assert call.kwargs.get("trim") is False

    def test_b_em_dash_and_colon_preserved(self, kokoro_settings):
        """Test B — Em-dash (U+2014) y números con dos puntos NO son tocados.

        La normalización solo colapsa whitespace, no otros caracteres.
        Tras el chunking, el em-dash puede caer en cualquier chunk; verificamos
        que la CONCATENACIÓN de todos los chunks recibidos preserva los
        caracteres clave (em-dash, "18:00").
        """
        input_text = "Tu cita es a las 18:00 — no te olvides"

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(input_text)

                received_chunks = [c.args[0] for c in mock_instance.create.call_args_list]

                # Reconstruir el texto concatenando chunks (con espacios entre ellos)
                reconstructed = " ".join(received_chunks)

                # Em-dash presente en la concatenación (no se pierde)
                assert "\u2014" in reconstructed
                # "18:00" presente (los dos puntos no son whitespace)
                assert "18:00" in reconstructed
                # Sin espacios múltiples introducidos en ningún chunk
                for chunk in received_chunks:
                    assert "  " not in chunk

    def test_c_trim_false_always_passed(self, kokoro_settings):
        """Test C — En TODAS las llamadas a ``create()`` se pasa ``trim=False``.

        Verifica en múltiples invocaciones que el kwarg ``trim`` es siempre
        ``False`` (no ``True`` ni ``None``).
        """
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize("primera llamada")
                client.synthesize("segunda\nllamada\ncon\nnewlines")
                client.synthesize("tercera con   espacios   múltiples")

                # 3 invocaciones, todas con trim=False
                assert mock_instance.create.call_count == 3
                for call in mock_instance.create.call_args_list:
                    assert "trim" in call.kwargs, "Falta kwarg 'trim'"
                    assert call.kwargs["trim"] is False, (
                        f"Se esperaba trim=False, se obtuvo {call.kwargs['trim']!r}"
                    )

    def test_d_tabs_and_multiple_spaces_collapsed(self, kokoro_settings):
        """Test D — Tabs y secuencias de espacios múltiples → un solo espacio."""
        input_text = "hola\t\tmundo   multiple   espacios"

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(input_text)

                received_text = mock_instance.create.call_args.args[0]

                # Texto exacto esperado (post-normalización)
                assert received_text == "hola mundo multiple espacios"

    def test_e_synthesize_stream_inherits_normalization(self, kokoro_settings):
        """Test E — ``synthesize_stream`` también normaliza el texto antes de ``create``.

        Verifica que el texto multilinea que entra a ``synthesize_stream``
        llega normalizado (sin newlines) a ``Kokoro.create``.
        """
        input_text = "línea uno\nlínea dos\nlínea tres"

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                # Consumir el iterador para forzar la materialización
                chunks = list(client.synthesize_stream(input_text))

                # Hubo al menos un chunk (no falló)
                assert len(chunks) >= 1
                # El texto que llegó a create fue normalizado
                received_text = mock_instance.create.call_args.args[0]
                assert "\n" not in received_text
                assert received_text == "línea uno línea dos línea tres"
                # Además, trim=False se propaga a través de synthesize_stream
                assert mock_instance.create.call_args.kwargs.get("trim") is False


# ──────────────────────────────────────────────────────────────────
# Tests de chunking de texto largo (fix kokoro IndexError 510 phonemes)
# ──────────────────────────────────────────────────────────────────
#
# Estos tests cubren el fix de la spec
# ``specs/bug_kokoro_chunking_510_phonemes.md``. El handler ahora splitea
# el texto normalizado por puntuación fuerte + separadores (lookbehind)
# y concatena los arrays de samples de cada llamada a ``Kokoro.create()``.


@pytest.mark.unit
class TestKokoroChunking:
    """Suite: chunking de texto largo en KokoroTTSClient (fix IndexError 510)."""

    # Texto del log de prod (reconstruido con el patrón del bug: agenda
    # lunes→domingo con horarios, em-dash, en-dash, slash, dos puntos).
    # ~678 chars — disparaba IndexError por exceder MAX_PHONEME_LENGTH=510.
    _PROD_LONG_AGENDA = (
        "Te resumo tu agenda de la semana que viene (lunes 29/6 al domingo 5/7): "
        "Lunes 29/6 18:00–20:00 — Estudio Teclab (Teclab Placeholder) "
        "20:00 a 21:00 — Ejercicio / Entrenamiento. "
        "Martes 30/6 09:00 a 10:30 — Reunión de equipo. "
        "Miércoles 1/7 libre. "
        "Jueves 2/7 14:00 a 15:00 — Llamada con cliente. "
        "Viernes 3/7 19:00 a 20:30 — Cine con amigos. "
        "Sábado 4/7 10:00 a 13:00 — Estudio Teclab. "
        "Domingo 5/7 descanso."
    )

    def test_a_prod_long_agenda_multiple_chunks(self, kokoro_settings):
        """Test A — Texto largo del log de prod (~678 chars) → múltiples chunks.

        Verifica:
        - ``create`` fue llamada MÁS DE UNA VEZ (múltiples chunks).
        - Ningún chunk recibido excede ~1500 chars (safety net del fix).
        - El PCM resultante no es None y tiene bytes.
        - Cada llamada incluye ``trim=False``.
        """
        # Verificar longitud aproximada del fixture (~678 chars ± overhead)
        assert len(self._PROD_LONG_AGENDA) > 400, (
            f"Fixture demasiado corto ({len(self._PROD_LONG_AGENDA)} chars), "
            f"no ejercería el path de chunking"
        )

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                result = client.synthesize(self._PROD_LONG_AGENDA)

                # create fue llamada MÁS DE UNA VEZ
                assert mock_instance.create.call_count > 1, (
                    f"Se esperaban múltiples chunks, se obtuvo call_count="
                    f"{mock_instance.create.call_count}"
                )

                # Ningún chunk excede 1500 chars (safety net)
                received_chunks = [c.args[0] for c in mock_instance.create.call_args_list]
                for i, chunk in enumerate(received_chunks):
                    assert len(chunk) <= 1500, (
                        f"Chunk {i} excede 1500 chars: {len(chunk)}"
                    )

                # Resultado: bytes PCM no vacíos
                assert result is not None
                assert isinstance(result, bytes)
                assert len(result) > 0

                # Cada llamada tuvo trim=False
                for call in mock_instance.create.call_args_list:
                    assert call.kwargs.get("trim") is False

    def test_b_split_preserves_punctuation_signs(self, kokoro_settings):
        """Test B — Split por em-dash, en-dash, dos puntos y slash.

        Verifica que los chunks resultantes preservan los signos (lookbehind:
        el signo queda en el chunk anterior, no se pierde).
        """
        # Texto diseñado con cada signo del regex de split:
        # ``.,;:!?—–/``
        input_text = (
            "Lunes 29/6: 18:00–20:00 — Estudio Teclab. "
            "Martes 30/6: libre. "
            "Miércoles 1/7: ¿descanso? "
            "Jueves 2/7: 19:00/20:00 — cena rápida."
        )

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(input_text)

                received_chunks = [c.args[0] for c in mock_instance.create.call_args_list]
                reconstructed = " ".join(received_chunks)

                # Lookbehind: los signos quedan PEGADOS al chunk anterior.
                # Verificar que aparecen en algún chunk recibido (preservados):
                # - Dos puntos en "29/6:", "30/6:", etc.
                assert any("29/6:" in c for c in received_chunks), (
                    f"'29/6:' (con dos puntos) no aparece en ningún chunk. "
                    f"Chunks: {received_chunks}"
                )
                # - En-dash (U+2013) en "18:00–20:00"
                assert any("18:00\u201320:00" in c for c in received_chunks), (
                    f"En-dash no preservado. Chunks: {received_chunks}"
                )
                # - Em-dash (U+2014) en "— Estudio"
                assert any("\u2014" in c for c in received_chunks), (
                    f"Em-dash no preservado. Chunks: {received_chunks}"
                )
                # - Punto final "libre." queda en chunk anterior
                assert any("libre." in c for c in received_chunks), (
                    f"'libre.' (con punto) no preservado. Chunks: {received_chunks}"
                )
                # - Signo de interrogación "¿descanso?" queda en chunk anterior
                assert any("¿descanso?" in c for c in received_chunks), (
                    f"'¿descanso?' no preservado. Chunks: {received_chunks}"
                )
                # - Slash en "19:00/20:00"
                assert any("19:00/20:00" in c for c in received_chunks), (
                    f"'19:00/20:00' (con slash) no preservado. Chunks: {received_chunks}"
                )

                # Reconstrucción = texto normalizado (mismo contenido)
                expected_normalized = re.sub(r"\s+", " ", input_text).strip()
                reconstructed_normalized = re.sub(r"\s+", " ", reconstructed).strip()
                assert reconstructed_normalized == expected_normalized

                # create fue llamada MÁS DE UNA VEZ (texto tiene múltiples signos)
                assert mock_instance.create.call_count > 1

    def test_c_short_text_single_chunk(self, kokoro_settings):
        """Test C — Texto corto sin puntuación → 1 solo chunk, 1 sola llamada.

        Verifica que el split NO introduce chunks espurios para textos
        cortos SIN puntuación que dispare el split. La spec menciona
        "texto corto → 1 solo chunk"; esto ocurre cuando el texto no
        contiene signos seguidos de whitespace (los signos del set
        ``.,;:!?—–/`` solo generan split si están seguidos de un espacio).

        NOTA: un input con coma + espacio SÍ genera 2 chunks ("Hola mundo,"
        + "¿cómo estás?") — eso es el comportamiento correcto del lookbehind
        y se cubre en test_b_split_preserves_punctuation_signs.
        """
        # Texto SIN puntuación que dispare el split (sin comas/puntos/etc.
        # seguidos de espacio).
        input_text = "Hola mundo"

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(input_text)

                # create fue llamada EXACTAMENTE UNA VEZ
                assert mock_instance.create.call_count == 1
                received_text = mock_instance.create.call_args.args[0]

                # El chunk es el texto normalizado completo
                expected = re.sub(r"\s+", " ", input_text).strip()
                assert received_text == expected

    def test_d_safety_net_splits_long_chunk_without_punctuation(self, kokoro_settings):
        """Test D — Safety net: chunk > 1500 chars SIN puntuación → split por espacios.

        Construye un texto de ~2000 chars sin puntuación (solo palabras con
        espacios) y verifica que el safety net lo parte en ≥2 chunks, cada
        uno con len ≤ 1500.
        """
        # Texto de 2000 chars SIN puntuación (no aparece ``.,;:!?—–/``)
        # → el split principal no parte nada → safety net debe actuar.
        word = "palabra"
        input_text = (" ".join([word] * 400)).strip()  # 400 * 7 + 399 espacios ≈ 3199 chars
        # Confirmar que NO hay signos de split
        assert not re.search(r"[.,;:!?—–/]", input_text)
        # Confirmar que excede el safety net
        assert len(input_text) > 1500

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                client.synthesize(input_text)

                received_chunks = [c.args[0] for c in mock_instance.create.call_args_list]

                # Safety net partió el texto en múltiples chunks
                assert mock_instance.create.call_count > 1, (
                    f"Safety net no actuó. call_count={mock_instance.create.call_count}, "
                    f"len(texto)={len(input_text)}"
                )

                # Ningún chunk excede 1500 chars
                for i, chunk in enumerate(received_chunks):
                    assert len(chunk) <= 1500, (
                        f"Chunk {i} excede 1500 chars: {len(chunk)}"
                    )

                # Reconstrucción preserva todas las palabras (sin pérdida)
                reconstructed = " ".join(received_chunks)
                assert reconstructed.count(word) == input_text.count(word)

    def test_e_synthesize_stream_inherits_chunking(self, kokoro_settings):
        """Test E — ``synthesize_stream`` hereda el chunking de ``synthesize``.

        Con texto largo multilinea, ``synthesize_stream`` debe producir
        múltiples chunks de salida (yield) Y ``create`` debe haber sido
        llamada múltiples veces.
        """
        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=_fake_create)

                client = KokoroTTSClient(kokoro_settings)
                # synthesize_stream delega en synthesize → hereda el chunking
                chunks = list(client.synthesize_stream(self._PROD_LONG_AGENDA))

                # Hubo al menos 1 chunk de salida PCM
                assert len(chunks) >= 1
                # Y ``create`` se llamó múltiples veces (texto largo → N chunks)
                assert mock_instance.create.call_count > 1, (
                    f"synthesize_stream no heredó chunking. "
                    f"call_count={mock_instance.create.call_count}"
                )

                # Cada chunk de salida ≤ 4096 bytes (contrato de synthesize_stream)
                for chunk in chunks:
                    assert len(chunk) <= 4096

                # La concatenación de todos los chunks PCM == PCM total
                reconstructed_pcm = b"".join(chunks)
                # El mock _fake_create retorna 100 samples float32 → 200 bytes
                expected_total_bytes = (
                    mock_instance.create.call_count * 100 * 2  # 100 samples × 2 bytes
                )
                assert len(reconstructed_pcm) == expected_total_bytes

    def test_f_audio_concatenation_correct_length(self, kokoro_settings):
        """Test F — Concatenación de audio: distintos tamaños por chunk → suma correcta.

        Mockea ``create`` para que cada llamada retorne un array de longitud
        DISTINTA. Verifica que el PCM final tiene el tamaño exacto igual a
        la suma de las longitudes de todos los chunks (la concatenación
        con ``np.concatenate`` es correcta).
        """
        # Tamaños de cada "chunk" que retornará el mock (samples float32).
        chunk_sizes = [50, 120, 30, 200, 75]
        expected_total_samples = sum(chunk_sizes)
        expected_total_bytes = expected_total_samples * 2  # int16 = 2 bytes/sample

        def varying_create(text, voice, speed, lang, trim=None):
            """Retorna arrays de tamaños según el orden de llamada."""
            # Usar una lista mutable como contador de invocaciones
            if not hasattr(varying_create, "_counter"):
                varying_create._counter = 0
            idx = varying_create._counter
            varying_create._counter += 1
            size = chunk_sizes[idx % len(chunk_sizes)]
            samples = np.zeros(size, dtype=np.float32)
            return samples, 24000

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=varying_create)

                # Reset counter (por si pytest reusa el mock entre tests)
                varying_create._counter = 0

                client = KokoroTTSClient(kokoro_settings)
                # Texto corto que produce exactamente 1 chunk para aislar
                # el test de F (1 sola llamada → 1 array de 50 samples)
                result = client.synthesize("Hola mundo")

                # 1 sola llamada → 50 samples → 100 bytes
                assert mock_instance.create.call_count == 1
                assert len(result) == chunk_sizes[0] * 2

        # ── Test 2: texto largo que produce MÚLTIPLES llamadas → concatenación ──
        varying_create._counter = 0  # reset

        with patch("handlers.kokoro_tts_client.Path.exists", return_value=True):
            with patch("handlers.kokoro_tts_client.Kokoro") as MockKokoro:
                mock_instance = MockKokoro.return_value
                mock_instance.create = MagicMock(side_effect=varying_create)

                client = KokoroTTSClient(kokoro_settings)
                result = client.synthesize(self._PROD_LONG_AGENDA)

                # create fue llamada N veces (1 por chunk del split)
                assert mock_instance.create.call_count > 1
                n_calls = mock_instance.create.call_count

                # El PCM total debe ser EXACTAMENTE la suma de los tamaños
                # de los arrays retornados (round-robin sobre chunk_sizes).
                expected_bytes_for_n_calls = (
                    sum(chunk_sizes[: n_calls % len(chunk_sizes)])  # ciclo parcial
                    + sum(chunk_sizes) * (n_calls // len(chunk_sizes))  # ciclos completos
                ) * 2
                assert len(result) == expected_bytes_for_n_calls, (
                    f"Concatenación incorrecta. "
                    f"len(result)={len(result)}, esperado={expected_bytes_for_n_calls}, "
                    f"n_calls={n_calls}"
                )
