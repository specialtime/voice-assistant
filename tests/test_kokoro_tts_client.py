"""Tests unitarios para KokoroTTSClient.

Verifica síntesis de voz local con Kokoro-ONNX, lazy-load del modelo,
conversión float32→int16, manejo de errores, y que no se filtren
secretos en logs.

Todos los tests son ``@pytest.mark.unit`` — sin red, sin disco,
sin modelo real. Se mockea ``kokoro_onnx.Kokoro`` con ``unittest.mock``.
"""

from __future__ import annotations

import logging
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


def _fake_create(text, voice, speed, lang):
    """Retorna (samples float32, sample_rate 24000) como Kokoro real."""
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
        def fake_create_out_of_range(text, voice, speed, lang):
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
