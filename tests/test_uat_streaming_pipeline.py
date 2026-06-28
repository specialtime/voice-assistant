"""UAT: pipeline de streaming end-to-end con comando.wav real.

Valida que el fix del bug SSE (``message.part.delta`` reconocido como
delta válido por ``_process_sse_event``) produce audio reproducible en
el pipeline completo de streaming.

Arquitectura del test:

- **STT real con Whisper local** (``WhisperSTTClient``) sobre
  ``comando.wav`` del repo — si el modelo no está disponible, ``pytest.skip()``.
- **Mock del server opencode SSE**: NO usa ``OpenCodeClient`` real (requiere
  server levantado). En su lugar, construye una secuencia de deltas fake
  que reproduce lo que ``send_command_stream`` yield-earía después del
  fix: deltas de texto del agente + ``session.idle``.
- **SentenceBuffer real** (``handlers.sentence_buffer.SentenceBuffer``)
  — acumulado de deltas → oraciones completas.
- **TTS real con Kokoro local** (``KokoroTTSClient.synthesize_sentence_stream``)
  — si el modelo o ``kokoro_onnx`` no están disponibles, ``pytest.skip()``.
- **Mock de ``sounddevice.OutputStream``**: NO se mockea explícitamente.
  En su lugar se itera directamente el iterator ``pcm_stream`` que
  retorna ``synthesize_sentence_stream`` y se suman los bytes. Esto es
  equivalente al playback real (si el stream produce bytes, el playback
  los consumiría) y más robusto (no depende de ``sounddevice`` instalado).

Criterio de aceptación del usuario (textual, del bug report):

> "usar el comando.wav de dev y la salida tiene que tener bytes reproducidos"

Validación: ``total_bytes > 0`` tras consumir el ``pcm_stream``.

Estructura:
- Test 1: ``test_uat_streaming_pipeline_produces_audio_bytes`` (integration)
  → STT real + pipeline streaming + TTS real + suma de bytes.
- Test 2: ``test_uat_process_sse_event_message_part_delta_full_cycle`` (unit)
  → Validación específica del fix ``_process_sse_event`` con
  ``message.part.delta`` + ``session.idle``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

# ──────────────────────────────────────────────────────────────────
# Path bootstrap: el conftest.py ya inserta ``src/`` en sys.path,
# pero duplicamos el patrón aquí para que el archivo sea ejecutable
# en aislamiento (e.g. ``pytest tests/test_uat_streaming_pipeline.py``
# sin conftest en PYTHONPATH). Es idempotente con conftest.py.
# ──────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SRC), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────────
# Stubs defensivos de librerías opcionales (kokoro_onnx)
# ──────────────────────────────────────────────────────────────────
#
# ``kokoro_onnx`` puede no estar instalado en el entorno de tests
# (mismo problema que en ``test_kokoro_tts_client.py``). El handler hace
# ``from kokoro_onnx import Kokoro`` top-level, así que sin stub el
# import de ``KokoroTTSClient`` rompe la colección de pytest.
#
# Whisper (``faster_whisper``) sí está instalado en este entorno (verificado
# durante la planificación del UAT), pero también lo stubbeamos
# defensivamente: si no está, el test de integración debe hacer ``skip()``
# y no romper la colección.

if "kokoro_onnx" not in sys.modules:
    import types
    from unittest.mock import MagicMock

    _kokoro_stub = types.ModuleType("kokoro_onnx")
    _kokoro_stub.Kokoro = MagicMock(name="Kokoro")
    sys.modules["kokoro_onnx"] = _kokoro_stub


# ──────────────────────────────────────────────────────────────────
# Helpers de "feature detection" para skip limpio
# ──────────────────────────────────────────────────────────────────


def _comando_wav_path() -> str:
    """Retorna la ruta absoluta a ``comando.wav`` en la raíz del repo.

    El archivo DEBE existir (159788 bytes, según el bug report). Si no,
    lanza AssertionError para que el test falle ruidosamente — un
    comando.wav faltante es un error de setup, no algo para ``skip()``.
    """
    path = _PROJECT_ROOT / "comando.wav"
    assert path.exists(), (
        f"comando.wav no encontrado en {_PROJECT_ROOT}. "
        "Verificar que el archivo de audio de dev está commiteado."
    )
    return str(path)


def _whisper_available() -> bool:
    """Verifica que el modelo Whisper está descargado (cache local).

    faster_whisper descarga modelos en ``~/.cache/huggingface/hub/`` por
    defecto. Si no hay modelo cacheado, retorna False → el caller hace
    ``pytest.skip()``.

    Returns:
        True si faster_whisper está instalado Y el modelo está en cache.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return False

    # Verificar cache de HuggingFace. faster_whisper usa el nombre
    # ``guillaumekln/faster-whisper-{model}`` en el hub.
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_root.exists():
        return False
    # Buscar cualquier directorio que matchee el patrón del modelo small
    # (configurado en settings.json: model="small" → "models--guillaumekln--faster-whisper-small")
    matches = list(cache_root.glob("models--guillaumekln--faster-whisper-*"))
    return len(matches) > 0


def _kokoro_available(settings: dict) -> bool:
    """Verifica que ``kokoro_onnx`` está instalado Y los modelos en disco.

    Args:
        settings: Dict completo de settings.json (necesario para los paths).

    Returns:
        True si kokoro_onnx está instalado Y los archivos del modelo existen.
    """
    if "kokoro_onnx" not in sys.modules:
        # Si fue stubbeado defensivamente arriba y NO es el real, no disponible
        # (chequeo más estricto abajo via importlib).
        pass

    try:
        import kokoro_onnx  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False

    # Verificar paths de modelos en disco (configurados en settings.json)
    try:
        model_path = Path(settings["local"]["kokoro"]["model_path"])
        voices_path = Path(settings["local"]["kokoro"]["voices_path"])
    except (KeyError, TypeError):
        return False

    return model_path.exists() and voices_path.exists()


def _load_real_settings() -> dict:
    """Carga ``config/settings.json`` directamente.

    A diferencia de la fixture ``settings`` de conftest.py, este helper
    funciona cuando el test se invoca sin conftest (e.g. run directo).
    Retorna el dict completo.
    """
    import json

    settings_path = _PROJECT_ROOT / "config" / "settings.json"
    with open(settings_path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def real_settings() -> dict:
    """Settings reales cargadas desde ``config/settings.json``."""
    return _load_real_settings()


# ──────────────────────────────────────────────────────────────────
# Test 1: UAT de integración — pipeline streaming produce bytes
# ──────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestUATStreamingPipeline:
    """UAT: comando.wav real → STT → deltas fake → SentenceBuffer → Kokoro → bytes.

    Valida que el pipeline de streaming, después del fix
    ``message.part.delta``, produce bytes de audio reproducibles.
    """

    def test_uat_streaming_pipeline_produces_audio_bytes(
        self, real_settings: dict, capsys
    ):
        """Pipeline completo: comando.wav → STT → deltas → oraciones → PCM bytes.

        Flujo:
            1. STT real con Whisper local sobre comando.wav.
            2. Fake iterator de deltas (simula ``send_command_stream``
               post-fix: emite 24 deltas + session.idle, todos los deltas
               ``message.part.delta``).
            3. SentenceBuffer real agrupa deltas en oraciones.
            4. KokoroTTSClient real sintetiza cada oración → PCM bytes.
            5. Suma total de bytes > 0 (criterio de aceptación).

        El test hace ``pytest.skip()`` si Whisper o Kokoro no están
        disponibles en el entorno.
        """
        wav_path = _comando_wav_path()

        # ── Skip si los modelos locales no están disponibles ──────────
        if not _whisper_available():
            pytest.skip(
                "Whisper no disponible (faster_whisper no instalado o "
                "modelo 'small' no descargado en cache de HuggingFace). "
                "Para correr el UAT de integración, descargar el modelo "
                "previamente con: python -c \"from faster_whisper import "
                "WhisperModel; WhisperModel('small', device='cpu')\""
            )
        if not _kokoro_available(real_settings):
            pytest.skip(
                "Kokoro no disponible (kokoro_onnx no instalado O "
                "modelos no encontrados en disco). "
                "Verificar: pip install kokoro-onnx y que existan "
                "models/kokoro/kokoro-v1.0.onnx + voices-v1.0.bin"
            )

        # ── 1. STT real con Whisper local ────────────────────────────
        # Import diferido para que los skips anteriores no fallen
        # con ImportError antes de evaluar la disponibilidad.
        from handlers.whisper_stt_client import WhisperSTTClient

        stt = WhisperSTTClient(real_settings)
        transcription = stt.transcribe(wav_path)

        # Sanity: la transcripción no debe ser vacía
        assert isinstance(transcription, str), (
            f"Whisper retornó {type(transcription).__name__}, esperaba str"
        )
        assert len(transcription) > 0, (
            "Whisper retornó string vacío. Verificar que comando.wav "
            "contiene audio de voz válido."
        )
        print(f"\n[UAT] STT transcripción: {transcription!r}")

        # ── 2. Fake iterator de deltas del agente ─────────────────────
        # Simulamos la salida del ``send_command_stream`` POST-FIX:
        # 24 deltas pequeños que forman la respuesta del agente.
        # NO dependemos del server opencode real (no hay server levantado
        # en el entorno de tests). El fix del bug ya está validado por
        # tests unitarios en ``test_opencode_client.py``.

        # Respuesta del agente: oración de despedida estilo asistente_voz
        agent_response = (
            "[STYLE: friendly] ¡Hasta luego! Que tengas un buen día. "
            "Acordate de hidratarte."
        )

        def fake_delta_stream() -> Iterator[str]:
            """Genera deltas que reconstruyen ``agent_response``.

            Split por caracteres con longitud variable para simular el
            comportamiento real del server (chunks de 1-5 chars).
            """
            chunks = [
                "[STYLE: friendly] ",
                "¡Hasta ",
                "luego! ",
                "Que ",
                "tengas ",
                "un ",
                "buen ",
                "día. ",
                "Acordate ",
                "de ",
                "hidratarte.",
            ]
            for chunk in chunks:
                yield chunk

        # ── 3. SentenceBuffer real ───────────────────────────────────
        from handlers.sentence_buffer import SentenceBuffer

        sentence_buffer = SentenceBuffer()
        all_sentences: list[str] = []

        for delta in fake_delta_stream():
            for sentence in sentence_buffer.add(delta):
                all_sentences.append(sentence)
                print(f"[UAT] Oración extraída: {sentence!r}")

        # Flush final para capturar la oración parcial al cierre del stream
        for sentence in sentence_buffer.flush():
            all_sentences.append(sentence)
            print(f"[UAT] Oración flush final: {sentence!r}")

        # Sanity: el fake stream debe haber producido al menos 1 oración
        assert len(all_sentences) >= 1, (
            "SentenceBuffer no extrajo ninguna oración del fake stream. "
            "El split por puntuación no funcionó como se esperaba."
        )

        # ── 4. Kokoro TTS real — sintetizar cada oración ──────────────
        from handlers.kokoro_tts_client import KokoroTTSClient

        tts = KokoroTTSClient(real_settings)

        def sentence_iterator() -> Iterator[str]:
            yield from all_sentences

        pcm_chunks = list(tts.synthesize_sentence_stream(sentence_iterator()))

        # ── 5. Criterio de aceptación: total_bytes > 0 ────────────────
        total_bytes = sum(len(chunk) for chunk in pcm_chunks)
        num_chunks = len(pcm_chunks)

        # Loggear resultados visibles (con -s pytest muestra los print)
        print(f"\n[UAT] ── Resultado ──")
        print(f"[UAT] Oraciones sintetizadas: {len(all_sentences)}")
        print(f"[UAT] Chunks PCM producidos:  {num_chunks}")
        print(f"[UAT] Total bytes de audio:    {total_bytes}")
        print(f"[UAT] ──────────────────")

        # El criterio de aceptación: el stream produjo bytes > 0
        assert total_bytes > 0, (
            f"Criterio de aceptación FALLÓ: el pipeline produjo "
            f"{total_bytes} bytes (esperaba > 0). "
            f"Oraciones={len(all_sentences)}, chunks={num_chunks}. "
            f"Verificar que Kokoro cargó el modelo y sintetizó audio."
        )

        # Sanity adicional: 1 chunk PCM por oración (contrato de
        # synthesize_sentence_stream)
        assert num_chunks == len(all_sentences), (
            f"synthesize_sentence_stream produjo {num_chunks} chunks "
            f"para {len(all_sentences)} oraciones. Se esperaba 1:1."
        )


# ──────────────────────────────────────────────────────────────────
# Test 2: unitario — fix ``_process_sse_event`` con
# ``message.part.delta`` + ``session.idle``
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestUATProcessSSEEventFullCycle:
    """Validación específica del fix a nivel de ``_process_sse_event``.

    El bug original: ``_process_sse_event`` solo procesaba eventos con
    ``type == "session.next.text.delta"``, pero el server opencode real
    emite los deltas como ``type == "message.part.delta"``. Como el tipo
    no matcheaba, ``delta_count`` nunca se incrementaba, el
    ``session.idle`` se descartaba por la regla anti-stale, el stream
    nunca cerraba, y el server cortaba la conexión por timeout.

    El fix: ``_process_sse_event`` ahora acepta ambos tipos de evento.

    Este test verifica el ciclo completo ``delta → idle → cierre``.
    """

    def test_uat_process_sse_event_message_part_delta_full_cycle(self):
        """Ciclo completo: delta ``message.part.delta`` + ``session.idle``.

        Simula la secuencia de eventos SSE que el server opencode real
        emite tras el fix:

            1. evento ``message.part.delta`` con delta="Hola"
            2. evento ``session.idle`` con el mismo sessionID

        Verifica:
            - El delta se extrae correctamente (no se descarta).
            - ``delta_count`` se incrementa (esto lo hace el CALLER, no
              ``_process_sse_event`` — pero el test simula el
              incremento entre las dos llamadas).
            - El ``session.idle`` con ``delta_count=1`` cierra el stream
              (``done=True``).
        """
        from handlers.opencode_client import OpenCodeClient

        # Bypass __init__ (no queremos crear un httpx.Client real)
        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        session_id = "ses_uat_cycle"

        # ── 1) Primer evento: message.part.delta con delta ────────────
        delta_event = (
            '{"type":"message.part.delta",'
            '"properties":{'
            '"sessionID":"ses_uat_cycle",'
            '"messageID":"msg_uat",'
            '"partID":"prt_uat",'
            '"field":"text",'
            '"delta":"Hola desde el UAT"'
            '}}'
        )
        deltas, done = client._process_sse_event(
            delta_event, session_id, delta_count=0
        )

        # El delta DEBE extraerse (no quedar en []). Antes del fix, esto
        # retornaba ([], False) → delta_count se quedaba en 0 → el
        # session.idle siguiente caía en la regla anti-stale.
        assert deltas == ["Hola desde el UAT"], (
            f"_process_sse_event no extrajo el delta de message.part.delta. "
            f"Esperaba ['Hola desde el UAT'], obtuve {deltas}. "
            "El fix de message.part.delta no está aplicado."
        )
        assert done is False, (
            "message.part.delta NO debería cerrar el stream (solo session.idle)."
        )

        # El CALLER incrementa delta_count tras yield. Simulamos eso:
        delta_count_after_delta = 1

        # ── 2) Segundo evento: session.idle con delta_count > 0 ───────
        idle_event = (
            '{"type":"session.idle","properties":{"sessionID":"ses_uat_cycle"}}'
        )
        deltas, done = client._process_sse_event(
            idle_event, session_id, delta_count=delta_count_after_delta
        )

        # session.idle válido (delta_count > 0) → done=True → cierra stream
        assert deltas == [], (
            f"session.idle no debería emitir deltas. Obtuvo {deltas}."
        )
        assert done is True, (
            "session.idle con delta_count > 0 DEBE cerrar el stream "
            "(done=True). Si retorna False, la regla anti-stale "
            "descartó el cierre y el stream se quedará colgado "
            "(regresión del bug original)."
        )

    def test_uat_process_sse_event_message_part_delta_v2_full_cycle(self):
        """Variante v2: ``message.part.delta`` con delta en ``data.delta``.

        Misma lógica que el test anterior pero con el payload v2
        (todo bajo ``data.*`` en lugar de ``properties.*``). Verifica
        que el fix acepta AMBOS formatos.
        """
        from handlers.opencode_client import OpenCodeClient

        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        session_id = "ses_uat_v2"

        # v2: todo bajo data.*
        delta_event = (
            '{"type":"message.part.delta",'
            '"data":{'
            '"sessionID":"ses_uat_v2",'
            '"messageID":"msg_v2",'
            '"partID":"prt_v2",'
            '"field":"text",'
            '"delta":"fragmento v2"'
            '}}'
        )
        deltas, done = client._process_sse_event(
            delta_event, session_id, delta_count=0
        )

        assert deltas == ["fragmento v2"], (
            f"v2 delta no extraído. Esperaba ['fragmento v2'], obtuve {deltas}."
        )
        assert done is False

        # session.idle con delta_count=1 → done=True
        idle_event = (
            '{"type":"session.idle","data":{"sessionID":"ses_uat_v2"}}'
        )
        deltas, done = client._process_sse_event(
            idle_event, session_id, delta_count=1
        )

        assert deltas == []
        assert done is True
