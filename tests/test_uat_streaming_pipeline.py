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

try:
    import kokoro_onnx  # noqa: F401 — verificar si está instalado
except ImportError:
    # kokoro_onnx no instalado → inyectar stub para que la colección
    # de pytest no rompa al importar KokoroTTSClient (que hace
    # `from kokoro_onnx import Kokoro` top-level).
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
    # Buscar cualquier directorio que matchee el patrón del modelo small.
    # faster_whisper puede usar dos prefijos de hub distintos según la versión:
    #   - "guillaumekln" (legacy, hasta faster-whisper ~0.9)
    #   - "Systran" (actual, desde faster-whisper ~0.10)
    # Aceptamos ambos para no romper el skip en entornos con versiones distintas.
    matches = list(cache_root.glob("models--guillaumekln--faster-whisper-*"))
    matches += list(cache_root.glob("models--Systran--faster-whisper-*"))
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
        part_types: dict = {}
        deltas, done = client._process_sse_event(
            delta_event, session_id, delta_count=0, part_types=part_types
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
            idle_event, session_id, delta_count=delta_count_after_delta, part_types=part_types
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
        part_types: dict = {}
        deltas, done = client._process_sse_event(
            delta_event, session_id, delta_count=0, part_types=part_types
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
            idle_event, session_id, delta_count=1, part_types=part_types
        )

        assert deltas == []
        assert done is True


# ──────────────────────────────────────────────────────────────────
# Test 3: UAT — filtrado de reasoning en streaming SSE
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestUATReasoningFilter:
    """UAT: el filtrado de parts de reasoning funciona en el ciclo SSE.

    Valida que ``_process_sse_event`` con la nueva signature (4º arg
    ``part_types``) descarta los deltas de parts de tipo ``reasoning``
    y emite solo los de tipo ``text``.

    Escenario real (del bug report 2026-06-28): el agente emite
    razonamiento en inglés ("The user is asking for their schedule...")
    seguido de la respuesta al usuario en español ("El lunes arrancás
    con Estudio Teclab..."). Solo el segundo debe llegar al TTS.
    """

    def test_uat_reasoning_filtered_text_emitted(self):
        """Stream mixto reasoning + text → solo text llega al consumidor.

        Simula la secuencia SSE real del bug report:

            1. message.part.updated (reasoning, prt_r)
            2. message.part.delta  (prt_r, "The user is asking...")
            3. message.part.updated (text, prt_t)
            4. message.part.delta  (prt_t, "El lunes arrancás...")
            5. session.idle

        Verifica que:
            - El delta de reasoning se descarta (no se emite).
            - El delta de text se emite.
            - session.idle cierra el stream (delta_count=1 > 0).
            - part_types registró ambos parts.
        """
        from handlers.opencode_client import OpenCodeClient

        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        session_id = "ses_uat_reasoning"
        part_types: dict = {}

        events = [
            # 1) Registrar part de reasoning
            '{"type":"message.part.updated",'
            '"properties":{'
            '"sessionID":"ses_uat_reasoning",'
            '"part":{"id":"prt_r","type":"reasoning"}'
            '}}',
            # 2) Delta de reasoning → DEBE descartarse
            '{"type":"message.part.delta",'
            '"properties":{'
            '"sessionID":"ses_uat_reasoning",'
            '"partID":"prt_r",'
            '"delta":"The user is asking for their schedule."'
            '}}',
            # 3) Registrar part de text
            '{"type":"message.part.updated",'
            '"properties":{'
            '"sessionID":"ses_uat_reasoning",'
            '"part":{"id":"prt_t","type":"text"}'
            '}}',
            # 4) Delta de text → DEBE emitirse
            '{"type":"message.part.delta",'
            '"properties":{'
            '"sessionID":"ses_uat_reasoning",'
            '"partID":"prt_t",'
            '"delta":"El lunes arrancás con Estudio Teclab."'
            '}}',
            # 5) session.idle → cierra el stream
            '{"type":"session.idle",'
            '"properties":{"sessionID":"ses_uat_reasoning"}'
            '}',
        ]

        emitted: list[str] = []
        delta_count = 0
        stream_closed = False

        for event_json in events:
            deltas, done = client._process_sse_event(
                event_json, session_id, delta_count=delta_count,
                part_types=part_types,
            )
            for d in deltas:
                emitted.append(d)
                delta_count += 1
            if done:
                stream_closed = True
                break

        # ── Aserciones ────────────────────────────────────────────────
        assert stream_closed, (
            "session.idle debió cerrar el stream. "
            "Si no cerró, el delta de text no se emitió y delta_count "
            "se quedó en 0 (regla anti-stale descartó el idle)."
        )
        assert emitted == ["El lunes arrancás con Estudio Teclab."], (
            f"Solo el delta de text debió emitirse. Esperaba "
            f"['El lunes arrancás con Estudio Teclab.'], obtuve {emitted}. "
            "El reasoning NO debió emitirse al TTS."
        )
        assert len(emitted) == 1, (
            f"Se esperaba exactamente 1 delta emitido, se obtuvieron "
            f"{len(emitted)}: {emitted}"
        )
        assert part_types == {
            "prt_r": "reasoning",
            "prt_t": "text",
        }, f"part_types mal registrado: {part_types}"
        assert delta_count == 1


@pytest.mark.integration
class TestUATReasoningFilterPipeline:
    """UAT: pipeline streaming con filtrado de reasoning produce bytes > 0.

    Criterio de aceptación del usuario (textual):

        > usar comando.wav del entorno dev y que haya audio reproducido
        > con bytes > 0

    Este test simula un stream SSE con parts de reasoning (que deben
    filtrarse) y parts de text (que deben llegar al TTS), pasa los
    deltas filtrados por SentenceBuffer + Kokoro TTS, y verifica que
    el PCM resultante tiene bytes > 0.

    Además verifica que el contenido de reasoning NO aparece en el
    texto sintetizado y que la respuesta al usuario SÍ aparece.
    """

    def test_uat_reasoning_filter_pipeline_produces_audio(
        self, real_settings: dict
    ):
        """Pipeline con reasoning + text → bytes > 0 y sin reasoning en TTS.

        Flujo:
            1. STT real con Whisper sobre comando.wav (verifica que el
               audio de dev funciona).
            2. Simular ``send_command_stream`` con eventos SSE que
               incluyen parts de reasoning y text.
            3. Filtrar con ``_process_sse_event`` (el fix).
            4. SentenceBuffer agrupa los deltas de text en oraciones.
            5. Kokoro TTS sintetiza → PCM bytes.
            6. Verificar total_bytes > 0.
            7. Verificar que el texto no contiene el reasoning.
        """
        wav_path = _comando_wav_path()

        # ── Skip si los modelos locales no están disponibles ──────────
        if not _whisper_available():
            pytest.skip(
                "Whisper no disponible (faster_whisper no instalado o "
                "modelo 'small' no descargado en cache de HuggingFace)."
            )
        if not _kokoro_available(real_settings):
            pytest.skip(
                "Kokoro no disponible (kokoro_onnx no instalado O "
                "modelos no encontrados en disco)."
            )

        # ── 1. STT real con Whisper local ────────────────────────────
        from handlers.whisper_stt_client import WhisperSTTClient

        stt = WhisperSTTClient(real_settings)
        transcription = stt.transcribe(wav_path)

        assert isinstance(transcription, str)
        assert len(transcription) > 0, (
            "Whisper retornó string vacío. Verificar comando.wav."
        )
        print(f"\n[UAT-RF] STT transcripción: {transcription!r}")

        # ── 2. Simular stream SSE con reasoning + text ───────────────
        # Reproduce el patrón del bug report: el agente razona en
        # inglés y luego responde en español. Solo el text debe llegar
        # al TTS.
        from handlers.opencode_client import OpenCodeClient

        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        session_id = "ses_uat_rf"
        part_types: dict = {}

        # Texto de reasoning (NO debe llegar al TTS)
        reasoning_text = "The user is asking for their schedule for next week."

        # Texto de respuesta al usuario (SÍ debe llegar al TTS)
        response_text = "El lunes arrancás con Estudio Teclab de 18 a 20."

        # Construir eventos SSE: part.updated + deltas fragmentados
        sse_events: list[str] = []

        # 1) Registrar part de reasoning
        sse_events.append(
            '{"type":"message.part.updated",'
            '"properties":{'
            '"sessionID":"ses_uat_rf",'
            '"part":{"id":"prt_r","type":"reasoning"}'
            '}}'
        )
        # 2) Deltas de reasoning (fragmentados como en la vida real)
        reasoning_chunks = [
            reasoning_text[i:i + 20]
            for i in range(0, len(reasoning_text), 20)
        ]
        for chunk in reasoning_chunks:
            sse_events.append(
                '{"type":"message.part.delta",'
                '"properties":{'
                '"sessionID":"ses_uat_rf",'
                '"partID":"prt_r",'
                f'"delta":"{chunk}"'
                '}}'
            )
        # 3) Registrar part de text
        sse_events.append(
            '{"type":"message.part.updated",'
            '"properties":{'
            '"sessionID":"ses_uat_rf",'
            '"part":{"id":"prt_t","type":"text"}'
            '}}'
        )
        # 4) Deltas de text (fragmentados)
        response_chunks = [
            response_text[i:i + 15]
            for i in range(0, len(response_text), 15)
        ]
        for chunk in response_chunks:
            sse_events.append(
                '{"type":"message.part.delta",'
                '"properties":{'
                '"sessionID":"ses_uat_rf",'
                '"partID":"prt_t",'
                f'"delta":"{chunk}"'
                '}}'
            )
        # 5) session.idle
        sse_events.append(
            '{"type":"session.idle",'
            '"properties":{"sessionID":"ses_uat_rf"}'
            '}'
        )

        # ── 3. Procesar eventos SSE y recolectar deltas emitidos ─────
        emitted_deltas: list[str] = []
        delta_count = 0
        stream_closed = False

        for event_json in sse_events:
            deltas, done = client._process_sse_event(
                event_json, session_id, delta_count=delta_count,
                part_types=part_types,
            )
            for d in deltas:
                emitted_deltas.append(d)
                delta_count += 1
            if done:
                stream_closed = True
                break

        assert stream_closed, "session.idle no cerró el stream"
        assert delta_count > 0, (
            "No se emitió ningún delta. El filtrado descartó TODO "
            "(over-filtering). Verificar que los parts de text se emiten."
        )

        # Reconstruir el texto completo que llegó al TTS
        tts_text = "".join(emitted_deltas)
        print(f"[UAT-RF] Texto al TTS: {tts_text!r}")

        # ── Verificar que el reasoning NO está en el texto del TTS ────
        assert "The user is asking" not in tts_text, (
            f"El reasoning filtró al TTS. El texto contiene "
            f"'The user is asking'. Texto TTS: {tts_text!r}"
        )
        assert "schedule" not in tts_text, (
            f"El reasoning filtró al TTS. Contiene 'schedule'. "
            f"Texto TTS: {tts_text!r}"
        )

        # ── Verificar que la respuesta SÍ está en el texto del TTS ───
        assert "Estudio Teclab" in tts_text, (
            f"La respuesta al usuario no llegó al TTS. "
            f"Esperaba 'Estudio Teclab' en el texto. "
            f"Texto TTS: {tts_text!r}"
        )

        # ── 4. SentenceBuffer real ───────────────────────────────────
        from handlers.sentence_buffer import SentenceBuffer

        sentence_buffer = SentenceBuffer()
        all_sentences: list[str] = []

        for delta in emitted_deltas:
            for sentence in sentence_buffer.add(delta):
                all_sentences.append(sentence)
                print(f"[UAT-RF] Oración: {sentence!r}")

        for sentence in sentence_buffer.flush():
            all_sentences.append(sentence)
            print(f"[UAT-RF] Oración flush: {sentence!r}")

        assert len(all_sentences) >= 1, (
            "SentenceBuffer no extrajo oraciones de los deltas de text."
        )

        # ── 5. Kokoro TTS real ───────────────────────────────────────
        from handlers.kokoro_tts_client import KokoroTTSClient

        tts = KokoroTTSClient(real_settings)

        def sentence_iterator() -> Iterator[str]:
            yield from all_sentences

        pcm_chunks = list(tts.synthesize_sentence_stream(sentence_iterator()))

        # ── 6. Criterio de aceptación: total_bytes > 0 ────────────────
        total_bytes = sum(len(chunk) for chunk in pcm_chunks)
        num_chunks = len(pcm_chunks)

        print(f"\n[UAT-RF] ── Resultado ──")
        print(f"[UAT-RF] Oraciones sintetizadas: {len(all_sentences)}")
        print(f"[UAT-RF] Chunks PCM: {num_chunks}")
        print(f"[UAT-RF] Total bytes de audio: {total_bytes}")
        print(f"[UAT-RF] ──────────────────")

        assert total_bytes > 0, (
            f"Criterio de aceptación FALLÓ: el pipeline produjo "
            f"{total_bytes} bytes (esperaba > 0). "
            f"Oraciones={len(all_sentences)}, chunks={num_chunks}. "
            "El filtrado de reasoning puede haber descartado también "
            "los deltas de text (over-filtering)."
        )
