"""Tests unitarios focalizados en el wrapper ``pcm_stream_with_speaking_transition``.

Contexto del fix (commit ``fe33e29`` — ``fix/overlay-speaking-premature``)
==========================================================================
El bug original era que en el path streaming dentro de ``run_pipeline``
(``src/main.py`` ~líneas 230-269), la transición a ``STATE_SPEAKING`` +
``overlay.set_state("speaking")`` se disparaba **antes** de que existiera
audio real: en el momento en que se obtenía el iterador lazy de
``send_command_stream()``, no cuando el primer chunk PCM real estaba
disponible. Resultado: el overlay quedaba en "Hablando..." (verde)
durante la fase de STT + agente + buffering + síntesis, que puede
durar varios segundos, y "Procesando..." (amarillo) era invisible.

El fix introdujo un generador wrapper local
``pcm_stream_with_speaking_transition()`` que:

1. Itera el ``pcm_stream`` original.
2. En el primer chunk **real** (no vacío), adquiere ``self._lock``,
   chequea cancelación, y si OK setea ``self._state = STATE_SPEAKING`` +
   ``self._overlay.set_state("speaking")`` + emite log
   "→ SPEAKING (gen=%d, primer PCM real)".
3. Hace ``yield`` de cada chunk (incluido el primero).

Este archivo contiene los 4 casos del contrato del wrapper:

- **Caso 1**: happy path — transición al primer chunk real.
- **Caso 2**: cancelación antes del primer chunk — sin transición.
- **Caso 3**: chunks vacíos iniciales — transición solo al primer no-vacío.
- **Caso 4**: path síncrono intacto — transición ocurre ANTES del TTS.

Estrategia de testing
=====================
El wrapper es un closure local dentro de ``run_pipeline``. No es un símbolo
importable, por lo que lo testeamos **a través del orquestador**, controlando:

- ``synthesize_sentence_stream`` → inyectamos un iter de chunks PCM fake.
- ``play_audio_stream`` → reemplazamos su ``side_effect`` para que ejecute
  el generador pasado (no sea MagicMock que ignora argumentos). Esto fuerza
  la iteración real del wrapper, que es lo único que dispara la transición.

Como ``play_audio_stream`` en otros tests es MagicMock (no-op), el wrapper
nunca se itera y la transición nunca se observa en la suite previa. Estos
tests llenan ese hueco de cobertura.
"""

import logging
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Asegurar que la raíz del proyecto está en sys.path (por si conftest no se ejecutó)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# Stub de kokoro_onnx — registrado a nivel de módulo (idempotente).
# Mismo patrón que ``tests/test_state_machine.py`` y
# ``tests/test_kokoro_tts_client.py``.
# ──────────────────────────────────────────────────────────────────
if "kokoro_onnx" not in sys.modules:
    _kokoro_stub = types.ModuleType("kokoro_onnx")
    _kokoro_stub.Kokoro = MagicMock(name="Kokoro")
    sys.modules["kokoro_onnx"] = _kokoro_stub


def _drain_stream(stream):
    """Consume un generador/iterador y retorna la lista de chunks emitidos.

    Usado por los tests para forzar la iteración real del wrapper
    ``pcm_stream_with_speaking_transition`` y poder observar sus efectos
    sobre ``_state`` y ``_overlay``. Sin este consumo explícito,
    ``play_audio_stream`` (mockeado) ignoraría el generador.
    """
    return list(stream)


# ──────────────────────────────────────────────────────────────────
# Helpers de patching compartidos por los 4 tests
# ──────────────────────────────────────────────────────────────────
def _build_assistant(env_keys, mock_settings, mock_overlay, monkeypatch, streaming_enabled):
    """Construye un ``VoiceAssistant`` con todas las dependencias mockeadas.

    Idéntico al patrón de ``test_state_machine.py::patched_assistant``
    excepto que permite elegir el valor de ``_streaming_enabled`` antes
    de retornar la instancia.
    """
    monkeypatch.chdir(_PROJECT_ROOT)

    with patch("main.AzureTTSClient"), \
         patch("main.GeminiTTSClient"), \
         patch("main.OpenCodeClient"), \
         patch("main.GeminiSTTClient"), \
         patch("main.WhisperSTTClient"), \
         patch("main.PiperTTSClient"), \
         patch("main.KokoroTTSClient"), \
         patch("main.AudioManager"), \
         patch("main.load_dotenv"):

        from main import VoiceAssistant

        assistant = VoiceAssistant()
        assistant._settings = mock_settings
        assistant._streaming_enabled = streaming_enabled
        return assistant


@pytest.fixture
def env_keys(monkeypatch):
    """Setea las env vars necesarias para que el constructor cree los 4 clientes."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "fake_azure_key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "southamericaeast")
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "fake_opencode_pass")
    monkeypatch.setenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")


@pytest.mark.unit
class TestSpeakingTransitionWrapper:
    """Verifica el contrato del wrapper ``pcm_stream_with_speaking_transition``.

    Cada test configura ``_local_tts.synthesize_sentence_stream`` y
    ``_audio.play_audio_stream`` para forzar la iteración real del wrapper
    y observar la transición de estado.
    """

    # ── Caso 1: happy path ──────────────────────────────────────

    def test_wrapper_triggers_speaking_only_on_first_real_chunk(
        self, env_keys, mock_settings, mock_overlay, monkeypatch
    ):
        """Caso 1 (happy path): el wrapper dispara la transición a
        ``STATE_SPEAKING`` + ``overlay.set_state("speaking")`` **solo al
        primer chunk PCM real**, no antes, y no la repite en chunks
        subsiguientes.

        Setup:
            - 3 oraciones con puntuación final (3 yields PCM reales).
            - ``play_audio_stream`` consume explícitamente el wrapper pasado.

        Post-fix c115c48: el código ya no llama a ``synthesize_sentence_stream``
        sino al helper de fallback, que invoca ``_local_tts.synthesize`` por
        oración. Mockeamos ``synthesize`` con ``side_effect`` que retorna un
        PCM distinto por oración (3 invocaciones → 3 chunks downstream).
        """
        assistant = _build_assistant(
            env_keys, mock_settings, mock_overlay, monkeypatch, streaming_enabled=True
        )
        # Estado inicial: PROCESSING (asumimos que toggle() ya pasó por IDLE→RECORDING→PROCESSING)
        assistant._state = assistant.STATE_PROCESSING

        # STT OK
        assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # OpenCode streaming: deltas sintéticos con 3 oraciones terminadas en "."
        def delta_iter():
            yield "[STYLE: cheerful] Hola. "
            yield "Mundo. "
            yield "Chau. "
        assistant._opencode.send_command_stream.return_value = delta_iter()

        # Post-fix: el helper llama ``_local_tts.synthesize`` por oración.
        # 3 oraciones → 3 invocaciones → 3 yields PCM (uno por oración).
        pcm_per_sentence = [b"\x01" * 100, b"\x02" * 100, b"\x03" * 100]
        pcm_chunks = list(pcm_per_sentence)  # copia: aserción final

        def synth_side_effect(text, style_hint=""):
            return pcm_per_sentence.pop(0)

        assistant._local_tts.synthesize.side_effect = synth_side_effect

        # Capturar estado cuando se llame play_audio_stream (DEBE ser PROCESSING
        # porque la transición no debería haber ocurrido aún)
        state_at_play_call = []
        captured_stream = []

        def play_audio_stream_consuming(stream):
            # En el momento que play_audio_stream es invocado, el wrapper aún
            # no se iteró → el estado debe seguir siendo PROCESSING.
            state_at_play_call.append(assistant._state)
            # Consumir el wrapper para forzar la transición al primer chunk
            captured_stream.append(_drain_stream(stream))
            # Después del drain, el estado debe haber mutado a SPEAKING
            state_at_play_call.append(assistant._state)

        assistant._audio.play_audio_stream.side_effect = play_audio_stream_consuming

        assistant.run_pipeline("/tmp/fake.wav")

        # Verificaciones ──────────────────────────────────────────
        # 1) play_audio_stream fue llamado una vez
        assistant._audio.play_audio_stream.assert_called_once()
        # 2) Al momento de invocar play_audio_stream, el estado era PROCESSING
        #    (la transición NO ocurrió antes del primer PCM).
        assert state_at_play_call[0] == assistant.STATE_PROCESSING, (
            f"Estado al llamar play_audio_stream debe ser PROCESSING, "
            f"se obtuvo {state_at_play_call[0]!r}"
        )
        # 3) Tras consumir el wrapper, el estado es SPEAKING
        assert state_at_play_call[-1] == assistant.STATE_SPEAKING, (
            f"Estado tras consumir wrapper debe ser SPEAKING, "
            f"se obtuvo {state_at_play_call[-1]!r}"
        )
        # 4) overlay.set_state("speaking") se llamó exactamente UNA vez
        speaking_calls = [
            c for c in assistant._overlay.set_state.call_args_list
            if c.args and c.args[0] == "speaking"
        ]
        assert len(speaking_calls) == 1, (
            f"set_state('speaking') debe llamarse 1 vez, "
            f"se llamó {len(speaking_calls)}: {speaking_calls}"
        )
        # 5) El wrapper emitió TODOS los chunks downstream (incluido el primero)
        assert captured_stream[0] == pcm_chunks, (
            f"Wrapper debe emitir todos los chunks, se obtuvo {captured_stream[0]!r}"
        )

    # ── Caso 2: cancelación antes del primer chunk ───────────────

    def test_wrapper_aborts_cleanly_when_cancelled_before_first_chunk(
        self, env_keys, mock_settings, mock_overlay, monkeypatch
    ):
        """Caso 2: si ``_pipeline_generation`` cambia antes del primer chunk,
        el wrapper aborta sin setear ``STATE_SPEAKING`` ni llamar
        ``overlay.set_state("speaking")``. Estado queda intacto.

        Setup:
            - ``play_audio_stream`` simula que el usuario interrumpió
              (toggle → ``_pipeline_generation += 1``) ANTES de iterar el wrapper.
            - Esto fuerza el chequeo de cancelación dentro del wrapper.
        """
        assistant = _build_assistant(
            env_keys, mock_settings, mock_overlay, monkeypatch, streaming_enabled=True
        )
        # Estado inicial: PROCESSING
        assistant._state = assistant.STATE_PROCESSING
        gen_at_start = assistant._pipeline_generation  # captura (típicamente 0)

        assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # OpenCode streaming: deltas OK (no disparan cancelación por sí solos)
        def delta_iter():
            yield "[STYLE: cheerful] Hola. "
        assistant._opencode.send_command_stream.return_value = delta_iter()

        # Kokoro streaming: chunks PCM reales (NO cancela por sí solo)
        pcm_chunks = [b"\x01" * 100, b"\x02" * 100]
        assistant._local_tts.synthesize_sentence_stream.return_value = iter(pcm_chunks)

        # Capturar el wrapper y simular cancelación ANTES de iterarlo
        captured_stream = []
        wrapper_capture = []

        def play_audio_stream_simulating_cancel(stream):
            # Guardar el wrapper sin consumirlo aún
            wrapper_capture.append(stream)
            # Simular toggle() del usuario: incrementa generation
            assistant._pipeline_generation += 1
            # AHORA consumir el wrapper — debe abortar inmediatamente
            # porque el chequeo de cancelación dentro del lock detecta
            # que generation != la capturada al inicio.
            captured_stream.append(_drain_stream(stream))

        assistant._audio.play_audio_stream.side_effect = play_audio_stream_simulating_cancel

        assistant.run_pipeline("/tmp/fake.wav")

        # Verificaciones ──────────────────────────────────────────
        # 1) play_audio_stream fue invocado (capturamos el wrapper)
        assert wrapper_capture, "play_audio_stream no fue llamado"
        # 2) overlay.set_state("speaking") NUNCA se llamó
        speaking_calls = [
            c for c in assistant._overlay.set_state.call_args_list
            if c.args and c.args[0] == "speaking"
        ]
        assert speaking_calls == [], (
            f"set_state('speaking') no debe llamarse en cancelación, "
            f"se llamó {len(speaking_calls)}: {speaking_calls}"
        )
        # 3) El estado NO mutó a SPEAKING
        #    (debe seguir en PROCESSING — el wrapper abortó antes de transicionar).
        #    Nota: el finally NO lo resetea porque la generación difiere.
        assert assistant._state != assistant.STATE_SPEAKING, (
            f"Estado no debe ser SPEAKING tras cancelación, "
            f"se obtuvo {assistant._state!r}"
        )
        # 4) Generación se incrementó (simulando toggle())
        assert assistant._pipeline_generation > gen_at_start
        # 5) El wrapper no emitió chunks downstream (abortó antes del primer yield
        #    útil). Esto valida que la cancelación corta el flujo de audio.
        #    Nota: el wrapper es un generador; si aborta dentro del for-loop,
        #    simplemente termina sin yield. list() sobre un gen vacío → [].
        assert captured_stream[0] == [], (
            f"Wrapper cancelado no debe emitir chunks, "
            f"se obtuvo {captured_stream[0]!r}"
        )

    # ── Caso 3: chunks vacíos iniciales ──────────────────────────

    def test_wrapper_skips_empty_initial_chunks(
        self, env_keys, mock_settings, mock_overlay, monkeypatch
    ):
        """Caso 3: si los primeros yields del helper son ``b""`` (vacíos),
        la transición NO se dispara hasta que llegue un chunk no-vacío.

        Esto modela el caso real donde ``_azure_tts.synthesize_stream`` puede
        emitir chunks vacíos como padding, o donde ``_local_tts.synthesize``
        retorne ``b""`` (silencio). El wrapper debe ignorarlos y esperar al
        primer chunk no-vacío.

        Post-fix c115c48: el helper invoca ``_local_tts.synthesize`` por
        oración. El propio helper (``_synthesize_sentence_stream_with_fallback``)
        filtra los ``b""`` (no los yield-ea) — solo se yield-ean los PCM no
        vacíos. Mockeamos ``synthesize`` con un ``side_effect`` que retorna
        ``b""`` para las 2 primeras oraciones y PCM real para las 2 últimas.
        El wrapper debe transicionar a SPEAKING solo al primer chunk no-vacío.
        """
        assistant = _build_assistant(
            env_keys, mock_settings, mock_overlay, monkeypatch, streaming_enabled=True
        )
        assistant._state = assistant.STATE_PROCESSING

        assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # 4 oraciones: 2 vacías (silencio) + 2 reales
        def delta_iter():
            yield "[STYLE: cheerful] Primera. "
            yield "Segunda. "
            yield "Tercera. "
            yield "Cuarta. "
        assistant._opencode.send_command_stream.return_value = delta_iter()

        # ``_local_tts.synthesize`` retorna ``b""`` para las 2 primeras
        # oraciones y bytes reales para las 2 últimas. El helper filtra los
        # ``b""`` (no los yield-ea al wrapper) → solo llegan 2 chunks al
        # wrapper (los reales).
        def synth_side_effect(text, style_hint=""):
            if text in ("Primera.", "Segunda."):
                return b""  # silencio
            return {  # oraciones reales
                "Tercera.": b"\x01" * 100,
                "Cuarta.": b"\x02" * 100,
            }[text]

        assistant._local_tts.synthesize.side_effect = synth_side_effect

        # Lo que el wrapper debería emitir downstream: solo los 2 chunks
        # no-vacíos (el helper filtra los ``b""``).
        expected_downstream = [b"\x01" * 100, b"\x02" * 100]

        # Estado al pasar a play_audio_stream y después del drain
        state_progression = []
        captured_stream = []

        def play_audio_stream_consuming(stream):
            state_progression.append(("before_iter", assistant._state))
            captured_stream.append(_drain_stream(stream))
            state_progression.append(("after_iter", assistant._state))

        assistant._audio.play_audio_stream.side_effect = play_audio_stream_consuming

        assistant.run_pipeline("/tmp/fake.wav")

        # Verificaciones ──────────────────────────────────────────
        # 1) Antes de iterar: estado PROCESSING (aún no se vio el primer chunk real)
        assert state_progression[0] == ("before_iter", assistant.STATE_PROCESSING)
        # 2) Tras iterar: estado SPEAKING (porque el wrapper eventualmente vio un chunk real)
        assert state_progression[-1] == ("after_iter", assistant.STATE_SPEAKING)
        # 3) set_state("speaking") se llamó EXACTAMENTE 1 vez
        #    (los chunks vacíos no la dispararon).
        speaking_calls = [
            c for c in assistant._overlay.set_state.call_args_list
            if c.args and c.args[0] == "speaking"
        ]
        assert len(speaking_calls) == 1, (
            f"set_state('speaking') debe llamarse 1 vez (solo al primer chunk real), "
            f"se llamó {len(speaking_calls)}: {speaking_calls}"
        )
        # 4) El wrapper emitió SOLO los chunks no-vacíos downstream
        #    (el helper filtra ``b""`` antes de yield-earlos al wrapper).
        assert captured_stream[0] == expected_downstream, (
            f"Wrapper debe emitir solo chunks no-vacíos, "
            f"se obtuvo {captured_stream[0]!r}"
        )

    # ── Caso 4: path síncrono intacto ────────────────────────────

    def test_sync_pipeline_sets_speaking_before_tts(
        self, env_keys, mock_settings, mock_overlay, monkeypatch
    ):
        """Caso 4: el path síncrono (``_run_sync_pipeline``) sigue
        seteando ``STATE_SPEAKING`` ANTES de invocar ``synthesize``.

        El fix (commit ``fe33e29``) **NO debe** tocar el path síncrono: ahí
        ``synthesize`` retorna PCM completo de forma bloqueante, por lo que
        la transición ANTES del TTS es correcta. Este test verifica que el
        contrato del path síncrono no cambió.

        Estrategia: inyectamos un ``side_effect`` en ``_local_tts.synthesize``
        que captura el estado del orquestador en el momento de la llamada.
        Si el estado es ``SPEAKING``, el path síncrono funciona como antes.
        """
        assistant = _build_assistant(
            env_keys, mock_settings, mock_overlay, monkeypatch, streaming_enabled=False
        )

        assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo"
        )

        # Capturar el estado cuando synthesize es invocado
        state_during_synthesize = []

        def capture_state_synthesize(*args, **kwargs):
            state_during_synthesize.append(assistant._state)
            return b"\x00" * 48000

        assistant._local_tts.synthesize.side_effect = capture_state_synthesize

        assistant.run_pipeline("/tmp/fake.wav")

        # Verificaciones ──────────────────────────────────────────
        # 1) synthesize fue invocado
        assistant._local_tts.synthesize.assert_called_once()
        # 2) En el momento de synthesize, el estado era SPEAKING
        #    (la transición ocurrió ANTES de TTS, en el path síncrono)
        assert state_during_synthesize[0] == assistant.STATE_SPEAKING, (
            f"Estado durante synthesize debe ser SPEAKING (path síncrono), "
            f"se obtuvo {state_during_synthesize[0]!r}"
        )
        # 3) set_state("speaking") se llamó exactamente 1 vez
        speaking_calls = [
            c for c in assistant._overlay.set_state.call_args_list
            if c.args and c.args[0] == "speaking"
        ]
        assert len(speaking_calls) == 1, (
            f"set_state('speaking') debe llamarse 1 vez en path síncrono, "
            f"se llamó {len(speaking_calls)}: {speaking_calls}"
        )
        # 4) El log del path síncrono tiene el formato original ("→ SPEAKING (gen=%d)")
        #    (NO el "primer PCM real" introducido por el fix en el path streaming).
        #    Esto verifica que el fix solo tocó el path streaming.
        import logging as logging_mod
        # Nota: no usamos caplog aquí porque el path síncrono usa el mismo logger
        # que el streaming. Verificamos que NO esté el mensaje del wrapper:
        sync_log_marker = "→ SPEAKING (gen=%d)"  # formato original síncrono
        streaming_log_marker = "primer PCM real"   # introducido por el fix
        # La presencia del streaming_log_marker en logs NO es testeable directamente
        # aquí porque el path síncrono no entra al wrapper. Pero verificamos que
        # el path síncrono NO haya disparado el wrapper de streaming.
        assistant._local_tts.synthesize_sentence_stream.assert_not_called()
        # El sync marker debe haber sido emitido (logger.info("→ SPEAKING ..."))
        # pero como no usamos caplog, validamos indirectamente: el estado mutó
        # correctamente, lo cual solo ocurre si el bloque síncrono se ejecutó.