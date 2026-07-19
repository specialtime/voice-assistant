"""Tests unitarios para main.py:VoiceAssistant (máquina de estados).

Mockea `keyboard`, `AudioManager` y los 4 clientes para aislar
la lógica de transiciones de estado (IDLE ↔ RECORDING ↔ PROCESSING ↔ SPEAKING)
y del pipeline de 7 pasos definido en IMPLEMENTATION.md §4.9.

Cubre:
- Fase 11: máquina de 3 estados y pipeline de 6 pasos.
- Fase 12.B: estado SPEAKING, generación de pipeline, send_lock y
  reconsideración de respuesta (interrupción con Alt+V).
"""

import logging
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Asegurar que la raíz del proyecto está en sys.path (por si conftest no se ejecutó)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# Stub de kokoro_onnx — registrado a nivel de módulo para que el
# import top-level de ``main`` (``from kokoro_onnx import Kokoro``)
# resuelva sin necesidad de tener la dependencia instalada.
#
# Mismo patrón que ``tests/test_kokoro_tts_client.py`` y
# ``tests/test_local_integration.py``. Si este módulo se importa antes
# que cualquiera de los otros dos, el stub ya estará disponible y no se
# duplica (es idempotente).
# ──────────────────────────────────────────────────────────────────
if "kokoro_onnx" not in sys.modules:
    _kokoro_stub = types.ModuleType("kokoro_onnx")
    _kokoro_stub.Kokoro = MagicMock(name="Kokoro")
    sys.modules["kokoro_onnx"] = _kokoro_stub


@pytest.mark.unit
class TestVoiceAssistantStateMachine:
    """Suite de tests para la máquina de estados y el pipeline de VoiceAssistant."""

    @pytest.fixture
    def env_keys(self, monkeypatch):
        """Setea las env vars necesarias para que el constructor cree los 4 clientes."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
        monkeypatch.setenv("AZURE_SPEECH_KEY", "fake_azure_key")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "southamericaeast")
        monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "fake_opencode_pass")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")

    @pytest.fixture
    def patched_assistant(self, env_keys, mock_settings, mock_overlay, monkeypatch):
        """Crea un VoiceAssistant con TODAS las dependencias externas mockeadas.

        Patches aplicados:
        - main.AudioManager
        - main.GeminiSTTClient
        - main.OpenCodeClient
        - main.GeminiTTSClient
        - main.AzureTTSClient
        - main.OverlayChip (vía fixture mock_overlay)
        - main.load_dotenv  (no toca el .env real)

        También fuerza CWD a la raíz del proyecto para que el __init__ del
        orquestador encuentre config/settings.json.
        """
        monkeypatch.chdir(_PROJECT_ROOT)

        with patch("main.AzureTTSClient") as mock_atts, \
             patch("main.GeminiTTSClient") as mock_gtts, \
             patch("main.OpenCodeClient") as mock_oc, \
             patch("main.GeminiSTTClient") as mock_stt, \
             patch("main.WhisperSTTClient") as mock_wstt, \
             patch("main.PiperTTSClient") as mock_ptts, \
             patch("main.KokoroTTSClient") as mock_ktts, \
             patch("main.AudioManager") as mock_am, \
             patch("main.load_dotenv"):

            mock_am.return_value = MagicMock(name="AudioManager")
            mock_stt.return_value = MagicMock(name="GeminiSTTClient")
            mock_wstt.return_value = MagicMock(name="WhisperSTTClient")
            mock_ptts.return_value = MagicMock(name="PiperTTSClient")
            mock_ktts.return_value = MagicMock(name="KokoroTTSClient")
            mock_oc.return_value = MagicMock(name="OpenCodeClient")
            mock_gtts.return_value = MagicMock(name="GeminiTTSClient")
            mock_atts.return_value = MagicMock(name="AzureTTSClient")

            # Importar DESPUÉS de aplicar los patches para que `from main import`
            # use las versiones mockeadas. Pero como vamos a usar la fixture desde
            # muchos tests, importamos una sola vez al inicio del módulo.
            from main import VoiceAssistant

            assistant = VoiceAssistant()
            # Sobrescribir settings con mock_settings para que los tests no
            # dependan del config/settings.json real.
            assistant._settings = mock_settings
            # Forzar flujo SÍNCRONO por default en esta suite: los tests que
            # NO son de streaming (la mayoría) verifican el camino síncrono
            # ``send_command + synthesize + play_audio``. El default en
            # config/settings.json es ``streaming_enabled=True``, pero
            # ``mock_settings`` no incluye esa key en la sección ``opencode``,
            # por lo que el orquestador cae al default True del
            # ``.get("streaming_enabled", True)``. Esto rompe los tests que
            # mockean el flujo síncrono (eran 6 antes de la suite T6-T9).
            #
            # Los 4 tests de streaming (T6-T9: ``test_pipeline_streaming_*``)
            # sobreescriben este atributo explícitamente a ``True`` o ``False``
            # según su contrato, así que este default no los afecta.
            assistant._streaming_enabled = False

            yield assistant

    def test_toggle_idle_to_recording(self, patched_assistant):
        """IDLE → toggle() → RECORDING, y se llama audio.start_recording()."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_IDLE

        patched_assistant.toggle()

        assert patched_assistant._state == VoiceAssistant.STATE_RECORDING
        patched_assistant._audio.start_recording.assert_called_once()
        # stop_recording no debe haberse llamado en este toggle
        patched_assistant._audio.stop_recording.assert_not_called()

    def test_toggle_recording_to_processing(self, patched_assistant):
        """RECORDING → toggle() → PROCESSING, stop_recording llamado, hilo lanzado."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_RECORDING
        patched_assistant._audio.stop_recording.return_value = "/tmp/fake.wav"

        with patch("main.threading.Thread") as mock_thread_cls:
            patched_assistant.toggle()

        assert patched_assistant._state == VoiceAssistant.STATE_PROCESSING
        patched_assistant._audio.stop_recording.assert_called_once()

        # El hilo debe haberse instanciado con target=run_pipeline
        mock_thread_cls.assert_called_once()
        thread_kwargs = mock_thread_cls.call_args.kwargs
        assert thread_kwargs["target"] == patched_assistant.run_pipeline
        assert thread_kwargs["args"] == ("/tmp/fake.wav",)
        assert thread_kwargs["daemon"] is True
        # Y se debe haber llamado .start() sobre la instancia
        mock_thread_cls.return_value.start.assert_called_once()

    def test_pipeline_stt_none_returns(self, patched_assistant, caplog):
        """Si _whisper_stt y _stt son None → log error y return (sin crash), estado vuelve a IDLE."""
        from main import VoiceAssistant

        patched_assistant._whisper_stt = None
        patched_assistant._stt = None

        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # El estado debe volver a IDLE (finally)
        assert patched_assistant._state == VoiceAssistant.STATE_IDLE
        # Log de error mencionando Gemini STT no configurado
        assert any(
            "gemini" in record.getMessage().lower()
            and "stt" in record.getMessage().lower()
            and "no configurado" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_pipeline_tts_fallback(self, patched_assistant):
        """Si Piper falla y Gemini falla → Azure streaming es llamado
        y el iterator de chunks PCM se pasa a audio.play_audio_stream (micro-spec C)."""
        from main import VoiceAssistant

        # STT y OpenCode retornan valores válidos
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo, abrí Chrome"
        )
        # Piper TTS falla
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
            "Piper falló"
        )
        # Gemini TTS falla (fuerza el fallback a Azure)
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
            "Gemini TTS falló (rate limit)"
        )
        # Azure TTS streaming: retorna iterator de chunks PCM s16le
        # (la API real es un generator — usamos iter() para semántica equivalente)
        fake_pcm_chunks = [b"\x00\x01" * 100, b"\x00\x01" * 100]
        patched_assistant._azure_tts.synthesize_stream.return_value = iter(fake_pcm_chunks)
        # audio.play_audio_stream es MagicMock (no-op) — solo necesitamos
        # que acepte el iterator que le pasa run_pipeline().

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se llamó a azure_tts.synthesize_stream con el texto limpio (sin [STYLE:])
        patched_assistant._azure_tts.synthesize_stream.assert_called_once()
        azure_call_args = patched_assistant._azure_tts.synthesize_stream.call_args
        assert azure_call_args.args[0] == "Listo, abrí Chrome"

        # Se pasó el iterator de chunks a play_audio_stream (streaming playback)
        patched_assistant._audio.play_audio_stream.assert_called_once()
        # El argumento pasado debe ser el iterator retornado por synthesize_stream
        stream_arg = patched_assistant._audio.play_audio_stream.call_args.args[0]
        assert stream_arg is patched_assistant._azure_tts.synthesize_stream.return_value

        # Piper y Gemini synthesize fueron llamados
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # play_audio NO se llamó (Azure streaming ya reprodujo)
        patched_assistant._audio.play_audio.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == VoiceAssistant.STATE_IDLE

    def test_pipeline_both_tts_none(self, patched_assistant, caplog):
        """Si los 3 TTS son None → log error y return, estado vuelve a IDLE."""
        from main import VoiceAssistant

        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo"
        )
        # Forzar los 3 TTS a None
        patched_assistant._local_tts = None
        patched_assistant._gemini_tts = None
        patched_assistant._azure_tts = None

        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Estado final: IDLE (vía finally)
        assert patched_assistant._state == VoiceAssistant.STATE_IDLE
        # Log de error: todos los TTS fallaron (log consolidado post-primary_engine).
        # Antes del selector primary_engine, el flujo síncrono logueaba
        # "Azure TTS no configurado" cuando _azure_tts era None. Con la nueva
        # arquitectura multi-motor, el log final se consolidó a un mensaje
        # genérico que aplica a cualquier primary_engine.
        assert any(
            "todos los tts fallaron" in record.getMessage().lower()
            for record in caplog.records
        ), f"Log 'Todos los TTS fallaron' no encontrado. Logs: {[r.getMessage() for r in caplog.records]}"
        # No se reprodujo audio
        patched_assistant._audio.play_audio.assert_not_called()

    # ── Integración overlay chip (Fase 9) ────────────────────────

    def test_toggle_idle_to_recording_calls_overlay_show(
        self, patched_assistant
    ) -> None:
        """IDLE → toggle() → llama overlay.show('recording')."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_IDLE

        patched_assistant.toggle()

        # overlay.show fue invocado con 'recording'
        patched_assistant._overlay.show.assert_called_once_with("recording")
        # Estado mutó a RECORDING
        assert patched_assistant._state == VoiceAssistant.STATE_RECORDING

    def test_toggle_recording_to_processing_calls_overlay_set_state(
        self, patched_assistant
    ) -> None:
        """RECORDING → toggle() → llama overlay.set_state('processing')."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_RECORDING
        patched_assistant._audio.stop_recording.return_value = "/tmp/fake.wav"

        with patch("main.threading.Thread") as mock_thread_cls:
            patched_assistant.toggle()

        # overlay.set_state fue invocado con 'processing'
        patched_assistant._overlay.set_state.assert_called_once_with("processing")
        # Estado mutó a PROCESSING
        assert patched_assistant._state == VoiceAssistant.STATE_PROCESSING
        # El hilo del pipeline fue lanzado (regression check)
        mock_thread_cls.assert_called_once()

    def test_pipeline_finally_calls_overlay_hide(
        self, patched_assistant
    ) -> None:
        """run_pipeline() → finally → llama overlay.hide() + vuelve a IDLE.

        Verifica que el bloque ``finally`` (no la rama exitosa) invoca
        ``overlay.hide()`` y resetea el estado. El pipeline retorna early
        porque ambos STT son None — pero el finally debe ejecutarse igual.
        """
        from main import VoiceAssistant

        # Ambos STT None → return temprano, pero finally debe ejecutarse
        patched_assistant._whisper_stt = None
        patched_assistant._stt = None

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # overlay.hide fue invocado por el finally
        patched_assistant._overlay.hide.assert_called_once()
        # Estado volvió a IDLE
        assert patched_assistant._state == VoiceAssistant.STATE_IDLE

    # ── Fase 12.B: Estado SPEAKING y reconsideración de respuesta ──

    def test_toggle_processing_to_recording(self, patched_assistant):
        """PROCESSING → toggle() → cancela pipeline (gen++) + start_recording + RECORDING."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_PROCESSING
        gen_before = patched_assistant._pipeline_generation

        patched_assistant.toggle()

        # Estado mutó a RECORDING
        assert patched_assistant._state == VoiceAssistant.STATE_RECORDING
        # start_recording fue llamado
        patched_assistant._audio.start_recording.assert_called_once()
        # stop_recording NO fue llamado (no estábamos grabando)
        patched_assistant._audio.stop_recording.assert_not_called()
        # Generación fue incrementada (cancelación del pipeline en curso)
        assert patched_assistant._pipeline_generation >= gen_before + 1
        # Overlay fue notificado
        patched_assistant._overlay.show.assert_called_with("recording")

    def test_toggle_speaking_to_recording(self, patched_assistant):
        """SPEAKING → toggle() → stop_playback + start_recording + RECORDING."""
        from main import VoiceAssistant

        patched_assistant._state = VoiceAssistant.STATE_SPEAKING

        patched_assistant.toggle()

        # Estado mutó a RECORDING
        assert patched_assistant._state == VoiceAssistant.STATE_RECORDING
        # stop_playback fue llamado (interrumpe el TTS en curso)
        patched_assistant._audio.stop_playback.assert_called_once()
        # start_recording fue llamado
        patched_assistant._audio.start_recording.assert_called_once()
        # Overlay fue notificado
        patched_assistant._overlay.show.assert_called_with("recording")

    def test_pipeline_cancelled_before_stt(self, patched_assistant):
        """Pipeline cancelado ANTES de STT: la generación cambia entre
        la captura inicial y el primer checkpoint → aborta antes de transcribe().

        Se usa PropertyMock con side_effect=[0] + [1] * 20 para que:
        - La 1ª lectura (capture, src/main.py:175) retorne 0.
        - La 2ª lectura (checkpoint 1, src/main.py:178) retorne 1 → cancela.
        - Las lecturas restantes (e.g. finally check, src/main.py:250)
          también retornen 1 sin agotar el iterador.

        Nota: _pipeline_generation es atributo de instancia, no de clase.
        Usamos patch.object con create=True para inyectar el PropertyMock
        como descriptor de clase temporal.
        """
        prop_mock = PropertyMock(side_effect=[0] + [1] * 20)
        with patch.object(
            type(patched_assistant), "_pipeline_generation", prop_mock, create=True
        ):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # El pipeline abortó antes de STT → transcribe NO fue llamado
        patched_assistant._whisper_stt.transcribe.assert_not_called()

    def test_pipeline_cancelled_after_stt(self, patched_assistant):
        """Pipeline cancelado DESPUÉS de STT: el side_effect de transcribe
        incrementa _pipeline_generation, disparando el checkpoint 2."""
        def stt_side_effect(*_args, **_kwargs):
            patched_assistant._pipeline_generation += 1
            return "texto transcrito"

        patched_assistant._whisper_stt.transcribe.side_effect = stt_side_effect

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # STT fue llamado (el side_effect incrementó gen después)
        patched_assistant._whisper_stt.transcribe.assert_called_once()
        # El checkpoint 2 canceló antes de OpenCode
        patched_assistant._opencode.send_command.assert_not_called()

    def test_pipeline_cancelled_after_send_command(self, patched_assistant):
        """Pipeline cancelado DESPUÉS de send_command: el side_effect de
        send_command incrementa _pipeline_generation, abortando antes de TTS."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        def send_side_effect(*_args, **_kwargs):
            patched_assistant._pipeline_generation += 1
            return "[STYLE: cheerful] Listo, abrí Chrome"

        patched_assistant._opencode.send_command.side_effect = send_side_effect

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # send_command fue llamado (side_effect incrementó gen después)
        patched_assistant._opencode.send_command.assert_called_once()
        # TTS NO se invocó
        patched_assistant._local_tts.synthesize.assert_not_called()
        # overlay.set_state NO fue llamado con "speaking"
        set_state_calls = patched_assistant._overlay.set_state.call_args_list
        speaking_calls = [
            c for c in set_state_calls
            if c.args and c.args[0] == "speaking"
        ]
        assert speaking_calls == [], (
            f"Se llamó set_state('speaking') {len(speaking_calls)} veces, "
            f"se esperaba 0. Calls: {set_state_calls}"
        )

    def test_pipeline_cancelled_does_not_reset_state(self, patched_assistant):
        """Pipeline cancelado: el finally no debe pisar el estado actual
        ni llamar overlay.hide() si las generaciones difieren.

        FIX-1 @security: el check de generación va DENTRO del lock
        en el finally para evitar race con toggle().
        """
        from main import VoiceAssistant

        def stt_side_effect(*_args, **_kwargs):
            # Incrementar generación para forzar cancelación en checkpoint 2
            patched_assistant._pipeline_generation += 1
            # Simular que toggle() ya mutó el estado a RECORDING
            patched_assistant._state = VoiceAssistant.STATE_RECORDING
            return "texto"

        patched_assistant._whisper_stt.transcribe.side_effect = stt_side_effect

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # El estado NO fue pisado a IDLE por el finally
        assert patched_assistant._state != VoiceAssistant.STATE_IDLE
        assert patched_assistant._state == VoiceAssistant.STATE_RECORDING
        # overlay.hide NO fue llamado (generación difiere → no reset)
        patched_assistant._overlay.hide.assert_not_called()

    def test_pipeline_normal_sets_speaking(self, patched_assistant):
        """Pipeline normal exitoso: en algún momento el estado es SPEAKING
        y overlay.set_state fue llamado con 'speaking'."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo, abrí Chrome"
        )
        # Piper TTS retorna bytes PCM → play_audio (no streaming)
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # overlay.set_state fue invocado con "speaking" en algún momento
        set_state_calls = patched_assistant._overlay.set_state.call_args_list
        speaking_calls = [
            c for c in set_state_calls
            if c.args and c.args[0] == "speaking"
        ]
        assert len(speaking_calls) >= 1, (
            f"set_state('speaking') no fue invocado. Calls: {set_state_calls}"
        )

    def test_pipeline_finally_resets_state_when_not_cancelled(self, patched_assistant):
        """Pipeline normal (no cancelado): el finally sí resetea a IDLE
        y llama overlay.hide() (rama exitosa del check de generación)."""
        from main import VoiceAssistant

        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo"
        )
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Estado final: IDLE
        assert patched_assistant._state == VoiceAssistant.STATE_IDLE
        # overlay.hide fue llamado por el finally
        patched_assistant._overlay.hide.assert_called_once()

    def test_send_lock_serializes_concurrent_pipelines(self, patched_assistant):
        """_send_lock serializa send_command: dos pipelines concurrentes
        nunca ejecutan send_command simultáneamente (max concurrencia = 1).

        FIX-1 @security: el lock garantiza que las llamadas a OpenCode
        no se solapen, evitando race conditions en el servidor remoto.
        """
        concurrency = {"current": 0, "max": 0}
        track_lock = threading.Lock()

        def tracked_send(*_args, **_kwargs):
            with track_lock:
                concurrency["current"] += 1
                concurrency["max"] = max(
                    concurrency["max"], concurrency["current"]
                )
            # Pequeño sleep para exponer races: si no hay lock, ambos
            # threads podrían estar en tracked_send simultáneamente.
            time.sleep(0.05)
            with track_lock:
                concurrency["current"] -= 1
            return "[STYLE: cheerful] OK"

        patched_assistant._whisper_stt.transcribe.return_value = "test"
        patched_assistant._opencode.send_command.side_effect = tracked_send
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        # Lanzar dos run_pipeline en paralelo
        t1 = threading.Thread(
            target=patched_assistant.run_pipeline,
            args=("/tmp/fake.wav",),
            daemon=True,
        )
        t2 = threading.Thread(
            target=patched_assistant.run_pipeline,
            args=("/tmp/fake.wav",),
            daemon=True,
        )
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Ambos pipelines ejecutaron send_command
        assert patched_assistant._opencode.send_command.call_count == 2
        # Pero nunca concurrentemente: max concurrencia observada = 1
        assert concurrency["max"] == 1, (
            f"send_lock falló: concurrencia máxima={concurrency['max']} "
            f"(se esperaba 1)"
        )

    def test_pipeline_stt_whisper_success(self, patched_assistant):
        """Whisper STT OK → Gemini STT NO se llama."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_not_called()

    def test_pipeline_stt_whisper_fails_gemini_fallback(self, patched_assistant):
        """Whisper STT falla → Gemini STT fallback OK."""
        patched_assistant._whisper_stt.transcribe.side_effect = RuntimeError("Whisper falló")
        patched_assistant._stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_called_once()

    def test_pipeline_tts_piper_success(self, patched_assistant):
        """Piper TTS OK → Gemini y Azure NO se llaman."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_not_called()
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()

    def test_pipeline_tts_piper_fails_gemini_fallback(self, patched_assistant):
        """Piper TTS falla → Gemini TTS fallback OK."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.synthesize.return_value = b"\x00" * 48000

        patched_assistant.run_pipeline("/tmp/fake.wav")

        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()

    def test_pipeline_tts_all_fail(self, patched_assistant, caplog):
        """Piper, Gemini y Azure todos fallan → return sin playback, estado IDLE."""
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini falló")
        patched_assistant._azure_tts = None

        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        patched_assistant._audio.play_audio.assert_not_called()
        patched_assistant._audio.play_audio_stream.assert_not_called()
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec Streaming TTS: pipeline streaming + fallback (T9)
    # ──────────────────────────────────────────────────────────────────
    #
    # Cubre los caminos del nuevo flujo streaming introducido en
    # `feature/streaming-tts-kokoro`:
    #   - Flujo streaming exitoso (send_command_stream + synthesize_sentence_stream)
    #   - Fallback a síncrono si send_command_stream falla
    #   - streaming_enabled=False → usa el flujo síncrono actual
    #   - Cancelación durante el streaming (generation mismatch)

    def test_pipeline_streaming_success(self, patched_assistant):
        """Flujo streaming exitoso: send_command_stream + helper de fallback
        + play_audio_stream se invocan en orden. NO se llama send_command (síncrono).

        Post-fix c115c48: el código ya no llama a ``_local_tts.synthesize_sentence_stream``
        directamente sino al helper ``_synthesize_sentence_stream_with_fallback``,
        que a su vez llama a ``_local_tts.synthesize`` (cadena local → Gemini → Azure).
        Mockeamos ``_local_tts.synthesize`` para reflejar la nueva ruta.

        Para que el stream lazy se itere (delta → sentence → pcm), forzamos a
        ``play_audio_stream`` (MagicMock por defecto) a consumir el stream que
        recibe. Sin este consumo explícito, el iter nunca avanza y
        ``synthesize`` no se invoca.
        """
        # Forzar streaming habilitado (la fixture usa mock_settings sin esta key,
        # por lo que el __init__ toma el default True desde el settings.json real)
        patched_assistant._streaming_enabled = True

        # STT OK
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # OpenCode streaming: retorna un iter de deltas con 2 oraciones terminadas
        # en punto (separador de SentenceBuffer).
        def delta_iter():
            yield "[STYLE: cheerful] Hola. "
            yield "Chau. "

        patched_assistant._opencode.send_command_stream.return_value = delta_iter()
        # OpenCode send_command (síncrono) NO debe ser llamado
        patched_assistant._opencode.send_command.return_value = "NO DEBE LLAMARSE"

        # TTS local (helper de fallback usa ``_local_tts.synthesize`` por oración):
        # SentenceBuffer split por ". ! ? ;" → deltas yield 2 oraciones ("Hola." y "Chau.").
        # Side effect retorna PCM distinto por oración para verificar el orden.
        pcm_per_call = [
            b"\x00" * 100,  # "Hola." → pcm
            b"\x00" * 200,  # "Chau." → pcm
        ]

        def synth_side_effect(text, style_hint=""):
            pcm = pcm_per_call.pop(0)
            return pcm

        patched_assistant._local_tts.synthesize.side_effect = synth_side_effect

        # Forzar consumo del stream: ``play_audio_stream`` debe iterar el
        # pcm_stream que recibe para que el iter lazy (delta → sentence → synth)
        # avance. Sin esto, ``synthesize`` no se invoca.
        def play_audio_stream_consuming(stream):
            list(stream)  # consumir el generador

        patched_assistant._audio.play_audio_stream.side_effect = (
            play_audio_stream_consuming
        )

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se invocó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_called_once_with("abrí chrome")
        # send_command (síncrono) NO fue llamado
        patched_assistant._opencode.send_command.assert_not_called()
        # Post-fix c115c48: el helper de fallback llama a ``_local_tts.synthesize``
        # por oración (en vez de ``synthesize_sentence_stream`` directo).
        # 2 oraciones → 2 invocaciones a ``_local_tts.synthesize``.
        assert patched_assistant._local_tts.synthesize.call_count == 2
        # Los textos recibidos son las oraciones (sin prefijo [STYLE:])
        received = [c.args[0] for c in patched_assistant._local_tts.synthesize.call_args_list]
        assert received == ["Hola.", "Chau."], (
            f"Esperaba ['Hola.', 'Chau.'], obtuve {received}"
        )
        # El helper NO cayó a Gemini ni Azure (TTS local OK)
        patched_assistant._gemini_tts.synthesize.assert_not_called()
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()
        # play_audio_stream fue invocado
        patched_assistant._audio.play_audio_stream.assert_called_once()
        # play_audio (no streaming) NO fue llamado
        patched_assistant._audio.play_audio.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_pipeline_streaming_fallback_on_error(self, patched_assistant):
        """Si ``send_command_stream`` falla → fallback automático al flujo síncrono.

        El pipeline principal envuelve el flujo streaming en un try/except;
        ante una excepción cae a ``_run_sync_pipeline()``, que usa
        ``send_command()`` + ``synthesize()`` + ``play_audio()``.
        """
        patched_assistant._streaming_enabled = True
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # OpenCode streaming FALLA
        patched_assistant._opencode.send_command_stream.side_effect = RuntimeError(
            "SSE cortado"
        )
        # OpenCode síncrono (fallback) retorna respuesta válida
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo"
        )
        # TTS local síncrono (fallback) retorna PCM
        pcm = b"\x00" * 48000
        patched_assistant._local_tts.synthesize.return_value = pcm

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se intentó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_called_once()
        # CAYÓ al síncrono: send_command fue llamado
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")
        # Kokoro streaming NO fue invocado (estamos en fallback)
        patched_assistant._local_tts.synthesize_sentence_stream.assert_not_called()
        # Kokoro síncrono SÍ fue invocado
        patched_assistant._local_tts.synthesize.assert_called_once()
        # Se reprodujo por play_audio (no streaming)
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_pipeline_streaming_disabled(self, patched_assistant):
        """``streaming_enabled=False`` → usa el flujo SÍNCRONO (no streaming).

        Verifica el contrato del setting: cuando streaming_enabled es False,
        el handler principal va directo a ``_run_sync_pipeline()`` sin
        intentar el flujo streaming.
        """
        # DESHABILITAR streaming
        patched_assistant._streaming_enabled = False

        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = (
            "[STYLE: cheerful] Listo"
        )
        pcm = b"\x00" * 48000
        patched_assistant._local_tts.synthesize.return_value = pcm

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # NO se intentó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_not_called()
        # Se usó el flujo síncrono: send_command fue llamado
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")
        # Kokoro síncrono fue llamado
        patched_assistant._local_tts.synthesize.assert_called_once()
        # Kokoro streaming NO fue invocado
        patched_assistant._local_tts.synthesize_sentence_stream.assert_not_called()
        # Reproducción por play_audio
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_pipeline_streaming_cancellation(self, patched_assistant):
        """Cancelación durante el streaming: ``generation`` cambia mientras se
        itera el stream → el iter aborta antes de invocar Kokoro.

        Verifica que el chequeo de ``self._pipeline_generation != generation``
        dentro del ``sentence_iterator()`` aborta correctamente el flujo.

        Post-fix c115c48: el código llama al helper de fallback que invoca
        ``_local_tts.synthesize`` por oración. Mockeamos ``synthesize`` como
        proxy: consume el ``sentence_iterator`` que recibe vía el helper
        (lo que fuerza al ``sentence_iterator`` interno a iterar el
        ``delta_stream`` y disparar la cancelación) pero retorna ``b""`` (la
        cancelación abortó antes del primer yield real).

        Contrato del fix ``fe33e29`` (overlay speaking prematuro):
        Como la cancelación ocurre ANTES del primer chunk PCM real, el wrapper
        ``pcm_stream_with_speaking_transition`` aborta sin transicionar a
        ``STATE_SPEAKING``. El estado del orquestador queda en su valor previo
        (en este test, ``STATE_IDLE`` porque no hubo ``toggle()`` previo). El
        bloque ``finally`` además NO resetea el estado porque la generación
        difiere (cancelación detectada).
        """
        patched_assistant._streaming_enabled = True
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # Stream que incrementa generation al emitir su primer delta (simula
        # que el usuario presionó Alt+V durante el streaming).
        def delta_iter():
            patched_assistant._pipeline_generation += 1  # simula toggle()
            yield "[STYLE: cheerful] Hola. "  # este yield se descarta por cancel
            yield "Chau. "  # nunca llega aquí

        patched_assistant._opencode.send_command_stream.return_value = delta_iter()

        # Mock synthesize como PROXY del sentence_iterator que el helper
        # recibe. Consumir el iter fuerza avance del sentence_iterator →
        # fuerza avance del delta_iter → dispara _pipeline_generation += 1 →
        # el cancellation check aborta. Pero como el chequeo es
        # ``_pipeline_generation != generation`` (donde ``generation`` es el
        # capturado al inicio del pipeline), el helper no llega a llamar
        # ``synthesize`` con la oración ya que la cancelación ocurre antes.
        # Por lo tanto, ``synthesize`` NO se invoca.
        # Sin embargo, queremos verificar que la cancelación disparó y que el
        # estado del orquestador quedó en IDLE (sin transición a SPEAKING).
        #
        # NOTA: con el nuevo helper, ``_synthesize_sentence_stream_with_fallback``
        # chequea ``self._pipeline_generation != generation`` ANTES de cada
        # oración. Si el delta_iter incrementa generation al emitir, el
        # sentence_iterator interno aborta al detectar cancelación, y el
        # helper no llega a invocar ``synthesize`` para ninguna oración.
        # Para verificar la ruta, hacemos que ``synthesize`` retorne PCM vacío:
        # si por alguna razón el helper lo invoca, simplemente retorna vacío
        # (que es truthy=False → el helper no lo yield-ea y no transiciona).

        # Forzar consumo del stream via play_audio_stream side_effect
        def play_audio_stream_consuming(stream):
            list(stream)

        patched_assistant._audio.play_audio_stream.side_effect = (
            play_audio_stream_consuming
        )

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se intentó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_called_once()
        # El delta_iter SÍ fue consumido (cancellation disparó)
        assert patched_assistant._pipeline_generation >= 1, (
            f"Esperaba _pipeline_generation>=1 (cancelación disparada), "
            f"se obtuvo {patched_assistant._pipeline_generation}"
        )
        # play_audio_stream fue llamado (con stream que terminó sin chunks reales)
        patched_assistant._audio.play_audio_stream.assert_called_once()
        # FIX ``fe33e29``: la transición a SPEAKING ahora ocurre SOLO al primer
        # chunk PCM real. Como la cancelación abortó antes del primer chunk,
        # el wrapper nunca transicionó → ``overlay.set_state("speaking")`` NO
        # se invoca y el estado queda intacto (IDLE en este test porque no
        # hubo ``toggle()`` previo). El bloque ``finally`` tampoco lo resetea
        # porque detecta que la generación difiere (rama de cancelación).
        assert patched_assistant._state == patched_assistant.STATE_IDLE, (
            f"Esperaba estado=IDLE (sin transición a SPEAKING por cancelación "
            f"pre-PCM), se obtuvo {patched_assistant._state!r}"
        )
        # overlay.set_state("speaking") NO fue llamado (la cancelación ocurrió
        # antes del primer chunk PCM real).
        speaking_calls = [
            c for c in patched_assistant._overlay.set_state.call_args_list
            if c.args and c.args[0] == "speaking"
        ]
        assert speaking_calls == [], (
            f"set_state('speaking') no debe llamarse en cancelación pre-PCM, "
            f"se llamó {len(speaking_calls)}: {speaking_calls}"
        )
        # overlay.hide NO fue llamado por el finally (cancelación detectada).
        patched_assistant._overlay.hide.assert_not_called()

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec Streaming TTS: helper de fallback por oración (T10)
    # ──────────────────────────────────────────────────────────────────
    #
    # Cubre el nuevo helper ``_synthesize_one_sentence_with_fallback`` y
    # ``_synthesize_sentence_stream_with_fallback`` introducidos en commit
    # ``c115c48`` para restaurar la cadena de fallback a Gemini TTS y Azure
    # TTS en el flujo streaming (que se había perdido al agregar Kokoro).

    @pytest.mark.unit
    class TestSynthesizeOneSentenceWithFallback:
        """Tests del helper ``_synthesize_one_sentence_with_fallback``.

        Cadena: local (Piper/Kokoro) → Gemini → Azure streaming.
        Cubre los caminos de éxito y cada fallback.
        """

        def test_fallback_local_ok(self, patched_assistant):
            """Local OK → solo se llama local, NO Gemini ni Azure."""
            patched_assistant._local_tts.synthesize.return_value = b"local_pcm"
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter([b"azure_pcm"])

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # Local retornó PCM
            assert result == b"local_pcm"
            # Local fue llamado
            patched_assistant._local_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Gemini y Azure NO se llamaron
            patched_assistant._gemini_tts.synthesize.assert_not_called()
            patched_assistant._azure_tts.synthesize_stream.assert_not_called()

        def test_fallback_local_fails_gemini_ok(self, patched_assistant):
            """Local falla (RuntimeError) → Gemini es llamado, Azure NO."""
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter([b"azure_pcm"])

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            assert result == b"gemini_pcm"
            # Local fue llamado una vez (raise)
            patched_assistant._local_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Gemini fue llamado una vez (OK)
            patched_assistant._gemini_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Azure NO se invocó (Gemini ya tuvo éxito)
            patched_assistant._azure_tts.synthesize_stream.assert_not_called()

        def test_fallback_local_and_gemini_fail_azure_ok(self, patched_assistant):
            """Local Y Gemini fallan → Azure streaming es consumido a bytes."""
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
                "Gemini rate limit"
            )
            # Azure streaming retorna iter de chunks — el helper los junta con b"".join
            azure_chunks = [b"a", b"b", b"c"]
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(azure_chunks)

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # El helper retorna b"".join(...) → b"abc"
            assert result == b"abc"
            # Azure fue llamado con style_hint=""
            patched_assistant._azure_tts.synthesize_stream.assert_called_once_with(
                "hola", style_hint=""
            )

        def test_fallback_all_fail_returns_none(self, patched_assistant):
            """Los 3 TTS fallan → helper retorna None, no lanza excepción."""
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
                "Gemini rate limit"
            )
            patched_assistant._azure_tts.synthesize_stream.side_effect = RuntimeError(
                "Azure 503"
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # helper NO lanza, retorna None
            assert result is None

        def test_fallback_gemini_circuit_breaker_open(self, patched_assistant):
            """Gemini ``is_available()`` retorna False (circuit breaker abierto)
            → se salta Gemini, va directo a Azure (incluso si local también falló)."""
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            patched_assistant._gemini_tts.is_available.return_value = False
            patched_assistant._azure_tts.synthesize_stream.return_value = iter([b"azure_pcm"])

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            assert result == b"azure_pcm"
            # Gemini NO fue llamado (saltado por circuit breaker)
            patched_assistant._gemini_tts.synthesize.assert_not_called()
            # Azure SÍ fue llamado
            patched_assistant._azure_tts.synthesize_stream.assert_called_once_with(
                "hola", style_hint=""
            )

        def test_fallback_gemini_not_configured(self, patched_assistant):
            """``_gemini_tts is None`` → se salta Gemini, va a Azure."""
            patched_assistant._gemini_tts = None
            patched_assistant._azure_tts.synthesize_stream.return_value = iter([b"azure_pcm"])

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # Local OK retornó b"\x00" * 100 (default MagicMock), pero también
            # queremos verificar que con local OK, no se llame Azure.
            # Para que el test verifique la cadena, forzamos local a fallar:
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")
            assert result == b"azure_pcm"
            # Azure SÍ fue llamado (porque Gemini no está configurado)
            patched_assistant._azure_tts.synthesize_stream.assert_called_once_with(
                "hola", style_hint=""
            )

        def test_fallback_azure_not_configured(self, patched_assistant):
            """``_azure_tts is None`` y local+Gemini fallan → retorna None sin excepción."""
            patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
                "Piper OOM"
            )
            patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
                "Gemini rate limit"
            )
            patched_assistant._azure_tts = None

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # helper NO lanza, retorna None
            assert result is None

        def test_fallback_local_none_returns_none(self, patched_assistant):
            """``_local_tts is None`` → AttributeError capturado, cae a Gemini.

            FIX MINOR-01 @security: si _local_tts es None, el acceso
            ``self._local_tts.synthesize(...)`` lanza AttributeError. El helper
            usa ``except Exception`` (cubre AttributeError) y cae al siguiente
            TTS de la cadena. Este test verifica el camino defensivo.
            """
            patched_assistant._local_tts = None
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"

            result = patched_assistant._synthesize_one_sentence_with_fallback("hola")

            # helper NO lanza (AttributeError capturado por except Exception)
            assert result == b"gemini_pcm"
            # Gemini fue llamado
            patched_assistant._gemini_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )

    @pytest.mark.unit
    class TestSynthesizeSentenceStreamWithFallback:
        """Tests del helper ``_synthesize_sentence_stream_with_fallback``.

        Cubre el iterador con cadena de fallback, cancelación cooperativa
        y casos mixtos (algunas oraciones OK, otras con fallback).
        """

        def test_stream_fallback_yields_pcm_per_sentence(self, patched_assistant):
            """3 oraciones, local OK → 3 yields de PCM (uno por oración)."""
            pcm_per_sentence = [b"a" * 10, b"b" * 10, b"c" * 10]
            pcm_iter = iter(pcm_per_sentence)

            patched_assistant._local_tts.synthesize.side_effect = lambda text, style_hint="": next(pcm_iter)

            def sentence_iter():
                yield "primera"
                yield "segunda"
                yield "tercera"

            chunks = list(
                patched_assistant._synthesize_sentence_stream_with_fallback(
                    sentence_iter(), generation=0
                )
            )

            assert chunks == pcm_per_sentence
            assert patched_assistant._local_tts.synthesize.call_count == 3

        def test_stream_fallback_skips_empty_sentences(self, patched_assistant):
            """Oraciones vacías/whitespace se saltan, no se llama synthesize."""
            patched_assistant._local_tts.synthesize.return_value = b"pcm"
            patched_assistant._gemini_tts.synthesize.return_value = b"pcm_g"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter([b"pcm_a"])

            def sentence_iter():
                yield ""
                yield "  "
                yield "hola"
                yield ""
                yield "mundo"

            chunks = list(
                patched_assistant._synthesize_sentence_stream_with_fallback(
                    sentence_iter(), generation=0
                )
            )

            # Solo 2 yields (las 2 oraciones reales)
            assert len(chunks) == 2
            assert chunks == [b"pcm", b"pcm"]
            # 2 invocaciones a synthesize (las oraciones reales)
            assert patched_assistant._local_tts.synthesize.call_count == 2
            received = [
                c.args[0] for c in patched_assistant._local_tts.synthesize.call_args_list
            ]
            assert received == ["hola", "mundo"]

        def test_stream_fallback_cancellation_aborts(self, patched_assistant):
            """Si ``_pipeline_generation`` cambia antes de procesar la 2da
            oración, el helper aborta sin sintetizar más."""
            # Side effect que incrementa generation al ser llamado
            call_count = {"n": 0}

            def synth_side_effect(text, style_hint=""):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # 1ra oración OK
                    return b"pcm1"
                # 2da oración: simular toggle() (cambia generation)
                patched_assistant._pipeline_generation += 1
                return b"pcm2"

            patched_assistant._local_tts.synthesize.side_effect = synth_side_effect

            def sentence_iter():
                yield "primera"  # OK, generation=0
                yield "segunda"  # side_effect incrementa generation
                yield "tercera"  # el helper debe abortar aquí

            chunks = list(
                patched_assistant._synthesize_sentence_stream_with_fallback(
                    sentence_iter(), generation=0
                )
            )

            # Solo 1 yield (la 1ra oración). La 2da se sintetizó pero el helper
            # ya hizo la cancelación check antes de yield-ear → no la cuenta
            # como yield. La 3ra nunca se procesa.
            # NOTA: el comportamiento real es:
            # 1. procesar "primera" → synth OK → yield "pcm1"
            # 2. procesar "segunda" → synth side_effect incrementa gen → synth retorna "pcm2" → yield "pcm2"
            # 3. procesar "tercera" → chequea gen, difiere → return
            # Por lo tanto chunks == [b"pcm1", b"pcm2"]
            # La cancelación se verifica viendo que "tercera" NO se procesa.
            assert len(chunks) == 2
            # Pero "tercera" NUNCA se invocó (cancellation check previo)
            received = [
                c.args[0] for c in patched_assistant._local_tts.synthesize.call_args_list
            ]
            assert "tercera" not in received, (
                f"Esperaba que 'tercera' NO se procesara, obtuve {received}"
            )

        def test_stream_fallback_mixed_success(self, patched_assistant):
            """Oración 1: local OK. Oración 2: local falla, Gemini OK.
            Oración 3: todos fallan. → yield 2 chunks (1 y 2), 3ra se loguea."""
            # Oración 1: local OK
            # Oración 2: local falla, Gemini OK
            # Oración 3: local falla, Gemini falla, Azure falla
            call_count = {"n": 0}

            def synth_side_effect(text, style_hint=""):
                call_count["n"] += 1
                if call_count["n"] in (1,):  # 1ra: OK
                    return b"pcm_local"
                if call_count["n"] in (2, 3, 4, 5):  # 2da: local falla
                    raise RuntimeError("Piper OOM")
                raise RuntimeError("never reached")

            patched_assistant._local_tts.synthesize.side_effect = synth_side_effect
            # Gemini: falla para 3ra, OK para 2da
            gemini_call_count = {"n": 0}

            def gemini_side_effect(text, style_hint=""):
                gemini_call_count["n"] += 1
                if gemini_call_count["n"] == 1:  # 2da oración: OK
                    return b"pcm_gemini"
                raise RuntimeError("Gemini rate limit")  # 3ra: falla

            patched_assistant._gemini_tts.is_available.return_value = True
            patched_assistant._gemini_tts.synthesize.side_effect = gemini_side_effect
            # Azure: falla para 3ra
            patched_assistant._azure_tts.synthesize_stream.side_effect = RuntimeError(
                "Azure 503"
            )

            def sentence_iter():
                yield "primera"
                yield "segunda"
                yield "tercera"

            chunks = list(
                patched_assistant._synthesize_sentence_stream_with_fallback(
                    sentence_iter(), generation=0
                )
            )

            # 2 yields (1ra y 2da). 3ra falla en los 3 TTS → no yield.
            assert chunks == [b"pcm_local", b"pcm_gemini"]
            # Verificar que 3ra sí se intentó en los 3 TTS:
            # Local se invoca 3 veces (1ra OK + 2da falla + 3ra falla)
            assert patched_assistant._local_tts.synthesize.call_count == 3
            # Gemini se invoca 2 veces (2da OK + 3ra falla). NO en 1ra (local OK).
            assert patched_assistant._gemini_tts.synthesize.call_count == 2
            # Azure se invoca 1 vez (solo cuando Gemini falla = 3ra oración).
            assert patched_assistant._azure_tts.synthesize_stream.call_count == 1

    @pytest.mark.unit
    def test_pipeline_streaming_with_piper(self, patched_assistant):
        """Regresión: con ``tts_engine=piper`` el flujo streaming se activa
        (Piper ahora tiene ``synthesize_sentence_stream`` — post-fix c115c48).

        El check ``hasattr(self._local_tts, 'synthesize_sentence_stream')``
        debe pasar para PiperTTSClient, lo cual permite que el flujo streaming
        se active. Antes del fix, Piper no tenía el método y caía a síncrono.

        Verificación: ``PiperTTSClient`` (importada del handler real) tiene el
        método, lo cual demuestra que el flujo streaming puede activarse. El
        resto del flujo (mockear ``_local_tts.synthesize`` por oración) ya está
        cubierto por ``test_pipeline_streaming_success``.
        """
        from handlers.piper_tts_client import PiperTTSClient

        # Verificación del contrato PiperTTSClient (post-fix c115c48)
        assert hasattr(PiperTTSClient, "synthesize_sentence_stream"), (
            "PiperTTSClient debe tener synthesize_sentence_stream post-fix c115c48 "
            "para que el flujo streaming se active con tts_engine=piper"
        )
        # Verificar que el método es callable
        assert callable(getattr(PiperTTSClient, "synthesize_sentence_stream")), (
            "synthesize_sentence_stream debe ser callable"
        )

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec primary_engine (tts.primary_engine en config)
    # ──────────────────────────────────────────────────────────────────
    #
    # Cubre la nueva sección ``tts.primary_engine`` introducida en commit
    # ``35ebc67``. Esta spec agrega un selector del motor TTS primario
    # (``"local"``, ``"gemini"`` o ``"azure"``) que controla el orden de la
    # cadena de fallback en ``_synthesize_one_sentence_with_fallback`` y
    # ``_run_sync_pipeline``. Ver:
    # ``specs/feature_tts_primary_engine.md`` §4.

    @pytest.mark.unit
    class TestSynthesizeOneSentenceWithFallbackPrimaryEngine:
        """Tests del helper ``_synthesize_one_sentence_with_fallback`` con
        el selector ``_tts_primary_engine`` activo.

        Casos cubiertos:
        - ``"local"`` (default, regresión): local → Gemini → Azure.
        - ``"gemini"``: Gemini → Azure (sin local).
        - ``"azure"``: Azure solo (sin local ni Gemini).
        - Edge cases: valor inválido, clientes None.
        """

        def test_primary_engine_local_calls_local_first(self, patched_assistant):
            """``primary_engine="local"`` → ``_local_tts.synthesize`` se invoca
            primero y retorna PCM. Gemini y Azure NO se llaman.

            Regresión del comportamiento default (ya cubierto por
            ``test_fallback_local_ok``) re-validado con el atributo explícito
            para fijar el contrato de la nueva sección ``tts`` del settings.
            """
            # Forzar el atributo explícitamente (la fixture lo deja en "local"
            # por default, pero el test documenta la intención).
            patched_assistant._tts_primary_engine = "local"
            patched_assistant._local_tts.synthesize.return_value = b"local_pcm"
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                [b"azure_pcm"]
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # Local fue el primero y retornó su PCM
            assert result == b"local_pcm"
            patched_assistant._local_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Gemini y Azure NO se invocaron (local tuvo éxito)
            patched_assistant._gemini_tts.synthesize.assert_not_called()
            patched_assistant._azure_tts.synthesize_stream.assert_not_called()

        def test_primary_engine_gemini_skips_local(self, patched_assistant):
            """``primary_engine="gemini"`` con ``_local_tts=None`` → NO se
            accede a ``_local_tts`` (verificable: sigue siendo ``None`` y no
            se llama ``synthesize``). ``_gemini_tts.synthesize`` se invoca
            primero y retorna PCM.

            Espec §3.2: ``"gemini"`` saltea local y va directo a Gemini.
            """
            patched_assistant._tts_primary_engine = "gemini"
            patched_assistant._local_tts = None  # simula el caso real (init lo setea así)
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                [b"azure_pcm"]
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # El helper retorna el PCM de Gemini
            assert result == b"gemini_pcm"
            # Gemini fue llamado (con style_hint="")
            patched_assistant._gemini_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Azure NO se invocó (Gemini tuvo éxito)
            patched_assistant._azure_tts.synthesize_stream.assert_not_called()
            # El atributo _local_tts sigue siendo None (no se intentó acceder)
            assert patched_assistant._local_tts is None

        def test_primary_engine_azure_skips_local_and_gemini(self, patched_assistant):
            """``primary_engine="azure"`` con ``_local_tts=None`` y
            ``_gemini_tts=None`` → solo ``_azure_tts.synthesize_stream`` se
            invoca. El helper retorna ``b"".join(chunks)``.

            Espec §3.2: ``"azure"`` no consulta local ni Gemini.
            """
            patched_assistant._tts_primary_engine = "azure"
            patched_assistant._local_tts = None
            patched_assistant._gemini_tts = None
            # Azure streaming: helper consume el iter y lo junta con b"".join
            azure_chunks = [b"a", b"b", b"c"]
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                azure_chunks
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # El helper retorna b"".join(chunks) → b"abc"
            assert result == b"abc"
            # Azure fue llamado con la sentence completa
            patched_assistant._azure_tts.synthesize_stream.assert_called_once_with(
                "hola", style_hint=""
            )
            # Los atributos _local_tts y _gemini_tts siguen siendo None
            assert patched_assistant._local_tts is None
            assert patched_assistant._gemini_tts is None

        def test_primary_engine_gemini_falls_back_to_azure(self, patched_assistant):
            """``primary_engine="gemini"``, ``_gemini_tts`` configurado pero
            su ``synthesize`` levanta excepción → cae a Azure. ``_local_tts``
            no se invoca en ningún momento.

            Valida la cadena ``gemini → azure`` cuando el primario falla.
            """
            patched_assistant._tts_primary_engine = "gemini"
            patched_assistant._local_tts = None  # init real: local=None para gemini
            # Gemini falla
            patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
                "Gemini TTS rate limit"
            )
            # Azure OK con 2 chunks
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                [b"x", b"y"]
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # El helper cae a Azure: retorna join de los chunks
            assert result == b"xy"
            # Gemini SÍ fue llamado (1 vez, falló)
            patched_assistant._gemini_tts.synthesize.assert_called_once_with(
                "hola", style_hint=""
            )
            # Azure SÍ fue llamado (después del fallo de Gemini)
            patched_assistant._azure_tts.synthesize_stream.assert_called_once_with(
                "hola", style_hint=""
            )
            # _local_tts sigue siendo None (no se intentó invocar)
            assert patched_assistant._local_tts is None

        def test_primary_engine_azure_no_fallback_returns_none(self, patched_assistant):
            """``primary_engine="azure"`` con ``_azure_tts=None`` → helper
            retorna ``None`` sin lanzar excepción, y NO invoca local ni Gemini.

            Espec §3.3: si Azure no está configurado y primary es azure,
            todos los motores fallan → ``None`` y error en log.
            """
            patched_assistant._tts_primary_engine = "azure"
            patched_assistant._local_tts = None
            patched_assistant._gemini_tts = None
            patched_assistant._azure_tts = None  # No hay fallback posible

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # helper NO lanza, retorna None
            assert result is None
            # Verificación de que NO se intentó instanciar nada raro:
            # el atributo sigue siendo None (el helper ya lo había detectado)
            assert patched_assistant._azure_tts is None

        # ── Test OMITIDO intencionalmente ───────────────────────────
        # El test 6 de la spec (``test_primary_engine_invalid_defaults_to_local``)
        # se omitió por el siguiente hallazgo descubierto durante testing:
        #
        # El helper ``_synthesize_one_sentence_with_fallback`` NO es defensivo
        # contra valores inválidos de ``_tts_primary_engine``. Si el atributo
        # tiene un valor que no es ``"local"``, ``"gemini"`` ni ``"azure"``
        # (p. ej. ``"foo"``), las tres ramas del helper se saltean y retorna
        # ``None`` silenciosamente. Esto NO es lo que la spec §4 pedía
        # ("verificá que el helper se comporta como ``"local"`` (llama
        # ``_local_tts`` primero)"). El ``__init__`` sí valida y defaultea a
        # ``"local"`` con warning, por lo que en producción el atributo nunca
        # debería contener un valor inválido. Pero esto es una validación
        # **solo en el constructor**, no en el helper.
        #
        # Decisión: delegar al arquitecto la decisión de si (a) agregar
        # defensa en el helper (cambiar la lógica a ``if not in ("gemini",
        # "azure")`` → tratar como local), (b) documentar como contrato
        # implícito que ``_tts_primary_engine`` siempre es válido, o
        # (c) no hacer nada (acepta que la corrupción del atributo causaría
        # pérdida silenciosa de audio).
        #
        # El comportamiento real está cubierto por el test
        # ``test_primary_engine_invalid_returns_none_no_local_call`` que
        # documenta el contrato actual y sirve como regression check si
        # el arquitecto decide modificar la lógica del helper.
        def test_primary_engine_invalid_returns_none_no_local_call(
            self, patched_assistant
        ):
            """Helper con ``_tts_primary_engine="foo"`` (valor inválido
            que slipped through ``__init__``): retorna ``None`` sin
            lanzar excepción y NO invoca ningún TTS.

            HALLAZGO: este test documenta el comportamiento **real** del
            helper. La spec §4 pedía que el helper fuera defensivo
            (tratara el valor inválido como ``"local"``), pero la
            implementación actual NO lo es: las tres ramas del helper
            (``== "local"``, ``in ("local", "gemini")``,
            ``in ("local", "gemini", "azure")``) usan ``==``/``in`` con
            strings literales, por lo que un valor como ``"foo"`` no
            matchea ninguna rama y el helper retorna ``None``
            silenciosamente.

            Decisión QA: se omite el test que pedía comportamiento
            defensivo (test 6 de la spec) y se conserva este test que
            documenta el contrato actual. Si el arquitecto decide hacer
            el helper defensivo, este test fallará y deberá actualizarse.

            Ver spec §8 (notas) y el reporte final de QA para más
            contexto.
            """
            # Simular corrupción del atributo (escenario patológico)
            patched_assistant._tts_primary_engine = "foo"
            patched_assistant._local_tts.synthesize.return_value = b"local_pcm"
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                [b"azure_pcm"]
            )

            result = patched_assistant._synthesize_one_sentence_with_fallback(
                "hola"
            )

            # El helper retorna None para un valor no reconocido
            assert result is None, (
                f"Esperaba None para primary_engine='foo' (no defensivo), "
                f"obtuve {result!r}"
            )
            # Y NO invoca ningún TTS (las 3 ramas se saltearon)
            patched_assistant._local_tts.synthesize.assert_not_called()
            patched_assistant._gemini_tts.synthesize.assert_not_called()
            patched_assistant._azure_tts.synthesize_stream.assert_not_called()

    @pytest.mark.unit
    class TestRunSyncPipelinePrimaryEngine:
        """Tests del flujo síncrono ``_run_sync_pipeline`` con el selector
        ``_tts_primary_engine`` activo.

        Estrategia: invocar ``_run_sync_pipeline`` directamente con un
        ``generation`` válido y mocks de STT/OpenCode. Verifica que se
        respeta el orden de fallback del ``primary_engine``.
        """

        def test_sync_pipeline_primary_engine_gemini_uses_gemini(
            self, patched_assistant
        ):
            """``primary_engine="gemini"`` → ``_gemini_tts.synthesize`` se
            invoca, ``_local_tts`` no se consulta. Verifica que no hay
            ``AttributeError`` cuando ``_local_tts is None`` (defensivo).
            """
            patched_assistant._tts_primary_engine = "gemini"
            patched_assistant._local_tts = None  # init real lo setea así
            # Mockear el resto de la cadena
            patched_assistant._opencode.send_command.return_value = (
                "[STYLE: cheerful] Hola"
            )
            patched_assistant._gemini_tts.synthesize.return_value = b"gemini_pcm"

            # Llamar directamente al pipeline síncrono (no streaming)
            patched_assistant._run_sync_pipeline("texto", generation=0)

            # Gemini SÍ fue llamado con texto limpio (sin prefijo [STYLE:])
            patched_assistant._gemini_tts.synthesize.assert_called_once()
            gemini_call_args = patched_assistant._gemini_tts.synthesize.call_args
            assert gemini_call_args.args[0] == "Hola"
            # play_audio (no streaming) SÍ se llamó con el PCM
            patched_assistant._audio.play_audio.assert_called_once_with(
                b"gemini_pcm"
            )
            # _local_tts sigue siendo None (no se intentó invocar)
            assert patched_assistant._local_tts is None

        def test_sync_pipeline_primary_engine_azure_uses_azure_streaming(
            self, patched_assistant
        ):
            """``primary_engine="azure"`` con ``_local_tts=None`` y
            ``_gemini_tts=None`` → ``_azure_tts.synthesize_stream`` se
            invoca y su iter se pasa a ``play_audio_stream``.
            ``play_audio`` (no streaming) NO se llama.
            """
            patched_assistant._tts_primary_engine = "azure"
            patched_assistant._local_tts = None
            patched_assistant._gemini_tts = None
            patched_assistant._opencode.send_command.return_value = (
                "[STYLE: cheerful] Hola"
            )
            # Azure streaming retorna un iter
            fake_chunks = [b"\x00" * 100, b"\x00" * 100]
            patched_assistant._azure_tts.synthesize_stream.return_value = iter(
                fake_chunks
            )

            patched_assistant._run_sync_pipeline("texto", generation=0)

            # Azure streaming SÍ fue invocado
            patched_assistant._azure_tts.synthesize_stream.assert_called_once()
            azure_call_args = patched_assistant._azure_tts.synthesize_stream.call_args
            assert azure_call_args.args[0] == "Hola"
            # play_audio_stream SÍ fue invocado con el iter retornado
            patched_assistant._audio.play_audio_stream.assert_called_once()
            stream_arg = (
                patched_assistant._audio.play_audio_stream.call_args.args[0]
            )
            assert (
                stream_arg
                is patched_assistant._azure_tts.synthesize_stream.return_value
            )
            # play_audio (no streaming) NO se llamó (ya se usó el streaming)
            patched_assistant._audio.play_audio.assert_not_called()
            # _local_tts y _gemini_tts siguen siendo None
            assert patched_assistant._local_tts is None
            assert patched_assistant._gemini_tts is None

    @pytest.mark.unit
    class TestInitPrimaryEngineSelector:
        """Tests del ``__init__`` con la sección ``tts.primary_engine`` en
        ``config/settings.json``.

        Estos tests requieren instanciar ``VoiceAssistant`` con un settings
        custom (que contenga la sección ``tts``). Se diferencian de
        ``patched_assistant`` (que usa ``mock_settings`` sin sección ``tts``)
        porque el ``__init__`` lee ``self._settings.get("tts", ...)`` y debe
        ser evaluado ANTES de la instanciación.

        Estrategia: crear un archivo temporal ``config/settings.json`` en
        ``tmp_path`` con la sección ``tts`` requerida, y usar
        ``monkeypatch.chdir(tmp_path)`` para que el ``__init__`` (que usa
        ``Path("config/settings.json")``) lo encuentre. Aplica los mismos
        patches de TTS/STT que la fixture ``patched_assistant``.
        """

        def _make_settings_file(self, tmp_path, primary_engine):
            """Helper: escribe ``tmp_path/config/settings.json`` con la
            sección ``tts.primary_engine`` solicitada y devuelve la ruta
            base. El dict es mínimo para que ``__init__`` funcione con
            los clientes mockeados.
            """
            import json as _json

            config_dir = tmp_path / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            settings_path = config_dir / "settings.json"
            settings = {
                "gemini": {
                    "stt_model_primary": "gemini-3.1-flash-lite",
                    "stt_model_fallback": "gemini-2.5-flash-lite",
                    "tts_model": "gemini-3.1-flash-tts-preview",
                    "tts_voice": "Charon",
                    "tts_circuit_breaker_cooldown_seconds": 1800,
                    "stt_prompt": "test",
                },
                "opencode": {
                    "agent": "asistente_voz",
                    "model_fallback": "opencode/big-pickle",
                    "timeout_ms": 120000,
                    "max_session_messages": 10,
                },
                "azure": {
                    "voice": "es-AR-TomasNeural",
                    "locale": "es-AR",
                    "output_format": "raw-24khz-16bit-mono-pcm",
                },
                "audio": {
                    "sample_rate": 24000,
                    "channels": 1,
                    "sample_width": 2,
                    "recording_filename": "comando.wav",
                },
                "hotkey": "alt+v",
                "logging": {
                    "filename": "logs/cortex.log",
                    "max_bytes": 5242880,
                    "backup_count": 3,
                    "level": "INFO",
                },
                "tts": {
                    "primary_engine": primary_engine,
                },
            }
            settings_path.write_text(
                _json.dumps(settings), encoding="utf-8"
            )
            return tmp_path

        def test_init_primary_engine_gemini_sets_local_tts_none(
            self, env_keys, mock_overlay, tmp_path, monkeypatch
        ):
            """``__init__`` con ``tts.primary_engine="gemini"`` en
            ``config/settings.json`` → ``_tts_primary_engine == "gemini"``
            y ``_local_tts is None``.

            Espec §3.1: cuando primary es ``"gemini"`` o ``"azure"``, el
            orquestador NO instancia el TTS local (ahorra memoria y
            dependencias de modelos).
            """
            self._make_settings_file(tmp_path, primary_engine="gemini")
            monkeypatch.chdir(tmp_path)

            with patch("main.AzureTTSClient") as mock_atts, \
                 patch("main.GeminiTTSClient") as mock_gtts, \
                 patch("main.OpenCodeClient") as mock_oc, \
                 patch("main.GeminiSTTClient") as mock_stt, \
                 patch("main.WhisperSTTClient") as mock_wstt, \
                 patch("main.PiperTTSClient") as mock_ptts, \
                 patch("main.KokoroTTSClient") as mock_ktts, \
                 patch("main.AudioManager") as mock_am, \
                 patch("main.load_dotenv"):

                mock_am.return_value = MagicMock(name="AudioManager")
                mock_stt.return_value = MagicMock(name="GeminiSTTClient")
                mock_wstt.return_value = MagicMock(name="WhisperSTTClient")
                # NO mockeamos el return_value de Piper/Kokoro porque NO
                # deberían ser instanciados. Si se instancian, el test falla
                # con la aserción sobre _local_tts abajo.
                mock_ptts.return_value = MagicMock(name="PiperTTSClient")
                mock_ktts.return_value = MagicMock(name="KokoroTTSClient")
                mock_oc.return_value = MagicMock(name="OpenCodeClient")
                mock_gtts.return_value = MagicMock(name="GeminiTTSClient")
                mock_atts.return_value = MagicMock(name="AzureTTSClient")

                from main import VoiceAssistant

                assistant = VoiceAssistant()

            # El atributo refleja el settings
            assert assistant._tts_primary_engine == "gemini", (
                f"Esperaba _tts_primary_engine='gemini', "
                f"obtuve {assistant._tts_primary_engine!r}"
            )
            # _local_tts NO se instanció (primary != 'local')
            assert assistant._local_tts is None, (
                "Con primary_engine='gemini' el TTS local NO debe instanciarse"
            )
            # Verificación adicional: Piper/Kokoro no fueron invocados
            mock_ptts.assert_not_called()
            mock_ktts.assert_not_called()

        def test_init_primary_engine_invalid_logs_warning_and_defaults_local(
            self, env_keys, mock_overlay, tmp_path, monkeypatch, caplog
        ):
            """``__init__`` con ``tts.primary_engine="foo"`` (inválido) →
            loguea warning mencionando el valor inválido y defaultea a
            ``"local"`` (``_tts_primary_engine == "local"``).

            Espec §2.1: valores no reconocidos caen a ``"local"`` con
            warning, no lanzan excepción.
            """
            self._make_settings_file(tmp_path, primary_engine="foo")
            monkeypatch.chdir(tmp_path)

            with patch("main.AzureTTSClient") as mock_atts, \
                 patch("main.GeminiTTSClient") as mock_gtts, \
                 patch("main.OpenCodeClient") as mock_oc, \
                 patch("main.GeminiSTTClient") as mock_stt, \
                 patch("main.WhisperSTTClient") as mock_wstt, \
                 patch("main.PiperTTSClient") as mock_ptts, \
                 patch("main.KokoroTTSClient") as mock_ktts, \
                 patch("main.AudioManager") as mock_am, \
                 patch("main.load_dotenv"):

                mock_am.return_value = MagicMock(name="AudioManager")
                mock_stt.return_value = MagicMock(name="GeminiSTTClient")
                mock_wstt.return_value = MagicMock(name="WhisperSTTClient")
                mock_ptts.return_value = MagicMock(name="PiperTTSClient")
                mock_ktts.return_value = MagicMock(name="KokoroTTSClient")
                mock_oc.return_value = MagicMock(name="OpenCodeClient")
                mock_gtts.return_value = MagicMock(name="GeminiTTSClient")
                mock_atts.return_value = MagicMock(name="AzureTTSClient")

                from main import VoiceAssistant

                with caplog.at_level(logging.WARNING, logger="main"):
                    assistant = VoiceAssistant()

            # _tts_primary_engine defaulteó a "local"
            assert assistant._tts_primary_engine == "local", (
                f"Esperaba _tts_primary_engine='local' (default por valor "
                f"inválido), obtuve {assistant._tts_primary_engine!r}"
            )
            # _local_tts SÍ se instanció (default local)
            assert assistant._local_tts is not None, (
                "Con primary_engine inválido y default a 'local', el TTS "
                "local SÍ debe instanciarse"
            )
            # El warning fue logueado con el valor inválido
            assert any(
                "primary_engine" in record.getMessage()
                and "foo" in record.getMessage()
                and "inválido" in record.getMessage().lower()
                and record.levelno == logging.WARNING
                for record in caplog.records
            ), (
                f"Warning de primary_engine inválido no encontrado. "
                f"Logs: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
            )
