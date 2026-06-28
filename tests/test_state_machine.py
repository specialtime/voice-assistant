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
        # Log de error de Azure no configurado
        assert any(
            "azure" in record.getMessage().lower()
            and "no configurado" in record.getMessage().lower()
            for record in caplog.records
        ), f"Log de Azure no configurado no encontrado. Logs: {[r.getMessage() for r in caplog.records]}"
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
        """Flujo streaming exitoso: send_command_stream + synthesize_sentence_stream
        + play_audio_stream se invocan en orden. NO se llama send_command (síncrono).
        """
        # Forzar streaming habilitado (la fixture usa mock_settings sin esta key,
        # por lo que el __init__ toma el default True desde el settings.json real)
        patched_assistant._streaming_enabled = True

        # STT OK
        patched_assistant._whisper_stt.transcribe.return_value = "abrí chrome"

        # OpenCode streaming: retorna un iter de deltas
        def delta_iter():
            yield "[STYLE: cheerful] Hola. "
            yield "Chau. "

        patched_assistant._opencode.send_command_stream.return_value = delta_iter()
        # OpenCode send_command (síncrono) NO debe ser llamado
        patched_assistant._opencode.send_command.return_value = "NO DEBE LLAMARSE"

        # Kokoro streaming: retorna un iter de PCM (1 chunk por oración)
        pcm_chunks = [b"\x00" * 100, b"\x00" * 100]  # 2 oraciones → 2 chunks
        patched_assistant._local_tts.synthesize_sentence_stream.return_value = iter(pcm_chunks)

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se invocó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_called_once_with("abrí chrome")
        # send_command (síncrono) NO fue llamado
        patched_assistant._opencode.send_command.assert_not_called()
        # Kokoro streaming fue invocado
        patched_assistant._local_tts.synthesize_sentence_stream.assert_called_once()
        # play_audio_stream recibió los chunks PCM
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

        Para que el test sea realista, mockeamos ``synthesize_sentence_stream``
        como PROXY del sentence_iterator que recibe: consume el iter (lo que
        fuerza al ``sentence_iterator`` interno a iterar el delta_stream y
        disparar la cancelación) pero produce 0 PCM (porque la cancelación
        abortó antes del primer yield).
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

        # Mock synthesize_sentence_stream como PROXY: consume el sentence_iterator
        # para forzar la iteración del delta_stream (y disparar la cancelación),
        # pero produce 0 PCM (la cancelación abortó antes del primer yield).
        def sentence_stream_proxy(sentences):
            # Consumir el iter → fuerza avance del sentence_iterator →
            # fuerza avance del delta_iter → dispara _pipeline_generation += 1
            # → el sentence_iterator interno aborta por cancellation check.
            consumed_sentences = list(sentences)
            # Retornar iter VACÍO: 0 oraciones llegaron a Kokoro.
            return iter([])

        patched_assistant._local_tts.synthesize_sentence_stream.side_effect = (
            sentence_stream_proxy
        )

        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Se intentó el flujo streaming
        patched_assistant._opencode.send_command_stream.assert_called_once()
        # Kokoro streaming SÍ se invocó (la pipeline todavía entra al with send_lock)
        patched_assistant._local_tts.synthesize_sentence_stream.assert_called_once()
        # El delta_iter SÍ fue consumido (cancellation disparó)
        assert patched_assistant._pipeline_generation >= 1, (
            f"Esperaba _pipeline_generation>=1 (cancelación disparada), "
            f"se obtuvo {patched_assistant._pipeline_generation}"
        )
        # play_audio_stream fue llamado (con iter vacío → no reproduce nada)
        patched_assistant._audio.play_audio_stream.assert_called_once()
        # El estado NO fue pisado a IDLE por el finally (generación difiere)
        assert patched_assistant._state != patched_assistant.STATE_IDLE, (
            f"Esperaba estado != IDLE (cancelación), se obtuvo {patched_assistant._state!r}"
        )
        # overlay.hide NO fue llamado (cancelación)
        patched_assistant._overlay.hide.assert_not_called()
