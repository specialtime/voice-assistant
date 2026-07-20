"""Tests de integración del pipeline STT/TTS local con cadena de failover.

Verifica end-to-end (sin red, sin GPU) que ``VoiceAssistant.run_pipeline``
compone correctamente la cadena de fallback del orquestador:

    STT:  Whisper (local, primario)  →  Gemini (cloud, fallback)
    TTS:  _local_tts (piper|kokoro, local primario)
        → Gemini (cloud, fallback 1)
        → Azure streaming (cloud, fallback 2)

Casos cubiertos (todos ``@pytest.mark.unit``):
    1. test_stt_whisper_primary_gemini_not_called
    2. test_stt_whisper_fails_gemini_fallback
    3. test_stt_both_fail_raises
    4. test_tts_piper_primary_no_cloud_called
    5. test_tts_piper_fails_gemini_fallback
    6. test_tts_piper_gemini_fail_azure_streaming
    7. test_tts_all_fail_no_playback
    8. test_pipeline_full_local_success
    9. Selector de motor TTS (piper | kokoro) — TestTTSEngineSelector

Implementación técnica:
    ``faster_whisper``, ``piper`` y ``kokoro_onnx`` no están instalados
    en este entorno (los wheels de ``av``/``onnxruntime`` no se
    construyen en Windows con Python 3.14). Para evitar errores de
    ``import`` top-level en los handlers locales, registramos
    ``MagicMock`` en ``sys.modules`` ANTES de importar ``main`` o
    cualquier handler. Esto preserva el contrato de los handlers (sus
    tests originales mockean con ``@patch`` dentro de cada test) y
    permite que el orquestador se instancie sin tocar dependencias
    reales.

    Patrón inspirado en ``tests/test_state_machine.py::patched_assistant``
    (misma idea, fixture local y auto-contenida para evitar acoplamiento
    entre archivos de tests).
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────
# Path bootstrap (por si conftest no fue recogido)
# ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SRC), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────────
# Stub de dependencias locales: faster_whisper + piper
#
# Los handlers ``whisper_stt_client`` y ``piper_tts_client`` hacen
# ``from faster_whisper import WhisperModel`` y ``import piper`` /
# ``from piper.download_voices import download_voice`` en el top-level
# del módulo. Si las wheels reales no están instaladas, esos imports
# rompen toda la colección de pytest ANTES de que el body del test
# corra.
#
# Solución: registrar MagicMock en sys.modules para que los imports
# resuelvan sin tocar el disco ni la red. Dentro de cada test se
# siguen usando ``@patch`` / atributos de instancia para inyectar
# comportamiento.
# ──────────────────────────────────────────────────────────────────


def _ensure_stub_modules() -> None:
    """Registra módulos stub para ``faster_whisper`` y ``piper`` en ``sys.modules``.

    Idempotente — si ya están registrados, no hace nada.
    """
    if "faster_whisper" not in sys.modules:
        fw_stub = types.ModuleType("faster_whisper")
        fw_stub.WhisperModel = MagicMock(name="WhisperModel")
        sys.modules["faster_whisper"] = fw_stub

    if "piper" not in sys.modules:
        piper_stub = types.ModuleType("piper")
        piper_stub.PiperVoice = MagicMock(name="PiperVoice")
        piper_stub.SynthesisConfig = MagicMock(name="SynthesisConfig")
        # piper.download_voices es submódulo (API real de piper-tts 1.4.2)
        piper_download_stub = types.ModuleType("piper.download_voices")
        piper_download_stub.download_voice = MagicMock(name="download_voice")
        piper_stub.download_voices = piper_download_stub
        sys.modules["piper"] = piper_stub
        sys.modules["piper.download_voices"] = piper_download_stub

    if "kokoro_onnx" not in sys.modules:
        kokoro_stub = types.ModuleType("kokoro_onnx")
        kokoro_stub.Kokoro = MagicMock(name="Kokoro")
        sys.modules["kokoro_onnx"] = kokoro_stub


_ensure_stub_modules()


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def env_keys(monkeypatch):
    """Setea las env vars mínimas para que ``VoiceAssistant`` cree todos los clientes."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "fake_azure_key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "southamericaeast")
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "fake_opencode_pass")
    monkeypatch.setenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")


@pytest.fixture
def local_settings(mock_settings: dict) -> dict:
    """Extiende ``mock_settings`` con la sección ``local.{whisper,piper}``.

    Los handlers reales leen ``settings['local']['whisper']`` y
    ``settings['local']['piper']``. Aunque aquí no se usen los
    handlers reales (se mockean vía ``patched_assistant``), este dict
    está disponible por si algún test quiere instanciar un handler
    de forma aislada.
    """
    settings = dict(mock_settings)
    settings["local"] = {
        "tts_engine": "piper",
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8",
            "language": "es",
            "beam_size": 5,
        },
        "piper": {
            "voice_model": "es_AR-daniela-high",
            "voices_dir": "models/piper-voices",
            "download_url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0",
            "length_scale": 1.0,
        },
        "kokoro": {
            "model_path": "models/kokoro/kokoro-v1.0.onnx",
            "voices_path": "models/kokoro/voices-v1.0.bin",
            "voice": "em_alex",
            "lang": "es",
            "speed": 1.0,
        },
    }
    return settings


@pytest.fixture
def patched_assistant(env_keys, local_settings, mock_overlay, monkeypatch):
    """Crea un ``VoiceAssistant`` con todas las dependencias externas mockeadas.

    Mocks aplicados (todos a nivel de ``main``):
        - AudioManager
        - GeminiSTTClient
        - WhisperSTTClient
        - OpenCodeClient
        - GeminiTTSClient
        - PiperTTSClient
        - AzureTTSClient
        - OverlayChip (vía fixture ``mock_overlay``)
        - load_dotenv (no toca el ``.env`` real)

    Cada mock se rellena con ``MagicMock`` ``name=...`` para que las
    aserciones con ``assert_called_*`` tengan nombres legibles.

    Yields:
        Instancia de ``VoiceAssistant`` lista para ejecutar ``run_pipeline``.
    """
    monkeypatch.chdir(_PROJECT_ROOT)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("main.AzureTTSClient", MagicMock(name="AzureTTSClient"))
        mp.setattr("main.GeminiTTSClient", MagicMock(name="GeminiTTSClient"))
        mp.setattr("main.OpenCodeClient", MagicMock(name="OpenCodeClient"))
        mp.setattr("main.GeminiSTTClient", MagicMock(name="GeminiSTTClient"))
        mp.setattr("main.WhisperSTTClient", MagicMock(name="WhisperSTTClient"))
        mp.setattr("main.PiperTTSClient", MagicMock(name="PiperTTSClient"))
        mp.setattr("main.KokoroTTSClient", MagicMock(name="KokoroTTSClient"))
        mp.setattr("main.AudioManager", MagicMock(name="AudioManager"))
        mp.setattr("main.load_dotenv", lambda: None)

        # Re-leer las clases mockeadas ya configuradas como ``return_value``
        from main import AzureTTSClient, GeminiTTSClient, OpenCodeClient, GeminiSTTClient
        from main import WhisperSTTClient, PiperTTSClient, KokoroTTSClient, AudioManager, VoiceAssistant

        AudioManager.return_value = MagicMock(name="AudioManagerInstance")
        GeminiSTTClient.return_value = MagicMock(name="GeminiSTTClientInstance")
        WhisperSTTClient.return_value = MagicMock(name="WhisperSTTClientInstance")
        OpenCodeClient.return_value = MagicMock(name="OpenCodeClientInstance")
        GeminiTTSClient.return_value = MagicMock(name="GeminiTTSClientInstance")
        PiperTTSClient.return_value = MagicMock(name="PiperTTSClientInstance")
        KokoroTTSClient.return_value = MagicMock(name="KokoroTTSClientInstance")
        AzureTTSClient.return_value = MagicMock(name="AzureTTSClientInstance")

        assistant = VoiceAssistant()
        # Sobrescribir settings con los sintéticos (no tocar el JSON real)
        assistant._settings = local_settings
        # Forzar flujo síncrono: estos tests verifican el camino síncrono
        # (send_command + synthesize + play_audio), no el streaming SSE.
        # El flujo streaming tiene cobertura dedicada en test_state_machine.py
        # (tests T6-T9: test_pipeline_streaming_*).
        # El default en config/settings.json es streaming_enabled=True, pero
        # este fixture no incluye esa key en local_settings['opencode'], por
        # lo que el orquestador cae al default True del .get(...).
        # FIX: setear explícitamente False para mantener los tests estables
        # sin requerir que el JSON real tenga streaming_enabled=false.
        assistant._streaming_enabled = False

        yield assistant


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _wire_successful_agent(assistant, text: str = "abrí chrome", response: str | None = None) -> None:
    """Helper: conecta STT-Whisper + agente para que el pipeline avance hasta TTS.

    Args:
        assistant: Instancia del orquestador.
        text: Texto que retornará Whisper STT.
        response: Texto que retornará el agente (incluye ``[STYLE: ...]``).
            Si es ``None``, usa uno por defecto.
    """
    if response is None:
        response = "[STYLE: cheerful] Listo, abrí Chrome"
    assistant._whisper_stt.transcribe.return_value = text
    assistant._opencode.send_command.return_value = response


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSTTFailoverChain:
    """Suite: cadena de fallback de STT (Whisper → Gemini)."""

    def test_stt_whisper_primary_gemini_not_called(self, patched_assistant):
        """Whisper OK → Gemini NUNCA se invoca. El agente recibe el texto de Whisper."""
        # Arrange
        _wire_successful_agent(patched_assistant, text="abrí chrome")
        # Piper OK para no chocar con la rama TTS (este test es de STT)
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._whisper_stt.transcribe.assert_called_once_with("/tmp/fake.wav")
        patched_assistant._stt.transcribe.assert_not_called()
        # El agente recibió el texto que Whisper transcribió
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")

    def test_stt_whisper_fails_gemini_fallback(self, patched_assistant):
        """Whisper lanza excepción → Gemini STT se invoca y retorna texto.

        El agente debe recibir el texto que Gemini transcribió.
        """
        # Arrange
        patched_assistant._whisper_stt.transcribe.side_effect = RuntimeError("CUDA OOM")
        patched_assistant._stt.transcribe.return_value = "abrí chrome"
        patched_assistant._opencode.send_command.return_value = "[STYLE: cheerful] Listo"
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_called_once_with("/tmp/fake.wav")
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")

    def test_stt_both_fail_raises(self, patched_assistant, caplog):
        """Whisper falla Y Gemini falla → pipeline termina sin crash, estado IDLE.

        Comportamiento esperado: si Gemini también lanza excepción, el
        ``except`` interno del bloque STT NO la captura (solo captura la
        de Whisper), así que la excepción propaga al ``except`` externo
        que loguea y va al ``finally``. El estado vuelve a IDLE.
        """
        # Arrange
        patched_assistant._whisper_stt.transcribe.side_effect = RuntimeError("Whisper falló")
        patched_assistant._stt.transcribe.side_effect = RuntimeError("Gemini también falló")

        # Act
        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert — sin crash, estado IDLE, agente nunca invocado
        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_called_once()
        patched_assistant._opencode.send_command.assert_not_called()
        patched_assistant._local_tts.synthesize.assert_not_called()
        assert patched_assistant._state == patched_assistant.STATE_IDLE


@pytest.mark.unit
class TestTTSFailoverChain:
    """Suite: cadena de fallback de TTS (Piper → Gemini → Azure streaming)."""

    def test_tts_piper_primary_no_cloud_called(self, patched_assistant):
        """Piper OK → Gemini y Azure NUNCA se invocan. Se reproduce con ``play_audio``."""
        # Arrange
        _wire_successful_agent(patched_assistant)
        pcm = b"\x00" * 48000
        patched_assistant._local_tts.synthesize.return_value = pcm

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        # Texto limpio (sin prefijo [STYLE:]) va a Piper
        piper_args = patched_assistant._local_tts.synthesize.call_args
        assert piper_args.args[0] == "Listo, abrí Chrome"
        # Gemini y Azure NO se invocan
        patched_assistant._gemini_tts.synthesize.assert_not_called()
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()
        # Se reproduce por play_audio (no streaming)
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_piper_fails_gemini_fallback(self, patched_assistant):
        """Piper falla → Gemini TTS OK → no se invoca Azure, sí ``play_audio``."""
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        pcm = b"\x00" * 48000
        patched_assistant._gemini_tts.synthesize.return_value = pcm
        patched_assistant._gemini_tts.is_available.return_value = True

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.is_available.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # Texto limpio va a Gemini (mismo que a Piper)
        gemini_args = patched_assistant._gemini_tts.synthesize.call_args
        assert gemini_args.args[0] == "Listo, abrí Chrome"
        # Azure NO se invoca
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()
        # Se reproduce por play_audio (no streaming)
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_piper_gemini_fail_azure_streaming(self, patched_assistant):
        """Piper y Gemini fallan → Azure streaming se invoca con ``play_audio_stream``.

        ``play_audio`` NO se llama (Azure streaming ya reprodujo).
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        # Azure streaming: retorna iterator de chunks PCM
        fake_pcm_chunks = [b"\x00\x01" * 100, b"\x00\x01" * 100]
        patched_assistant._azure_tts.synthesize_stream.return_value = iter(fake_pcm_chunks)

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.is_available.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # Azure streaming recibe el texto limpio
        patched_assistant._azure_tts.synthesize_stream.assert_called_once()
        azure_args = patched_assistant._azure_tts.synthesize_stream.call_args
        assert azure_args.args[0] == "Listo, abrí Chrome"
        # Se pasa el iterator de Azure a play_audio_stream
        patched_assistant._audio.play_audio_stream.assert_called_once_with(
            patched_assistant._azure_tts.synthesize_stream.return_value
        )
        # play_audio NO se llama (Azure streaming ya reprodujo)
        patched_assistant._audio.play_audio.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_all_fail_no_playback(self, patched_assistant, caplog):
        """Piper, Gemini y Azure todos fallan / no disponibles → sin playback, IDLE.

        Cubre dos sub-caminos:
          a) Azure configurado pero ``synthesize_stream`` lanza excepción.
          b) Azure es ``None`` (no configurado) → el bloque ``except``
             externo entra en juego (Gemini ``RuntimeError`` propaga).
        Aquí probamos (a) porque es la semántica más realista: las keys
        están configuradas (ver ``env_keys``) y aún así el servicio
        devuelve error.
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        patched_assistant._azure_tts.synthesize_stream.side_effect = RuntimeError(
            "Azure 5xx"
        )

        # Act
        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        patched_assistant._azure_tts.synthesize_stream.assert_called_once()
        # Sin playback por ninguna vía
        patched_assistant._audio.play_audio.assert_not_called()
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_azure_none_when_piper_and_gemini_fail(self, patched_assistant, caplog):
        """Variante de ``test_tts_all_fail_no_playback``: Azure NO configurado.

        Si ``_azure_tts is None``, el código retorna antes de invocar
        Azure y queda en estado IDLE. El código emite un único log
        genérico de fallo total (``"Todos los TTS fallaron..."``); no
        existe un log específico de "Azure no configurado" en esta
        versión (se omite intencionalmente para simplificar el manejo
        del failover).
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        # Forzar Azure None (escenario "no configurado")
        patched_assistant._azure_tts = None

        # Act
        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # Azure NO se invoca (es None)
        assert patched_assistant._azure_tts is None
        # Sin playback por ninguna vía
        patched_assistant._audio.play_audio.assert_not_called()
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Log genérico de fallo total (verifica que se diagnosticó el fallo)
        assert any(
            "Todos los TTS fallaron" in r.getMessage()
            for r in caplog.records
        ), (
            "Log 'Todos los TTS fallaron...' no encontrado. "
            f"Logs: {[r.getMessage() for r in caplog.records]}"
        )
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_gemini_circuit_breaker_open(self, patched_assistant):
        """Piper falla Y ``gemini.is_available() == False`` → Azure streaming.

        El bloque ``except`` interno de Gemini se dispara con
        ``RuntimeError("Gemini TTS circuit breaker abierto")``, saltando
        directo a Azure. Esto valida que el circuit breaker NO bloquea
        el failover (solo cambia el path).
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.is_available.return_value = False
        fake_pcm_chunks = [b"\x00\x01" * 50]
        patched_assistant._azure_tts.synthesize_stream.return_value = iter(fake_pcm_chunks)

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._gemini_tts.is_available.assert_called_once()
        # Gemini TTS NO se invoca porque el circuit breaker está abierto
        patched_assistant._gemini_tts.synthesize.assert_not_called()
        # Azure streaming sí
        patched_assistant._azure_tts.synthesize_stream.assert_called_once()
        patched_assistant._audio.play_audio_stream.assert_called_once()
        patched_assistant._audio.play_audio.assert_not_called()
        assert patched_assistant._state == patched_assistant.STATE_IDLE


@pytest.mark.unit
class TestPipelineFullLocal:
    """Suite: pipeline completo en modo local (sin tocar cloud)."""

    def test_pipeline_full_local_success(self, patched_assistant):
        """Whisper OK + Piper OK → NUNCA se llama Gemini STT/TTS ni Azure streaming.

        Es el caso "happy path" del modo local-first: todo el procesamiento
        se queda on-device y no hay tráfico cloud.
        """
        # Arrange
        _wire_successful_agent(patched_assistant, text="abrí chrome")
        patched_assistant._local_tts.synthesize.return_value = b"\x00" * 48000

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert — los 3 clientes cloud NO se invocan
        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_not_called()  # Gemini STT no se llama
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_not_called()  # Gemini TTS no se llama
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()  # Azure no se llama

        # El playback se hace por play_audio (no streaming) con bytes PCM
        patched_assistant._audio.play_audio.assert_called_once()
        patched_assistant._audio.play_audio_stream.assert_not_called()

        # Estado final: IDLE (rama exitosa del finally)
        assert patched_assistant._state == patched_assistant.STATE_IDLE

        # El overlay se ocultó (cleanup del finally)
        patched_assistant._overlay.hide.assert_called_once()


# ──────────────────────────────────────────────────────────────────
# Selector de motor TTS local (piper | kokoro)
#
# Estos tests verifican que ``VoiceAssistant.__init__`` elige el motor
# TTS correcto según ``settings['local']['tts_engine']``. Como el
# orquestador lee ``config/settings.json`` del disco durante la
# construcción, los tests mockean ``builtins.open`` para inyectar un
# settings sintético sin tocar el archivo real.
# ──────────────────────────────────────────────────────────────────


def _build_kokoro_settings_dict() -> dict:
    """Helper: settings con ``tts_engine='kokoro'`` para tests de selector."""
    return {
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
            "voice": "es-MX-JorgeNeural",
            "locale": "es-MX",
            "output_format": "raw-24khz-16bit-mono-pcm",
        },
        "local": {
            "tts_engine": "kokoro",
            "whisper": {
                "model": "small",
                "device": "cuda",
                "compute_type": "int8",
                "language": "es",
                "beam_size": 5,
            },
            "piper": {
                "voice_model": "es_AR-daniela-high",
                "voices_dir": "models/piper-voices",
                "length_scale": 1.0,
            },
            "kokoro": {
                "model_path": "models/kokoro/kokoro-v1.0.onnx",
                "voices_path": "models/kokoro/voices-v1.0.bin",
                "voice": "em_alex",
                "lang": "es",
                "speed": 1.0,
            },
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
            "level": "DEBUG",
        },
    }


def _build_piper_settings_dict() -> dict:
    """Helper: settings con ``tts_engine='piper'`` (default) para tests de selector."""
    s = _build_kokoro_settings_dict()
    s["local"]["tts_engine"] = "piper"
    return s


def _build_invalid_engine_settings_dict() -> dict:
    """Helper: settings con ``tts_engine='invalid'`` → debe caer a piper."""
    s = _build_kokoro_settings_dict()
    s["local"]["tts_engine"] = "invalid_engine_xyz"
    return s


@pytest.mark.unit
class TestTTSEngineSelector:
    """Suite: selector de motor TTS local (piper | kokoro).

    Verifica que ``VoiceAssistant.__init__`` instancia el cliente TTS
    correcto según ``settings['local']['tts_engine']`` y que el valor
    inválido cae con warning al default (piper).
    """

    def test_selector_piper_default(self, patched_assistant):
        """``tts_engine='piper'`` (default) → ``_local_tts`` se construye
        vía ``PiperTTSClient`` y ``KokoroTTSClient`` NUNCA se instancia."""
        # Assert: el selector eligió Piper, no Kokoro
        from main import PiperTTSClient, KokoroTTSClient

        PiperTTSClient.assert_called_once()
        KokoroTTSClient.assert_not_called()
        # El atributo ``_local_tts`` debe ser la instancia mock de Piper
        assert patched_assistant._local_tts is PiperTTSClient.return_value

    def test_selector_kokoro(self, env_keys, mock_overlay, monkeypatch):
        """``tts_engine='kokoro'`` → ``_local_tts`` se construye vía
        ``KokoroTTSClient`` y ``PiperTTSClient`` NUNCA se instancia."""
        import json
        from unittest.mock import mock_open

        monkeypatch.chdir(_PROJECT_ROOT)

        kokoro_settings = _build_kokoro_settings_dict()
        m = mock_open(read_data=json.dumps(kokoro_settings))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("main.AzureTTSClient", MagicMock(name="AzureTTSClient"))
            mp.setattr("main.GeminiTTSClient", MagicMock(name="GeminiTTSClient"))
            mp.setattr("main.OpenCodeClient", MagicMock(name="OpenCodeClient"))
            mp.setattr("main.GeminiSTTClient", MagicMock(name="GeminiSTTClient"))
            mp.setattr("main.WhisperSTTClient", MagicMock(name="WhisperSTTClient"))
            mp.setattr("main.PiperTTSClient", MagicMock(name="PiperTTSClient"))
            mp.setattr("main.KokoroTTSClient", MagicMock(name="KokoroTTSClient"))
            mp.setattr("main.AudioManager", MagicMock(name="AudioManager"))
            mp.setattr("main.load_dotenv", lambda: None)
            # Mockear open() para devolver JSON con tts_engine='kokoro'
            mp.setattr("builtins.open", m)

            from main import (
                VoiceAssistant,
                PiperTTSClient,
                KokoroTTSClient,
                AudioManager,
            )

            AudioManager.return_value = MagicMock(name="AudioManagerInstance")
            PiperTTSClient.return_value = MagicMock(name="PiperTTSClientInstance")
            KokoroTTSClient.return_value = MagicMock(name="KokoroTTSClientInstance")

            assistant = VoiceAssistant()

            # Assert: el selector eligió Kokoro, no Piper
            KokoroTTSClient.assert_called_once()
            PiperTTSClient.assert_not_called()
            assert assistant._local_tts is KokoroTTSClient.return_value

    def test_selector_invalid_falls_back_to_piper(
        self, env_keys, mock_overlay, monkeypatch, caplog
    ):
        """``tts_engine='invalid_xyz'`` → warning logueado y ``PiperTTSClient`` se instancia."""
        import json
        from unittest.mock import mock_open

        monkeypatch.chdir(_PROJECT_ROOT)

        invalid_settings = _build_invalid_engine_settings_dict()
        m = mock_open(read_data=json.dumps(invalid_settings))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("main.AzureTTSClient", MagicMock(name="AzureTTSClient"))
            mp.setattr("main.GeminiTTSClient", MagicMock(name="GeminiTTSClient"))
            mp.setattr("main.OpenCodeClient", MagicMock(name="OpenCodeClient"))
            mp.setattr("main.GeminiSTTClient", MagicMock(name="GeminiSTTClient"))
            mp.setattr("main.WhisperSTTClient", MagicMock(name="WhisperSTTClient"))
            mp.setattr("main.PiperTTSClient", MagicMock(name="PiperTTSClient"))
            mp.setattr("main.KokoroTTSClient", MagicMock(name="KokoroTTSClient"))
            mp.setattr("main.AudioManager", MagicMock(name="AudioManager"))
            mp.setattr("main.load_dotenv", lambda: None)
            mp.setattr("builtins.open", m)

            from main import (
                VoiceAssistant,
                PiperTTSClient,
                KokoroTTSClient,
                AudioManager,
            )

            AudioManager.return_value = MagicMock(name="AudioManagerInstance")
            PiperTTSClient.return_value = MagicMock(name="PiperTTSClientInstance")
            KokoroTTSClient.return_value = MagicMock(name="KokoroTTSClientInstance")

            with caplog.at_level(logging.WARNING, logger="main"):
                assistant = VoiceAssistant()

            # Assert: cayó a Piper con warning
            PiperTTSClient.assert_called_once()
            KokoroTTSClient.assert_not_called()
            assert assistant._local_tts is PiperTTSClient.return_value
            # El warning debe mencionar el engine inválido y el fallback
            assert any(
                "tts_engine" in r.getMessage().lower()
                and "inv" in r.getMessage().lower()
                and "piper" in r.getMessage().lower()
                for r in caplog.records
            ), (
                f"Warning de tts_engine inválido no encontrado. "
                f"Logs: {[r.getMessage() for r in caplog.records]}"
            )

    def test_tts_kokoro_primary_no_cloud_called(self, patched_assistant):
        """Kokoro (representado por ``_local_tts`` mockeado) OK → Gemini y Azure NO se invocan.
        Se reproduce con ``play_audio``.
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        pcm = b"\x00" * 48000
        patched_assistant._local_tts.synthesize.return_value = pcm

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        # Texto limpio (sin prefijo [STYLE:]) va al TTS local
        local_args = patched_assistant._local_tts.synthesize.call_args
        assert local_args.args[0] == "Listo, abrí Chrome"
        # Gemini y Azure NO se invocan
        patched_assistant._gemini_tts.synthesize.assert_not_called()
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()
        # Se reproduce por play_audio (no streaming)
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_kokoro_fails_gemini_fallback(self, patched_assistant):
        """``_local_tts`` (Kokoro) falla → Gemini TTS OK → no se invoca Azure."""
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
            "Kokoro falló"
        )
        pcm = b"\x00" * 48000
        patched_assistant._gemini_tts.synthesize.return_value = pcm
        patched_assistant._gemini_tts.is_available.return_value = True

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.is_available.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # Texto limpio va a Gemini
        gemini_args = patched_assistant._gemini_tts.synthesize.call_args
        assert gemini_args.args[0] == "Listo, abrí Chrome"
        # Azure NO se invoca
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()
        # Se reproduce por play_audio
        patched_assistant._audio.play_audio.assert_called_once_with(pcm)
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE

    def test_tts_kokoro_gemini_fail_azure_streaming(self, patched_assistant):
        """``_local_tts`` (Kokoro) y Gemini fallan → Azure streaming se invoca con
        ``play_audio_stream``.
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._local_tts.synthesize.side_effect = RuntimeError(
            "Kokoro falló"
        )
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError(
            "Gemini TTS falló"
        )
        patched_assistant._gemini_tts.is_available.return_value = True
        # Azure streaming: retorna iterator de chunks PCM
        fake_pcm_chunks = [b"\x00\x01" * 100, b"\x00\x01" * 100]
        patched_assistant._azure_tts.synthesize_stream.return_value = iter(
            fake_pcm_chunks
        )

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._local_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.is_available.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        patched_assistant._azure_tts.synthesize_stream.assert_called_once()
        azure_args = patched_assistant._azure_tts.synthesize_stream.call_args
        assert azure_args.args[0] == "Listo, abrí Chrome"
        # Se pasa el iterator de Azure a play_audio_stream
        patched_assistant._audio.play_audio_stream.assert_called_once()
        # play_audio NO se llama (Azure streaming ya reprodujo)
        patched_assistant._audio.play_audio.assert_not_called()
        # Estado final IDLE
        assert patched_assistant._state == patched_assistant.STATE_IDLE