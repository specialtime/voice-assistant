"""Tests de integración del pipeline STT/TTS local con cadena de failover.

Verifica end-to-end (sin red, sin GPU) que ``VoiceAssistant.run_pipeline``
compone correctamente la cadena de fallback del orquestador:

    STT:  Whisper (local, primario)  →  Gemini (cloud, fallback)
    TTS:  Piper  (local, primario)  →  Gemini (cloud, fallback 1)
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

Implementación técnica:
    ``faster_whisper`` y ``piper`` no están instalados en este entorno
    (los wheels de ``av``/``onnxruntime`` no se construyen en Windows con
    Python 3.14). Para evitar errores de ``import`` top-level en los
    handlers locales, registramos ``MagicMock`` en ``sys.modules`` ANTES
    de importar ``main`` o cualquier handler. Esto preserva el contrato
    de los handlers (sus tests originales mockean con ``@patch`` dentro
    de cada test) y permite que el orquestador se instancie sin tocar
    dependencias reales.

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
# ``from piper.download import ensure_voice_exists`` en el top-level
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
        # piper.download es submódulo
        piper_download_stub = types.ModuleType("piper.download")
        piper_download_stub.ensure_voice_exists = MagicMock(name="ensure_voice_exists")
        piper_stub.download = piper_download_stub
        sys.modules["piper"] = piper_stub
        sys.modules["piper.download"] = piper_download_stub


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
        mp.setattr("main.AudioManager", MagicMock(name="AudioManager"))
        mp.setattr("main.load_dotenv", lambda: None)

        # Re-leer las clases mockeadas ya configuradas como ``return_value``
        from main import AzureTTSClient, GeminiTTSClient, OpenCodeClient, GeminiSTTClient
        from main import WhisperSTTClient, PiperTTSClient, AudioManager, VoiceAssistant

        AudioManager.return_value = MagicMock(name="AudioManagerInstance")
        GeminiSTTClient.return_value = MagicMock(name="GeminiSTTClientInstance")
        WhisperSTTClient.return_value = MagicMock(name="WhisperSTTClientInstance")
        OpenCodeClient.return_value = MagicMock(name="OpenCodeClientInstance")
        GeminiTTSClient.return_value = MagicMock(name="GeminiTTSClientInstance")
        PiperTTSClient.return_value = MagicMock(name="PiperTTSClientInstance")
        AzureTTSClient.return_value = MagicMock(name="AzureTTSClientInstance")

        assistant = VoiceAssistant()
        # Sobrescribir settings con los sintéticos (no tocar el JSON real)
        assistant._settings = local_settings

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
        patched_assistant._piper_tts.synthesize.return_value = b"\x00" * 48000

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
        patched_assistant._piper_tts.synthesize.return_value = b"\x00" * 48000

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
        patched_assistant._piper_tts.synthesize.assert_not_called()
        assert patched_assistant._state == patched_assistant.STATE_IDLE


@pytest.mark.unit
class TestTTSFailoverChain:
    """Suite: cadena de fallback de TTS (Piper → Gemini → Azure streaming)."""

    def test_tts_piper_primary_no_cloud_called(self, patched_assistant):
        """Piper OK → Gemini y Azure NUNCA se invocan. Se reproduce con ``play_audio``."""
        # Arrange
        _wire_successful_agent(patched_assistant)
        pcm = b"\x00" * 48000
        patched_assistant._piper_tts.synthesize.return_value = pcm

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._piper_tts.synthesize.assert_called_once()
        # Texto limpio (sin prefijo [STYLE:]) va a Piper
        piper_args = patched_assistant._piper_tts.synthesize.call_args
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
        patched_assistant._piper_tts.synthesize.side_effect = RuntimeError("Piper falló")
        pcm = b"\x00" * 48000
        patched_assistant._gemini_tts.synthesize.return_value = pcm
        patched_assistant._gemini_tts.is_available.return_value = True

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._piper_tts.synthesize.assert_called_once()
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
        patched_assistant._piper_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        # Azure streaming: retorna iterator de chunks PCM
        fake_pcm_chunks = [b"\x00\x01" * 100, b"\x00\x01" * 100]
        patched_assistant._azure_tts.synthesize_stream.return_value = iter(fake_pcm_chunks)

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._piper_tts.synthesize.assert_called_once()
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
        patched_assistant._piper_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        patched_assistant._azure_tts.synthesize_stream.side_effect = RuntimeError(
            "Azure 5xx"
        )

        # Act
        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._piper_tts.synthesize.assert_called_once()
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
        Azure y queda en estado IDLE.
        """
        # Arrange
        _wire_successful_agent(patched_assistant)
        patched_assistant._piper_tts.synthesize.side_effect = RuntimeError("Piper falló")
        patched_assistant._gemini_tts.is_available.return_value = True
        patched_assistant._gemini_tts.synthesize.side_effect = RuntimeError("Gemini TTS falló")
        # Forzar Azure None (escenario "no configurado")
        patched_assistant._azure_tts = None

        # Act
        with caplog.at_level(logging.ERROR, logger="main"):
            patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert
        patched_assistant._piper_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_called_once()
        # Sin playback por ninguna vía
        patched_assistant._audio.play_audio.assert_not_called()
        patched_assistant._audio.play_audio_stream.assert_not_called()
        # Log de "Azure no configurado"
        assert any(
            "azure" in r.getMessage().lower() and "no configurado" in r.getMessage().lower()
            for r in caplog.records
        ), f"Log de Azure no configurado no encontrado. Logs: {[r.getMessage() for r in caplog.records]}"
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
        patched_assistant._piper_tts.synthesize.side_effect = RuntimeError("Piper falló")
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
        patched_assistant._piper_tts.synthesize.return_value = b"\x00" * 48000

        # Act
        patched_assistant.run_pipeline("/tmp/fake.wav")

        # Assert — los 3 clientes cloud NO se invocan
        patched_assistant._whisper_stt.transcribe.assert_called_once()
        patched_assistant._stt.transcribe.assert_not_called()  # Gemini STT no se llama
        patched_assistant._opencode.send_command.assert_called_once_with("abrí chrome")
        patched_assistant._piper_tts.synthesize.assert_called_once()
        patched_assistant._gemini_tts.synthesize.assert_not_called()  # Gemini TTS no se llama
        patched_assistant._azure_tts.synthesize_stream.assert_not_called()  # Azure no se llama

        # El playback se hace por play_audio (no streaming) con bytes PCM
        patched_assistant._audio.play_audio.assert_called_once()
        patched_assistant._audio.play_audio_stream.assert_not_called()

        # Estado final: IDLE (rama exitosa del finally)
        assert patched_assistant._state == patched_assistant.STATE_IDLE

        # El overlay se ocultó (cleanup del finally)
        patched_assistant._overlay.hide.assert_called_once()