"""Tests E2E (end-to-end) para Jarvis.

REQUIEREN:
- `.env` con `GEMINI_API_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`,
  `OPENCODE_SERVER_PASSWORD` (y opcionalmente `OPENCODE_BASE_URL`).
- Servidor `opencode serve --port 4096` levantado (para los tests que lo usen).

Los tests hacen `pytest.skip()` automático si las condiciones no se cumplen,
así que son seguros de correr en cualquier entorno (CI incluido).

Marcados con `@pytest.mark.e2e` — ejecutar con:
    pytest tests/ -m e2e -v
"""

import json
import os
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Asegurar que la raíz del proyecto está en sys.path para importar handlers/main
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# Helpers de detección de entorno
# ──────────────────────────────────────────────────────────────────
def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _has_azure_key() -> bool:
    return bool(os.getenv("AZURE_SPEECH_KEY")) and bool(
        os.getenv("AZURE_SPEECH_REGION")
    )


def _opencode_server_alive() -> bool:
    """Verifica si el servidor opencode responde en /global/health."""
    password = os.getenv("OPENCODE_SERVER_PASSWORD")
    base_url = os.getenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    if not password:
        return False
    try:
        import httpx

        response = httpx.get(
            f"{base_url}/global/health",
            auth=httpx.BasicAuth("opencode", password),
            timeout=2.0,
        )
        return response.status_code == 200
    except Exception:
        return False


def _load_settings() -> dict:
    """Carga config/settings.json desde la raíz del proyecto."""
    settings_path = _PROJECT_ROOT / "config" / "settings.json"
    with open(settings_path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────
# Fixtures de skip condicional
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def skip_if_no_gemini():
    if not _has_gemini_key():
        pytest.skip("GEMINI_API_KEY no configurada en .env — saltando test E2E")


@pytest.fixture
def skip_if_no_azure():
    if not _has_azure_key():
        pytest.skip(
            "AZURE_SPEECH_KEY/REGION no configuradas en .env — saltando test E2E"
        )


@pytest.fixture
def skip_if_no_opencode():
    if not _opencode_server_alive():
        pytest.skip(
            "OPENCODE_SERVER_PASSWORD no configurada o servidor opencode no responde "
            "en http://127.0.0.1:4096 — saltando test E2E"
        )


@pytest.fixture
def real_wav(tmp_path: Path) -> str:
    """Genera un WAV real de 1 segundo de silencio en tmp_path.

    24 kHz, 1 canal, 16-bit — formato esperado por GeminiSTTClient.
    """
    wav_path = tmp_path / "e2e_silence.wav"
    sample_rate = 24000
    n_samples = sample_rate  # 1 segundo

    audio_data = np.zeros(n_samples, dtype=np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return str(wav_path)


# ──────────────────────────────────────────────────────────────────
# Tests E2E
# ──────────────────────────────────────────────────────────────────
@pytest.mark.e2e
class TestE2EScenarios:
    """Suite E2E con API keys reales y servidor opencode.

    NOTA: estos tests consumen cuota de las APIs gratuitas. Ejecutar
    manualmente solo cuando se quiera validar el flujo completo contra
    los servicios reales.
    """

    def test_stt_real_transcription(self, skip_if_no_gemini, real_wav):
        """Envía WAV real a Gemini STT, valida que retorna string (puede ser vacío)."""
        from dotenv import load_dotenv
        from handlers.gemini_stt_client import GeminiSTTClient

        load_dotenv()
        settings = _load_settings()

        client = GeminiSTTClient(settings, os.environ["GEMINI_API_KEY"])
        result = client.transcribe(real_wav)

        # El audio es silencio, así que el texto puede ser cadena vacía o
        # caracteres de "no speech". Lo importante: la API respondió sin
        # excepción y el tipo es str.
        assert isinstance(result, str)
        # No debe ser None
        assert result is not None

    def test_tts_real_synthesis(self, skip_if_no_gemini):
        """Envía texto a Gemini TTS, valida que retorna bytes de audio no vacíos."""
        from dotenv import load_dotenv
        from handlers.gemini_tts_client import GeminiTTSClient

        load_dotenv()
        settings = _load_settings()

        client = GeminiTTSClient(settings, os.environ["GEMINI_API_KEY"])
        result = client.synthesize("Hola, esto es una prueba de Jarvis", "cheerful")

        assert isinstance(result, bytes)
        assert len(result) > 0, "Gemini TTS retornó 0 bytes"
        # 1 segundo de audio PCM 24kHz mono int16 = 48000 bytes.
        # Permitimos tolerancia amplia.
        assert len(result) > 1000

    def test_azure_real_synthesis(self, skip_if_no_azure):
        """Envía texto a Azure TTS, valida que retorna bytes de audio no vacíos."""
        from dotenv import load_dotenv
        from handlers.azure_tts_client import AzureTTSClient

        load_dotenv()
        settings = _load_settings()

        client = AzureTTSClient(
            settings,
            os.environ["AZURE_SPEECH_KEY"],
            os.environ["AZURE_SPEECH_REGION"],
        )
        result = client.synthesize("Hola, esto es una prueba de Jarvis.")

        assert isinstance(result, bytes)
        assert len(result) > 0, "Azure TTS retornó 0 bytes"

    def test_opencode_real_session(self, skip_if_no_opencode):
        """Crea sesión real en opencode, valida que retorna id y se cachea."""
        from dotenv import load_dotenv
        from handlers.opencode_client import OpenCodeClient

        load_dotenv()
        settings = _load_settings()

        client = OpenCodeClient(
            settings,
            os.environ["OPENCODE_SERVER_PASSWORD"],
            os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
        )

        # Primera llamada crea sesión
        sid1 = client.ensure_session()
        assert isinstance(sid1, str)
        assert len(sid1) > 0

        # Segunda llamada usa cache (mismo id, sin nuevo POST)
        sid2 = client.ensure_session()
        assert sid1 == sid2

    def test_full_pipeline_mock_tts(
        self, skip_if_no_gemini, skip_if_no_opencode, real_wav
    ):
        """STT real + OpenCode real + TTS mockeado.

        No reproducimos audio (mock del TTS) para que el test sea seguro
        de correr en CI sin hardware de audio.
        """
        from dotenv import load_dotenv
        import base64

        from handlers.gemini_stt_client import GeminiSTTClient
        from handlers.opencode_client import OpenCodeClient
        from handlers.gemini_tts_client import GeminiTTSClient
        from handlers.response_parser import parse_response

        load_dotenv()
        settings = _load_settings()

        stt = GeminiSTTClient(settings, os.environ["GEMINI_API_KEY"])
        oc = OpenCodeClient(
            settings,
            os.environ["OPENCODE_SERVER_PASSWORD"],
            os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
        )

        # 1) STT real
        text = stt.transcribe(real_wav)
        assert text is not None
        # Si la transcripción de silencio viene vacía, usar un fallback para
        # que opencode tenga algo con qué responder.
        text_to_send = text if text else "hola"

        # 2) OpenCode real
        response = oc.send_command(text_to_send)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # 3) Parse
        style_hint, clean_text = parse_response(response)
        # clean_text puede no tener prefijo [STYLE:], eso es válido
        assert isinstance(clean_text, str)

        # 4) TTS mockeado (no reproducimos audio en CI)
        with patch("handlers.gemini_tts_client.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            fake_pcm = b"FAKE_AUDIO_BYTES_FOR_E2E_TEST"
            audio_b64 = base64.b64encode(fake_pcm).decode("utf-8")
            mock_client.post.return_value = MagicMock(
                status_code=200,
                text="",
                content=b"",
                raise_for_status=MagicMock(),
                json=lambda: {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"inlineData": {"data": audio_b64}}
                                ]
                            }
                        }
                    ]
                },
            )

            tts = GeminiTTSClient(settings, os.environ["GEMINI_API_KEY"])
            pcm = tts.synthesize(clean_text, style_hint)

        assert pcm == fake_pcm
