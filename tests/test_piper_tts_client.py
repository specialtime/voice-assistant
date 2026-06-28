"""Tests unitarios para PiperTTSClient.

Mockea piper.PiperVoice y piper.download.ensure_voice_exists.
Sin red, sin disco, sin modelo real.
"""

import io
import os
import wave
from unittest.mock import MagicMock, patch

import pytest

from handlers.piper_tts_client import PiperTTSClient


@pytest.fixture
def piper_settings() -> dict:
    """Settings sintéticos con sección local.piper para tests."""
    return {
        "local": {
            "piper": {
                "voice_model": "es_AR-daniela-high",
                "voices_dir": "models/piper-voices",
                "download_url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0",
                "length_scale": 1.0,
            }
        }
    }


def _fake_synthesize(wav_file, text_list, length_scale=1.0):
    """Helper: escribe un WAV válido mínimo en el buffer.

    Cabecera de 44 bytes + 200 bytes PCM (100 samples s16le).
    """
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    wav_file.writeframes(b"\x00\x01" * 100)  # 200 bytes PCM


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_synthesize_success(piper_settings):
    """synthesize() retorna PCM crudo (sin cabecera WAV)."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        result = client.synthesize("Hola mundo")

        # Debe ser PCM crudo: 200 bytes (100 samples * 2 bytes)
        assert len(result) == 200
        # NO debe empezar con "RIFF" (cabecera WAV)
        assert not result.startswith(b"RIFF")


@pytest.mark.unit
def test_synthesize_lazy_load(piper_settings):
    """La voz NO se carga en __init__, sí en la 1ra llamada a synthesize()."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        # __init__ no debe llamar a load
        client = PiperTTSClient(piper_settings)
        mock_voice_cls.load.assert_not_called()

        # 1ra synthesize() sí debe llamar a load
        client.synthesize("Hola")
        mock_voice_cls.load.assert_called_once()


@pytest.mark.unit
def test_synthesize_downloads_voice_if_missing(piper_settings):
    """Si el ONNX no existe, ensure_voice_exists es llamado."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=False), \
         patch("handlers.piper_tts_client.ensure_voice_exists") as mock_download:
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        client.synthesize("Hola")

        mock_download.assert_called_once()
        # Verificar que se pasaron los argumentos correctos
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["lang"] == "es"
        assert call_kwargs["lang_locale"] == "es_AR"
        assert call_kwargs["speaker"] == "daniela"
        assert call_kwargs["quality"] == "high"


@pytest.mark.unit
def test_synthesize_no_download_if_exists(piper_settings):
    """Si el ONNX ya existe, ensure_voice_exists NO es llamado."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True), \
         patch("handlers.piper_tts_client.ensure_voice_exists") as mock_download:
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        client.synthesize("Hola")

        mock_download.assert_not_called()


@pytest.mark.unit
def test_synthesize_failure(piper_settings):
    """Si PiperVoice.synthesize lanza excepción → RuntimeError."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = RuntimeError("modelo roto")
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        with pytest.raises(RuntimeError, match="Piper TTS falló"):
            client.synthesize("Hola")


@pytest.mark.unit
def test_synthesize_stream_chunks(piper_settings):
    """synthesize_stream() yields chunks de hasta 4096 bytes PCM."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        chunks = list(client.synthesize_stream("Hola"))

        # Todos los chunks deben ser <= 4096 bytes
        for chunk in chunks:
            assert len(chunk) <= 4096

        # La suma de todos los chunks debe ser 200 bytes (el PCM total)
        total = sum(len(c) for c in chunks)
        assert total == 200


@pytest.mark.unit
def test_style_hint_ignored(piper_settings):
    """synthesize(text, 'cheerful') produce el mismo resultado que synthesize(text, '')."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        result_no_style = client.synthesize("Hola", "")
        result_with_style = client.synthesize("Hola", "cheerful")

        assert result_no_style == result_with_style


@pytest.mark.unit
def test_returns_pcm_not_wav(piper_settings):
    """El resultado NO empieza con 'RIFF' (cabecera WAV), es PCM crudo."""
    with patch("handlers.piper_tts_client.piper.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.os.path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        result = client.synthesize("Hola mundo")

        assert not result.startswith(b"RIFF")
        # Debe ser exactamente 200 bytes (100 samples * 2 bytes s16le)
        assert len(result) == 200
