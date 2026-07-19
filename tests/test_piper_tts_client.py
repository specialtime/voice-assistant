"""Tests unitarios para PiperTTSClient.

Mockea PiperVoice, download_voice y Path.exists.
Sin red, sin disco, sin modelo real.
"""

import io
import logging
import os
import wave
from pathlib import Path
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


def _fake_synthesize_wav(text, wav_file, syn_config=None):
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
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
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
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
        mock_voice_cls.load.return_value = mock_voice

        # __init__ no debe llamar a load
        client = PiperTTSClient(piper_settings)
        mock_voice_cls.load.assert_not_called()

        # 1ra synthesize() sí debe llamar a load
        client.synthesize("Hola")
        mock_voice_cls.load.assert_called_once()


@pytest.mark.unit
def test_synthesize_downloads_voice_if_missing(piper_settings):
    """Si el ONNX no existe, download_voice es llamado."""
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=False), \
         patch("handlers.piper_tts_client.Path.mkdir") as mock_mkdir, \
         patch("handlers.piper_tts_client.download_voice") as mock_download:
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        client.synthesize("Hola")

        mock_download.assert_called_once()
        # download_voice(voice_model, voices_dir) con Path
        call_args = mock_download.call_args[0]
        assert call_args[0] == "es_AR-daniela-high"
        assert call_args[1] == Path("models/piper-voices")


@pytest.mark.unit
def test_synthesize_no_download_if_exists(piper_settings):
    """Si el ONNX ya existe, download_voice NO es llamado."""
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True), \
         patch("handlers.piper_tts_client.download_voice") as mock_download:
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        client.synthesize("Hola")

        mock_download.assert_not_called()


@pytest.mark.unit
def test_synthesize_failure(piper_settings):
    """Si PiperVoice.synthesize_wav lanza excepción → RuntimeError."""
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = RuntimeError("modelo roto")
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        with pytest.raises(RuntimeError, match="Piper TTS falló"):
            client.synthesize("Hola")


@pytest.mark.unit
def test_synthesize_stream_chunks(piper_settings):
    """synthesize_stream() yields chunks de hasta 4096 bytes PCM."""
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
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
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        result_no_style = client.synthesize("Hola", "")
        result_with_style = client.synthesize("Hola", "cheerful")

        assert result_no_style == result_with_style


@pytest.mark.unit
def test_returns_pcm_not_wav(piper_settings):
    """El resultado NO empieza con 'RIFF' (cabecera WAV), es PCM crudo."""
    with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
         patch("handlers.piper_tts_client.Path.exists", return_value=True):
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
        mock_voice_cls.load.return_value = mock_voice

        client = PiperTTSClient(piper_settings)
        result = client.synthesize("Hola mundo")

        assert not result.startswith(b"RIFF")
        # Debe ser exactamente 200 bytes (100 samples * 2 bytes s16le)
        assert len(result) == 200


@pytest.mark.unit
@patch("handlers.piper_tts_client.PiperVoice")
@patch("handlers.piper_tts_client.download_voice")
@patch("handlers.piper_tts_client.Path.mkdir")
def test_no_secrets_logged(mock_mkdir, mock_download, mock_voice_cls, piper_settings, caplog):
    """Paths absolutos del usuario NO deben aparecer en logs de PiperTTSClient."""
    # Configurar voices_dir con un path "sensible"
    piper_settings["local"]["piper"]["voices_dir"] = "C:\\Users\\SECRET_USER_DO_NOT_LEAK_999\\models"

    mock_voice_cls.load.return_value = MagicMock()
    mock_voice_inst = MagicMock()
    mock_voice_cls.load.return_value = mock_voice_inst

    def fake_synthesize_wav(text, wav_file, syn_config=None):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x01" * 100)

    mock_voice_inst.synthesize_wav.side_effect = fake_synthesize_wav

    with caplog.at_level(logging.DEBUG, logger="handlers.piper_tts_client"):
        client = PiperTTSClient(piper_settings)
        client.synthesize("test")

    all_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "SECRET_USER_DO_NOT_LEAK_999" not in all_logs, (
        f"Path absoluto filtrado en logs: {[r.getMessage() for r in caplog.records]}"
    )


# ──────────────────────────────────────────────────────────────────
# Tests de synthesize_sentence_stream (Micro-Spec Streaming TTS — T8.bis)
# ──────────────────────────────────────────────────────────────────
#
# Cubre el contrato de ``PiperTTSClient.synthesize_sentence_stream()``
# introducido en ``fix/streaming-tts-fallback`` (commit c115c48) para
# permitir que Piper participe del flujo streaming.
# Análogo a la suite de Kokoro (test_kokoro_tts_client.py:744-862).


@pytest.mark.unit
class TestPiperSynthesizeSentenceStream:
    """Suite: synthesize_sentence_stream() — yield de PCM por oración."""

    def test_synthesize_sentence_stream_yields_pcm(self, piper_settings):
        """Iterator de 3 oraciones → 3 yields de PCM (uno por oración).

        Cada yield contiene el PCM completo de la oración. El consumidor
        (``play_audio_stream``) los reproduce en tiempo real.
        """
        with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
             patch("handlers.piper_tts_client.Path.exists", return_value=True):
            mock_voice = MagicMock()
            mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
            mock_voice_cls.load.return_value = mock_voice

            client = PiperTTSClient(piper_settings)

            def sentence_iter():
                yield "Primera oración."
                yield "Segunda oración."
                yield "Tercera oración."

            chunks = list(client.synthesize_sentence_stream(sentence_iter()))

            # 3 oraciones → 3 yields de PCM
            assert len(chunks) == 3
            for chunk in chunks:
                # _fake_synthesize_wav escribe 200 bytes PCM (100 samples s16le)
                assert isinstance(chunk, bytes)
                assert len(chunk) == 200
                # No debe empezar con "RIFF" (cabecera WAV)
                assert not chunk.startswith(b"RIFF")

            # synthesize_wav fue llamada 3 veces (1 por oración)
            assert mock_voice.synthesize_wav.call_count == 3
            # Los textos recibidos son las oraciones
            received = [c.args[0] for c in mock_voice.synthesize_wav.call_args_list]
            assert received == [
                "Primera oración.",
                "Segunda oración.",
                "Tercera oración.",
            ]

    def test_synthesize_sentence_stream_skips_empty(self, piper_settings):
        """Iterator con oraciones VACÍAS o whitespace → se saltean.

        El handler chequea ``sentence.strip()`` antes de invocar
        ``synthesize()``. ``""`` y ``"   "`` NO deben cargar el modelo
        NI invocar ``PiperVoice.synthesize_wav``.
        """
        with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
             patch("handlers.piper_tts_client.Path.exists", return_value=True):
            mock_voice = MagicMock()
            mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
            mock_voice_cls.load.return_value = mock_voice

            client = PiperTTSClient(piper_settings)

            def sentence_iter():
                yield ""        # vacía → skip
                yield "  "      # whitespace → skip
                yield "hola"    # real
                yield ""        # vacía → skip
                yield "mundo"   # real

            chunks = list(client.synthesize_sentence_stream(sentence_iter()))

            # Solo 2 yields (las 2 oraciones reales)
            assert len(chunks) == 2
            assert all(len(c) == 200 for c in chunks)

            # synthesize_wav fue llamada SOLO 2 veces (skip de vacías)
            assert mock_voice.synthesize_wav.call_count == 2
            received = [c.args[0] for c in mock_voice.synthesize_wav.call_args_list]
            assert received == ["hola", "mundo"]

    def test_synthesize_sentence_stream_lazy_load(self, piper_settings):
        """Iterator VACÍO → NO se carga el modelo Piper (lazy-load).

        El modelo solo debe cargarse cuando hay al menos UNA oración real
        para sintetizar. Un iter vacío es el caso del caller que no llegó
        a yield-ear ninguna oración (ej: cancelación durante el streaming).
        NO debe invocar ``PiperVoice.load`` NI ``PiperVoice.synthesize_wav``.
        """
        with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
             patch("handlers.piper_tts_client.Path.exists", return_value=True):
            mock_voice = MagicMock()
            mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
            mock_voice_cls.load.return_value = mock_voice

            client = PiperTTSClient(piper_settings)

            # Iter vacío
            def empty_iter():
                if False:
                    yield ""  # pragma: no cover

            chunks = list(client.synthesize_sentence_stream(empty_iter()))

            # Sin yields
            assert chunks == []
            # PiperVoice.load NO se invocó (modelo no se cargó)
            mock_voice_cls.load.assert_not_called()
            # synthesize_wav NO se invocó
            mock_voice.synthesize_wav.assert_not_called()

    def test_synthesize_sentence_stream_single_sentence(self, piper_settings):
        """Iterator de UNA oración → un yield de PCM.

        Caso degenerado: solo una oración para sintetizar.
        """
        with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
             patch("handlers.piper_tts_client.Path.exists", return_value=True):
            mock_voice = MagicMock()
            mock_voice.synthesize_wav.side_effect = _fake_synthesize_wav
            mock_voice_cls.load.return_value = mock_voice

            client = PiperTTSClient(piper_settings)

            def sentence_iter():
                yield "Hola mundo"

            chunks = list(client.synthesize_sentence_stream(sentence_iter()))

            # Exactamente 1 yield
            assert len(chunks) == 1
            assert len(chunks[0]) == 200
            assert mock_voice.synthesize_wav.call_count == 1

    def test_synthesize_sentence_stream_propagates_exception(self, piper_settings):
        """Si ``synthesize`` lanza excepción → el generator PROPAGA la excepción
        (no la traga). El caller (helper de fallback) depende de esto para
        detectar la falla y caer al siguiente TTS de la cadena.
        """
        with patch("handlers.piper_tts_client.PiperVoice") as mock_voice_cls, \
             patch("handlers.piper_tts_client.Path.exists", return_value=True):
            mock_voice = MagicMock()
            # synthesize_wav lanza RuntimeError → synthesize() la envuelve
            # en RuntimeError("Piper TTS falló: ...") — el generator debe propagarla.
            mock_voice.synthesize_wav.side_effect = RuntimeError("modelo roto")
            mock_voice_cls.load.return_value = mock_voice

            client = PiperTTSClient(piper_settings)

            def sentence_iter():
                yield "primera"

            with pytest.raises(RuntimeError, match="Piper TTS falló"):
                list(client.synthesize_sentence_stream(sentence_iter()))
