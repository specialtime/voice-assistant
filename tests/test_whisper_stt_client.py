"""Tests unitarios para handlers/whisper_stt_client.py.

Mockea `faster_whisper.WhisperModel` con `unittest.mock.patch` — sin GPU, sin red.
Cubre el contrato de `WhisperSTTClient` definido en specs/feature_local_stt_tts.md.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from handlers.whisper_stt_client import WhisperSTTClient


@pytest.fixture
def whisper_mock_settings(mock_settings: dict) -> dict:
    """Extiende mock_settings con la sección local.whisper."""
    settings = dict(mock_settings)
    settings["local"] = {
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8",
            "language": "es",
            "beam_size": 5,
        }
    }
    return settings


def _make_segment(text: str) -> MagicMock:
    """Crea un mock de segment de faster-whisper con atributo .text."""
    seg = MagicMock()
    seg.text = text
    return seg


@pytest.mark.unit
class TestWhisperSTTClient:
    """Suite de tests para WhisperSTTClient con faster_whisper mockeado."""

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_success(self, mock_model_cls, whisper_mock_settings, tmp_wav):
        """Mock WhisperModel.transcribe retorna segments con texto → retorna texto limpio (stripped)."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("  hola mundo  "), _make_segment(" desde whisper")],
            MagicMock(),  # _info
        )

        client = WhisperSTTClient(whisper_mock_settings)
        result = client.transcribe(tmp_wav)

        assert result == "hola mundo desde whisper"
        mock_model.transcribe.assert_called_once()

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_lazy_load(self, mock_model_cls, whisper_mock_settings, tmp_wav):
        """El modelo NO se carga en __init__, sí en la 1ra llamada a transcribe()."""
        mock_model_cls.return_value.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        # __init__ no debe instanciar WhisperModel
        assert mock_model_cls.call_count == 0
        client = WhisperSTTClient(whisper_mock_settings)
        assert mock_model_cls.call_count == 0  # sigue sin cargarse tras __init__

        # 1ra transcribe() → carga el modelo
        client.transcribe(tmp_wav)
        assert mock_model_cls.call_count == 1

        # 2da transcribe() → no recarga
        client.transcribe(tmp_wav)
        assert mock_model_cls.call_count == 1

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_file_not_found(self, mock_model_cls, whisper_mock_settings):
        """wav_path inexistente → FileNotFoundError. NO debe cargar el modelo."""
        client = WhisperSTTClient(whisper_mock_settings)

        with pytest.raises(FileNotFoundError, match="Audio no encontrado"):
            client.transcribe("C:\\nonexistent\\file.wav")

        # El modelo NO debe haberse cargado
        mock_model_cls.assert_not_called()

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_model_failure(self, mock_model_cls, whisper_mock_settings, tmp_wav):
        """Mock model.transcribe lanza excepción → RuntimeError("Whisper STT falló")."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        client = WhisperSTTClient(whisper_mock_settings)

        with pytest.raises(RuntimeError, match="Whisper STT falló"):
            client.transcribe(tmp_wav)

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_uses_config_language(self, mock_model_cls, whisper_mock_settings, tmp_wav):
        """Verificar que model.transcribe() recibe language='es' y beam_size=5 del config."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        mock_model.transcribe.assert_called_once_with(
            tmp_wav,
            language="es",
            beam_size=5,
        )

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_no_secrets_logged(self, mock_model_cls, whisper_mock_settings, caplog):
        """Paths absolutos del usuario no se loguean en DEBUG."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        secret_path = "C:\\Users\\SECRET_USER_DO_NOT_LEAK\\test.wav"

        # Mock os.path.exists para que pase la validación de archivo
        with patch.object(os.path, "exists", return_value=True):
            client = WhisperSTTClient(whisper_mock_settings)

            with caplog.at_level(logging.DEBUG, logger="handlers.whisper_stt_client"):
                client.transcribe(secret_path)

        all_logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "SECRET_USER_DO_NOT_LEAK" not in all_logs, (
            f"Path de usuario filtrado en logs: {[r.getMessage() for r in caplog.records]}"
        )
