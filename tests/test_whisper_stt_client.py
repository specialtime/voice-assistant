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
    """Extiende mock_settings con la sección local.whisper completa.

    Incluye los 5 parámetros de precisión agregados en feature/stt-accuracy-improvements:
    vad_filter, vad_min_silence_duration_ms, initial_prompt, hotwords,
    condition_on_previous_text.
    """
    settings = dict(mock_settings)
    settings["local"] = {
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8_float16",
            "language": "es",
            "beam_size": 5,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 500,
            "initial_prompt": "Comandos de voz en español rioplatense.",
            "hotwords": "Chrome VSCode opencode",
            "condition_on_previous_text": False,
        }
    }
    return settings


@pytest.fixture
def whisper_mock_settings_vad_disabled(mock_settings: dict) -> dict:
    """Variante del mock_settings con vad_filter=False (deshabilitado).

    Usada por tests que verifican el branch de VAD deshabilitado.
    """
    settings = dict(mock_settings)
    settings["local"] = {
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8_float16",
            "language": "es",
            "beam_size": 5,
            "vad_filter": False,
            "vad_min_silence_duration_ms": 500,
            "initial_prompt": "Comandos de voz en español rioplatense.",
            "hotwords": "Chrome VSCode opencode",
            "condition_on_previous_text": False,
        }
    }
    return settings


@pytest.fixture
def whisper_mock_settings_legacy(mock_settings: dict) -> dict:
    """Variante con solo los campos originales (settings.json previo a accuracy-improvements).

    Sin vad_filter, vad_min_silence_duration_ms, initial_prompt, hotwords ni
    condition_on_previous_text — sirve para verificar que el handler aplica
    defaults sensatos cuando faltan estos campos.
    """
    settings = dict(mock_settings)
    settings["local"] = {
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8_float16",
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
        """Mock WhisperModel.transcribe retorna segments con texto → retorna texto limpio (stripped).

        Verifica además que transcribe() recibe los 5 kwargs nuevos de precisión:
        vad_filter, vad_parameters, initial_prompt, hotwords, condition_on_previous_text.
        """
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("  hola mundo  "), _make_segment(" desde whisper")],
            MagicMock(),  # _info
        )

        client = WhisperSTTClient(whisper_mock_settings)
        result = client.transcribe(tmp_wav)

        assert result == "hola mundo desde whisper"
        mock_model.transcribe.assert_called_once()
        # Verificar explícitamente los kwargs nuevos de precisión
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == dict(min_silence_duration_ms=500)
        assert kwargs["initial_prompt"] == "Comandos de voz en español rioplatense."
        assert kwargs["hotwords"] == "Chrome VSCode opencode"
        assert kwargs["condition_on_previous_text"] is False

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
        """Verifica todos los kwargs que recibe transcribe() Y los kwargs del constructor WhisperModel.

        - Constructor WhisperModel recibe (model, device, compute_type) del config.
        - model.transcribe() recibe los 7 kwargs: language, beam_size, vad_filter,
          vad_parameters, initial_prompt, hotwords, condition_on_previous_text.
        """
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        # Constructor recibe compute_type del config (no a transcribe())
        mock_model_cls.assert_called_once_with(
            "small", device="cuda", compute_type="int8_float16"
        )
        # transcribe() recibe language + los 5 kwargs nuevos de precisión
        mock_model.transcribe.assert_called_once_with(
            tmp_wav,
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt="Comandos de voz en español rioplatense.",
            hotwords="Chrome VSCode opencode",
            condition_on_previous_text=False,
        )

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_vad_filter_passed(
        self, mock_model_cls, whisper_mock_settings, tmp_wav
    ):
        """vad_filter=True → transcribe() recibe vad_filter=True y vad_parameters con min_silence_duration_ms."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == dict(min_silence_duration_ms=500)
        assert "min_silence_duration_ms" in kwargs["vad_parameters"]
        assert kwargs["vad_parameters"]["min_silence_duration_ms"] == 500

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_vad_filter_disabled(
        self, mock_model_cls, whisper_mock_settings_vad_disabled, tmp_wav
    ):
        """vad_filter=False → transcribe() recibe vad_filter=False y vad_parameters=None."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings_vad_disabled)
        client.transcribe(tmp_wav)

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["vad_filter"] is False
        assert kwargs["vad_parameters"] is None

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_initial_prompt_passed(
        self, mock_model_cls, whisper_mock_settings, tmp_wav
    ):
        """initial_prompt del config llega tal cual a model.transcribe()."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["initial_prompt"] == "Comandos de voz en español rioplatense."

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_hotwords_passed(
        self, mock_model_cls, whisper_mock_settings, tmp_wav
    ):
        """hotwords del config llega tal cual a model.transcribe()."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["hotwords"] == "Chrome VSCode opencode"

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_condition_on_previous_text_false(
        self, mock_model_cls, whisper_mock_settings, tmp_wav
    ):
        """condition_on_previous_text=False del config llega a model.transcribe()."""
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("test")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings)
        client.transcribe(tmp_wav)

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["condition_on_previous_text"] is False

    @patch("handlers.whisper_stt_client.WhisperModel")
    def test_transcribe_defaults_when_config_missing(
        self, mock_model_cls, whisper_mock_settings_legacy, tmp_wav
    ):
        """Settings viejo (sin los 5 campos nuevos) → defaults sensatos sin crashear.

        Defaults aplicados cuando los campos faltan:
        - vad_filter=False
        - vad_parameters=None
        - initial_prompt=None
        - hotwords=None
        - condition_on_previous_text=True
        """
        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = (
            [_make_segment("hola")],
            MagicMock(),
        )

        client = WhisperSTTClient(whisper_mock_settings_legacy)
        result = client.transcribe(tmp_wav)

        # No debe crashear y retorna texto limpio
        assert result == "hola"
        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["vad_filter"] is False
        assert kwargs["vad_parameters"] is None
        assert kwargs["initial_prompt"] is None
        assert kwargs["hotwords"] is None
        assert kwargs["condition_on_previous_text"] is True
        # Los kwargs originales del config también deben llegar
        assert kwargs["language"] == "es"
        assert kwargs["beam_size"] == 5

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