"""Tests unitarios para handlers/gemini_stt_client.py.

Mockea `httpx.Client` con `unittest.mock.patch` — sin red.
Cubre el contrato de `GeminiSTTClient` definido en IMPLEMENTATION.md §4.4,
incluyendo failover 429 y validación de tamaño.
"""

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from handlers.gemini_stt_client import GeminiSTTClient


def _ok_response(json_data: dict) -> MagicMock:
    """Crea un mock de httpx.Response con status 200 y JSON arbitrario."""
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.raise_for_status = MagicMock()  # no-op
    response.json.return_value = json_data
    return response


def _err_response(status_code: int, text: str = "error") -> MagicMock:
    """Crea un mock de httpx.Response que lanza HTTPStatusError en raise_for_status."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} error",
        request=MagicMock(),
        response=response,
    )
    return response


def _transcription_payload(text: str) -> dict:
    """Payload JSON esperado de la API Gemini STT."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                }
            }
        ]
    }


@pytest.mark.unit
class TestGeminiSTTClient:
    """Suite de tests para GeminiSTTClient con httpx mockeado."""

    @patch("handlers.gemini_stt_client.httpx.Client")
    def test_transcribe_success(self, mock_client_cls, mock_settings, tmp_wav):
        """Respuesta 200 del modelo primario → retorna texto limpio (stripped)."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _ok_response(
            _transcription_payload("  abrí chrome  ")
        )

        client = GeminiSTTClient(mock_settings, "fake_key")
        result = client.transcribe(tmp_wav)

        assert result == "abrí chrome"
        # Una sola llamada: el primario funcionó
        assert mock_client.post.call_count == 1

    @patch("handlers.gemini_stt_client.httpx.Client")
    def test_transcribe_failover_429(
        self, mock_client_cls, mock_settings, tmp_wav, caplog
    ):
        """Primario 429 → fallback 200 → usa fallback, log warning de failover."""
        mock_client = mock_client_cls.return_value

        # 1ª llamada: 429 (primario falla)
        # 2ª llamada: 200 (fallback OK)
        mock_client.post.side_effect = [
            _err_response(429, "Rate limit exceeded"),
            _ok_response(_transcription_payload("hola")),
        ]

        client = GeminiSTTClient(mock_settings, "fake_key")

        with caplog.at_level(logging.WARNING):
            result = client.transcribe(tmp_wav)

        assert result == "hola"
        assert mock_client.post.call_count == 2
        # Verificar que se logueó un warning relacionado con el failover
        assert any(
            "Failover" in record.getMessage() or "fallback" in record.getMessage().lower()
            for record in caplog.records
        )

    @patch("handlers.gemini_stt_client.httpx.Client")
    def test_transcribe_both_fail(self, mock_client_cls, mock_settings, tmp_wav):
        """Primario y fallback fallan → RuntimeError con mensaje claro."""
        mock_client = mock_client_cls.return_value
        # Ambas llamadas devuelven 429
        mock_client.post.return_value = _err_response(429)

        client = GeminiSTTClient(mock_settings, "fake_key")

        with pytest.raises(RuntimeError, match="STT falló"):
            client.transcribe(tmp_wav)

        assert mock_client.post.call_count == 2

    def test_transcribe_oversized_audio(self, mock_settings, tmp_path):
        """WAV > 20MB → ValueError ANTES de tocar la red."""
        # Crear archivo fake > 20MB (no necesita ser WAV válido para el check
        # porque la validación de tamaño ocurre antes de decodificar/llamar API)
        big_wav = tmp_path / "huge.wav"
        big_wav.write_bytes(b"\x00" * (21 * 1024 * 1024))  # 21 MB

        # Patch httpx.Client para que no se haga ninguna request real
        with patch("handlers.gemini_stt_client.httpx.Client") as mock_client_cls:
            client = GeminiSTTClient(mock_settings, "fake_key")

            with pytest.raises(ValueError, match=r"demasiado grande|20"):
                client.transcribe(str(big_wav))

            # CRÍTICO: no debe haberse hecho ninguna request
            mock_client_cls.return_value.post.assert_not_called()

    @patch("handlers.gemini_stt_client.httpx.Client")
    def test_no_api_key_logged(
        self, mock_client_cls, mock_settings, tmp_wav, caplog
    ):
        """La API key NO debe aparecer en ningún log record del cliente STT."""
        secret_key = "SECRET_GEMINI_KEY_DO_NOT_LEAK_777777"

        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _ok_response(
            _transcription_payload("test")
        )

        with caplog.at_level(logging.DEBUG, logger="handlers.gemini_stt_client"):
            client = GeminiSTTClient(mock_settings, secret_key)
            # Forzar varias operaciones que loguean (init + transcribe OK)
            client.transcribe(tmp_wav)

        # La secret NO debe aparecer en ningún mensaje de log
        all_logs = "\n".join(record.getMessage() for record in caplog.records)
        assert secret_key not in all_logs, (
            f"API key filtrada en logs: {[r.getMessage() for r in caplog.records]}"
        )
