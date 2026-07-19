"""Tests unitarios para handlers/gemini_tts_client.py y handlers/azure_tts_client.py.

Mockea `httpx.Client` con `unittest.mock.patch` — sin red.
Cubre los contratos definidos en IMPLEMENTATION.md §4.6 (Gemini TTS) y §4.7 (Azure TTS).
"""

import base64
import logging
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from handlers.gemini_tts_client import GeminiTTSClient
from handlers.azure_tts_client import AzureTTSClient


# ──────────────────────────────────────────────────────────────────
# Helpers compartidos
# ──────────────────────────────────────────────────────────────────
def _json_response(json_data: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.content = b""
    response.raise_for_status = MagicMock()
    response.json.return_value = json_data
    return response


def _content_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.content = content
    response.raise_for_status = MagicMock()
    response.json.return_value = {}
    return response


def _err_response(status_code: int, text: str = "error") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.content = b""
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} error",
        request=MagicMock(),
        response=response,
    )
    return response


def _gemini_tts_payload(pcm_bytes: bytes) -> dict:
    """Payload JSON esperado de Gemini TTS con inlineData base64."""
    audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm",
                                "data": audio_b64,
                            }
                        }
                    ]
                }
            }
        ]
    }


# ──────────────────────────────────────────────────────────────────
# Gemini TTS
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestGeminiTTSClient:
    """Tests para GeminiTTSClient (handler primario de TTS)."""

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_gemini_synthesize_success(self, mock_client_cls, mock_settings):
        """Mock 200 con inlineData base64 → retorna bytes PCM decodificados."""
        mock_client = mock_client_cls.return_value
        fake_pcm = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        mock_client.post.return_value = _json_response(_gemini_tts_payload(fake_pcm))

        client = GeminiTTSClient(mock_settings, "fake_key")
        result = client.synthesize("hola", "cheerful")

        assert result == fake_pcm
        assert isinstance(result, bytes)

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_gemini_synthesize_with_style(self, mock_client_cls, mock_settings):
        """El prompt compuesto es 'Say {style_hint}: {text}'."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _json_response(
            _gemini_tts_payload(b"\x00\x00")
        )

        client = GeminiTTSClient(mock_settings, "fake_key")
        client.synthesize("abrí Chrome", "cheerful")

        # Inspeccionar el body enviado
        body = mock_client.post.call_args.kwargs["json"]
        prompt_text = body["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Say cheerful: abrí Chrome"

        # voiceConfig debe tener la voz configurada
        voice_name = (
            body["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"]
        )
        assert voice_name == mock_settings["gemini"]["tts_voice"]

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_gemini_synthesize_without_style(self, mock_client_cls, mock_settings):
        """Sin style_hint → prompt 'Say: {text}' (con los dos puntos pero sin adjetivo)."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _json_response(
            _gemini_tts_payload(b"\x00")
        )

        client = GeminiTTSClient(mock_settings, "fake_key")
        client.synthesize("hola", "")

        body = mock_client.post.call_args.kwargs["json"]
        prompt_text = body["contents"][0]["parts"][0]["text"]
        # Sin estilo: "Say: hola" (formato §4.6)
        assert prompt_text == "Say: hola"

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_gemini_synthesize_failure(self, mock_client_cls, mock_settings):
        """Mock 500 → RuntimeError('Gemini TTS falló')."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _err_response(500, "Server Error")

        client = GeminiTTSClient(mock_settings, "fake_key")

        with pytest.raises(RuntimeError, match="Gemini TTS falló"):
            client.synthesize("hola", "cheerful")

    # ──────────────────────────────────────────────────────────────────
    # Circuit Breaker — Micro-Spec A
    # ──────────────────────────────────────────────────────────────────
    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_is_available_true_when_circuit_closed(self, mock_client_cls, mock_settings):
        """Estado inicial: circuit cerrado → is_available() retorna True."""
        client = GeminiTTSClient(mock_settings, "fake_key")

        assert client._circuit_open is False
        assert client.is_available() is True

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_circuit_breaker_opens_on_429(self, mock_client_cls, mock_settings):
        """Mock HTTP 429 en synthesize() → _circuit_open=True y _circuit_open_until en el futuro."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _err_response(429, "Rate limit exceeded")

        client = GeminiTTSClient(mock_settings, "fake_key")

        assert client._circuit_open is False
        before_ts = time.time()

        with pytest.raises(RuntimeError, match="Gemini TTS falló"):
            client.synthesize("hola", "cheerful")

        # Tras el 429, el circuito debe estar abierto
        assert client._circuit_open is True
        assert client._circuit_open_until > before_ts
        # Y el cooldown debe ser ~300s en el futuro
        assert client._circuit_open_until >= before_ts + client._circuit_cooldown_seconds - 1

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_circuit_breaker_blocks_second_call(
        self, mock_client_cls, mock_settings
    ):
        """Tras un 429, la 2da llamada a synthesize() lanza RuntimeError SIN hacer HTTP."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _err_response(429, "Rate limit exceeded")

        client = GeminiTTSClient(mock_settings, "fake_key")

        # 1ª llamada: 429 → abre circuito
        with pytest.raises(RuntimeError, match="Gemini TTS falló"):
            client.synthesize("hola", "cheerful")
        assert client._circuit_open is True
        first_call_count = mock_client.post.call_count

        # 2ª llamada: debe lanzar "circuit breaker abierto" sin tocar el HTTP client
        with pytest.raises(RuntimeError, match="Gemini TTS circuit breaker abierto"):
            client.synthesize("hola de nuevo", "cheerful")

        # CRÍTICO: el client.post NO debe haberse llamado de nuevo
        assert mock_client.post.call_count == first_call_count

    @patch("handlers.gemini_tts_client.httpx.Client")
    def test_circuit_breaker_closes_after_cooldown(
        self, mock_client_cls, mock_settings
    ):
        """Mock time.time() para simular 300s pasados → is_available() cierra el circuito."""
        import time as time_module

        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _err_response(429, "Rate limit exceeded")

        client = GeminiTTSClient(mock_settings, "fake_key")

        # 1ª llamada: abre circuito
        with pytest.raises(RuntimeError, match="Gemini TTS falló"):
            client.synthesize("hola", "cheerful")
        assert client._circuit_open is True
        assert client.is_available() is False  # aún en cooldown

        # Simular que pasaron >300s (cooldown completo)
        future_ts = client._circuit_open_until + 1
        with patch.object(time_module, "time", return_value=future_ts):
            assert client.is_available() is True
            # El circuito debe haberse cerrado
            assert client._circuit_open is False


# ──────────────────────────────────────────────────────────────────
# Azure TTS (fallback)
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestAzureTTSClient:
    """Tests para AzureTTSClient (handler fallback de TTS)."""

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_success(self, mock_client_cls, mock_settings):
        """Mock 200 con audio bytes → retorna bytes de audio sin modificar."""
        mock_client = mock_client_cls.return_value
        fake_audio = b"ID3\x03\x00\x00\x00fake_mp3_bytes"
        mock_client.post.return_value = _content_response(fake_audio)

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        result = client.synthesize("hola")

        assert result == fake_audio
        assert isinstance(result, bytes)

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_no_express_as(self, mock_client_cls, mock_settings):
        """El SSML NO debe contener <mstts:express-as> (requisito anti-SSML §4.7)."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _content_response(b"audio")

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        client.synthesize("hola mundo")

        # Inspeccionar el body enviado
        ssml_body = mock_client.post.call_args.kwargs["content"]
        assert isinstance(ssml_body, str)

        # CRÍTICO: no debe haber <mstts:express-as>
        assert "<mstts:express-as" not in ssml_body
        assert "mstts" not in ssml_body.lower() or "<mstts" not in ssml_body

        # Sí debe estar el wrapper mínimo <speak> + <voice>
        assert "<speak" in ssml_body
        assert "<voice" in ssml_body
        assert "</speak>" in ssml_body
        assert "</voice>" in ssml_body

        # Voz configurada presente
        assert "es-AR-TomasNeural" in ssml_body

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_xml_escape(self, mock_client_cls, mock_settings):
        """Los caracteres &, <, > se escapan dentro del SSML."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _content_response(b"audio")

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        client.synthesize("texto con & < y > caracteres especiales")

        ssml_body = mock_client.post.call_args.kwargs["content"]

        # Caracteres originales NO deben aparecer sin escapar dentro del texto
        # (pueden aparecer como parte de los tags <speak>, <voice>, etc.)
        # Verificamos que las versiones escapadas SÍ están presentes:
        assert "&amp;" in ssml_body
        assert "&lt;" in ssml_body
        assert "&gt;" in ssml_body

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_failure(self, mock_client_cls, mock_settings):
        """Mock 500 → RuntimeError('Azure TTS falló')."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _err_response(500, "Server Error")

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")

        with pytest.raises(RuntimeError, match="Azure TTS falló"):
            client.synthesize("hola")


# ──────────────────────────────────────────────────────────────────
# Seguridad: API keys no logueadas
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
def test_no_api_keys_logged(mock_settings, caplog):
    """Las API keys de Gemini y Azure NO deben aparecer en logs al sintetizar."""
    secret_gemini = "SECRET_GEMINI_KEY_DO_NOT_LEAK_777"
    secret_azure = "SECRET_AZURE_KEY_DO_NOT_LEAK_888"

    # Gemini TTS
    with caplog.at_level(logging.DEBUG, logger="handlers.gemini_tts_client"):
        with patch("handlers.gemini_tts_client.httpx.Client") as mc1:
            mc1.return_value.post.return_value = _json_response(
                _gemini_tts_payload(b"\x00")
            )
            g_client = GeminiTTSClient(mock_settings, secret_gemini)
            g_client.synthesize("test", "calm")

    # Azure TTS
    with caplog.at_level(logging.DEBUG, logger="handlers.azure_tts_client"):
        with patch("handlers.azure_tts_client.httpx.Client") as mc2:
            mc2.return_value.post.return_value = _content_response(b"audio")
            a_client = AzureTTSClient(mock_settings, secret_azure, "southamericaeast")
            a_client.synthesize("test")

    all_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_gemini not in all_logs, (
        f"Gemini key filtrada: {[r.getMessage() for r in caplog.records]}"
    )
    assert secret_azure not in all_logs, (
        f"Azure key filtrada: {[r.getMessage() for r in caplog.records]}"
    )


# ──────────────────────────────────────────────────────────────────
# Azure TTS — synthesize_stream() con style_hint="" (Micro-Spec fallback)
# ──────────────────────────────────────────────────────────────────
#
# Cubre el fix introducido en ``fix/streaming-tts-fallback`` (commit c115c48):
# cuando ``synthesize_stream`` se invoca con ``style_hint=""`` (caso real desde
# el helper de fallback de main.py:380 cuando ``parse_response`` no encuentra
# prefijo [STYLE:]), el SSML generado NO debe incluir ``<mstts:express-as>``
# (que produciría ``style=""`` inválido).


def _make_azure_stream_response(chunks):
    """Helper: construye un mock del response de ``self._client.stream(...)``.

    ``httpx.Client.stream`` retorna un context manager. ``iter_bytes()`` es
    el método que el código llama dentro del with para leer chunks PCM.
    """
    mock_stream = MagicMock(name="AzureStreamResponse")
    mock_stream.__enter__.return_value = mock_stream
    mock_stream.__exit__.return_value = False
    mock_stream.raise_for_status = MagicMock()
    mock_stream.iter_bytes.return_value = iter(chunks)
    return mock_stream


@pytest.mark.unit
class TestAzureSynthesizeStreamStyleHint:
    """Tests para el fix de ``AzureTTSClient.synthesize_stream()`` con ``style_hint``."""

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_stream_no_style_hint(
        self, mock_client_cls, mock_settings
    ):
        """``synthesize_stream("hola", style_hint="")`` → SSML SIN ``<mstts:express-as>``.

        Verifica el bug del spec §1.3: con ``style_hint=""`` se generaba
        ``<mstts:express-as style="">`` (SSML inválido). El fix produce un
        wrapper mínimo, igual que ``synthesize()``.
        """
        mock_client = mock_client_cls.return_value
        fake_pcm_chunks = [b"\x00\x01" * 100, b"\x00\x01" * 100]
        mock_client.stream.return_value = _make_azure_stream_response(fake_pcm_chunks)

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        list(client.synthesize_stream("hola", style_hint=""))

        # Inspeccionar el body (content kwarg) enviado a httpx.Client.stream
        assert mock_client.stream.call_count == 1
        call_kwargs = mock_client.stream.call_args.kwargs
        ssml_body = call_kwargs["content"]
        assert isinstance(ssml_body, str)

        # CRÍTICO: no debe haber <mstts:express-as> (ni con style vacío ni válido)
        assert "<mstts:express-as" not in ssml_body, (
            f"SSML contiene <mstts:express-as> con style_hint='': {ssml_body!r}"
        )
        # Y tampoco el prefijo mstts suelto (xmlns:mstts no debería estar)
        assert "mstts" not in ssml_body.lower(), (
            f"SSML contiene 'mstts' con style_hint='': {ssml_body!r}"
        )

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_stream_with_style_hint(
        self, mock_client_cls, mock_settings
    ):
        """``synthesize_stream("hola", style_hint="cheerful")`` → SSML CON ``<mstts:express-as>``.

        Regresión: no romper el caso con style. La rama ``if style_hint:`` debe
        seguir generando el wrapper completo con ``xmlns:mstts`` y el tag.
        """
        mock_client = mock_client_cls.return_value
        fake_pcm_chunks = [b"\x00\x01" * 100]
        mock_client.stream.return_value = _make_azure_stream_response(fake_pcm_chunks)

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        list(client.synthesize_stream("hola", style_hint="cheerful"))

        call_kwargs = mock_client.stream.call_args.kwargs
        ssml_body = call_kwargs["content"]
        assert isinstance(ssml_body, str)

        # Debe incluir el wrapper con <mstts:express-as style="cheerful">
        assert '<mstts:express-as style="cheerful">' in ssml_body
        # Y el namespace xmlns:mstts
        assert "xmlns:mstts" in ssml_body
        assert "mstts:express-as" in ssml_body

    @patch("handlers.azure_tts_client.httpx.Client")
    def test_azure_synthesize_stream_no_style_hint_has_minimal_wrapper(
        self, mock_client_cls, mock_settings
    ):
        """``synthesize_stream("hola", style_hint="")`` → SSML con ``<speak>`` y ``<voice>`` solamente.

        El wrapper mínimo debe contener la voz configurada y NO debe llevar
        namespaces innecesarios (``xmlns``, ``xmlns:mstts``).
        """
        mock_client = mock_client_cls.return_value
        fake_pcm_chunks = [b"\x00\x01" * 100]
        mock_client.stream.return_value = _make_azure_stream_response(fake_pcm_chunks)

        client = AzureTTSClient(mock_settings, "fake_key", "southamericaeast")
        list(client.synthesize_stream("hola", style_hint=""))

        call_kwargs = mock_client.stream.call_args.kwargs
        ssml_body = call_kwargs["content"]
        assert isinstance(ssml_body, str)

        # Wrapper mínimo presente
        assert "<speak" in ssml_body
        assert "</speak>" in ssml_body
        assert "<voice" in ssml_body
        assert "</voice>" in ssml_body
        # Voz configurada presente
        assert "es-AR-TomasNeural" in ssml_body
        # El texto escapado presente
        assert "hola" in ssml_body
        # NO debe llevar xmlns (no se usan namespaces cuando no hay express-as)
        assert "xmlns=" not in ssml_body, (
            f"SSML lleva xmlns innecesario con style_hint='': {ssml_body!r}"
        )
