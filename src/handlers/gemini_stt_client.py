"""Cliente de transcripción de voz (STT) usando Gemini (Google AI Studio).

Implementa GeminiSTTClient que envía .wav a la API REST de Gemini
vía generateContent con inline_data (audio/wav base64) + stt_prompt.
Incluye failover automático: gemini-3.1-flash-lite → gemini-2.5-flash-lite.
"""

import base64
import logging

import httpx

logger = logging.getLogger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
_STT_ENDPOINT = "/v1beta/models/{model}:generateContent"
_MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB


class GeminiSTTClient:
    """Cliente para transcripción de audio con Gemini STT.

    Envía archivos .wav codificados en base64 como inline_data a la API
    generateContent de Gemini. Soporta failover automático entre modelo
    primario (gemini-3.1-flash-lite) y fallback (gemini-2.5-flash-lite).

    Attributes:
        settings: Dict con configuración gemini (modelos y stt_prompt).
        api_key: API key de Google AI Studio.
    """

    def __init__(self, settings: dict, api_key: str) -> None:
        """Inicializa el cliente HTTP para Gemini STT.

        Args:
            settings: Dict completo de settings.json (usa settings['gemini']).
            api_key: API key de Google AI Studio (NO se loguea).
        """
        self.settings = settings
        self.api_key = api_key

        self._client = httpx.Client(
            base_url=_GEMINI_BASE_URL,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )

        logger.debug(
            "GeminiSTTClient inicializado — primario=%s, fallback=%s, timeout=60s",
            settings["gemini"]["stt_model_primary"],
            settings["gemini"]["stt_model_fallback"],
        )

    def transcribe(self, wav_path: str) -> str:
        """Transcribe un archivo .wav a texto usando Gemini STT.

        Lee el archivo WAV, lo codifica a base64 y lo envía al modelo
        primario gemini-3.1-flash-lite. Si falla por rate-limit (429),
        error de servidor (5xx) o timeout, reintenta automáticamente con
        el modelo fallback gemini-2.5-flash-lite.

        Args:
            wav_path: Ruta absoluta al archivo .wav a transcribir.

        Returns:
            Texto transcrito (str), limpio y sin espacios extra.

        Raises:
            ValueError: Si el archivo supera los 20 MB.
            RuntimeError: Si ambos modelos (primario y fallback) fallan.
        """
        # --- Leer y validar tamaño del audio ---
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        audio_size_mb = len(audio_bytes) / (1024 * 1024)
        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio demasiado grande para inline_data (máx 20MB). "
                f"Tamaño actual: {audio_size_mb:.1f} MB"
            )

        # Codificar a base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        logger.debug(
            "Audio cargado — path=%s, tamaño=%.2f MB, base64=%.2f KB",
            wav_path,
            audio_size_mb,
            len(audio_b64) / 1024,
        )

        # --- Construir payload base ---
        stt_prompt = self.settings["gemini"]["stt_prompt"]
        payload = {
            "contents": [{
                "parts": [
                    {"text": stt_prompt},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ]
            }]
        }

        # --- Intentar con modelo primario ---
        primary_model = self.settings["gemini"]["stt_model_primary"]
        try:
            text = self._call_gemini(primary_model, payload)
            truncated = text[:100] + "..." if len(text) > 100 else text
            logger.debug(
                "STT primario OK — modelo=%s, texto='%s'",
                primary_model,
                truncated,
            )
            return text.strip()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning(
                "Error en modelo primario (%s) — %s: %s. Iniciando failover...",
                primary_model,
                type(exc).__name__,
                exc,
            )

        # --- Failover: modelo fallback ---
        fallback_model = self.settings["gemini"]["stt_model_fallback"]
        logger.warning(
            "Failover a %s tras error en %s",
            fallback_model,
            primary_model,
        )

        try:
            text = self._call_gemini(fallback_model, payload)
            truncated = text[:100] + "..." if len(text) > 100 else text
            logger.debug(
                "STT fallback OK — modelo=%s, texto='%s'",
                fallback_model,
                truncated,
            )
            return text.strip()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error(
                "Error en modelo fallback (%s) — %s: %s",
                fallback_model,
                type(exc).__name__,
                exc,
            )
            raise RuntimeError(
                "STT falló: ambos modelos primario y fallback no respondieron"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _call_gemini(self, model: str, payload: dict) -> str:
        """Envía el payload a la API de Gemini y extrae el texto transcrito.

        Args:
            model: ID del modelo a usar (ej. 'gemini-3.1-flash-lite').
            payload: Dict con la estructura del request body.

        Returns:
            Texto transcrito extraído de candidates[0].content.parts[0].text.

        Raises:
            httpx.HTTPStatusError: Si la API responde con error HTTP.
            httpx.TimeoutException: Si la request excede el timeout.
            httpx.RequestError: Si hay error de red/conexión.
            RuntimeError: Si la respuesta no tiene la estructura esperada.
        """
        response = self._client.post(
            _STT_ENDPOINT.format(model=model),
            json=payload,
        )
        response.raise_for_status()

        try:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "Gemini STT respuesta inesperada — modelo=%s, body=%s",
                model,
                response.text[:300],
            )
            raise RuntimeError(
                f"Gemini STT ({model}) respuesta inesperada"
            ) from exc

        return text
