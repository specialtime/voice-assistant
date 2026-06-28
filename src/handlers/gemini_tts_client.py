"""Cliente de síntesis de voz usando Gemini TTS (Google AI Studio).

Implementa GeminiTTSClient que llama a la API REST de Gemini
gemini-3.1-flash-tts-preview para generar audio PCM 24 kHz mono s16le
a partir de texto con control de estilo por lenguaje natural.
"""

import base64
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_GEMINI_TTS_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiTTSClient:
    """Cliente para síntesis de voz con Gemini TTS.

    Attributes:
        settings: Dict con configuración (gemini.tts_model, gemini.tts_voice).
        api_key: API key de Google AI Studio.
    """

    def __init__(self, settings: dict, api_key: str) -> None:
        """Inicializa el cliente HTTP para Gemini TTS.

        Args:
            settings: Dict con claves 'gemini.tts_model' y 'gemini.tts_voice'.
            api_key: API key de Google AI Studio (NO se loguea).
        """
        self.settings = settings
        self.api_key = api_key

        self._client = httpx.Client(
            base_url=_GEMINI_TTS_BASE_URL,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

        logger.debug(
            "GeminiTTSClient inicializado — modelo=%s, voz=%s",
            settings["gemini"]["tts_model"],
            settings["gemini"]["tts_voice"],
        )

        # Circuit breaker para 429 de quota excedida (Micro-Spec A)
        self._circuit_open: bool = False
        self._circuit_open_until: float = 0.0
        self._circuit_cooldown_seconds: float = settings["gemini"].get("tts_circuit_breaker_cooldown_seconds", 300.0)

    def is_available(self) -> bool:
        """Retorna True si el circuit breaker está cerrado (puede intentar Gemini TTS).

        Si está abierto pero pasó el cooldown, lo cierra y retorna True."""
        if not self._circuit_open:
            return True
        if time.time() >= self._circuit_open_until:
            self._circuit_open = False
            logger.info("Gemini TTS circuit breaker cerrado — reintentando")
            return True
        return False

    def synthesize(self, text: str, style_hint: str = "") -> bytes:
        """Sintetiza texto a voz usando Gemini TTS.

        Compone un prompt en lenguaje natural (ej. 'Say cheerfully: ...')
        y lo envía a gemini-3.1-flash-tts-preview. Retorna los bytes PCM
        24 kHz mono s16le decodificados de la respuesta base64.

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Estilo opcional (ej. 'cheerful', 'sad', etc.).

        Returns:
            Bytes PCM 24 kHz mono s16le.

        Raises:
            RuntimeError: Si la API responde con error o timeout.
        """
        # Componer prompt según §4.6
        if style_hint:
            prompt = f"Say {style_hint}: {text}"
        else:
            prompt = f"Say: {text}"

        # Log debug del prompt compuesto (truncado)
        truncated_prompt = prompt[:150] + "..." if len(prompt) > 150 else prompt
        logger.debug("Gemini TTS prompt='%s'", truncated_prompt)

        model = self.settings["gemini"]["tts_model"]

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.settings["gemini"]["tts_voice"],
                        }
                    }
                }
            }
        }

        if not self.is_available():
            logger.warning("Gemini TTS circuit breaker abierto — omitiendo llamada")
            raise RuntimeError("Gemini TTS circuit breaker abierto")

        try:
            response = self._client.post(
                f"/v1beta/models/{model}:generateContent",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Gemini TTS HTTP error — status=%s, response=%s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            if exc.response.status_code == 429:
                self._circuit_open = True
                self._circuit_open_until = time.time() + self._circuit_cooldown_seconds
                logger.warning(
                    "Gemini TTS circuit breaker abierto por 429 — cooldown %ss",
                    self._circuit_cooldown_seconds,
                )
            raise RuntimeError("Gemini TTS falló") from exc
        except httpx.TimeoutException as exc:
            logger.error("Gemini TTS timeout — texto='%s'", truncated_prompt)
            raise RuntimeError("Gemini TTS falló") from exc
        except httpx.RequestError as exc:
            logger.error("Gemini TTS request error — %s", exc)
            raise RuntimeError("Gemini TTS falló") from exc

        # Extraer base64 de la respuesta
        try:
            data = response.json()
            audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "Gemini TTS respuesta inesperada — %s",
                response.text[:300],
            )
            raise RuntimeError("Gemini TTS falló") from exc

        # Decodificar base64 a PCM bytes
        pcm_bytes = base64.b64decode(audio_b64)

        # Éxito → cerrar circuito si estaba abierto
        self._circuit_open = False

        logger.debug(
            "Gemini TTS recibidos %d bytes PCM",
            len(pcm_bytes),
        )

        return pcm_bytes
