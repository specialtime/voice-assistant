"""Cliente de síntesis de voz usando Azure Cognitive Services (fallback).

Implementa AzureTTSClient que llama a la REST API de Azure TTS
con un wrapper SSML mínimo (solo <speak> + <voice>, sin <mstts:express-as>)
para generar audio PCM crudo 24 kHz mono s16le con voz es-AR-TomasNeural.
"""

import logging
import xml.sax.saxutils
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)


class AzureTTSClient:
    """Cliente para síntesis de voz con Azure TTS (fallback).

    Attributes:
        settings: Dict con configuración (azure.voice, azure.locale, azure.output_format).
        key: Azure Speech subscription key.
        region: Azure Speech region (ej. southamericaeast).
    """

    def __init__(self, settings: dict, key: str, region: str) -> None:
        """Inicializa el cliente HTTP para Azure TTS.

        Args:
            settings: Dict con claves 'azure.voice', 'azure.locale', 'azure.output_format'.
            key: Azure Speech subscription key (NO se loguea).
            region: Azure Speech region (ej. 'southamericaeast').
        """
        self.settings = settings
        self.key = key
        self.region = region

        self._client = httpx.Client(timeout=httpx.Timeout(30.0))

        logger.debug(
            "AzureTTSClient inicializado — voz=%s, locale=%s, region=%s",
            settings["azure"]["voice"],
            settings["azure"]["locale"],
            region,
        )

    def synthesize(self, text: str) -> bytes:
        """Sintetiza texto a voz usando Azure TTS REST API.

        Envía el texto envuelto en un SSML mínimo (solo <speak> + <voice>,
        SIN <mstts:express-as>) a la API de Azure. Retorna los bytes de audio
        PCM crudo 24 kHz mono s16le.

        Args:
            text: Texto limpio a sintetizar.

        Returns:
            Bytes de audio PCM crudo 24 kHz mono s16le.

        Raises:
            RuntimeError: Si la API responde con error o timeout.
        """
        # Escapar caracteres XML especiales antes de insertar en SSML
        escaped_text = xml.sax.saxutils.escape(text)

        # Wrapper SSML mínimo — sin <mstts:express-as> (requisito §4.7)
        ssml = (
            f'<speak version="1.0" xml:lang="{self.settings["azure"]["locale"]}">'
            f'<voice name="{self.settings["azure"]["voice"]}">'
            f"{escaped_text}"
            f"</voice>"
            f"</speak>"
        )

        # Log debug del texto (truncado)
        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Azure TTS texto='%s' (%d chars)", truncated, len(text))

        endpoint = (
            f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self.settings["azure"]["output_format"],
        }

        try:
            response = self._client.post(endpoint, headers=headers, content=ssml)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Azure TTS HTTP error — status=%s, response=%s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            raise RuntimeError("Azure TTS falló") from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "Azure TTS timeout — texto='%s'",
                truncated,
            )
            raise RuntimeError("Azure TTS falló") from exc
        except httpx.RequestError as exc:
            logger.error("Azure TTS request error — %s", exc)
            raise RuntimeError("Azure TTS falló") from exc

        audio_bytes = response.content

        logger.debug(
            "Azure TTS recibidos %d bytes PCM",
            len(audio_bytes),
        )

        return audio_bytes

    def synthesize_stream(self, text: str, style_hint: str = "") -> Iterator[bytes]:
        """Versión streaming de synthesize(). Usa httpx.stream() para leer el body
        en chunks y hacer yield de bytes PCM conforme llegan.

        Mismo SSML, mismos headers, mismo endpoint que synthesize().
        Diferencia: no espera el body completo; hace yield de chunks de 4096 bytes.

        Args:
            text: Texto limpio a sintetizar.

        Yields:
            Bytes PCM crudo 24kHz mono s16le en chunks de hasta 4096 bytes.

        Raises:
            RuntimeError: Si la API responde con error HTTP o de red.
        """
        # Escapar caracteres XML especiales antes de insertar en SSML
        escaped_text = xml.sax.saxutils.escape(text)

        # SSML condicional: si hay style_hint, incluir <mstts:express-as>;
        # si no, wrapper mínimo (igual que synthesize()) para evitar style="" inválido.
        if style_hint:
            ssml = (
                f'<speak version="1.0" '
                f'xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xmlns:mstts="https://www.w3.org/2001/mstts" '
                f'xml:lang="{self.settings["azure"]["locale"]}">\n'
                f'    <voice name="{self.settings["azure"]["voice"]}">\n'
                f'        <mstts:express-as style="{style_hint}">\n'
                f'            {escaped_text}\n'
                f'        </mstts:express-as>\n'
                f'    </voice>\n'
                f'</speak>'
            )
        else:
            ssml = (
                f'<speak version="1.0" '
                f'xml:lang="{self.settings["azure"]["locale"]}">\n'
                f'    <voice name="{self.settings["azure"]["voice"]}">\n'
                f'        {escaped_text}\n'
                f'    </voice>\n'
                f'</speak>'
            )

        # Log debug del texto (truncado)
        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Azure TTS stream texto='%s' (%d chars) style=%s", truncated, len(text), style_hint)

        endpoint = (
            f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self.settings["azure"]["output_format"],
        }

        try:
            with self._client.stream("POST", endpoint, headers=headers, content=ssml) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=4096):
                    yield chunk
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Azure TTS stream HTTP error — status=%s",
                exc.response.status_code,
            )
            raise RuntimeError("Azure TTS falló") from exc
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error("Azure TTS stream error — %s", exc)
            raise RuntimeError("Azure TTS falló") from exc
