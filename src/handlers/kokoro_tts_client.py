"""Cliente de síntesis de voz usando Kokoro-ONNX (local, CPU).

Implementa KokoroTTSClient que usa kokoro-onnx para generar audio PCM
a partir de texto. No requiere API key ni conexión a internet (tras
descarga manual de los archivos de modelo).
"""

import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

# Kokoro produce float32 24kHz. AudioManager.play_audio espera PCM crudo
# s16le (int16 little-endian). Hay que convertir.
_SAMPLE_RATE = 24000  # Kokoro siempre produce 24kHz


class KokoroTTSClient:
    """Cliente para síntesis de voz con Kokoro local.

    Attributes:
        settings: Dict con configuración local.kokoro (model_path, voices_path, voice, lang, speed).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente TTS local Kokoro.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['kokoro']).
        """
        self.settings = settings
        self._kokoro: Kokoro | None = None  # lazy-load

        cfg = settings["local"]["kokoro"]
        logger.debug(
            "KokoroTTSClient inicializado — voice=%s, lang=%s, model=%s",
            cfg["voice"], cfg["lang"], os.path.basename(cfg["model_path"]),
        )

    def _ensure_model_loaded(self) -> None:
        """Carga el modelo Kokoro si aún no está cargado (lazy-load).

        A diferencia de Piper, NO hay auto-download. Si los archivos
        model_path o voices_path no existen, lanza RuntimeError con
        instrucciones de descarga.

        Raises:
            RuntimeError: Si los archivos de modelo no existen.
        """
        if self._kokoro is not None:
            return
        cfg = self.settings["local"]["kokoro"]
        model_path = Path(cfg["model_path"])
        voices_path = Path(cfg["voices_path"])

        if not model_path.exists():
            raise RuntimeError(
                f"Modelo Kokoro no encontrado: {model_path.name}. "
                f"Descargar kokoro-v1.0.onnx desde "
                f"https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )
        if not voices_path.exists():
            raise RuntimeError(
                f"Voces Kokoro no encontradas: {voices_path.name}. "
                f"Descargar voices-v1.0.bin desde "
                f"https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )

        logger.info("Cargando modelo Kokoro local — model=%s...", model_path.name)
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        logger.info("Modelo Kokoro cargado OK")

    def synthesize(self, text: str, style_hint: str = "") -> bytes:
        """Sintetiza texto a voz usando Kokoro local.

        Genera audio PCM crudo 24kHz mono s16le a partir de texto.
        El style_hint se ignora (Kokoro no soporta estilos SSML).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado (compatibilidad de firma con GeminiTTSClient).

        Returns:
            Bytes PCM crudo s16le (sin cabecera WAV) — compatible con AudioManager.play_audio().

        Raises:
            RuntimeError: Si la síntesis falla o el modelo no está descargado.
        """
        self._ensure_model_loaded()

        cfg = self.settings["local"]["kokoro"]
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=cfg["voice"],
                speed=cfg["speed"],
                lang=cfg["lang"],
            )
        except Exception as exc:
            logger.error("Kokoro TTS falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Kokoro TTS falló: {exc}") from exc

        # Kokoro retorna np.ndarray float32 en [-1.0, 1.0].
        # Convertir a int16 PCM little-endian para AudioManager.play_audio().
        samples_clipped = np.clip(samples, -1.0, 1.0)
        samples_int16 = (samples_clipped * 32767).astype(np.int16)
        pcm_bytes = samples_int16.tobytes()  # little-endian en x86/x64

        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Kokoro TTS OK — texto='%s', %d bytes PCM", truncated, len(pcm_bytes))
        return pcm_bytes

    def synthesize_stream(self, text: str, style_hint: str = "") -> Iterator[bytes]:
        """Versión streaming: sintetiza y hace yield de chunks PCM.

        Kokoro no soporta streaming nativo, así que sintetiza todo y
        divide en chunks de 4096 bytes (compatible con play_audio_stream).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado.

        Yields:
            Bytes PCM crudo (sin cabecera WAV) en chunks de hasta 4096 bytes.
        """
        pcm_bytes = self.synthesize(text, style_hint)
        chunk_size = 4096
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i:i + chunk_size]
