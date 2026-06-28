"""Cliente de síntesis de voz usando Kokoro-ONNX (local, CPU).

Implementa KokoroTTSClient que usa kokoro-onnx para generar audio PCM
a partir de texto. No requiere API key ni conexión a internet (tras
descarga manual de los archivos de modelo).
"""

import logging
import os
import re
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

    def _split_text(self, text: str) -> list[str]:
        """Splitea texto en chunks seguros para Kokoro (<510 phonemas por batch).

        kokoro-onnx 0.5.0 tiene MAX_PHONEME_LENGTH=510. Cuando el texto
        genera >510 phonemas, `_split_phonemes` upstream solo splitea en
        `[.,!?;]` — no splitea en `:`, `—`, `–`, `/`. Un batch excede 510
        → trima → `voice[510]` out of bounds → IndexError.

        Estrategia:
        1. Colapsar whitespace (mismo paso que en synthesize()).
        2. Split por puntuación fuerte + separadores (lookbehind): preserva
           el signo en el chunk anterior. Incluye em-dash, en-dash, slash
           y dos puntos (útiles para horarios `12:30`, rangos `lunes—martes`,
           separadores `14/15`).
        3. Safety net: chunks muy largos → split por espacios para evitar
           un único chunk que aún supere el límite.

        Args:
            text: Texto crudo a sintetizar.

        Returns:
            Lista de chunks no vacíos, cada uno seguro para una llamada
            a `Kokoro.create()` sin superar 510 phonemas.
        """
        normalized = re.sub(r"\s+", " ", text).strip()
        # Split después de puntuación fuerte + separadores (em-dash, en-dash, slash, dos puntos)
        chunks = re.split(r"(?<=[.,;:!?—–/])\s+", normalized)
        chunks = [c.strip() for c in chunks if c.strip()]
        # Safety net: chunks muy largos → split por espacios
        # Heurística: ~3-4 chars por phonema → 1500 chars ≈ <510 phonemas.
        MAX_CHARS = 1500
        safe_chunks: list[str] = []
        for c in chunks:
            if len(c) > MAX_CHARS:
                words = c.split(" ")
                current = ""
                for w in words:
                    if current and len(current) + len(w) + 1 > MAX_CHARS:
                        safe_chunks.append(current.strip())
                        current = w
                    else:
                        current = (current + " " + w).strip() if current else w
                if current:
                    safe_chunks.append(current.strip())
            else:
                safe_chunks.append(c)
        return safe_chunks

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

        # Colapsar cualquier secuencia de whitespace (newlines, tabs, espacios múltiples) a un solo espacio.
        # Previene el WARNING "words count mismatch" de phonemizer y artefactos de audio
        # (kokoro-onnx 0.5.0 no normaliza newlines internos — ver PR upstream #185).
        # Además, spliteamos el texto en chunks seguros (<510 phonemas cada uno) para evitar
        # el IndexError upstream cuando un batch excede MAX_PHONEME_LENGTH (ver issue #184).
        chunks = self._split_text(text)
        logger.debug("Kokoro TTS — %d chunks tras split", len(chunks))
        first_chunk = chunks[0] if chunks else ""
        logger.debug(
            "Kokoro TTS — texto normalizado (chunk 1)='%s'",
            first_chunk[:120] + ("..." if len(first_chunk) > 120 else ""),
        )

        try:
            audio_parts: list[np.ndarray] = []
            for i, chunk in enumerate(chunks):
                samples_part, _ = self._kokoro.create(
                    chunk,
                    voice=cfg["voice"],
                    speed=cfg["speed"],
                    lang=cfg["lang"],
                    trim=False,  # evitar corte agresivo de la última sílaba
                )
                audio_parts.append(samples_part)
                logger.debug(
                    "Kokoro TTS — chunk %d/%d sintetizado (%d samples)",
                    i + 1, len(chunks), len(samples_part),
                )
            samples = np.concatenate(audio_parts)
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

    def synthesize_sentence_stream(self, sentences: Iterator[str]) -> Iterator[bytes]:
        """Sintetiza oraciones una a una y hace yield de PCM completo por oración.

        Diferencia con synthesize_stream: recibe un Iterator[str] de oraciones
        (no un texto completo) y sintetiza cada oración independientemente,
        haciendo yield del PCM completo de cada una. Esto permite que el
        playback empiece antes de que terminen de llegar todas las oraciones.

        Args:
            sentences: Iterator que yields oraciones (str) una a una.

        Yields:
            Bytes PCM crudo s16le (sin cabecera WAV) — un yield por oración.
        """
        for sentence in sentences:
            if not sentence.strip():
                continue
            pcm_bytes = self.synthesize(sentence, style_hint="")
            yield pcm_bytes
