"""Cliente de síntesis de voz usando Piper TTS (local, CPU).

Implementa PiperTTSClient que usa piper-tts (ONNX Runtime) para generar
audio PCM a partir de texto. No requiere API key ni conexión a internet
(tras descarga inicial de la voz).
"""

import io
import logging
import os
import wave
from pathlib import Path
from typing import Iterator

from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

logger = logging.getLogger(__name__)

_WAV_HEADER_SIZE = 44  # Cabecera WAV estándar (PCM s16le)
_PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


class PiperTTSClient:
    """Cliente para síntesis de voz con Piper local.

    Attributes:
        settings: Dict con configuración local.piper (voice_model, voices_dir, length_scale).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente TTS local.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['piper']).
        """
        self.settings = settings
        self._voice: PiperVoice | None = None  # lazy-load

        cfg = settings["local"]["piper"]
        logger.debug(
            "PiperTTSClient inicializado — voice=%s, voices_dir=%s",
            cfg["voice_model"], os.path.basename(cfg["voices_dir"]),
        )

    def _ensure_voice_loaded(self) -> None:
        """Descarga y carga la voz de Piper si aún no está cargada (lazy-load)."""
        if self._voice is not None:
            return
        cfg = self.settings["local"]["piper"]
        voices_dir = Path(cfg["voices_dir"])
        voice_model = cfg["voice_model"]

        # Rutas: download_voice descarga {voice}.onnx y {voice}.onnx.json directo a voices_dir
        onnx_path = voices_dir / f"{voice_model}.onnx"
        json_path = voices_dir / f"{voice_model}.onnx.json"

        # Descargar si no existe
        if not onnx_path.exists():
            logger.info("Descargando voz Piper — voice=%s...", voice_model)
            voices_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice_model, voices_dir)
            logger.info("Voz Piper descargada OK")

        # Cargar modelo
        logger.info("Cargando voz Piper local — voice=%s...", voice_model)
        self._voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
        logger.info("Voz Piper cargada OK")

    def synthesize(self, text: str, style_hint: str = "") -> bytes:
        """Sintetiza texto a voz usando Piper local.

        Genera audio PCM crudo 24kHz mono s16le a partir de texto.
        El style_hint se ignora (Piper no soporta estilos).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado (compatibilidad de firma con GeminiTTSClient).

        Returns:
            Bytes PCM crudo s16le (sin cabecera WAV) — compatible con AudioManager.play_audio().

        Raises:
            RuntimeError: Si la síntesis falla.
        """
        self._ensure_voice_loaded()

        cfg = self.settings["local"]["piper"]
        try:
            # Piper sintetiza a un buffer WAV via synthesize_wav
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self._voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=SynthesisConfig(length_scale=cfg["length_scale"]),
                )
            wav_bytes = buffer.getvalue()
        except Exception as exc:
            logger.error("Piper TTS falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Piper TTS falló: {exc}") from exc

        # Extraer PCM crudo (saltar cabecera WAV de 44 bytes)
        pcm_bytes = wav_bytes[_WAV_HEADER_SIZE:]

        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Piper TTS OK — texto='%s', %d bytes PCM", truncated, len(pcm_bytes))
        return pcm_bytes

    def synthesize_stream(self, text: str, style_hint: str = "") -> Iterator[bytes]:
        """Versión streaming: sintetiza y hace yield de chunks PCM.

        Piper no soporta streaming nativo, así que sintetiza todo y
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
