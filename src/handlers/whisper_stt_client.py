"""Cliente de transcripción de voz (STT) usando faster-whisper (local, GPU).

Implementa WhisperSTTClient que carga un modelo Whisper localmente
(faster-whisper / CTranslate2) y transcribe archivos .wav a texto.
No requiere API key ni conexión a internet (tras descarga inicial del modelo).
"""

import logging
import os

# Registrar DLLs de CUDA (cuBLAS, cuDNN, cuDNN) para ctranslate2 en Windows.
# Estas libs se instalan via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) pero
# Windows no las encuentra automaticamente porque estan en subdirectorios
# de site-packages/nvidia/. os.add_dll_directory() las hace visibles a
# LoadLibraryEx que usa ctranslate2 internamente.
# Si las libs no estan instaladas (CPU-only), el registro se omite silenciosamente
# y faster_whisper intentara cargar normalmente (fallando con su mensaje habitual).
try:
    import nvidia
    _nvidia_base = list(nvidia.__path__)[0]
    for _lib_name in ("cublas", "cudnn", "cuda_nvrtc"):
        _bin_dir = os.path.join(_nvidia_base, _lib_name, "bin")
        if os.path.isdir(_bin_dir):
            os.add_dll_directory(_bin_dir)
            # Tambien agregar al PATH: ctranslate2 usa LoadLibraryEx que en algunos
            # casos consulta PATH ademas de los directorios registrados.
            os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
except (ImportError, OSError):
    pass  # nvidia-* no instalado o path invalido — CPU fallback

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperSTTClient:
    """Cliente para transcripción de audio con Whisper local.

    Carga el modelo una vez en __init__ (lazy-load diferido al primer
    transcribe() para no bloquear el startup si no hay GPU).

    Attributes:
        settings: Dict con configuración local.whisper (model, device, compute_type, language, beam_size).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente STT local.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['whisper']).
        """
        self.settings = settings
        self._model: WhisperModel | None = None  # lazy-load

        cfg = settings["local"]["whisper"]
        logger.debug(
            "WhisperSTTClient inicializado — model=%s, device=%s, compute_type=%s",
            cfg["model"], cfg["device"], cfg["compute_type"],
        )

    def _ensure_model_loaded(self) -> None:
        """Carga el modelo Whisper si aún no está cargado (lazy-load)."""
        if self._model is not None:
            return
        cfg = self.settings["local"]["whisper"]
        logger.info("Cargando modelo Whisper local — model=%s, device=%s...", cfg["model"], cfg["device"])
        self._model = WhisperModel(
            cfg["model"],
            device=cfg["device"],
            compute_type=cfg["compute_type"],
        )
        logger.info("Modelo Whisper cargado OK")

    def transcribe(self, wav_path: str) -> str:
        """Transcribe un archivo .wav a texto usando Whisper local.

        Lee el WAV, lo pasa al modelo Whisper y retorna el texto transcrito.
        Aplica limpieza: strip() de espacios.

        Args:
            wav_path: Ruta absoluta al archivo .wav a transcribir.

        Returns:
            Texto transcrito (str), limpio y sin espacios extra.

        Raises:
            FileNotFoundError: Si wav_path no existe.
            RuntimeError: Si el modelo falla al transcribir.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio no encontrado: {wav_path}")

        self._ensure_model_loaded()

        cfg = self.settings["local"]["whisper"]
        try:
            segments, _info = self._model.transcribe(
                wav_path,
                language=cfg["language"],
                beam_size=cfg["beam_size"],
            )
            # segments es un generator — consumir y concatenar
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception as exc:
            logger.error("Whisper STT falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Whisper STT falló: {exc}") from exc

        truncated = text[:100] + "..." if len(text) > 100 else text
        logger.debug("Whisper STT OK — texto='%s'", truncated)
        return text
