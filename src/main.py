"""Loop principal del Asistente de Voz (Jarvis).

Implementa la máquina de estados IDLE / RECORDING / PROCESSING,
el listener global de teclado (Alt+V) y el pipeline de 6 pasos
que se conectará en fases posteriores.
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import keyboard
from dotenv import load_dotenv

from handlers.audio_manager import AudioManager
from handlers.gemini_stt_client import GeminiSTTClient
from handlers.opencode_client import OpenCodeClient
from handlers.gemini_tts_client import GeminiTTSClient
from handlers.azure_tts_client import AzureTTSClient
from handlers.response_parser import _strip_markdown, parse_response
from handlers.sentence_buffer import SentenceBuffer
from handlers.overlay import OverlayChip
from handlers.whisper_stt_client import WhisperSTTClient
from handlers.piper_tts_client import PiperTTSClient
from handlers.kokoro_tts_client import KokoroTTSClient

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Orquestador principal del asistente de voz.

    Máquina de estados de 4 estados (idle, recording, processing, speaking).
    Responde al hotkey Alt+V para toggle entre estados y
    lanza el pipeline de procesamiento en un hilo separado.

    Attributes:
        STATE_IDLE: Constante del estado ocioso.
        STATE_RECORDING: Constante del estado grabando.
        STATE_PROCESSING: Constante del estado procesando.
        STATE_SPEAKING: Constante del estado hablando.
    """

    STATE_IDLE = "idle"
    STATE_RECORDING = "recording"
    STATE_PROCESSING = "processing"
    STATE_SPEAKING = "speaking"

    def __init__(self) -> None:
        """Inicializa el asistente: carga .env, settings, AudioManager.

        Los clientes STT, OpenCode, TTS se inicializan en None y se
        conectarán en Fases 3, 4 y 5 respectivamente.
        """
        load_dotenv()

        settings_path = Path("config/settings.json")
        with open(settings_path, encoding="utf-8") as f:
            self._settings: dict = json.load(f)

        self._state: str = self.STATE_IDLE
        self._lock = threading.Lock()
        self._pipeline_generation: int = 0  # generation counter para cancelación
        self._send_lock = threading.Lock()  # anti-concurrencia HTTP en send_command
        self._audio = AudioManager(self._settings)

        # Clientes STT (local primario + cloud fallback)
        gemini_key = os.getenv("GEMINI_API_KEY")
        azure_key = os.getenv("AZURE_SPEECH_KEY")
        azure_region = os.getenv("AZURE_SPEECH_REGION", "southamericaeast")
        opencode_password = os.getenv("OPENCODE_SERVER_PASSWORD")
        opencode_base_url = os.getenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")

        # STT: Whisper local (primario), Gemini (fallback)
        self._whisper_stt = WhisperSTTClient(self._settings)
        self._stt = GeminiSTTClient(self._settings, gemini_key) if gemini_key else None

        # TTS local: selector de motor (piper | kokoro)
        tts_engine = self._settings.get("local", {}).get("tts_engine", "piper")
        if tts_engine not in ("piper", "kokoro"):
            logger.warning("tts_engine='%s' inválido, usando 'piper' por defecto", tts_engine)
            tts_engine = "piper"

        if tts_engine == "kokoro":
            self._local_tts = KokoroTTSClient(self._settings)
            logger.info("TTS local: Kokoro (selector)")
        else:
            self._local_tts = PiperTTSClient(self._settings)
            logger.info("TTS local: Piper (selector)")

        # TTS cloud fallback (sin cambios)
        self._gemini_tts = GeminiTTSClient(self._settings, gemini_key) if gemini_key else None
        self._azure_tts = AzureTTSClient(self._settings, azure_key, azure_region) if azure_key else None

        # OpenCode (sin cambios)
        self._opencode = OpenCodeClient(self._settings, opencode_password or "", opencode_base_url) if opencode_base_url else None
        self._streaming_enabled: bool = self._settings.get("opencode", {}).get("streaming_enabled", True)

        if not gemini_key:
            logger.warning("GEMINI_API_KEY no configurada — Gemini STT/TTS fallback no disponible")
        if not azure_key:
            logger.warning("AZURE_SPEECH_KEY no configurada — Azure TTS fallback no disponible")
        if not opencode_base_url:
            logger.warning("OPENCODE_BASE_URL no configurada — agente no disponible")

        logger.info("VoiceAssistant inicializado — hotkey=%s", self._settings["hotkey"])

        # Overlay chip visual (feedback de estado en pantalla)
        self._overlay = OverlayChip()
        self._overlay.start()


    # ── Propiedades ────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Estado actual de la máquina de estados."""
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        """Establece el estado actual."""
        self._state = value

    # ── Máquina de estados ────────────────────────────────────────

    def toggle(self) -> None:
        """Handler del hotkey. Implementa la máquina de estados de 4 estados.

        Transiciones:
            IDLE      → start_recording() + RECORDING
            RECORDING → stop_recording() + run_pipeline(hilo) + PROCESSING
            PROCESSING→ cancela pipeline (generation++) + start_recording() + RECORDING
            SPEAKING  → stop_playback() + cancela pipeline (generation++) + start_recording() + RECORDING
        """
        with self._lock:
            logger.debug("toggle — estado actual: %s", self._state)

            if self._state == self.STATE_IDLE:
                self._audio.start_recording()
                self._state = self.STATE_RECORDING
                logger.info("→ RECORDING")
                self._overlay.show("recording")

            elif self._state == self.STATE_RECORDING:
                wav_path = self._audio.stop_recording()
                self._state = self.STATE_PROCESSING
                self._pipeline_generation += 1  # NUEVO — nueva generación
                logger.info("→ PROCESSING (wav=%s, gen=%d)", wav_path, self._pipeline_generation)
                self._overlay.set_state("processing")
                threading.Thread(
                    target=self.run_pipeline,
                    args=(wav_path,),
                    daemon=True,
                ).start()

            elif self._state == self.STATE_PROCESSING:
                # NUEVO — cancelar pipeline y volver a grabar
                self._pipeline_generation += 1
                self._audio.start_recording()
                self._state = self.STATE_RECORDING
                logger.info("→ RECORDING (interrumpió procesamiento, gen=%d)", self._pipeline_generation)
                self._overlay.show("recording")

            elif self._state == self.STATE_SPEAKING:
                # NUEVO — interrumpir playback y volver a grabar
                self._pipeline_generation += 1
                self._audio.stop_playback()
                self._audio.start_recording()
                self._state = self.STATE_RECORDING
                logger.info("→ RECORDING (interrumpió playback, gen=%d)", self._pipeline_generation)
                self._overlay.show("recording")

    # ── Pipeline ──────────────────────────────────────────────────

    def run_pipeline(self, wav_path: str) -> None:
        """Ejecuta el pipeline completo de procesamiento de voz.

        Con cancelación cooperativa via generation counter: si el usuario
        interrumpe (Alt+V durante PROCESSING o SPEAKING), la generación
        global incrementa y este pipeline aborta en el próximo checkpoint.

        Flujo de 7 pasos:
            1. STT: transcribe wav a texto
            2. Agente: envía texto a OpenCode y obtiene respuesta (con send_lock)
            3. Parse: extrae [STYLE: ...] y texto limpio
            4. Transición a SPEAKING (antes del TTS)
            5-6. TTS: sintetiza con Gemini TTS (fallback Azure TTS streaming)
            7. Playback: reproduce PCM por altavoces

        Args:
            wav_path: Ruta al archivo WAV grabado.
        """
        generation = self._pipeline_generation  # capturar al inicio
        try:
            # 1. STT — Whisper local (primario) → Gemini (fallback)
            if self._pipeline_generation != generation:
                logger.info("Pipeline (gen=%d) cancelado antes de STT", generation)
                return

            text = None
            try:
                text = self._whisper_stt.transcribe(wav_path)
                logger.debug("STT Whisper OK: %s", text[:100])
            except Exception as e:
                logger.warning("Whisper STT falló (%s), intentando Gemini fallback", e)
                if self._stt is None:
                    logger.error("Gemini STT no configurado (GEMINI_API_KEY faltante)")
                    return
                text = self._stt.transcribe(wav_path)
                logger.debug("STT Gemini fallback OK: %s", text[:100])

            if not text:
                logger.error("STT retornó texto vacío")
                return

            # 2. Agente — streaming o síncrono según config
            if self._pipeline_generation != generation:
                logger.info("Pipeline (gen=%d) cancelado después de STT", generation)
                return
            if self._opencode is None:
                logger.error("OpenCode no configurado (OPENCODE_SERVER_PASSWORD faltante)")
                return

            if self._streaming_enabled and self._local_tts is not None and hasattr(self._local_tts, 'synthesize_sentence_stream'):
                # ── Flujo streaming ──
                try:
                    with self._send_lock:
                        if self._pipeline_generation != generation:
                            logger.info("Pipeline (gen=%d) cancelado mientras esperaba send_lock", generation)
                            return
                        delta_stream = self._opencode.send_command_stream(text)

                        # 3+4. Transición a SPEAKING antes del primer audio
                        with self._lock:
                            if self._pipeline_generation != generation:
                                return
                            self._state = self.STATE_SPEAKING
                            self._overlay.set_state("speaking")
                            logger.info("→ SPEAKING (gen=%d, streaming)", generation)

                        # 5. Pipeline streaming: deltas → oraciones → Kokoro → playback
                        sentence_buffer = SentenceBuffer()

                        def sentence_iterator():
                            for delta in delta_stream:
                                if self._pipeline_generation != generation:
                                    logger.info("Pipeline (gen=%d) cancelado durante streaming", generation)
                                    return
                                for sentence in sentence_buffer.add(delta):
                                    yield _strip_markdown(sentence)
                            # flush final
                            if self._pipeline_generation != generation:
                                return
                            for sentence in sentence_buffer.flush():
                                yield _strip_markdown(sentence)

                        pcm_stream = self._local_tts.synthesize_sentence_stream(sentence_iterator())
                        self._audio.play_audio_stream(pcm_stream)
                        logger.debug("Streaming pipeline completado (gen=%d)", generation)

                except Exception as e:
                    logger.warning("Streaming falló (%s: %s), fallback a síncrono", type(e).__name__, e)
                    # Fallback al flujo síncrono
                    self._run_sync_pipeline(text, generation)
            else:
                # ── Flujo síncrono (no streaming) ──
                self._run_sync_pipeline(text, generation)

        except Exception as e:
            logger.exception("Error en pipeline: %s", e)
        finally:
            # FIX-1 @security: check DENTRO del lock — evita race condition
            # (ventana entre check y adquisición del lock donde toggle() puede
            # incrementar generation y pisar el estado del nuevo flujo).
            with self._lock:
                if self._pipeline_generation == generation:
                    self._overlay.hide()
                    self._state = self.STATE_IDLE
                    logger.info("→ IDLE (gen=%d)", generation)
                else:
                    logger.info("Pipeline (gen=%d) cancelado — no se resetea el estado", generation)

    # ── Pipeline síncrono (fallback) ──────────────────────────────

    def _run_sync_pipeline(self, text: str, generation: int) -> None:
        """Ejecuta el pipeline síncrono (no streaming) como fallback.

        Flujo: send_command → parse_response → SPEAKING → synthesize → play_audio.
        Incluye todos los chequeos de _pipeline_generation y el send_lock.

        Args:
            text: Texto transcrito del usuario.
            generation: Número de generación para cancelación cooperativa.
        """
        # 2. Agente (cerebro) — con send_lock anti-concurrencia
        if self._pipeline_generation != generation:
            logger.info("Pipeline (gen=%d) cancelado después de STT", generation)
            return
        if self._opencode is None:
            logger.error("OpenCode no configurado (OPENCODE_SERVER_PASSWORD faltante)")
            return
        with self._send_lock:
            if self._pipeline_generation != generation:
                logger.info("Pipeline (gen=%d) cancelado mientras esperaba send_lock", generation)
                return
            response = self._opencode.send_command(text)
        logger.debug("Agente respondió: %s", response[:100])

        # 3. Parsear respuesta
        if self._pipeline_generation != generation:
            logger.info("Pipeline (gen=%d) cancelado después de send_command", generation)
            return
        style_hint, clean_text = parse_response(response)
        logger.debug("Parseado — style=%s, text=%s", style_hint, clean_text[:100])

        # 4. Transición a SPEAKING antes del TTS+playback
        if self._pipeline_generation != generation:
            logger.info("Pipeline (gen=%d) cancelado antes de TTS", generation)
            return
        with self._lock:
            if self._pipeline_generation != generation:
                return
            self._state = self.STATE_SPEAKING
            self._overlay.set_state("speaking")
            logger.info("→ SPEAKING (gen=%d)", generation)

        # 5 + 6. TTS — local (primario) → Gemini (fallback 1) → Azure streaming (fallback 2)
        pcm_bytes = None
        try:
            pcm_bytes = self._local_tts.synthesize(clean_text, style_hint)
            logger.debug("TTS local OK — %d bytes", len(pcm_bytes))
        except Exception as e:
            logger.warning("TTS local falló (%s), intentando Gemini fallback", e)
            try:
                if self._gemini_tts is None:
                    raise RuntimeError("Gemini TTS no configurado")
                if not self._gemini_tts.is_available():
                    raise RuntimeError("Gemini TTS circuit breaker abierto")
                pcm_bytes = self._gemini_tts.synthesize(clean_text, style_hint)
                logger.debug("TTS Gemini fallback OK — %d bytes", len(pcm_bytes))
            except Exception as e2:
                logger.warning("Gemini TTS falló (%s), intentando Azure streaming fallback", e2)
                if self._azure_tts is None:
                    logger.error("Azure TTS no configurado (AZURE_SPEECH_KEY faltante)")
                    return
                # Streaming Azure: reproducir en tiempo real (latencia baja al primer sample)
                self._audio.play_audio_stream(self._azure_tts.synthesize_stream(clean_text, style_hint))
                logger.debug("TTS Azure streaming OK")
                # pcm_bytes se queda en None → paso 7 se salta (ya reproducido)

        # 7. Playback (local o Gemini — no streaming)
        if pcm_bytes:
            self._audio.play_audio(pcm_bytes)
            logger.debug("Playback completado")

    # ── Loop principal ────────────────────────────────────────────

    def run(self) -> None:
        """Registra el hotkey global y bloquea el hilo principal.

        Escucha la tecla definida en settings['hotkey'] ("alt+v")
        y ejecuta self.toggle() en cada pulsación.
        """
        hotkey = self._settings["hotkey"]
        keyboard.add_hotkey(hotkey, self.toggle)
        logger.info("Jarvis escuchando... presioná %s", hotkey)
        keyboard.wait()


def setup_logging() -> None:
    """Configura logging global con RotatingFileHandler + consola opcional.

    Lee la configuración de config/settings.json sección "logging".
    Crea logs/ si no existe. Si hay TTY (consola interactiva para debug
    manual), también añade un StreamHandler a stderr. Si no hay TTY
    (pythonw.exe en producción), solo file handler.
    """
    settings_path = Path("config/settings.json")
    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)

    log_cfg = settings.get("logging", {})
    filename = log_cfg.get("filename", "logs/cortex.log")
    max_bytes = log_cfg.get("max_bytes", 5242880)
    backup_count = log_cfg.get("backup_count", 3)
    level_str = log_cfg.get("level", "DEBUG")
    level = getattr(logging, level_str.upper(), logging.DEBUG)

    # Crear directorio logs/ si no existe
    log_path = Path(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # File handler con rotación
    file_handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler solo si hay TTY (debug manual con python.exe)
    if sys.stderr is not None and sys.stderr.isatty():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


if __name__ == "__main__":
    setup_logging()
    assistant = VoiceAssistant()
    assistant.run()
