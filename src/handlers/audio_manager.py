"""Módulo de captura y reproducción de audio.

Implementa AudioManager usando sounddevice para I/O de micrófono y altavoces.
Grabación 24 kHz mono s16le a buffer RAM, vuelco a WAV.
"""

import logging
import os
import queue
import time as _time
import wave
from threading import Event, Lock
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioManager:
    """Gestiona la captura de micrófono y reproducción por altavoces.

    Attributes:
        settings: Dict con configuración de audio (sample_rate, channels, sample_width, recording_filename).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa AudioManager con la configuración de audio.

        Args:
            settings: Dict con claves 'sample_rate', 'channels', 'sample_width', 'recording_filename'
                      dentro de settings['audio'].
        """
        self._settings = settings
        self._lock = Lock()
        self._stream: Optional[sd.InputStream] = None
        self._frames: list[bytes] = []
        self._recording = False
        self._stop_playback_event = Event()  # señaliza interrupción al productor
        self._playback_stream: Optional[sd.OutputStream] = None  # ref al stream actual
        self._playback_lock = Lock()  # protege _playback_stream

        logger.debug(
            "AudioManager inicializado — sample_rate=%s, channels=%s",
            self._settings["audio"]["sample_rate"],
            self._settings["audio"]["channels"],
        )

    def start_recording(self) -> None:
        """Inicia captura de micrófono a buffer en RAM.

        Abre un InputStream de sounddevice con samplerate=24000, channels=1, dtype='int16'.
        Los frames se acumulan en una lista interna (self._frames) como objetos bytes.
        """
        sample_rate = self._settings["audio"]["sample_rate"]
        channels = self._settings["audio"]["channels"]

        def callback(indata, frames, time_info, status) -> None:
            """Callback del stream: acumula frames de audio en el buffer."""
            if status:
                logger.warning("Audio input status: %s", status)
            self._frames.append(indata.copy().tobytes())

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        self._recording = True
        logger.debug("Grabación iniciada — %s Hz, %s canal(es)", sample_rate, channels)

    def stop_recording(self) -> str:
        """Detiene captura, vuelca buffer a archivo WAV.

        Cierra el InputStream, escribe todos los frames acumulados en un archivo WAV
        con formato 24 kHz mono s16le usando el módulo wave de stdlib.

        Returns:
            Ruta absoluta al archivo .wav generado.
        """
        if not self._recording or self._stream is None:
            logger.warning("stop_recording llamado sin grabación activa")
            return ""

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False

        filename = self._settings["audio"]["recording_filename"]
        sample_rate = self._settings["audio"]["sample_rate"]
        channels = self._settings["audio"]["channels"]
        sample_width = self._settings["audio"]["sample_width"]

        # Concatenar todos los frames acumulados
        audio_data = b"".join(self._frames)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        abs_path = os.path.abspath(filename)
        logger.debug(
            "Grabación finalizada — %s (%d bytes, %s Hz, %d canales, %d bytes/sample)",
            abs_path,
            len(audio_data),
            sample_rate,
            channels,
            sample_width,
        )
        return abs_path

    def play_audio(self, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
        """Reproduce bytes PCM por altavoces.

        Convierte los bytes a un array numpy int16 y los reproduce usando
        sounddevice.play. Bloquea hasta que termina la reproducción.

        Args:
            pcm_bytes: Bytes PCM en formato s16le (raw, sin cabecera).
            sample_rate: Frecuencia de muestreo del audio (default: 24000).
        """
        self._stop_playback_event.clear()  # reset para este playback
        if not pcm_bytes:
            logger.warning("play_audio recibió 0 bytes, omitiendo reproducción")
            return

        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
        logger.debug(
            "Reproducción completada — %d samples, %s Hz",
            len(audio_array),
            sample_rate,
        )

    def play_audio_stream(self, pcm_stream: Iterator[bytes], sample_rate: int = 24000) -> None:
        """Reproduce un stream de bytes PCM en tiempo real (latencia baja).

        Patrón productor-consumidor con cola + buffer acumulador:
        - El hilo caller (productor) itera pcm_stream y alimenta una queue.Queue
          con q.put(timeout=2.0) — bloquea al productor hasta que hay espacio
          (va a la velocidad del consumidor = tiempo real de audio).
          Si el callback muere, levanta queue.Full tras 2s → cerrar stream.
        - Un sd.OutputStream (consumidor) con blocksize=1024 corre ~23 veces/seg.
          El callback mantiene un buffer interno np.array que ACUMULA samples de
          múltiples chunks. En cada invocación consume `frames` samples del frente.
          Underrun real (cola vacía + buffer insuficiente) → silencio.
        - Bloquea hasta que el stream se agota Y el playback termina (o timeout dinámico con safety net).

        Args:
            pcm_stream: Iterator que yields bytes PCM s16le (raw, sin cabecera).
            sample_rate: Frecuencia de muestreo (default 24000).
        """
        self._stop_playback_event.clear()  # reset para este playback
        q = queue.Queue(maxsize=256)  # absorbe burst inicial (~98 chunks en 1s)
        _SENTINEL = None
        _buffer = np.array([], dtype=np.int16)  # buffer acumulador de samples
        _done = False  # flag: se recibió sentinel

        def callback(outdata, frames, time_info, status):
            nonlocal _buffer, _done
            # Acumular chunks hasta tener >= frames samples (o cola vacía / done)
            while len(_buffer) < frames and not _done:
                try:
                    data = q.get_nowait()
                except queue.Empty:
                    break  # underrun — salir con lo que tengas
                if data is _SENTINEL:
                    _done = True
                    break
                arr = np.frombuffer(data, dtype=np.int16)
                _buffer = np.concatenate([_buffer, arr])
            # Consumir `frames` samples del frente del buffer
            if len(_buffer) >= frames:
                outdata[:] = _buffer[:frames].reshape(-1, 1)
                _buffer = _buffer[frames:]
            elif len(_buffer) > 0:
                # Buffer parcial (< frames) → rellenar resto con silencio
                n = len(_buffer)
                outdata[:n] = _buffer.reshape(-1, 1)
                outdata[n:] = 0
                _buffer = np.array([], dtype=np.int16)
            elif _done:
                raise sd.CallbackStop  # fin: buffer vacío + sentinel recibido
            else:
                outdata.fill(0)  # underrun total → silencio

        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=1024,  # ~23 callbacks/seg a 24kHz → consumo fluido de cola
            callback=callback,
        )
        # FIX-3 @security: iniciar ANTES de guardar ref — si start() falla, no dejar ref sucia
        try:
            stream.start()
        except Exception as exc:
            logger.error("No se pudo iniciar OutputStream: %s", exc)
            return
        with self._playback_lock:
            self._playback_stream = stream
        _total_samples_pushed = 0
        try:
            for chunk in pcm_stream:
                if self._stop_playback_event.is_set():  # NUEVO — chequear interrupción
                    logger.info("Playback stream interrumpido por stop_playback()")
                    break
                try:
                    q.put(chunk, timeout=2.0)  # bloquea hasta 2s esperando espacio
                    _total_samples_pushed += len(np.frombuffer(chunk, dtype=np.int16))
                except queue.Full:
                    logger.error("Audio stream: callback no consume — cerrando stream")
                    break  # el finally manda sentinel
        except Exception:
            logger.exception("Productor stream falló")
        finally:
            q.put(_SENTINEL)  # GARANTIZAR sentinel
        # Timeout dinámico anti-deadlock: el callback consume a tiempo
        # real de audio, no a velocidad de red. Estimar duración real del
        # playback y darle margen. Safety net superior configurable para
        # evitar espera infinita si el callback se clava.
        _estimated_duration = _total_samples_pushed / sample_rate
        _safety_net = self._settings.get("audio", {}).get("streaming_playback_safety_net_seconds", 600)
        _start_wait = _time.time()
        _deadline = _start_wait + _estimated_duration * 1.3 + 10.0
        _capped_deadline = min(_deadline, _start_wait + _safety_net)
        logger.debug(
            "Streaming playback: deadline dinámico %.1fs (estimado %.1fs + margen, safety_net %ss)",
            _capped_deadline - _start_wait, _estimated_duration, _safety_net,
        )
        while stream.active and _time.time() < _capped_deadline:
            _time.sleep(0.05)
        if stream.active:
            logger.warning(
                "Streaming playback timeout — forzando stop tras %.1fs",
                _time.time() - _start_wait,
            )
        with self._playback_lock:  # limpiar ref
            self._playback_stream = None
        # FIX-2 @security: try/except — stop_playback() puede haber cerrado el stream ya
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            logger.warning("Error cerrando stream en finally: %s", exc)
        logger.debug("Streaming playback completado — %s Hz", sample_rate)

    def stop_playback(self) -> None:
        """Detiene cualquier playback en curso (play_audio o play_audio_stream).

        Setea _stop_playback_event para que el productor de play_audio_stream
        deje de iterar el stream. Detiene el sd.OutputStream si hay uno activo.
        Llama sd.stop() para detener play_audio (no-streaming).

        No bloquea: las llamadas a stream.stop() y sd.stop() son no-bloqueantes.
        El productor y el callback terminan asíncronamente.
        """
        self._stop_playback_event.set()
        with self._playback_lock:
            if self._playback_stream is not None:
                try:
                    self._playback_stream.stop()
                    self._playback_stream.close()
                except Exception as exc:
                    logger.warning("Error deteniendo playback stream: %s", exc)
                self._playback_stream = None
            try:
                sd.stop()
            except Exception as exc:
                logger.warning("Error en sd.stop(): %s", exc)
