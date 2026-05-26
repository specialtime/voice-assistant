from __future__ import annotations

import tempfile
import threading
import uuid
import wave
from pathlib import Path


class AudioRecorder:
    STOP_THREAD_TIMEOUT_SECONDS = 3

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        temp_dir: Path | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self._frames: list[bytes] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._pyaudio = None
        self._format = None

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            self._frames.append(self._stream.read(self.chunk_size, exception_on_overflow=False))

    def start_recording(self) -> None:
        if self.is_recording:
            return

        try:
            import pyaudio  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("pyaudio is not installed") from exc

        self._frames = []
        self._stop_event.clear()
        self._pyaudio = pyaudio.PyAudio()
        self._format = pyaudio.paInt16
        self._stream = self._pyaudio.open(
            format=self._format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self.is_recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop_recording(self) -> Path:
        if not self.is_recording:
            raise RuntimeError("No active recording is in progress")

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.STOP_THREAD_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError("Audio recording thread did not stop in time")

        self._stream.stop_stream()
        self._stream.close()

        sample_size = self._pyaudio.get_sample_size(self._format)
        self._pyaudio.terminate()

        self.is_recording = False
        output_file = self.temp_dir / f"voice_input_{uuid.uuid4().hex}.wav"
        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(sample_size)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(self._frames))

        return output_file
