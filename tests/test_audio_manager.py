"""Tests unitarios para handlers/audio_manager.py.

Mockea `sounddevice` con `unittest.mock.patch` — no se accede al
hardware real de micrófono/altavoces.
Cubre el contrato de `AudioManager` definido en IMPLEMENTATION.md §4.3.
"""

import os
import threading
import time
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from handlers.audio_manager import AudioManager


@pytest.mark.unit
class TestAudioManager:
    """Suite de tests para AudioManager con sounddevice completamente mockeado."""

    @patch("handlers.audio_manager.sd")
    def test_start_recording_sets_state(self, mock_sd, mock_settings):
        """start_recording abre un InputStream con parámetros correctos y lo inicia."""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        am = AudioManager(mock_settings)
        am.start_recording()

        # Verifica que InputStream se construyó con los parámetros esperados
        mock_sd.InputStream.assert_called_once()
        kwargs = mock_sd.InputStream.call_args.kwargs
        assert kwargs["samplerate"] == 24000
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        assert callable(kwargs["callback"])

        # Verifica que se llamó .start() sobre el stream
        mock_stream.start.assert_called_once()

        # Estado interno coherente
        assert am._recording is True
        assert am._stream is mock_stream

    @patch("handlers.audio_manager.sd")
    def test_stop_recording_returns_wav_path(
        self, mock_sd, mock_settings, tmp_path, monkeypatch
    ):
        """stop_recording vuelca el buffer a WAV y retorna una ruta absoluta válida."""
        # Cambiar CWD a tmp_path para que 'comando.wav' se cree allí
        monkeypatch.chdir(tmp_path)

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        am = AudioManager(mock_settings)
        am.start_recording()

        # Inyectar 2 frames de 1 segundo cada uno (48000 bytes c/u)
        one_sec = np.zeros(24000, dtype=np.int16).tobytes()
        am._frames = [one_sec, one_sec]

        wav_path = am.stop_recording()

        # Path absoluto y archivo existente
        assert os.path.isabs(wav_path)
        assert os.path.exists(wav_path)
        assert wav_path.endswith("comando.wav")

        # Validar formato del WAV
        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

        # Estado interno limpio
        assert am._recording is False
        assert am._stream is None

    @patch("handlers.audio_manager.sd")
    def test_play_audio_calls_sounddevice(self, mock_sd, mock_settings):
        """play_audio llama a sd.play y sd.wait con los argumentos correctos."""
        am = AudioManager(mock_settings)
        pcm_bytes = np.zeros(24000, dtype=np.int16).tobytes()  # 1 segundo

        am.play_audio(pcm_bytes, sample_rate=24000)

        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

        # El primer arg posicional es el array numpy int16
        call_args = mock_sd.play.call_args
        audio_array = call_args.args[0]
        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.int16
        assert len(audio_array) == 24000
        # Samplerate puede estar en args[1] o kwargs['samplerate']
        samplerate = (
            call_args.kwargs.get("samplerate")
            or (call_args.args[1] if len(call_args.args) > 1 else None)
        )
        assert samplerate == 24000

    @patch("handlers.audio_manager.sd")
    def test_wav_format_correct(self, mock_sd, mock_settings, tmp_path, monkeypatch):
        """Vuelco de buffer fake → WAV con formato exacto 24kHz/1ch/16bit + samples correctos."""
        monkeypatch.chdir(tmp_path)

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        am = AudioManager(mock_settings)
        am.start_recording()

        # Buffer fake con samples no-cero para validar roundtrip
        fake_samples = np.arange(24000, dtype=np.int16)  # 0..23999
        am._frames = [fake_samples.tobytes()]

        wav_path = am.stop_recording()

        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 24000

            # Roundtrip: los samples leídos deben coincidir con los escritos
            read_frames = wf.readframes(wf.getnframes())
            read_array = np.frombuffer(read_frames, dtype=np.int16)
            np.testing.assert_array_equal(read_array, fake_samples)

class FakeOutputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.blocksize = kwargs.get("blocksize", 1024)
        self._active = True
    def start(self):
        pass
    @property
    def active(self):
        return self._active
    def stop(self):
        self._active = False
    def close(self):
        pass
    def invoke(self, frames):
        outdata = np.zeros((frames, 1), dtype=np.int16)
        try:
            self.callback(outdata, frames, None, None)
            return False, outdata
        except sd.CallbackStop:
            self._active = False
            return True, outdata


@pytest.mark.unit
class TestPlayAudioStream:
    @patch("handlers.audio_manager._time.sleep")
    @patch("handlers.audio_manager._time.time", side_effect=lambda: 1000.0)
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_accumulates_small_chunks_no_silence(self, mock_time, mock_sleep, mock_settings):
        am = AudioManager(mock_settings)
        chunks = [np.arange(i*1024, (i+1)*1024, dtype=np.int16).tobytes() for i in range(4)]
        pcm_stream = iter(chunks)
        captured = {}
        original_init = FakeOutputStream.__init__
        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self
        with patch.object(FakeOutputStream, "__init__", spy_init):
            t = threading.Thread(target=am.play_audio_stream, args=(pcm_stream,), daemon=True)
            t.start()
            time.sleep(0.3)
            fake = captured["stream"]
            stopped1, outdata1 = fake.invoke(2048)
            assert not stopped1
            np.testing.assert_array_equal(outdata1.flatten()[:2048], np.arange(2048, dtype=np.int16))
            stopped2, outdata2 = fake.invoke(2048)
            np.testing.assert_array_equal(outdata2.flatten()[:2048], np.arange(2048, 4096, dtype=np.int16))
            t.join(timeout=2)

    @patch("handlers.audio_manager._time.sleep")
    @patch("handlers.audio_manager._time.time", side_effect=lambda: 1000.0)
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_consumes_frames_from_front_in_order(self, mock_time, mock_sleep, mock_settings):
        am = AudioManager(mock_settings)
        chunk = np.arange(4096, dtype=np.int16).tobytes()
        pcm_stream = iter([chunk])
        captured = {}
        original_init = FakeOutputStream.__init__
        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self
        with patch.object(FakeOutputStream, "__init__", spy_init):
            t = threading.Thread(target=am.play_audio_stream, args=(pcm_stream,), daemon=True)
            t.start()
            time.sleep(0.3)
            fake = captured["stream"]
            for i in range(4):
                stopped, outdata = fake.invoke(1024)
                expected = np.arange(i*1024, (i+1)*1024, dtype=np.int16)
                np.testing.assert_array_equal(outdata.flatten(), expected)
            t.join(timeout=2)

    @patch("handlers.audio_manager._time.sleep")
    @patch("handlers.audio_manager._time.time", side_effect=lambda: 1000.0)
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_stops_on_sentinel_when_buffer_empty(self, mock_time, mock_sleep, mock_settings):
        am = AudioManager(mock_settings)
        chunk = np.arange(512, dtype=np.int16).tobytes()
        pcm_stream = iter([chunk])
        captured = {}
        original_init = FakeOutputStream.__init__
        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self
        with patch.object(FakeOutputStream, "__init__", spy_init):
            t = threading.Thread(target=am.play_audio_stream, args=(pcm_stream,), daemon=True)
            t.start()
            time.sleep(0.3)
            fake = captured["stream"]
            stopped1, outdata1 = fake.invoke(1024)
            assert not stopped1
            np.testing.assert_array_equal(outdata1.flatten()[:512], np.arange(512, dtype=np.int16))
            assert np.all(outdata1.flatten()[512:] == 0)
            stopped2, outdata2 = fake.invoke(1024)
            assert stopped2 is True
            t.join(timeout=2)

    @patch("handlers.audio_manager._time.sleep")
    @patch("handlers.audio_manager._time.time", side_effect=lambda: 1000.0)
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_underrun_produces_silence(self, mock_time, mock_sleep, mock_settings):
        """Underrun real (cola vacía + buffer insuficiente) → silencio.

        Nota: _time es alias al módulo time global, así que mockear
        _time.sleep contamina time.sleep. Usamos threading.Event().wait()
        para el delay real del productor, que no se ve afectado por el mock.
        El _time.time se mockea con side_effect que retorna un float real
        (1000.0) para evitar TypeError en la comparación
        `_time.time() < _deadline` de audio_manager.py:214.
        """
        am = AudioManager(mock_settings)
        def slow_stream():
            # threading.Event().wait() NO usa time.sleep → no se mockea
            threading.Event().wait(0.5)
            yield np.arange(1024, dtype=np.int16).tobytes()
        pcm_stream = slow_stream()
        captured = {}
        original_init = FakeOutputStream.__init__
        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self
        with patch.object(FakeOutputStream, "__init__", spy_init):
            t = threading.Thread(target=am.play_audio_stream, args=(pcm_stream,), daemon=True)
            t.start()
            threading.Event().wait(0.1)  # delay real, no mockeado
            fake = captured["stream"]
            stopped, outdata = fake.invoke(1024)
            assert not stopped
            assert np.all(outdata == 0)  # silencio total por underrun
            t.join(timeout=3)


@pytest.mark.unit
class TestStopPlayback:
    """Tests para stop_playback() y la integración con play_audio/play_audio_stream (Fase 12.A).

    Cubre:
    - stop_playback() llama sd.stop() para play_audio (no-streaming).
    - stop_playback() cierra el _playback_stream activo de play_audio_stream.
    - stop_playback() es no-op seguro cuando no hay stream activo.
    - play_audio() limpia el _stop_playback_event al inicio.
    - play_audio_stream() rompe el loop productor si el evento está seteado.
    """

    @patch("handlers.audio_manager.sd")
    def test_stop_playback_stops_sd_play(self, mock_sd, mock_settings):
        """stop_playback() llama sd.stop() para detener play_audio (no-streaming)."""
        am = AudioManager(mock_settings)

        am.stop_playback()

        # sd.stop() fue invocado una vez (para play_audio en curso)
        mock_sd.stop.assert_called_once()
        # El evento fue seteado
        assert am._stop_playback_event.is_set()

    @patch("handlers.audio_manager.sd")
    def test_stop_playback_stops_stream(self, mock_sd, mock_settings):
        """stop_playback() cierra y limpia el _playback_stream activo."""
        am = AudioManager(mock_settings)
        mock_stream = MagicMock(name="PlaybackStream")
        am._playback_stream = mock_stream

        am.stop_playback()

        # El stream activo fue detenido y cerrado
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        # La referencia fue limpiada
        assert am._playback_stream is None
        # sd.stop() también fue llamado (para play_audio)
        mock_sd.stop.assert_called_once()

    @patch("handlers.audio_manager.sd")
    def test_stop_playback_no_op_when_idle(self, mock_sd, mock_settings):
        """stop_playback() sin playback activo: no levanta excepción y llama sd.stop()."""
        am = AudioManager(mock_settings)
        # Estado idle: sin stream activo
        am._playback_stream = None

        # No debe levantar excepción
        am.stop_playback()

        # sd.stop() se llama de todas formas (envuelto en try/except)
        mock_sd.stop.assert_called_once()
        # El evento fue seteado (para notificar a productores)
        assert am._stop_playback_event.is_set()

    @patch("handlers.audio_manager.sd")
    def test_play_audio_clears_stop_event(self, mock_sd, mock_settings):
        """play_audio() limpia _stop_playback_event al inicio.

        Garantiza que un stop_playback() previo no cancele el próximo playback.
        """
        am = AudioManager(mock_settings)
        # Simular que hubo una interrupción previa
        am._stop_playback_event.set()

        am.play_audio(b"\x00\x00" * 100)

        # El clear() al inicio reseteó el evento
        assert not am._stop_playback_event.is_set()
        # sd.play() y sd.wait() fueron invocados normalmente
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    @patch("handlers.audio_manager.sd")
    def test_play_audio_stream_checks_stop_event(self, mock_sd, mock_settings):
        """play_audio_stream() rompe el loop productor si el evento está seteado.

        El productor chequea self._stop_playback_event.is_set() en cada iteración
        y hace break inmediatamente, no consumiendo todos los chunks del stream.

        Implementación: el side_effect de OutputStream.start() setea el evento
        después del clear() inicial, asegurando que la primera iteración del
        loop productor vea el evento seteado y rompa.
        """
        am = AudioManager(mock_settings)

        # Mockear OutputStream: active=False para que el polling loop salga
        # inmediatamente, y start() con side_effect para setear el evento
        # después del clear() inicial de play_audio_stream().
        mock_stream_instance = MagicMock(name="OutputStreamInstance")
        mock_stream_instance.active = False

        def start_and_interrupt():
            # Setea el evento DESPUÉS del clear() inicial del método,
            # simulando una llamada a stop_playback() durante el setup.
            am._stop_playback_event.set()

        mock_stream_instance.start.side_effect = start_and_interrupt
        mock_sd.OutputStream.return_value = mock_stream_instance

        # Generador que cuenta cuántos chunks se llegan a consumir/iterar
        chunks_yielded = []

        def chunk_gen():
            for i in range(10):
                chunks_yielded.append(i)
                yield np.zeros(1024, dtype=np.int16).tobytes()

        am.play_audio_stream(chunk_gen())

        # El productor rompió antes de consumir todos los chunks
        assert len(chunks_yielded) < 10, (
            f"Productor consumió todos los chunks ({len(chunks_yielded)}), "
            f"se esperaba que rompa por _stop_playback_event"
        )
        # El evento sigue seteado (no fue limpiado por stop_playback aquí)
        assert am._stop_playback_event.is_set()

