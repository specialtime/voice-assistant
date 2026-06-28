"""Tests unitarios para el fix de streaming playback timeout dinámico (FIX-3 stream).

Cubre la lógica introducida en `src/handlers/audio_manager.py` `play_audio_stream`
donde el deadline fijo de 30s fue reemplazado por:
    _deadline = _start_wait + estimated_duration * 1.3 + 10.0
con un safety net superior configurable
(`settings["audio"]["streaming_playback_safety_net_seconds"]`, default 600s).

Mockea completamente `sounddevice` vía un ``FakeOutputStream`` que respeta el
contrato del callback real (``sd.CallbackStop`` cuando se termina el audio) y
mockea ``_time`` para simular avance temporal sin demoras reales.

Importante: ``play_audio_stream`` se ejecuta **sincrónicamente** en estos tests
(no en un hilo aparte). El callback se invoca desde **otro hilo del test**
(misma idea que PortAudio: callback asíncrono). El productor y el polling loop
del método viven en el hilo principal del test; los callbacks viven en un
worker thread que consume ``queue`` y levanta ``CallbackStop`` cuando termina.
"""

import logging
import threading
import time as _real_time
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from handlers.audio_manager import AudioManager


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


class FakeOutputStream:
    """OutputStream fake que respeta el contrato del callback real.

    El callback puede levantar ``sd.CallbackStop`` cuando termina el audio; en
    ese caso ``active`` se setea en ``False`` (mockea ``stream.active`` como
    property). El método ``tick(callback_calls=...)`` permite al test forzar
    invocaciones del callback N veces. Útil para sincronizar productor y
    consumidor sin esperas reales.

    Atributos:
        callback: el callable pasado por play_audio_stream
        blocksize: blocksize simulado (default 1024 — alinea con la realidad)
        _active: True hasta que el callback levante ``sd.CallbackStop``
        callback_stops_raised: contador de ``sd.CallbackStop`` levantados
        ever_callback_stop: si True, ``active`` se queda en False para siempre
            luego del primer CallbackStop (modo producción).
    """

    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.blocksize = kwargs.get("blocksize", 1024)
        self._active = True
        self.callback_stops_raised = 0
        self.ever_callback_stop = kwargs.pop("ever_callback_stop", True)

    def start(self):
        pass

    @property
    def active(self):
        return self._active

    def stop(self):
        self._active = False

    def close(self):
        pass

    def tick(self, callback_calls: int = 1, frames: int = None):
        """Invoca el callback ``callback_calls`` veces. Devuelve lista de tuplas (stopped, outdata).

        ``frames`` por defecto se calcula como ``blocksize`` del stream.
        """
        if frames is None:
            frames = self.blocksize
        results = []
        for _ in range(callback_calls):
            outdata = np.zeros((frames, 1), dtype=np.int16)
            stopped = False
            try:
                self.callback(outdata, frames, None, None)
            except sd.CallbackStop:
                self._active = False
                stopped = True
                self.callback_stops_raised += 1
            results.append((stopped, outdata))
            if stopped and self.ever_callback_stop:
                break
        return results


def _make_pcm_stream(
    total_seconds: float,
    sample_rate: int = 24000,
    chunk_samples: int = 2048,
) -> Iterator[bytes]:
    """Genera un PCM stream sintético de ``total_seconds`` segundos.

    Cada chunk es un array int16 con patrón sinusoidal (valores en rango [-16000, 16000])
    para evitar overflow de ``np.arange`` cuando el stream es largo
    (60s × 24000Hz = 1,440,000 samples > rango int16). La cantidad total de
    bytes coincide con el cálculo del spec:
        total_bytes = total_seconds * sample_rate * 2
    """
    total_samples = int(total_seconds * sample_rate)
    chunk_bytes = chunk_samples * 2  # int16 = 2 bytes
    # Generamos una sinusoide de baja frecuencia (1Hz) × amplitud 16000.
    # sample_rate_total_samples_phases = sample_rate / 1.0
    # No usamos np.arange(sample_idx, ...) porque desborda para >32k samples.
    # El primer sample del chunk es ``start_idx``, avanzamos en pasos de a 1.
    emitted = 0
    start_idx = 0
    while emitted < total_samples:
        n = min(chunk_samples, total_samples - emitted)
        # Genera n samples: senoidal sobre [start_idx, start_idx+n)
        phase = (np.arange(n, dtype=np.float64) + start_idx) / sample_rate
        chunk = (np.sin(2 * np.pi * phase) * 16000).astype(np.int16).tobytes()
        assert len(chunk) == n * 2, f"chunk size mismatch: {len(chunk)} != {n*2}"
        start_idx += n
        emitted += n
        yield chunk


def _make_settings_dict(safety_net_seconds=None, sample_rate=24000):
    """Settings sintéticos con override opcional del safety net."""
    s = {
        "gemini": {
            "stt_model_primary": "gemini-3.1-flash-lite",
            "stt_model_fallback": "gemini-2.5-flash-lite",
            "tts_model": "gemini-3.1-flash-tts-preview",
            "tts_voice": "Charon",
            "tts_circuit_breaker_cooldown_seconds": 1800,
            "stt_prompt": "Transcribe el siguiente audio al español rioplatense.",
        },
        "opencode": {
            "agent": "asistente_voz",
            "model_fallback": "opencode/big-pickle",
            "timeout_ms": 120000,
            "max_session_messages": 10,
        },
        "azure": {
            "voice": "es-AR-TomasNeural",
            "locale": "es-AR",
            "output_format": "audio-24khz-48kbitrate-mono-mp3",
        },
        "audio": {
            "sample_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
            "recording_filename": "comando.wav",
        },
        "hotkey": "alt+v",
        "logging": {
            "filename": "logs/cortex.log",
            "max_bytes": 5242880,
            "backup_count": 3,
            "level": "INFO",
        },
    }
    if safety_net_seconds is not None:
        s["audio"]["streaming_playback_safety_net_seconds"] = safety_net_seconds
    return s


class _ControllableTime:
    """Mock de ``_time`` que avanza en respuesta a ``time.sleep``.

    ``time()`` devuelve un flotante creciente controlado por ``advance(seconds)``
    o implícitamente por cada ``sleep(s)`` (que adelanta el reloj en ``s``
    segundos). Esto evita esperas reales Y permite que el polling loop
    ``while stream.active and _time.time() < _deadline`` termine por su cuenta
    cuando el deadline vence, sin necesidad de que el test haga ``advance``
    en un loop.

    CRÍTICO para evitar carrera por GIL: ``sleep`` también cede el control al
    sistema operativo durante ``min(seconds, 0.001)`` segundos reales (vía
    ``Event().wait``), dándole tiempo al callback worker thread para invocar
    callbacks. Sin esto, el polling loop (que es puramente Python con body
    chico) puede monopolizar la GIL durante miles de iteraciones sin soltar,
    evitando que el callback worker processe el sentinel antes de que el
    deadline venza.

    Con esto el test puede validar que el warning dice "forzando stop tras X.Xs"
    sin esperar X.Xs reales.
    """

    def __init__(self, start: float = 0.0):
        self._now = [start]

    def time(self) -> float:
        return self._now[0]

    def sleep(self, seconds: float) -> None:
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            s = 0.0
        self._now[0] += s
        # Ceder la GIL al sistema por una cantidad mínima de tiempo real,
        # para que el callback worker thread tenga oportunidad de correr.
        # 1ms es suficiente: el callback tick es ~0.1ms.
        _real_time.sleep(0.001)

    def advance(self, seconds: float) -> None:
        self._now[0] += seconds


def _extract_deadline_log(caplog) -> _ControllableTime | None:
    """Parsea el log debug de audio_manager que contiene el deadline dinámico.

    El log tiene el formato:
        'Streaming playback: deadline dinámico %.1fs (estimado %.1fs + margen, safety_net %ss)'
    Devuelve el número de segundos o None si no aparece.
    """
    for record in caplog.records:
        msg = record.getMessage()
        if "deadline dinámico" in msg:
            # Parseo simple: el primer %.1fs es lo que queremos
            # Formato real: "Streaming playback: deadline dinámico X.Xs (estimado Y.Ys + margen, safety_net Zs)"
            try:
                prefix = "deadline dinámico "
                i = msg.index(prefix) + len(prefix)
                end = msg.index("s", i)
                return float(msg[i:end])
            except (ValueError, IndexError):
                continue
    return None


def _wait_thread_drain(thread: threading.Thread, timeout: float = 5.0):
    """Joinea un thread con timeout real (no mockeado) y assert no quedó vivo."""
    thread.join(timeout=timeout)
    assert not thread.is_alive(), (
        f"Thread did not finish in {timeout}s — el productor/poll loop está colgado"
    )


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStreamingPlaybackDynamicTimeout:
    """Suite del fix de deadline dinámico + safety net para `play_audio_stream`.

    Patrón: ``play_audio_stream`` se llama SINCRÓNICAMENTE desde el test
    (thread principal). Un **callback worker thread** invoca el callback
    del FakeOutputStream de forma continua (simula PortAudio). Esto evita
    race conditions donde el polling loop «gane» al callback real.
    """

    # ───────────────────────────────────────────────────────────────────────
    # Test 1 — audio largo (~60s) no se corta
    # ───────────────────────────────────────────────────────────────────────
    @patch("handlers.audio_manager._time")
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_streaming_playback_completes_without_timeout_for_long_audio(
        self, mock_time_mod, caplog
    ):
        """Stream de ~60s: el deadline dinámico debe superar 30s y el callback
        termina antes de que se agote. NO se debe loguear el warning de timeout.
        """
        # Arrange — reloj arranca en 0
        t = _ControllableTime(start=0.0)
        mock_time_mod.time.side_effect = t.time
        mock_time_mod.sleep.side_effect = t.sleep

        settings = _make_settings_dict(safety_net_seconds=600)
        am = AudioManager(settings)

        # Stream de 60 segundos exactos
        pcm_stream = _make_pcm_stream(total_seconds=60.0, sample_rate=24000)

        captured: dict = {}
        original_init = FakeOutputStream.__init__

        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self

        expected_deadline_seconds = 60.0 * 1.3 + 10.0  # = 88.0s

        with patch.object(FakeOutputStream, "__init__", spy_init):
            with caplog.at_level(logging.DEBUG, logger="handlers.audio_manager"):
                # Arrancamos el callback worker en otro thread para que
                # consuma la cola a buen ritmo. Stream termina → CallbackStop →
                # stream.active=False → worker sale solo.
                # Llamamos a play_audio_stream en el hilo principal. Necesita
                # stream ya creado, pero el constructor guarda la ref ANTES
                # del push loop. Truco: el callback worker requiere ``captured``
                # ya populado. Hacemos un primer run "discard" para obtener
                # el FakeOutputStream del constructor. Mejor: que el worker
                # arranque con un predicado sobre ``captured``.
                def lazy_worker():
                    # Espera a que el stream sea capturado.
                    while "stream" not in captured:
                        _real_time.sleep(0.001)
                    fake = captured["stream"]
                    while fake.active:
                        fake.tick(8, frames=1024)
                        # Sin sleep: corremos tan rápido como podamos.

                worker = threading.Thread(target=lazy_worker, daemon=True)
                worker.start()

                # Run sincrónico — bloquea hasta que stream termine o timeout.
                am.play_audio_stream(pcm_stream, sample_rate=24000)

                # Avisamos al worker que pare (por si está en loop sobre active
                # cuando ya no hay nada que consumir — improbable).
                worker.join(timeout=2.0)

        # Assert 1: NO hubo warning de timeout
        timeout_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Streaming playback timeout" in r.getMessage()
        ]
        assert timeout_warnings == [], (
            f"Timeout warning logueado para audio de 60s: {[r.getMessage() for r in timeout_warnings]}"
        )

        # Assert 2: el log de deadline dinámico se emitió y fue > 30s
        deadline_seconds = _extract_deadline_log(caplog)
        assert deadline_seconds is not None, (
            "No se logueó el debug 'deadline dinámico'. Logs:\n"
            + "\n".join(r.getMessage() for r in caplog.records)
        )
        assert deadline_seconds > 30.0, (
            f"Deadline dinámico ({deadline_seconds:.1f}s) no superó el viejo "
            f"umbral fijo de 30s — el fix no se aplicó correctamente"
        )

        # Assert 3: el valor calculado coincide con la fórmula del spec
        assert abs(deadline_seconds - expected_deadline_seconds) < 1.0, (
            f"Deadline ({deadline_seconds:.1f}s) ≠ 60*1.3+10 = {expected_deadline_seconds:.1f}s"
        )

    # ───────────────────────────────────────────────────────────────────────
    # Test 2 — audio corto sin regresión
    # ───────────────────────────────────────────────────────────────────────
    @patch("handlers.audio_manager._time")
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_streaming_playback_short_audio_no_regression(self, mock_time_mod, caplog):
        """Stream de ~2s completa sin timeout y sin warning.
        Verifica que la nueva lógica no rompió el path de audios cortos.
        """
        t = _ControllableTime(start=100.0)
        mock_time_mod.time.side_effect = t.time
        mock_time_mod.sleep.side_effect = t.sleep

        settings = _make_settings_dict(safety_net_seconds=600)
        am = AudioManager(settings)

        pcm_stream = _make_pcm_stream(total_seconds=2.0, sample_rate=24000)

        captured: dict = {}
        original_init = FakeOutputStream.__init__

        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self

        with patch.object(FakeOutputStream, "__init__", spy_init):
            with caplog.at_level(logging.DEBUG, logger="handlers.audio_manager"):
                def lazy_worker():
                    while "stream" not in captured:
                        _real_time.sleep(0.001)
                    fake = captured["stream"]
                    while fake.active:
                        fake.tick(4, frames=1024)

                worker = threading.Thread(target=lazy_worker, daemon=True)
                worker.start()

                am.play_audio_stream(pcm_stream, sample_rate=24000)
                worker.join(timeout=2.0)

        # Assert — no hubo timeout
        timeout_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Streaming playback timeout" in r.getMessage()
        ]
        assert timeout_warnings == [], (
            f"Timeout warning para audio corto (regresión): {[r.getMessage() for r in timeout_warnings]}"
        )

        # El deadline dinámico para 2s debe ser: 2 * 1.3 + 10 = 12.6s
        deadline_seconds = _extract_deadline_log(caplog)
        assert deadline_seconds is not None
        assert 12.0 < deadline_seconds < 14.0, (
            f"Deadline ({deadline_seconds:.1f}s) ≠ 2*1.3+10 = 12.6s (esperado ~12.6s)"
        )

    # ───────────────────────────────────────────────────────────────────────
    # Test 3 — safety net capa el deadline cuando el callback se clava
    # ───────────────────────────────────────────────────────────────────────
    @patch("handlers.audio_manager._time")
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_streaming_playback_safety_net_caps_deadline(self, mock_time_mod, caplog):
        """safety_net_seconds=5 + callback que NUNCA levanta CallbackStop →
        warning de timeout con tiempo ~5s (no 30s, no infinito).
        """
        # Arrange — el reloj avanza implícitamente vía _time.sleep(0.05)
        # en el polling loop. El deadline es safety_net=5s → ~100 iteraciones.
        t = _ControllableTime(start=0.0)
        mock_time_mod.time.side_effect = t.time
        mock_time_mod.sleep.side_effect = t.sleep

        settings = _make_settings_dict(safety_net_seconds=5)
        am = AudioManager(settings)

        pcm_stream = _make_pcm_stream(total_seconds=2.0, sample_rate=24000)

        captured: dict = {}
        original_init = FakeOutputStream.__init__

        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self

        with patch.object(FakeOutputStream, "__init__", spy_init):
            # NO hay callback worker — simulamos callback clavado.
            # El polling loop vivirá hasta que el deadline (safety_net=5s)
            # expire. El sleep mockeado avanza el reloj 0.05s por iteración.
            with caplog.at_level(logging.DEBUG, logger="handlers.audio_manager"):
                am.play_audio_stream(pcm_stream, sample_rate=24000)
                # Retorna cuando el polling loop sale por timeout.

        # Assert — hubo warning de timeout
        timeout_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Streaming playback timeout" in r.getMessage()
        ]
        assert len(timeout_warnings) >= 1, (
            f"No se logueó 'Streaming playback timeout' con callback clavado. Logs:\n"
            + "\n".join(f"[{r.levelname}] {r.getMessage()}" for r in caplog.records)
        )

        # Parseamos el tiempo del warning: "forzando stop tras %.1fs"
        msg = timeout_warnings[-1].getMessage()
        assert "forzando stop tras" in msg
        try:
            prefix = "forzando stop tras "
            i = msg.index(prefix) + len(prefix)
            end = msg.index("s", i)
            elapsed_seconds = float(msg[i:end])
        except (ValueError, IndexError):
            pytest.fail(f"No se pudo parsear 'forzando stop tras' del warning: {msg!r}")

        # Assert — el elapsed es ~5s (safety_net) y NO es 30s ni infinito
        assert 4.0 < elapsed_seconds < 8.0, (
            f"Tiempo hasta timeout ({elapsed_seconds:.2f}s) no está cerca del "
            f"safety_net (5s). ¿El safety net no se está aplicando?"
        )
        assert elapsed_seconds < 25.0, (
            f"Tiempo hasta timeout ({elapsed_seconds:.2f}s) parece el viejo "
            f"deadline fijo de 30s — el fix no se aplicó"
        )

        # Además, el deadline dinámico logueado debe haber sido CAPADO al safety net
        deadline_seconds = _extract_deadline_log(caplog)
        assert deadline_seconds is not None
        # safety_net=5, deadline dinâmico = 2*1.3+10 = 12.6s → capado a 5s
        assert deadline_seconds <= 6.0, (
            f"Deadline reportado ({deadline_seconds:.1f}s) no fue capeado por el "
            f"safety_net de 5s — el cap no funciona"
        )

    # ───────────────────────────────────────────────────────────────────────
    # Test 4 — stop_playback_event interrumpe el productor
    # ───────────────────────────────────────────────────────────────────────
    @patch("handlers.audio_manager.sd")
    def test_streaming_playback_stop_event_interrupts(self, mock_sd, mock_settings):
        """Setear _stop_playback_event durante el loop productor: el productor
        rompe antes de consumir todos los chunks del stream.
        """
        am = AudioManager(mock_settings)

        # OutputStream mockeado: active=False para que el polling salga
        # inmediatamente. Lo importante es verificar que el productor
        # respeta el stop event.
        mock_stream_instance = MagicMock(name="OutputStreamInstance")
        mock_stream_instance.active = False

        chunks_yielded: List[int] = []

        def interrupt_after_first_chunk():
            # Seteamos el evento DESPUÉS del clear() inicial del método y
            # DESPUÉS de que el primer chunk sea pusheado. Esto simula
            # un stop_playback() durante el push loop.
            am._stop_playback_event.set()

        mock_stream_instance.start.side_effect = interrupt_after_first_chunk
        mock_sd.OutputStream.return_value = mock_stream_instance

        def chunk_gen():
            for i in range(20):
                chunks_yielded.append(i)
                yield np.zeros(1024, dtype=np.int16).tobytes()

        am.play_audio_stream(chunk_gen())

        # Assert — productor cortó antes de consumir todos los chunks
        assert len(chunks_yielded) < 20, (
            f"Productor consumió todos los chunks ({len(chunks_yielded)}/20), "
            f"se esperaba que rompiera por _stop_playback_event"
        )
        assert am._stop_playback_event.is_set()

    # ───────────────────────────────────────────────────────────────────────
    # Test 5 — stream vacío no crashea (deadline debe ser 10s, no negativo)
    # ───────────────────────────────────────────────────────────────────────
    @patch("handlers.audio_manager._time")
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_streaming_playback_zero_samples(self, mock_time_mod, caplog):
        """Stream vacío (0 chunks). El productor no pushea nada, el deadline
        dinámico es 0 * 1.3 + 10 = 10s. No debe crashear.
        """
        t = _ControllableTime(start=42.0)
        mock_time_mod.time.side_effect = t.time
        mock_time_mod.sleep.side_effect = t.sleep

        settings = _make_settings_dict(safety_net_seconds=600)
        am = AudioManager(settings)

        # Generador que no yield NADA
        def empty_stream() -> Iterator[bytes]:
            return
            yield  # noqa: unreachable — sólo para que sea generador

        captured: dict = {}
        original_init = FakeOutputStream.__init__

        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self

        with patch.object(FakeOutputStream, "__init__", spy_init):
            with caplog.at_level(logging.DEBUG, logger="handlers.audio_manager"):
                # Necesitamos un callback worker que invoque el callback del
                # FakeOutputStream. Sin él, el stream queda active=True y el
                # polling loop itera hasta que el deadline venza (safety_net=600s
                # → 600/0.05 = 12000 iteraciones, lentísimo). Mejor: arrancar
                # un worker que termina el stream rápido (no hay datos → el
                # callback levantará CallbackStop).
                def lazy_worker():
                    while "stream" not in captured:
                        _real_time.sleep(0.001)
                    fake = captured["stream"]
                    while fake.active:
                        fake.tick(4, frames=1024)

                worker = threading.Thread(target=lazy_worker, daemon=True)
                worker.start()

                # No debe crashear con stream vacío
                am.play_audio_stream(empty_stream(), sample_rate=24000)
                worker.join(timeout=2.0)

        # Assert — el deadline dinámico fue exactamente el piso de 10s
        deadline_seconds = _extract_deadline_log(caplog)
        assert deadline_seconds is not None, (
            f"No se logueó deadline dinámico. Logs:\n"
            + "\n".join(r.getMessage() for r in caplog.records)
        )
        # expected: 0 * 1.3 + 10 = 10s
        assert 9.5 < deadline_seconds < 10.5, (
            f"Deadline dinámico para stream vacío = {deadline_seconds:.1f}s, "
            f"esperado ~10s (piso mínimo)"
        )
        # No debe haber warning de timeout (safety_net=600 es mucho mayor a 10s)
        timeout_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Streaming playback timeout" in r.getMessage()
        ]
        assert timeout_warnings == [], (
            f"Timeout warning emitido para stream vacío: {[r.getMessage() for r in timeout_warnings]}"
        )


# ──────────────────────────────────────────────────────────────────
# Test bonus — secrets no se loguean en el path de streaming
# ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStreamingPlaybackSecretsNotLogged:
    """Regresión de secrets: confirma que play_audio_stream NO loguea API keys.

    El path de streaming playback no toca credenciales, pero verificamos que
    un settings con un secret visible no termina en logs. Patrón establecido
    por el repo (test_kokoro_tts_client::test_no_secrets_logged).
    """

    SENTINEL_API_KEY = "SECRET_STREAM_TEST_DO_NOT_LEAK_888"

    @patch("handlers.audio_manager._time")
    @patch("handlers.audio_manager.sd.OutputStream", FakeOutputStream)
    def test_streaming_playback_no_api_key_in_logs(self, mock_time_mod, caplog):
        # Arrange — settings con API keys sentinel
        t = _ControllableTime(start=0.0)
        mock_time_mod.time.side_effect = t.time
        mock_time_mod.sleep.side_effect = t.sleep

        settings = _make_settings_dict(safety_net_seconds=600)
        # Inyectar sentinels en todas las secciones que tienen API keys
        settings["gemini"]["_test_sentinel"] = self.SENTINEL_API_KEY
        settings["azure"]["_test_sentinel"] = self.SENTINEL_API_KEY

        am = AudioManager(settings)

        captured: dict = {}
        original_init = FakeOutputStream.__init__

        def spy_init(self, **kwargs):
            original_init(self, **kwargs)
            captured["stream"] = self

        with patch.object(FakeOutputStream, "__init__", spy_init):
            with caplog.at_level(logging.DEBUG, logger="handlers.audio_manager"):
                def lazy_worker():
                    while "stream" not in captured:
                        _real_time.sleep(0.001)
                    fake = captured["stream"]
                    while fake.active:
                        fake.tick(4, frames=1024)

                worker = threading.Thread(target=lazy_worker, daemon=True)
                worker.start()

                am.play_audio_stream(_make_pcm_stream(total_seconds=1.0), sample_rate=24000)
                worker.join(timeout=2.0)

        # Assert — el sentinel NO aparece en logs
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert self.SENTINEL_API_KEY not in all_logs, (
            f"Sentinel API key filtrada en logs del path streaming. "
            f"Primera ocurrencia: {[r.getMessage() for r in caplog.records if self.SENTINEL_API_KEY in r.getMessage()]}"
        )
