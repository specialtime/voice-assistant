# Bug: Streaming playback timeout corta audios largos

## Síntoma

Log recurrente:
```
handlers.audio_manager: Streaming playback timeout — forzando stop tras 30s
```
El playback de `play_audio_stream` se corta a los 30s aunque el audio sea más largo.

## Causa raíz

`src/handlers/audio_manager.py:228` — tras mandar el sentinel, el polling de `stream.active` usa un deadline fijo de 30s. El productor llena la cola a velocidad de red (burst), pero el callback consume a **tiempo real de audio** (~23 blocks/seg a 24kHz). Audios > 30s se cortan porque el callback aún no terminó de consumir el buffer.

## Diseño de la solución

**Estrategia: Timeout dinámico + safety net configurable.**

### Cambios en `src/handlers/audio_manager.py`

1. **Trackear samples totales pusheados** durante el loop del productor (líneas 211-219). Acumular en una variable local `_total_samples_pushed` sumando `len(np.frombuffer(chunk, dtype=np.int16))` por cada chunk (excluyendo el sentinel).

2. **Calcular deadline dinámico tras el sentinel** (reemplaza línea 228):
   ```python
   _estimated_duration = _total_samples_pushed / sample_rate
   _deadline = _time.time() + _estimated_duration * 1.3 + 10.0
   ```
   - Factor 1.3: margen por underruns que estiran el playback real.
   - +10s: piso mínimo para audios cortos (evita deadline negativo o ridículo).

3. **Safety net superior configurable**: leer de `settings["audio"].get("streaming_playback_safety_net_seconds", 600)`. Si el deadline dinámico excede `now + safety_net`, caparlo a `now + safety_net`. Esto evita esperas infinitas si el callback se clava por un bug de sounddevice.

4. **Log informativo** (no warning) cuando el deadline dinámico es > 30s, para distinguirlo del comportamiento anterior:
   ```python
   logger.debug("Streaming playback: deadline dinámico %.1fs (estimado %.1fs + margen)", _deadline - _time.time(), _estimated_duration)
   ```

5. **Mantener el warning** si se agota el deadline (sea dinámico o safety net):
   ```python
   logger.warning("Streaming playback timeout — forzando stop tras %.1fs", _time.time() - _start_wait)
   ```
   donde `_start_wait = _time.time()` se captura justo antes del loop de polling.

### Cambios en `config/settings.json`

Agregar dentro del bloque `"audio"`:
```json
"streaming_playback_safety_net_seconds": 600
```

**No tocar** otros campos del settings.

## Alcance (NO hacer)

- No refactorizar `play_audio` (non-streaming).
- No tocar `stop_playback()`.
- No cambiar el patrón productor-consumidor ni el blocksize.
- No modificar otros handlers.

## Complejidad

**Media** — lógica lineal, sin algoritmos complejos, pero toca el path crítico de playback. Agente: `@dev_senior`.

## Criterios de aceptación (UAT)

1. Audio de ~60s se reproduce completo sin corte.
2. Audio corto (< 5s) sigue funcionando sin regresión.
3. El log ya NO dice "forzando stop tras 30s" en audios largos válidos.
4. Si el callback se clava (simulado), el safety net corta a los 600s (no infinito).
5. `stop_playback()` del usuario sigue cortando inmediatamente.