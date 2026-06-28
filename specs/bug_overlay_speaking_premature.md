# Spec — Bug: Overlay "Hablando" prematuro + "Procesando" invisible

## Síntomas reportados
1. El indicador del overlay se pone en verde ("Hablando...") antes de que arranque la reproducción de audio.
2. El estado "Procesando..." (amarillo) ya no se ve — salta casi instantáneamente de grabando a hablando.

## Raíz
En `src/main.py`, flujo streaming (`run_pipeline`, líneas ~230-269), la transición a `STATE_SPEAKING` y `overlay.set_state("speaking")` se disparan **inmediatamente después** de obtener el iterador lazy de `send_command_stream()`, antes de que exista audio real:

```python
delta_stream = self._opencode.send_command_stream(text)  # iterador lazy
prompt_async_sent = True

# Transición a SPEAKING — PREMATURA
with self._lock:
    ...
    self._state = self.STATE_SPEAKING
    self._overlay.set_state("speaking")  # ← se dispara acá

# Recién acá empieza la cadena que produce audio:
pcm_stream = self._local_tts.synthesize_sentence_stream(sentence_iterator())
self._audio.play_audio_stream(pcm_stream)  # ← audio real sale acá
```

La cadena productora de audio (deltas SSE → SentenceBuffer → oración completa → Kokoro sintetiza → chunk PCM → queue → callback sounddevice) puede tardar segundos. Durante esa ventana el overlay miente: dice "Hablando" sin audio y "Procesando" nunca se ve.

## Fix
Mover la transición a `SPEAKING` al momento en que el **primer chunk PCM real** está disponible, mediante un generador wrapper que observa el primer yield del `pcm_stream` y dispara la transición dentro del lock.

### Archivo a modificar
- `src/main.py` — únicamente el bloque streaming dentro de `run_pipeline` (líneas ~230-269). **No tocar** `_run_sync_pipeline`, `audio_manager.py`, `kokoro_tts_client.py`, ni `overlay.py`.

### Diseño estricto
1. Eliminar la transición temprana a `SPEAKING` (bloque `with self._lock:` actual líneas 243-249).
2. Conservar el chequeo de cancelación `if self._pipeline_generation != generation: return` antes de iniciar el pipeline streaming (ese check queda, pero sin setear estado).
3. Envolver `pcm_stream` en un generador local `pcm_stream_with_speaking_transition()` que:
   - Itera `pcm_stream`.
   - En el primer chunk real (no vacío), adquiere `self._lock`, chequea cancelación, y si OK setea `self._state = STATE_SPEAKING` + `self._overlay.set_state("speaking")` + log "→ SPEAKING (gen=%d, primer PCM real)".
   - Si cancelado durante el primer chunk, retorna sin emitir nada.
   - Hace yield de cada chunk (incluido el primero).
4. Pasar el wrapper a `self._audio.play_audio_stream(...)`.
5. El overlay queda en "processing" (amarillo) durante toda la fase de STT + agente + buffering + síntesis, y salta a "speaking" (verde) recién cuando hay audio real.

### Variables / nombres
- Generador wrapper: `pcm_stream_with_speaking_transition` (closure local dentro de `run_pipeline`).
- Flag local: `speaking_set: bool = False`.
- Log message: `"→ SPEAKING (gen=%d, primer PCM real)"`.

### Restricciones
- **No refactorizar** nada fuera del bloque streaming de `run_pipeline`.
- **No modificar** `overlay.py`, `audio_manager.py`, `kokoro_tts_client.py`, `piper_tts_client.py`, ni tests existentes.
- **No cambiar** la lógica de cancelación cooperativa (generation counter).
- **No alterar** el flujo síncrono `_run_sync_pipeline` (su transición a SPEAKING antes del TTS es correcta ahí porque `synthesize` es bloqueante y retorna PCM real).

## Criterios de aceptación (UAT)
1. Al grabar un comando y soltar Alt+V, el overlay muestra "Procesando..." (amarillo) durante la fase de STT + agente + síntesis.
2. El overlay salta a "Hablando..." (verde) **solo cuando el audio empieza a sonar** (no antes).
3. Si el pipeline se cancela antes del primer chunk PCM, el overlay NO debe quedar en "speaking" — debe volver a "recording" o "idle" según corresponda.
4. El flujo síncrono (fallback) sigue funcionando igual — su transición a SPEAKING no se ve afectada.
5. Tests unitarios existentes de la máquina de estados siguen pasando.
6. Nuevo test unitario: verificar que el wrapper dispara la transición a SPEAKING solo al primer chunk PCM real, y no antes.

## Complejidad
**Media** — cambio acotado en un solo archivo, lógica lineal, sin decisiones arquitectónicas. Pero toca el pipeline core → delegar a `@dev_senior` para mayor seguridad.

## Branch
`fix/overlay-speaking-premature` (creada desde `master`).

## Delegación
- Implementación: `@dev_senior` (Micro-Spec arriba).
- Verificación UAT + tests automatizados: `@tester`.