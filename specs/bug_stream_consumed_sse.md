# Spec — Bug: `httpx.StreamConsumed` en streaming SSE de OpenCode

**Rama:** `fix/stream-consumed-sse`
**Tipo:** Bug / Debugging
**Agente destinatario:** `@debugger`
**Complejidad:** Media-Alta (concurrencia de streaming HTTP + generadores anidados)

## 1. Síntoma

El orquestador (`src/main.py`) cae del estado `SPEAKING` a `IDLE` inmediatamente
sin reproducir audio. El log muestra:

```
[ERROR] handlers.audio_manager: Productor stream falló
Traceback (most recent call last):
  ...
  File "...\src\handlers\opencode_client.py", line 344, in send_command_stream
    line_bytes = next(stream_response.iter_lines(), None)
  ...
  File "...\httpx\_models.py", line 940, in iter_raw
    raise StreamConsumed()
httpx.StreamConsumed: Attempted to read or stream some content, but the
content has already been streamed.
```

El error se reproduce en **cada** invocación (gen=1 y gen=2 del log), de forma
determinista. El playback reporta "completado" con 0 bytes reproducidos.

## 2. Contexto de reproducción

- **Entorno:** prod (`C:\Users\crist\voice-assistant`), logs en `logs/cortex.log`.
- **Pipeline:** STT Whisper local → OpenCode (SSE `/event`) → Kokoro TTS → playback.
- **Servidor OpenCode:** `http://127.0.0.1:57214`, sin auth (password vacía).
- **Input de voz:** "Hola, cómo..." (gen=1) y "Yeah." (gen=2). Ambos fallan igual.
- **STT funciona** correctamente (Whisper transcribe OK).
- **Falla** exclusivamente el consumo del stream SSE en `send_command_stream`.

## 3. Hipótesis del Arquitecto (NO resolver — delegar al debugger)

El error `httpx.StreamConsumed` se lanza cuando se intenta iterar el cuerpo de
una respuesta `httpx` más de una vez. En `send_command_stream` (líneas 312-464
de `src/handlers/opencode_client.py`) se usa:

```python
with self._client.stream("GET", "/event") as stream_response:
    ...
    line_bytes = next(stream_response.iter_lines(), None)   # línea 344
    ...
    for line_bytes in stream_response.iter_lines():          # línea 389
        ...
```

**Hipótesis principal:** `stream_response.iter_lines()` se está invocando dos
veces sobre el mismo objeto de respuesta. La primera llamada (línea 344, dentro
del bucle `while not connected_seen`) consume parcialmente el stream y deja el
iterador interno en estado "consumed". La segunda invocación (línea 389, `for`)
intenta re-iterar y `httpx` levanta `StreamConsumed`.

**Evidencia que respalda la hipótesis:**
1. El traceback apunta a la línea 344 (`next(stream_response.iter_lines(), None)`),
   no a la línea 389. Esto sugiere que la excepción se propaga desde el primer
   `next()` en una invocación posterior del generador, después de que el `for`
   de la línea 389 ya agotó/cerró el stream.
2. El log muestra `receive_response_body.failed exception=GeneratorExit()` justo
   antes del error. El `GeneratorExit` ocurre cuando el generador
   `send_command_stream` se cierra (por `return` o por GC) mientras el `with`
   aún tiene el stream abierto — httpx intenta drenar/cerrar el body y al
   re-entrar al iterador ya consumido lanza `StreamConsumed`.
3. El patrón "primer `next()` para consumir `server.connected`, luego `for` para
   el resto" es propenso a este bug: `iter_lines()` devuelve un generador que
   se consume una sola vez. Llamarlo dos veces crea dos generadores sobre el
   mismo `iter_raw()` subyacente, y el segundo falla.

**Hipótesis secundaria (menos probable):** el `with self._client.stream(...)`
se está saliendo del bloque antes de que el generador `send_command_stream`
termine de consumirse desde `main.py` (el generador es lazy y se consume fuera
del `with`). Al cerrarse el context manager, httpx marca el stream como
consumido/cerrado, y la siguiente iteración desde `main.py` levanta
`StreamConsumed`. Si esta fuera la causa, el fix requeriría materializar el
stream dentro del `with` o reestructurar el generador para no depender del
context manager externo.

## 4. Archivos involucrados

- **`src/handlers/opencode_client.py`** — método `send_command_stream` (líneas
  ~280-464). Es el único archivo de producción a tocar.
- **`src/main.py`** — `sentence_iterator` (líneas 254-265) consume el generador.
  No debería modificarse salvo que la hipótesis secundaria sea la correcta.
- **`src/handlers/audio_manager.py`** — `play_audio_stream` (línea 211). Solo
  es el consumidor final; no tocar.
- **`src/handlers/kokoro_tts_client.py`** — `synthesize_sentence_stream` (línea
  222). Solo es el consumidor intermedio; no tocar.

## 5. Alcance del fix

- **MUST:** Eliminar el `StreamConsumed` de forma determinista. El streaming
  debe emitir los deltas del agente y reproducir audio.
- **MUST:** Preservar el orden subscribe→prompt_async (race condition ya fixeado
  en commit `9790971`). No revertir ese cambio.
- **MUST:** Preservar el filtro anti-stale de `session.idle` con 0 deltas
  (commit `43c506e`). No revertir.
- **MUST NO:** No refactorizar el método entero. Cambio mínimo y quirúrgico.
- **MUST NO:** No tocar `main.py`, `audio_manager.py`, `kokoro_tts_client.py`.
- **SHOULD:** Mantener los logs DEBUG existentes para telemetría.

## 6. Criterios de aceptación (para @tester)

1. Al invocar el pipeline completo (grabación → STT → OpenCode → TTS → playback)
   no se registra `StreamConsumed` ni `Productor stream falló` en el log.
2. El estado `SPEAKING` emite audio real (no playback vacío de 0 bytes).
3. Los tests unitarios existentes de `opencode_client` (`pytest tests/ -m unit`)
   siguen pasando sin modificación de los mocks.
4. Si el debugger modifica la firma o el contrato de `send_command_stream`,
   los tests se actualizan en la misma rama.

## 7. Notas para el debugger

- **No ejecutar el pipeline en vivo** (requiere micrófono + servidor OpenCode
  levantado + API keys). Limitarse a tests unitarios con mocks de httpx.
- El repo tiene tests en `tests/test_opencode_client*.py` que mockean el
  cliente httpx. Usarlos como base para reproducir el bug de forma aislada.
- **Rama de trabajo:** `fix/stream-consumed-sse` (ya creada y con checkout).
- **Commitear** exclusivamente en esa rama con mensaje `fix(streaming): <detalle>`.