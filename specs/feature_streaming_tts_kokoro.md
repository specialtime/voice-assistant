# Spec: Streaming TTS con Kokoro + SSE de OpenCode

**Fecha:** 2026-06-28
**Rama:** `feature/streaming-tts-kokoro`
**Objetivo:** Reducir la latencia al primer audio hablado haciendo que Kokoro sintetice y reproduzca oraciones a medida que el agente genera texto, sin esperar la respuesta completa.

## 1. Problema

Pipeline actual (bloqueante en cada etapa):

```
STT → send_command (espera respuesta COMPLETA) → parse → Kokoro.synthesize (TODO) → play_audio (bloqueante)
```

Latencia total = T(STT) + T(agente completo) + T(Kokoro todo) + T(playback).

## 2. Solución

Pipeline streaming:

```
STT → prompt_async (204 inmediato) → SSE /event (tokens en vivo)
                                         ↓ acumular hasta oración completa
                                    buffer de oraciones
                                         ↓ por cada oración
                                    Kokoro.synthesize(oración) → encolar PCM
                                         ↓ hilo consumidor
                                    play_audio_stream (reproduce en tiempo real)
```

Latencia al primer audio = T(STT) + T(primer token) + T(primer oración) + T(Kokoro 1 oración).

## 3. Hallazgos del spike (OpenCode serve v1.17.11)

### 3.1 Endpoint SSE `/event`
- **GET `/event`** → `text/event-stream`.
- Formato: `data: {json}\n\n` (una línea `data:` por evento, separada por `\n\n`).
- **No requiere auth** para suscribirse (confirmado en vivo). El `OpenCodeClient` actual usa BasicAuth para POST; el stream SSE se puede consumir sin auth o con la misma auth (reusar).
- Primer evento: `{"type":"server.connected","properties":{}}`.
- Heartbeats periódicos: `{"type":"server.heartbeat","properties":{}}`.

### 3.2 Endpoint `POST /session/:id/prompt_async`
- Body igual a `/session/:id/message`: `{agent, parts: [{type:"text", text}]}`.
- Retorna `204 No Content` inmediatamente (no espera respuesta).
- Requiere BasicAuth (igual que el cliente actual).

### 3.3 Eventos SSE relevantes (schema legacy — campo `properties`)

| Evento | `type` | Campo clave | Significado |
|---|---|---|---|
| Text started | `session.next.text.started` | `properties.assistantMessageID`, `properties.textID` | El agente empezó a generar texto |
| **Text delta** | `session.next.text.delta` | **`properties.delta`** (string) | Fragmento de texto incremental |
| Text ended | `session.next.text.ended` | `properties.text` (texto completo) | Fin de un bloque de texto |
| Session idle | `session.idle` | `properties.sessionID` | La sesión terminó de procesar |
| Session error | `session.error` | `properties.error` | Error del agente/provider |
| Step started | `session.next.step.started` | — | Inicio de un step (puede haber tool calls antes del texto) |

**Estructura JSON de cada evento (legacy):**
```json
{
  "id": "evt_...",
  "type": "session.next.text.delta",
  "properties": {
    "timestamp": 1782672469771,
    "sessionID": "ses_...",
    "assistantMessageID": "msg_...",
    "textID": "txt_...",
    "delta": "Hola, "
  }
}
```

### 3.4 Consideraciones
- Puede haber eventos de tool calls (`session.next.tool.*`) antes/después del texto. Solo nos interesan los `text.delta` para TTS.
- `session.idle` marca el fin de toda la generación (todos los steps completados).
- El agente `asistente_voz` responde con formato `[STYLE: ...] texto`. El prefijo `[STYLE: ...]` llega en los primeros deltas. Hay que parsearlo del stream incremental antes de empezar a sintetizar.

## 4. Arquitectura

### 4.1 Nuevo método: `OpenCodeClient.send_command_stream()`

```python
def send_command_stream(self, text: str) -> Iterator[str]:
    """Envía prompt_async y hace yield de deltas de texto del stream SSE.
    
    Flujo:
    1. ensure_session()
    2. POST /session/:id/prompt_async (204 inmediato)
    3. GET /event (SSE stream) con httpx.stream()
    4. Parsear eventos SSE línea por línea
    5. Filtrar eventos de la sesión actual (properties.sessionID == self.session_id)
    6. Yield properties.delta de eventos session.next.text.delta
    7. Terminar al recibir session.idle o session.error
    
    Yields:
        str: cada delta de texto del agente.
    
    Raises:
        RuntimeError: si prompt_async falla o session.error.
    """
```

**Detalles de implementación:**
- Usar `httpx.Client.stream("GET", "/event", ...)` para el SSE.
- Parsear manualmente: acumular líneas `data: ...`, decodificar JSON, filtrar por `type` y `sessionID`.
- **Timeout**: el `httpx.Timeout` del cliente actual es 120s (de settings). Para el stream SSE, usar un timeout de lectura generoso (ej. 120s) pero con reconexión si se corta.
- **Failover**: si `prompt_async` devuelve error HTTP, caer a `send_command()` síncrono (fallback automático).
- **Race condition**: el stream `/event` es global (todas las sesiones). Filtrar por `properties.sessionID == self.session_id`. Si hay eventos de sesiones previas, ignorarlos.

### 4.2 Nuevo método: `KokoroTTSClient.synthesize_sentence_stream()`

```python
def synthesize_sentence_stream(self, sentences: Iterator[str]) -> Iterator[bytes]:
    """Sintetiza oraciones una a una y hace yield de PCM.
    
    Por cada oración del iterator:
    1. synthesize(oracion) → pcm_bytes
    2. yield pcm_bytes (chunk completo de la oración)
    
    El consumidor (play_audio_stream) reproduce en tiempo real.
    """
```

**Nota:** Kokoro no soporta streaming nativo, pero sintetizar oración por oración reduce la latencia al primer audio. Cada oración se sintetiza en ~100-300ms (CPU).

### 4.3 Nuevo módulo: `src/handlers/sentence_buffer.py`

Buffer que acumula deltas de texto y emite oraciones completas.

```python
class SentenceBuffer:
    """Acumula deltas de texto y emite oraciones completas.
    
    Split por puntuación fuerte: . ! ? ; (preservando el signo).
    Maneja el prefijo [STYLE: ...] al inicio del stream.
    """
    
    def __init__(self) -> None:
        self._buffer: str = ""
        self._style_hint: str = ""
        self._style_parsed: bool = False
    
    def add(self, delta: str) -> list[str]:
        """Agrega un delta y retorna lista de oraciones completas."""
        ...
    
    def flush(self) -> list[str]:
        """Retorna oraciones parciales restantes (para el final del stream)."""
        ...
    
    @property
    def style_hint(self) -> str:
        """Style hint extraído del prefijo [STYLE: ...]."""
        ...
```

**Lógica de split:**
- Patrón: `re.split(r"(?<=[.!?;])\s+", text)` — split después de puntuación fuerte.
- Acumular hasta que el buffer contenga al menos un delimitador de oración.
- El prefijo `[STYLE: ...]` se parsea del inicio del buffer acumulado (primeros ~30 chars). Una vez parseado, se setea `_style_parsed=True` y se descarta del buffer.

### 4.4 Modificación: `main.py` — `run_pipeline()` 

Nuevo flujo streaming (reemplaza pasos 2-7 cuando el TTS local es Kokoro):

```python
# 2. Agente (streaming) — con send_lock
with self._send_lock:
    delta_stream = self._opencode.send_command_stream(text)
    
    # 3+4. Transición a SPEAKING antes del primer audio
    with self._lock:
        self._state = self.STATE_SPEAKING
        self._overlay.set_state("speaking")
    
    # 5. Pipeline streaming: deltas → oraciones → Kokoro → playback
    sentence_buffer = SentenceBuffer()
    
    def sentence_iterator():
        for delta in delta_stream:
            if self._pipeline_generation != generation:
                return
            for sentence in sentence_buffer.add(delta):
                yield sentence
        # flush final
        for sentence in sentence_buffer.flush():
            yield sentence
    
    pcm_stream = self._local_tts.synthesize_sentence_stream(sentence_iterator())
    self._audio.play_audio_stream(pcm_stream)
```

**Fallback:** si `send_command_stream` falla (excepción), caer a `send_command()` síncrono + `synthesize()` + `play_audio()` (flujo actual).

### 4.5 Configuración

Nuevo campo en `config/settings.json`:

```json
"opencode": {
    ...
    "streaming_enabled": true,
    "streaming_timeout_seconds": 120
}
```

Si `streaming_enabled=false`, usar el flujo síncrono actual. Default: `true`.

## 5. Archivos a modificar/crear

| Archivo | Acción | Agente |
|---|---|---|
| `src/handlers/opencode_client.py` | Agregar `send_command_stream()` | `@dev_senior` |
| `src/handlers/sentence_buffer.py` | **Nuevo** — buffer de oraciones + parse de [STYLE] | `@dev_senior` |
| `src/handlers/kokoro_tts_client.py` | Agregar `synthesize_sentence_stream()` | `@dev` |
| `src/main.py` | Modificar `run_pipeline()` para flujo streaming + fallback | `@dev_senior` |
| `config/settings.json` | Agregar `streaming_enabled`, `streaming_timeout_seconds` | `@dev` |
| `tests/test_opencode_client.py` | Tests de `send_command_stream` (mock SSE) | `@tester` |
| `tests/test_sentence_buffer.py` | **Nuevo** — tests del buffer | `@tester` |
| `tests/test_kokoro_tts_client.py` | Tests de `synthesize_sentence_stream` | `@tester` |
| `tests/test_state_machine.py` | Tests del pipeline streaming + fallback | `@tester` |

## 6. Contratos estrictos

### 6.1 `OpenCodeClient.send_command_stream()`

- **Firma:** `def send_command_stream(self, text: str) -> Iterator[str]`
- **Yields:** `str` — cada delta de texto (`properties.delta`).
- **Termina:** al recibir `session.idle` (con `sessionID` matching) o `session.error`.
- **Raises:** `RuntimeError` si `prompt_async` falla con HTTP error.
- **NO** debe consumir el stream SSE si `prompt_async` falla — caer a excepción y dejar que el caller haga fallback.
- **Filtrado:** solo yield deltas donde `properties.sessionID == self.session_id`.
- **Reutilización de sesión:** usa `ensure_session()` igual que `send_command()`.
- **Compactación:** NO incrementa `_message_count` durante streaming. Incrementar al final (al recibir `session.idle`).
- **Auth:** el stream SSE `/event` se consume con la misma BasicAuth del cliente (por consistencia, aunque no la requiera).

### 6.2 `SentenceBuffer`

- **Firma:** `def add(self, delta: str) -> list[str]` — retorna lista de oraciones completas (puede ser vacía).
- **Firma:** `def flush(self) -> list[str]` — retorna oraciones parciales restantes.
- **Property:** `style_hint` → `str` (vacío si no hay prefijo).
- **Split:** `re.split(r"(?<=[.!?;])\s+", text)` — preserva signo en chunk anterior.
- **Prefijo [STYLE]:** regex `^\[STYLE:\s*(\w+)\]\s*` sobre el buffer acumulado. Se parsea una sola vez cuando el buffer tiene ≥10 chars o contiene `]`.
- **Markdown:** NO limpiar markdown aquí (el `response_parser._strip_markdown` se aplica por oración en el pipeline). **DECISIÓN:** aplicar `_strip_markdown` por oración dentro del `sentence_iterator` del pipeline, NO en el buffer.

### 6.3 `KokoroTTSClient.synthesize_sentence_stream()`

- **Firma:** `def synthesize_sentence_stream(self, sentences: Iterator[str]) -> Iterator[bytes]`
- **Por cada oración:** llamar `self.synthesize(oracion, style_hint="")` (ignora style, Kokoro no lo soporta).
- **Yields:** bytes PCM s16le completos de cada oración (no chunks de 4096 — el `play_audio_stream` los acumula).
- **Lazy:** no pre-cargar el modelo hasta la primera oración (`_ensure_model_loaded()` en la primera iteración).

## 7. Cancelación cooperativa

El pipeline streaming debe respetar el `_pipeline_generation` counter:
- Antes de yield cada oración a Kokoro, chequear `self._pipeline_generation != generation` → abortar.
- El `send_command_stream` (generator) se cierra cuando el caller deja de iterar (Python GC cierra el httpx stream).
- `play_audio_stream` ya chequea `_stop_playback_event` para interrumpir.

## 8. Fallback síncrono

Si `send_command_stream()` lanza excepción (timeout, error HTTP, SSE cortado):
```python
try:
    delta_stream = self._opencode.send_command_stream(text)
    # ... flujo streaming ...
except Exception as e:
    logger.warning("Streaming falló (%s), fallback a síncrono", e)
    response = self._opencode.send_command(text)  # síncrono
    style_hint, clean_text = parse_response(response)
    pcm_bytes = self._local_tts.synthesize(clean_text, style_hint)
    self._audio.play_audio(pcm_bytes)
```

## 9. Tests requeridos

### 9.1 `test_opencode_client.py` — `send_command_stream`
- Mock `httpx.Client.stream()` con eventos SSE simulados (líneas `data: {json}`).
- Test: yield deltas en orden, filtrar por sessionID, terminar en `session.idle`.
- Test: `session.error` → RuntimeError.
- Test: `prompt_async` HTTP error → RuntimeError (sin consumir SSE).
- Test: no filtra eventos de otras sesiones.
- Test: incrementa `_message_count` al final.

### 9.2 `test_sentence_buffer.py` (nuevo)
- Test: deltas parciales → oración completa al recibir puntuación.
- Test: prefijo `[STYLE: cheerful]` se extrae y se descarta del buffer.
- Test: sin prefijo → `style_hint=""`.
- Test: `flush()` retorna restos sin puntuación.
- Test: split por `.`, `!`, `?`, `;`.
- Test: oración muy larga sin puntuación → no se emite hasta flush.

### 9.3 `test_kokoro_tts_client.py` — `synthesize_sentence_stream`
- Mock `Kokoro.create()` → yield PCM por oración.
- Test: lazy-load del modelo en primera iteración.
- Test: iterator vacío → no carga modelo.

### 9.4 `test_state_machine.py` — pipeline streaming
- Test: flujo streaming completo (mock send_command_stream + synthesize_sentence_stream + play_audio_stream).
- Test: fallback a síncrono cuando send_command_stream falla.
- Test: cancelación (generation mismatch) aborta el iterator.
- Test: `streaming_enabled=false` usa flujo síncrono.

## 10. Out of scope

- No modificar el system prompt del agente `asistente_voz` (sigue devolviendo `[STYLE: ...] texto`).
- No modificar Azure TTS ni Gemini TTS (siguen siendo fallback no-streaming).
- No modificar el overlay.
- No refactorizar `send_command()` existente (se mantiene como fallback).