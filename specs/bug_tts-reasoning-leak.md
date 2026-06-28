# Bug: TTS lee el razonamiento interno del agente (reasoning leak)

**Estado:** Pendiente  
**Rama:** `fix/tts-reasoning-leak`  
**Complejidad:** Compleja / Core (afecta lógica de parsing SSE del cliente OpenCode)  
**Agente asignado:** `@dev_senior` (implementación), `@tester` (tests)

## Síntoma

Cuando el usuario hace una consulta (ej: "Día, dame mi agenda de la semana que viene"), el TTS sintetiza y reproduce TODO el stream de OpenCode, incluyendo:

1. La cadena de pensamiento del agente en inglés: `"The user is asking for their schedule for next week."`, `"Let me load the gog skill first, then follow the established workflow..."`, `"Wait, let me recalculate."`
2. IDs técnicos: emails largos de `group.calendar.google.com`, listas numeradas de calendarios.
3. Solo al final llega la respuesta útil al usuario: `"El lunes arrancás con Estudio Teclab de 18 a 20 y después entrenamiento de 20 a 21."`

El usuario escucha minutos de razonamiento interno antes de la respuesta real.

## Causa raíz

`src/handlers/opencode_client.py` método `_process_sse_event` (líneas 525-540) acepta **todos** los eventos `message.part.delta` sin distinguir el tipo de part al que pertenecen. OpenCode emite deltas tanto para parts de tipo `"text"` (respuesta al usuario) como para parts de tipo `"reasoning"` (cadena de pensamiento), `"tool"` (llamadas a herramientas), etc.

El cliente actual extrae el delta de cualquier `message.part.delta` y lo manda al TTS via el generator `send_command_stream`.

## Evidencia

Log `logs/cortex.log` (sesión 2026-06-28 17:55:12 → 17:57:02):

```
17:55:22 [DEBUG] handlers.opencode_client: Primer delta recibido: 'The'
17:55:23 [DEBUG] handlers.kokoro_tts_client: Kokoro TTS — texto normalizado (chunk 1)='The user is asking for their schedule for next week.'
17:55:25 [DEBUG] handlers.kokoro_tts_client: Kokoro TTS — texto normalizado (chunk 1)='Let me load the gog skill first,'
...
17:55:50 [DEBUG] handlers.kokoro_tts_client: Kokoro TTS — texto normalizado (chunk 1)='Teclab Placeholder - c7972ac25c7d03a95696474f0d58afa8a2747e6198287bff016fdf5008a3ca7f@group.calendar.google.com 2.'
...
17:56:55 [DEBUG] handlers.kokoro_tts_client: Kokoro TTS — texto normalizado (chunk 1)='El lunes arrancás con Estudio Teclab de 18 a 20 y después entrenamiento de 20 a 21.'
```

339 deltas emitidos, la mayoría corresponden a reasoning/tool output, no a la respuesta final.

## Solución de arquitectura

Filtrar deltas por tipo de part, siguiendo el mismo patrón que el código ACP oficial de OpenCode (PR [anomalyco/opencode#15614](https://github.com/anomalyco/opencode/pull/15614)).

### Mecanismo de OpenCode

OpenCode emite dos eventos SSE relacionados con parts:

1. **`message.part.updated`** — incluye el objeto `part` completo con `part.type` (`"text"`, `"reasoning"`, `"tool"`, `"file"`, `"step-start"`, `"step-finish"`, `"compaction"`, `"subtask"`) y `part.id`.
2. **`message.part.delta`** — incluye `partID` (referencia al part) + `field` + `delta` (texto incremental), pero **NO** incluye el tipo de part.

Para saber si un `message.part.delta` corresponde a un part de tipo `"text"` (respuesta al usuario) o `"reasoning"` (pensamiento), hay que haber registrado previamente el `partID → type` al recibir el `message.part.updated` correspondiente.

### Diseño del fix

**Archivo único a modificar:** `src/handlers/opencode_client.py`

**Cambios:**

1. **Nuevo estado en `send_command_stream`**: un diccionario `part_types: dict[str, str]` que mapea `partID → part.type`, poblado al recibir eventos `message.part.updated`.

2. **Extender `_process_sse_event`** con dos cambios:
   - Manejar `event_type == "message.part.updated"`: extraer `part.id` y `part.type` del payload (en `properties.part` o `data.part`), guardarlos en `part_types`, retornar `([], False)`. NO emitir delta.
   - En el handler existente de `message.part.delta`: antes de emitir el delta, buscar `partID` en `part_types`. Si el tipo es `"text"` → emitir delta. Si es `"reasoning"`, `"tool"`, o cualquier otro → **descartar** el delta (loguear a DEBUG). Si el `partID` es desconocido (no se vio el `message.part.updated` previo) → **emitir el delta por defecto** (preserva comportamiento actual para no romper servers que no emiten `part.updated`).

3. **Signature de `_process_sse_event`**: agregar parámetro `part_types: dict[str, str]` (mutable, se modifica in-place). El caller `send_command_stream` crea el dict vacío al inicio del stream y lo pasa en cada llamada.

4. **Compatibilidad hacia atrás**: si un server no emite `message.part.updated` antes de `message.part.delta` (caso edge), el `partID` no estará en `part_types` y el delta se emite por defecto. Esto preserva el comportamiento actual y no rompe tests existentes.

### Detalles del payload `message.part.updated`

Según el SDK de Rust (`MessagePartUpdatedProps`) y el PR #15614, el payload es:

```json
{
  "type": "message.part.updated",
  "properties": {
    "sessionID": "ses_...",
    "part": {
      "id": "prt_...",
      "type": "text" | "reasoning" | "tool" | "file" | ...,
      "text": "...",
      ...
    }
  }
}
```

Formato v2 alternativo (data en lugar de properties):

```json
{
  "type": "message.part.updated",
  "data": {
    "sessionID": "ses_...",
    "part": { "id": "prt_...", "type": "text", ... }
  }
}
```

El handler debe aceptar ambos formatos (`properties.part` o `data.part`), igual que ya hace para deltas.

### Detalles del payload `message.part.delta`

Ya soportado por el código actual. El `partID` está en `properties.partID` o `data.partID` (formato v1/v2). El fix agrega la lectura de `partID` para hacer el lookup en `part_types`.

### Tipos de part a FILTRAR (no emitir al TTS)

- `"reasoning"` — cadena de pensamiento del modelo
- `"tool"` — output de herramientas (bash, file ops, etc.)
- `"file"` — archivos adjuntos
- `"step-start"`, `"step-finish"` — marcadores de pasos
- `"compaction"` — marcador de compactación de contexto
- `"subtask"` — delegación a subagente

### Tipo de part a EMITIR al TTS

- `"text"` — respuesta en lenguaje natural al usuario
- **Desconocido** (partID no registrado) — emitir por defecto (compatibilidad)

## Alcance

- **Modificar:** `src/handlers/opencode_client.py` (métodos `send_command_stream` y `_process_sse_event`).
- **NO modificar:** `main.py`, `response_parser.py`, `kokoro_tts_client.py`, ni ningún otro handler. El fix es exclusivamente en el cliente OpenCode.
- **NO refactorizar** nada fuera del alcance de este ticket.

## Tests requeridos (`@tester`)

1. **Unit tests** en `tests/test_opencode_client.py`:
   - `message.part.updated` con `part.type="text"` → delta posterior se emite.
   - `message.part.updated` con `part.type="reasoning"` → delta posterior se descarta.
   - `message.part.updated` con `part.type="tool"` → delta posterior se descarta.
   - `message.part.delta` sin `message.part.updated` previo → delta se emite por defecto (compat).
   - Formato v2 (`data.part` en lugar de `properties.part`).
   - Stream mixto: reasoning + text → solo se emiten los deltas de text.

2. **UAT streaming** en `tests/test_uat_streaming_pipeline.py`:
   - Escenario: stream con parts de reasoning y text → el TTS solo recibe los deltas de text.
   - Preservar tests existentes (no romper los casos de `message.part.delta` ya cubiertos).
   - **Criterio de aceptación del usuario**: la UAT debe usar el `comando.wav` del entorno dev (mismo que usan los tests E2E existentes) y verificar que se reproduzca audio con **bytes > 0** (no audio vacío). El test debe fallar si el PCM resultante está vacío o si se filtraron TODOS los deltas (lo que indicaría over-filtering).

## Referencias

- [DeepWiki: Message and Part Structure](https://deepwiki.com/sst/opencode/2.2-message-and-part-structure)
- [PR anomalyco/opencode#15614](https://github.com/anomalyco/opencode/pull/15614) — distingue `agent_message_chunk` (text) de `agent_thought_chunk` (reasoning).
- [Rust SDK: MessagePartUpdatedProps](https://docs.rs/opencode-sdk-rs/latest/opencode_sdk_rs/resources/event/struct.MessagePartUpdatedProps.html)
- [Rust SDK: MessagePartDeltaProps](https://docs.rs/opencode-sdk-rs/latest/opencode_sdk_rs/resources/event/struct.MessagePartDeltaProps.html)