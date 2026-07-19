# Spec: Restore Gemini TTS y Azure TTS como fallback en pipeline streaming

**Fecha:** 2026-07-19
**Rama:** `fix/streaming-tts-fallback`
**Tipo:** Bug / Feature compleja (toca core del pipeline + 4 handlers TTS)
**Clasificación:** Complejo — lógica de fallback, manejo de errores, contrato de interfaz

## 1. Problema

Al agregar streaming + Kokoro (rama `feature/streaming-tts-kokoro`, mergeada en `3c6e057`), se perdió la cadena de fallback a Gemini TTS y Azure TTS cuando el flujo streaming está activo.

### 1.1 Síntoma concreto

`src/main.py:230` activa el flujo streaming solo si el TTS local tiene `synthesize_sentence_stream`:

```python
if self._streaming_enabled and self._local_tts is not None and hasattr(self._local_tts, 'synthesize_sentence_stream'):
    # Flujo streaming — SIN fallback a Gemini/Azure
```

Solo `KokoroTTSClient` implementa `synthesize_sentence_stream` (`kokoro_tts_client.py:208`). Si Kokoro falla en medio del streaming (modelo no descargado, OOM, error ONNX), el `except` en `main.py:281-300` solo loguea un warning y abandona — **no cae a Gemini ni Azure**.

### 1.2 Matriz de compatibilidad actual

| Config | Streaming | Fallback Gemini/Azure | Estado |
|---|---|---|---|
| `streaming=true` + `tts_engine=kokoro` | ✅ Activo | ❌ **Roto** — si Kokoro falla, no hay fallback | **BUG** |
| `streaming=true` + `tts_engine=piper` | ❌ No activa streaming (cae a síncrono) | ✅ Funciona vía `_run_sync_pipeline` | OK pero sin streaming |
| `streaming=false` | ❌ No streaming | ✅ Funciona vía `_run_sync_pipeline` | OK |

### 1.3 Causas raíz

1. **`PiperTTSClient` no tiene `synthesize_sentence_stream`** → no puede participar del flujo streaming (cae a síncrono).
2. **El `except` del flujo streaming** (`main.py:281-300`) no intenta fallback a Gemini/Azure si Kokoro falla.
3. **Bug latente en `AzureTTSClient.synthesize_stream`**: si se llama con `style_hint=""` (caso real desde `main.py:380` cuando `parse_response` no encuentra prefijo `[STYLE:]`), genera SSML inválido `<mstts:express-as style="">`. No se ha manifestado porque Azure es el último fallback y rara vez se llega.

## 2. Solución

### 2.1 Diseño: fallback por oración con cadena local → Gemini → Azure

En lugar de "cadena de fallback a nivel de stream" (que pierde las oraciones restantes si el TTS local falla en medio), se implementa **fallback por oración**: el helper itera oraciones una a una y, por cada oración, intenta sintetizarla con la cadena local → Gemini → Azure. Si todos fallan para una oración, se loguea error y se continúa con la siguiente.

**Ventajas:**
- Fallback granular: si Kokoro falla en la oración 5, las oraciones 1-4 ya se reprodujeron, la 5 va a Gemini, las 6+ siguen el flujo normal.
- No pierde oraciones: el helper controla la iteración, no el TTS client.
- Simple y robusto: no requiere buffering complejo ni tee de iterators.

### 2.2 Cambios por archivo

| Archivo | Acción | Complejidad |
|---|---|---|
| `src/handlers/piper_tts_client.py` | Agregar `synthesize_sentence_stream()` | Trivial |
| `src/handlers/azure_tts_client.py` | Fix `synthesize_stream()` con `style_hint=""` | Trivial |
| `src/main.py` | Nuevo helper `_synthesize_sentence_stream_with_fallback()` + refactor flujo streaming | Normal |
| `tests/test_piper_tts_client.py` | Tests de `synthesize_sentence_stream` | Testing |
| `tests/test_tts_clients.py` | Test de Azure `synthesize_stream` con `style_hint=""` | Testing |
| `tests/test_state_machine.py` | Tests del helper de fallback + regresión streaming | Testing |

**NO se agregan `synthesize_sentence_stream` a `GeminiTTSClient` ni `AzureTTSClient`** — el helper usa `synthesize()` directamente (control granular por oración). Sería código muerto.

## 3. Contratos estrictos

### 3.1 `PiperTTSClient.synthesize_sentence_stream()` — NUEVO

```python
def synthesize_sentence_stream(self, sentences: Iterator[str]) -> Iterator[bytes]:
    """Sintetiza oraciones una a una y hace yield de PCM completo por oración.
    
    Análogo a KokoroTTSClient.synthesize_sentence_stream. Por cada oración
    no vacía, llama a self.synthesize(sentence, style_hint="") y hace yield
    del PCM completo.
    
    Args:
        sentences: Iterator que yields oraciones (str) una a una.
    
    Yields:
        Bytes PCM crudo s16le (sin cabecera WAV) — un yield por oración.
    """
```

**Implementación exacta** (análoga a `kokoro_tts_client.py:222-226`):

```python
def synthesize_sentence_stream(self, sentences: Iterator[str]) -> Iterator[bytes]:
    for sentence in sentences:
        if not sentence.strip():
            continue
        pcm_bytes = self.synthesize(sentence, style_hint="")
        yield pcm_bytes
```

**Notas:**
- `style_hint=""` siempre (Piper no soporta estilos, igual que Kokoro).
- Lazy-load del modelo en la primera oración (heredado de `synthesize` → `_ensure_voice_loaded`).
- Oraciones vacías/whitespace se saltan con `continue`.

### 3.2 `AzureTTSClient.synthesize_stream()` — FIX SSML con style_hint vacío

**Comportamiento actual (bug):** si `style_hint=""`, genera:
```xml
<mstts:express-as style="">
    {text}
</mstts:express-as>
```
Esto es SSML inválido (style vacío). Azure puede rechazarlo o ignorar el style.

**Comportamiento fix:** si `style_hint` es vacío (falsy), NO incluir el wrapper `<mstts:express-as>`. Generar SSML mínimo (igual que `synthesize()`):

```xml
<speak version="1.0" xml:lang="{locale}">
    <voice name="{voice}">
        {escaped_text}
    </voice>
</speak>
```

**Implementación exacta** — reemplazar el bloque de construcción del SSML en `synthesize_stream` (líneas 137-148) por:

```python
escaped_text = xml.sax.saxutils.escape(text)

if style_hint:
    ssml = (
        f'<speak version="1.0" '
        f'xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="https://www.w3.org/2001/mstts" '
        f'xml:lang="{self.settings["azure"]["locale"]}">\n'
        f'    <voice name="{self.settings["azure"]["voice"]}">\n'
        f'        <mstts:express-as style="{style_hint}">\n'
        f'            {escaped_text}\n'
        f'        </mstts:express-as>\n'
        f'    </voice>\n'
        f'</speak>'
    )
else:
    ssml = (
        f'<speak version="1.0" '
        f'xml:lang="{self.settings["azure"]["locale"]}">\n'
        f'    <voice name="{self.settings["azure"]["voice"]}">\n'
        f'        {escaped_text}\n'
        f'    </voice>\n'
        f'</speak>'
    )
```

**Notas:**
- Cuando `style_hint` es vacío, no se incluyen los namespaces `xmlns` ni `xmlns:mstts` (no se usan).
- El log debug (línea 152) se mantiene igual.
- El resto del método (endpoint, headers, httpx.stream) no cambia.

### 3.3 `VoiceAssistant._synthesize_sentence_stream_with_fallback()` — NUEVO helper en main.py

```python
def _synthesize_sentence_stream_with_fallback(
    self, sentences: Iterator[str], generation: int
) -> Iterator[bytes]:
    """Itera oraciones y sintetiza cada una con cadena de fallback.
    
    Por cada oración:
    1. Intenta TTS local (self._local_tts.synthesize).
    2. Si falla, intenta Gemini TTS (self._gemini_tts.synthesize) si está
       configurado y el circuit breaker está cerrado.
    3. Si Gemini falla o no está, intenta Azure TTS streaming
       (self._azure_tts.synthesize_stream) consumido a bytes.
    4. Si todos fallan, loguea error y continúa con la siguiente oración
       (no aborta el stream completo).
    
    Chequea self._pipeline_generation != generation antes de cada oración
    para soportar cancelación cooperativa.
    
    Args:
        sentences: Iterator que yields oraciones (str) una a una.
        generation: Número de generación para cancelación cooperativa.
    
    Yields:
        Bytes PCM crudo s16le — un yield por oración sintetizada.
    """
```

**Implementación exacta:**

```python
def _synthesize_sentence_stream_with_fallback(
    self, sentences: Iterator[str], generation: int
) -> Iterator[bytes]:
    for sentence in sentences:
        if self._pipeline_generation != generation:
            logger.info("TTS fallback stream cancelado (gen=%d)", generation)
            return
        if not sentence.strip():
            continue
        pcm = self._synthesize_one_sentence_with_fallback(sentence)
        if pcm:
            yield pcm

def _synthesize_one_sentence_with_fallback(self, sentence: str) -> Optional[bytes]:
    """Sintetiza una oración con cadena local → Gemini → Azure.
    
    Retorna PCM bytes si algún TTS funciona, None si todos fallan.
    """
    # 1. TTS local (Piper o Kokoro)
    try:
        return self._local_tts.synthesize(sentence, style_hint="")
    except Exception as e:
        logger.warning("TTS local falló para oración (%s: %s), intentando Gemini", type(e).__name__, e)

    # 2. Gemini TTS (fallback 1)
    if self._gemini_tts is not None and self._gemini_tts.is_available():
        try:
            return self._gemini_tts.synthesize(sentence, style_hint="")
        except Exception as e:
            logger.warning("Gemini TTS falló (%s: %s), intentando Azure", type(e).__name__, e)
    elif self._gemini_tts is not None and not self._gemini_tts.is_available():
        logger.warning("Gemini TTS circuit breaker abierto — saltando a Azure")

    # 3. Azure TTS streaming (fallback 2) — consumir a bytes
    if self._azure_tts is not None:
        try:
            return b"".join(self._azure_tts.synthesize_stream(sentence, style_hint=""))
        except Exception as e:
            logger.error("Azure TTS falló (%s: %s) — sin más fallbacks para esta oración", type(e).__name__, e)

    logger.error("Todos los TTS fallaron para oración: '%s'", sentence[:80])
    return None
```

**Notas críticas:**
- `Optional` ya está importado en `main.py:15` — no agregar import.
- El helper NO lanza excepciones — siempre retorna `None` si todo falla. El caller decide qué hacer.
- `style_hint=""` en todas las llamadas: en streaming, el style_hint del `SentenceBuffer` no se propaga al TTS (las oraciones individuales no llevan style). Es fallback, la prioridad es que funcione. Documentado en spec original `feature_streaming_tts_kokoro.md` §6.2.
- Azure `synthesize_stream` se consume a bytes con `b"".join(...)` — para una oración, no es gran pérdida vs streaming nativo (una oración son ~1-3 segundos de audio).

### 3.4 Refactor del flujo streaming en `main.py:230-300`

**Cambio:** reemplazar la línea 263:
```python
pcm_stream = self._local_tts.synthesize_sentence_stream(sentence_iterator())
```
por:
```python
pcm_stream = self._synthesize_sentence_stream_with_fallback(sentence_iterator(), generation)
```

**El resto del flujo streaming NO cambia:**
- `pcm_stream_with_speaking_transition()` se mantiene igual.
- `play_audio_stream` se mantiene igual.
- El `except` en líneas 281-300 se mantiene igual (maneja fallas de `send_command_stream`, no del TTS).

**El `hasattr` check en línea 230 se mantiene** — ahora Piper también tiene `synthesize_sentence_stream`, así que el check pasa para ambos TTS locales. Si un futuro TTS local no lo tiene, cae a síncrono (comportamiento seguro).

## 4. Cancelación cooperativa

El helper `_synthesize_sentence_stream_with_fallback` respetar el `_pipeline_generation` counter:
- Antes de procesar cada oración, chequea `self._pipeline_generation != generation` → `return` (aborta el generator).
- Dentro de `_synthesize_one_sentence_with_fallback` NO se chequea generation (la síntesis de una oración es síncrona y rápida, no vale la pena interrumpirla a mitad).

## 5. Tests requeridos

### 5.1 `tests/test_piper_tts_client.py` — `synthesize_sentence_stream`

- **Test:** iterator con 3 oraciones → yield 3 chunks PCM (uno por oración).
- **Test:** oraciones vacías/whitespace se saltan (no se llama `synthesize`).
- **Test:** iterator vacío → no se carga el modelo (lazy-load).
- **Test:** `synthesize` lanza excepción → el generator propaga la excepción (no la traga).

### 5.2 `tests/test_tts_clients.py` — Azure `synthesize_stream` con `style_hint=""`

- **Test:** `synthesize_stream("hola", style_hint="")` → SSML SIN `<mstts:express-as>`.
- **Test:** `synthesize_stream("hola", style_hint="cheerful")` → SSML CON `<mstts:express-as style="cheerful">` (regresión, no romper el caso con style).
- **Test:** `synthesize_stream("hola", style_hint="")` → SSML tiene `<speak>` y `<voice>` (wrapper mínimo).

### 5.3 `tests/test_state_machine.py` — helper de fallback

- **Test:** TTS local OK → solo se llama `_local_tts.synthesize`, no Gemini ni Azure.
- **Test:** TTS local falla, Gemini OK → se llama `_local_tts.synthesize` (raise) y `_gemini_tts.synthesize`, no Azure.
- **Test:** TTS local y Gemini fallan, Azure OK → se llama Azure `synthesize_stream`.
- **Test:** Todos fallan → helper yield nada (stream vacío), no lanza excepción.
- **Test:** Gemini con circuit breaker abierto → se salta Gemini, va directo a Azure.
- **Test:** Gemini no configurado (`_gemini_tts is None`) → se saltea, va a Azure.
- **Test:** Cancelación: `_pipeline_generation != generation` → helper aborta (return) sin sintetizar más.
- **Test:** Oraciones vacías se saltan.

### 5.4 `tests/test_state_machine.py` — regresión flujo streaming

- **Test:** flujo streaming completo con Kokoro sigue funcionando (mock `_local_tts.synthesize` en lugar de `synthesize_sentence_stream`).
- **Test:** flujo streaming con Piper ahora funciona (Piper tiene `synthesize_sentence_stream`).
- **Test:** fallback a síncrono cuando `send_command_stream` falla antes de prompt_async (regresión).

## 6. Out of scope

- No modificar `GeminiTTSClient` (no se le agrega `synthesize_sentence_stream` — no se necesita).
- No modificar `KokoroTTSClient` (ya tiene `synthesize_sentence_stream`).
- No modificar `SentenceBuffer` ni `response_parser`.
- No modificar el system prompt del agente.
- No modificar `config/settings.json` (no hay nuevos campos).
- No modificar el flujo síncrono `_run_sync_pipeline` (ya tiene fallback correcto).

## 7. Dependencias

- `Optional` ya importado en `main.py:15`.
- `Iterator` ya importado en `main.py` (verificar, si no está, agregar `from typing import Iterator`).
- No nuevas dependencias pip.

## 8. Verificación

```powershell
# Tests unit (sin red)
.venv\Scripts\pytest tests/test_piper_tts_client.py -v -m unit
.venv\Scripts\pytest tests/test_tts_clients.py -v -m unit
.venv\Scripts\pytest tests/test_state_machine.py -v -m unit

# Tests de regresión streaming
.venv\Scripts\pytest tests/test_speaking_transition.py -v -m unit
.venv\Scripts\pytest tests/test_uat_streaming_pipeline.py -v -m unit

# Coverage
.venv\Scripts\pytest tests/ --cov=handlers --cov=main -v -m unit
```