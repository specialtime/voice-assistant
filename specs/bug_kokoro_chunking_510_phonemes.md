# Bug: Kokoro TTS — IndexError 510 phonemes en textos largos

## Estado
- **Rama:** `fix/kokoro-chunking-510-phonemes`
- **Severidad:** Alta (Kokoro cae a Gemini fallback en cada respuesta larga de prod)
- **Fecha:** 2026-06-28
- **Relacionado:** `specs/bug_kokoro_phonemizer_mismatch.md` (fix anterior de normalización)

## Síntomas reportados (producción)

```
2026-06-28 15:18:36 [WARNING] phonemizer: words count mismatch on 2000.0% of the lines (20/1)
2026-06-28 15:18:36 [WARNING] kokoro_onnx: Phonemes are too long, truncating to 510 phonemes
2026-06-28 15:18:36 [ERROR] handlers.kokoro_tts_client: Kokoro TTS falló — IndexError: index 510 is out of bounds for axis 0 with size 510
2026-06-28 15:18:36 [WARNING] __main__: TTS local falló (Kokoro TTS falló: index 510 is out of bounds...), intentando Gemini fallback
```

El usuario pregunta por todos sus eventos de la semana → el agente responde con un texto largo (678 chars) → Kokoro crashea → cae a Gemini fallback.

## Texto que disparó el bug (extraído del log de prod)

```
Te resumo tu agenda de la semana que viene (lunes 29/6 al domingo 5/7):
Lunes 29/6
18:00–20:00 — Estudio Teclab (Teclab Placeholder)
20:00 a 21:00 — Ejercicio / Entrenamiento
[... 678 chars total ...]
```

Tras la normalización del fix anterior, el texto queda en una sola línea de 678 chars. Kokoro lo phonemiza completo → genera >510 phonemas → `_split_phonemes` de kokoro-onnx no splitea en `:` ni `–` ni `/` (solo en `[.,!?;]`) → un batch excede `MAX_PHONEME_LENGTH` → trima a 510 → `voice[510]` está out of bounds → **IndexError**.

## Causa raíz

Bug upstream confirmado de `kokoro-onnx` 0.5.0:
- **Issue #184**: `IndexError: index 510 is out of bounds in _create_audio when phonemes are truncated to MAX_PHONEME_LENGTH`.
- **PR #185** (Open, Feb 2026): propone clamp del índice `voice[min(len(tokens), len(voice) - 1)]` + normalización de whitespace. **Aún NO mergeado**.
- **Comentario t-d-d (Abr 2026)**: el clamp convierte el crash en truncación silenciosa (pierde audio). El root cause es `_split_phonemes` solo splitea en `[.,!?;]`.

**Nuestro fix anterior** (normalización de whitespace) resolvió el `words count mismatch` para textos cortos multilinea, pero **no resuelve textos largos** porque el problema es el límite de 510 phonemas por batch, no el whitespace.

## Diseño del fix — Chunking en el handler

**Estrategia:** el handler splitea el texto normalizado en chunks por puntuación antes de llamar a `Kokoro.create()`, garantizando que ningún batch exceda ~500 phonemas (margen seguro bajo 510). Luego concatena el audio de todos los chunks.

**Alcance:** `src/handlers/kokoro_tts_client.py` únicamente.

### Cambio — Método `_split_text` + loop de síntesis

Añadir un método privado `_split_text(text: str) -> list[str]` que:

1. **Normaliza** el texto (reusa el `re.sub(r"\s+", " ", text).strip()` existente).
2. **Splitea** por puntuación de fin de oración y separadores fuertes: `re.split(r'(?<=[.,;:!?—–/])\s+', text)`. Esto splitea **después** de los signos de puntuación (lookbehind) preservándolos en el chunk anterior.
   - Incluye `—` (em-dash U+2014) y `–` (en-dash U+2013) porque el agente los usa como separadores.
   - Incluye `:` porque el agente usa formatos tipo `"Lunes 29/6:"`.
   - Incluye `/` porque aparece en fechas `"29/6"`.
3. **Filtrar** chunks vacíos.
4. **Safety net:** si algún chunk resultante sigue siendo muy largo (estimación heurística: >1500 chars → podría generar >510 phonemas), splitearlo por espacios en sub-chunks de ~1500 chars. Esto es un fallback defensivo — la mayoría de los casos se resuelven en el paso 2.

Modificar `synthesize()` para que:

1. Llame a `_split_text(text)` → obtiene `chunks: list[str]`.
2. Para cada chunk, llame a `self._kokoro.create(chunk, voice=..., speed=..., lang=..., trim=False)`.
3. Concatene los arrays de samples con `np.concatenate()`.
4. Convierta el array concatenado a PCM int16 (igual que ahora).

```python
def _split_text(self, text: str) -> list[str]:
    """Splitea texto en chunks seguros para Kokoro (<510 phonemas por batch).

    Splitea por puntuación de fin de oración y separadores fuertes,
    preservando los signos en el chunk anterior (lookbehind).
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    # Split después de puntuación fuerte + separadores (em-dash, en-dash, slash, dos puntos)
    chunks = re.split(r"(?<=[.,;:!?—–/])\s+", normalized)
    chunks = [c.strip() for c in chunks if c.strip()]
    # Safety net: chunks muy largos → split por espacios
    MAX_CHARS = 1500  # heurística: ~3-4 chars por phonema → <510 phonemas
    safe_chunks = []
    for c in chunks:
        if len(c) > MAX_CHARS:
            words = c.split(" ")
            current = ""
            for w in words:
                if len(current) + len(w) + 1 > MAX_CHARS:
                    if current:
                        safe_chunks.append(current.strip())
                    current = w
                else:
                    current = (current + " " + w).strip()
            if current:
                safe_chunks.append(current.strip())
        else:
            safe_chunks.append(c)
    return safe_chunks
```

```python
# En synthesize(), reemplazar la llamada única a create():
chunks = self._split_text(text)
logger.debug("Kokoro TTS — %d chunks tras split", len(chunks))

audio_parts = []
for i, chunk in enumerate(chunks):
    samples_part, _ = self._kokoro.create(
        chunk,
        voice=cfg["voice"],
        speed=cfg["speed"],
        lang=cfg["lang"],
        trim=False,
    )
    audio_parts.append(samples_part)
    logger.debug("Kokoro TTS — chunk %d/%d sintetizado (%d samples)", i + 1, len(chunks), len(samples_part))

samples = np.concatenate(audio_parts)
```

### Logging

- Log DEBUG del número de chunks tras el split.
- Log DEBUG por cada chunk sintetizado (índice + sample count).
- Preservar el log DEBUG del texto normalizado (primer chunk truncado a 120 chars).
- Preservar el log final `Kokoro TTS OK` con bytes PCM totales.

### Restricciones

- **NO refactorizar** nada fuera de `kokoro_tts_client.py`.
- **NO modificar** `requirements.txt` (no pinnear versión nueva de kokoro-onnx).
- **NO tocar** otros handlers TTS.
- Preservar la firma pública `synthesize(text, style_hint="") -> bytes`.
- Preservar `synthesize_stream()` — ya delega en `synthesize()`, hereda el chunking automáticamente.
- Preservar el manejo de errores (try/except → RuntimeError).
- Preservar la conversión float32 → int16.
- `trim=False` se mantiene en cada llamada a `create()` (fix anterior).

## Criterio de aceptación (UAT)

1. Sintetizar el texto exacto del log de prod (678 chars) **no** produce IndexError.
2. Sintetizar el texto exacto del log de prod **no** produce WARNING "Phonemes are too long, truncating to 510 phonemes".
3. El audio resultante contiene **todo** el texto (no se pierde contenido por truncación).
4. Textos cortos (1 oración) siguen funcionando igual que antes (1 chunk, 1 llamada a create).
5. Tests unitarios existentes siguen pasando.
6. Nuevos tests automatizados cubren:
   - Texto largo (678 chars) → múltiples chunks, sin IndexError.
   - Texto con em-dash, en-dash, dos puntos, slash → splitea correctamente.
   - Texto corto → 1 solo chunk.
   - Safety net: chunk muy largo sin puntuación → splitea por espacios.
7. No se filtran secrets en logs (preservar `test_no_secrets_logged`).

## Delegación

- **Fix:** `@debugger` (bug con causa raíz confirmada, fix acotado a un archivo)
- **Tests:** `@tester` (UAT automatizado)