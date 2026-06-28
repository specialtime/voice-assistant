# Bug: Kokoro TTS — phonemizer "words count mismatch" + corte abrupto al final

## Estado
- **Rama:** `fix/kokoro-phonemizer-mismatch`
- **Severidad:** Media (Kokoro activo en prod)
- **Fecha:** 2026-06-28

## Síntomas reportados (producción)

1. **WARNING en log:**
   ```
   2026-06-28 15:00:06 [WARNING] phonemizer: words count mismatch on 400.0% of the lines (4/1)
   ```
2. **Audio se corta de golpe al final** — la última palabra casi no termina de decirse.

## Texto que disparó el bug (extraído del log de prod)

```
Mañana lunes 29 de junio tenés:
18:00 a 20:00 — Estudio Teclab
20:00 a 21:00 — Ejercicio / Entrenamiento
El resto del dí...
```

Características relevantes del input:
- **Newlines internos** (`\n`) — 4 líneas.
- **Em-dash** `—` (U+2014, no ASCII).
- **Números con dos puntos** (`18:00`).
- **Tilde y ñ** (UTF-8, soportado por espeak-ng).

## Hipótesis de causa raíz

### H1 — Newlines no normalizados (principal)
`src/handlers/kokoro_tts_client.py` pasa el texto **crudo** (con `\n`) a `self._kokoro.create()`. El `normalize_text()` upstream de `kokoro-onnx` 0.5.0 solo hace `text.strip()` — **no colapsa newlines internos**.

- `phonemizer.phonemize()` recibe texto multilinea → el backend espeak-ng procesa línea por línea → el conteo de palabras se desincroniza → WARNING "words count mismatch on 400.0% (4/1)" (4 líneas vs 1 esperada).
- El PR upstream #185 (matteofrassi, Feb 2026) confirma: *"Normalize newlines and whitespace in input text before phonemization to prevent audio artifacts"*.

### H2 — `_split_phonemes` no splitea en newlines (contribuyente al corte)
`kokoro_onnx/__init__.py::_split_phonemes` solo splitea en `[.,!?;]`. Los newlines quedan dentro del batch de phonemas → artefactos en la concatenación de audio → corte abrupto.

### H3 — `trim=True` agresivo (contribuyente al corte)
`Kokoro.create()` aplica `trim_audio` por defecto, que trima silencios inicial (~2s) y final (~0.02s). Si el batch final ya viene truncado por H1/H2, el trim puede comerse parte de la última sílaba.

## Evidencia upstream (kokoro-onnx 0.5.0)

- **Issue #184**: `IndexError: index 510 is out of bounds` cuando phonemes ≥ 510 — confirma que `_split_phonemes` es frágil.
- **PR #185**: Fix propuesto = `re.sub(r"\s+", " ", text)` en `normalize_text()` + clamp de índice. **Aún NO mergeado** (estado Open, Feb 2026).
- **Comentario de t-d-d (Abr 2026)**: *"the index clamp turns the crash into a clean truncation... The root cause is _split_phonemes only breaking on [.,!?;]"*.

## Diseño del fix

**Alcance:** `src/handlers/kokoro_tts_client.py` únicamente. No tocar dependencias upstream.

### Cambio 1 — Normalización de whitespace antes de `create()`

En `synthesize()`, antes de llamar `self._kokoro.create(...)`, normalizar el texto:

```python
import re

# Colapsar cualquier secuencia de whitespace (newlines, tabs, espacios múltiples) a un solo espacio.
# Previene el WARNING "words count mismatch" de phonemizer y artefactos de audio.
text_normalized = re.sub(r"\s+", " ", text).strip()
```

Usar `text_normalized` en lugar de `text` en la llamada a `self._kokoro.create()`.

### Cambio 2 (opcional, evaluar) — Pasar `trim=False` o suavizar el corte final

Evaluar si `trim=True` (default) agrava el corte abrupto. Si después del Cambio 1 el audio sigue cortándose, probar `trim=False` en la llamada:

```python
samples, sample_rate = self._kokoro.create(
    text_normalized,
    voice=cfg["voice"],
    speed=cfg["speed"],
    lang=cfg["lang"],
    trim=False,  # evitar corte agresivo de la última sílaba
)
```

**Decisión:** delegar al `@debugger` la verificación empírica de si el Cambio 2 es necesario tras el Cambio 1.

### Cambio 3 — Logging de texto normalizado para diagnóstico

Añadir log DEBUG del texto normalizado (truncado) para futura telemetría:

```python
logger.debug("Kokoro TTS — texto normalizado='%s'", text_normalized[:120])
```

## Restricciones

- **NO refactorizar** nada fuera de `kokoro_tts_client.py`.
- **NO modificar** `requirements.txt` (no pinnear versión nueva de kokoro-onnx — el PR #185 no está mergeado).
- **NO tocar** otros handlers TTS (piper, azure, gemini).
- Preservar la firma pública `synthesize(text, style_hint="") -> bytes`.
- Preservar tests existentes en `tests/test_kokoro_tts_client.py` — si cambian mocks, actualizarlos.

## Criterio de aceptación (UAT)

1. Sintetizar el texto exacto del log de prod **no** produce WARNING "words count mismatch".
2. El audio resultante **no se corta** abruptamente al final — la última palabra se completa.
3. Tests unitarios existentes siguen pasando.
4. Nuevo test automatizado cubre el caso de texto multilinea con newlines.
5. No se filtran secrets en logs (preservar `test_no_secrets_logged`).

## Delegación

- **Debugging + fix:** `@debugger` (bug confirmado, requiere telemetría empírica para validar H1/H2/H3)
- **Tests:** `@tester` (UAT automatizado con texto multilinea)