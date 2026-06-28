# Spec: Mejoras de precisión STT (faster-whisper)

## Resumen

Aplicar recomendaciones de faster-whisper NO utilizadas actualmente para mejorar la precisión de transcripción del STT primario (Whisper local). Sin cambiar el modelo (`small`), sin aumentar VRAM, sin romper el pipeline existente.

## Problema

El cliente `WhisperSTTClient` (`src/handlers/whisper_stt_client.py`) llama a `model.transcribe()` pasando solo 2 parámetros: `language` y `beam_size`. faster-whisper expone varios parámetros que mejoran precisión y que están siendo ignorados:

1. **Sin VAD** — el audio con silencios/ruido genera alucinaciones ("Gracias por ver...", texto fantasma).
2. **Sin `initial_prompt`** — Whisper no recibe contexto de dominio (jerga técnica, estilo rioplatense). Gemini STT sí lo tenía via `stt_prompt`.
3. **Sin `hotwords`** — términos técnicos (Chrome, VSCode, PowerShell, opencode) se transcriben mal.
4. **`compute_type: "int8"`** — cuantización agresiva. `int8_float16` da mejor precisión con la misma VRAM.
5. **`condition_on_previous_text`** — default `True` en faster-whisper. Para comandos cortos de voz, propagar contexto entre segmentos genera loops de repetición.

## Decisiones de diseño

| Aspecto | Decisión | Justificación |
|---|---|---|
| `compute_type` | `int8_float16` (cambio de `int8`) | Pesos int8 + activaciones float16. Misma VRAM (~1GB con modelo small). Mejor precisión que int8 puro. Recomendado por README oficial de faster-whisper para GPU con INT8. |
| `vad_filter` | `True` | Filtra silencios con Silero VAD antes de transcribir. Reduce alucinaciones. Default del README: `min_silence_duration_ms=500` es razonable para comandos de voz cortos. |
| `initial_prompt` | Texto de contexto configurable en `settings.json` | Equivalente al `stt_prompt` de Gemini. Guía vocabulario y estilo. Ej: "Comandos de voz en español rioplatense. Términos técnicos: Chrome, VSCode, terminal, git, PowerShell, opencode." |
| `hotwords` | Lista configurable en `settings.json` | Prioriza términos técnicos que Whisper confunde. faster-whisper 1.2.1 soporta `hotwords` como string. |
| `condition_on_previous_text` | `False` (explicit) | Comandos de voz son cortos (1-10s). Arrastrar contexto entre segmentos causa loops de repetición. faster-whisper default es `True`, hay que desactivarlo explícitamente. |
| `beam_size` | Mantener `5` (sin cambio) | Mejora marginal subiendo a 8-10, más lento. No justifica el cambio. |
| Modelo | Mantener `small` (sin cambio) | Con 4GB VRAM, `medium` (~1.5GB int8) es posible pero el usuario eligió no cambiar modelo. Las otras mejoras ya dan suficiente ganancia. |

## Configuración nueva en `settings.json`

Sección `local.whisper` actual:
```json
"whisper": {
  "model": "small",
  "device": "cuda",
  "compute_type": "int8",
  "language": "es",
  "beam_size": 5
}
```

Sección `local.whisper` nueva:
```json
"whisper": {
  "model": "small",
  "device": "cuda",
  "compute_type": "int8_float16",
  "language": "es",
  "beam_size": 5,
  "vad_filter": true,
  "vad_min_silence_duration_ms": 500,
  "initial_prompt": "Comandos de voz en español rioplatense. Términos técnicos: Chrome, VSCode, terminal, git, PowerShell, opencode, Python, Docker.",
  "hotwords": "Chrome VSCode PowerShell opencode Python Docker terminal git",
  "condition_on_previous_text": false
}
```

**Notas:**
- `vad_min_silence_duration_ms`: 500ms — silencios más cortos que esto no se cortan. Para comandos de voz cortos, 500ms es razonable (default de faster-whisper para VAD no-batched es 2000ms, demasiado conservador).
- `initial_prompt`: string de contexto. Se pasa al primer segmento como prompt. No debe superar ~244 tokens (longitud máxima del prompt de Whisper).
- `hotwords`: string separado por espacios. faster-whisper los prioriza en el beam search.
- `condition_on_previous_text`: `false` explícito. faster-whisper default es `true`.

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `config/settings.json` | Agregar 5 campos nuevos a `local.whisper` + cambiar `compute_type` |
| `src/handlers/whisper_stt_client.py` | Pasar nuevos parámetros a `model.transcribe()` |
| `tests/test_whisper_stt_client.py` | Actualizar tests para verificar nuevos parámetros |
| `README.md` | Documentar nuevos parámetros de STT |

## Micro-Spec A: `src/handlers/whisper_stt_client.py`

**Complejidad:** Media (cableado de parámetros, sin lógica de negocio nueva)

**Agente:** `@dev_senior`

**Rama:** `feature/stt-accuracy-improvements`

**Cambio específico en `transcribe()` (líneas 96-101 actuales):**

Reemplazar:
```python
cfg = self.settings["local"]["whisper"]
try:
    segments, _info = self._model.transcribe(
        wav_path,
        language=cfg["language"],
        beam_size=cfg["beam_size"],
    )
    # segments es un generator — consumir y concatenar
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
except Exception as exc:
    logger.error("Whisper STT falló — %s: %s", type(exc).__name__, exc)
    raise RuntimeError(f"Whisper STT falló: {exc}") from exc
```

Por:
```python
cfg = self.settings["local"]["whisper"]
try:
    segments, _info = self._model.transcribe(
        wav_path,
        language=cfg["language"],
        beam_size=cfg["beam_size"],
        vad_filter=cfg.get("vad_filter", False),
        vad_parameters=dict(min_silence_duration_ms=cfg.get("vad_min_silence_duration_ms", 500))
            if cfg.get("vad_filter") else None,
        initial_prompt=cfg.get("initial_prompt"),
        hotwords=cfg.get("hotwords"),
        condition_on_previous_text=cfg.get("condition_on_previous_text", True),
    )
    # segments es un generator — consumir y concatenar
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
except Exception as exc:
    logger.error("Whisper STT falló — %s: %s", type(exc).__name__, exc)
    raise RuntimeError(f"Whisper STT falló: {exc}") from exc
```

**Reglas estrictas:**
- Usar `cfg.get(key, default)` para los campos nuevos — si faltan en settings.json, usar defaults de faster-whisper. Esto mantiene retrocompatibilidad si alguien tiene un settings.json viejo.
- `vad_parameters` solo se construye si `vad_filter` es `True`. Si es `False`, pasar `None` (default de faster-whisper).
- `initial_prompt` y `hotwords` pueden ser `None` (faster-whisper los acepta como `Optional[str]`).
- NO cambiar el método `_ensure_model_loaded()` — el `compute_type` se pasa en `WhisperModel()` y ya se lee de `cfg["compute_type"]`. Solo hay que cambiar el valor en settings.json.
- NO refactorizar nada fuera del método `transcribe()`.
- NO cambiar la firma pública `transcribe(wav_path: str) -> str`.
- Mantener el bloque `try/except` existente y el logging.

## Micro-Spec B: `config/settings.json`

**Complejidad:** Trivial

**Agente:** `@dev_senior` (mismo ticket que Micro-Spec A)

**Cambio:** Modificar la sección `local.whisper`:

1. Cambiar `"compute_type": "int8"` → `"compute_type": "int8_float16"`
2. Agregar después de `"beam_size": 5`:
```json
"vad_filter": true,
"vad_min_silence_duration_ms": 500,
"initial_prompt": "Comandos de voz en español rioplatense. Términos técnicos: Chrome, VSCode, terminal, git, PowerShell, opencode, Python, Docker.",
"hotwords": "Chrome VSCode PowerShell opencode Python Docker terminal git",
"condition_on_previous_text": false
```

## Micro-Spec C: `tests/test_whisper_stt_client.py`

**Complejidad:** Media

**Agente:** `@tester`

**Rama:** `feature/stt-accuracy-improvements`

**Cambios:**

1. **Actualizar fixture `whisper_mock_settings`** — agregar los 5 campos nuevos a la sección `whisper` del mock:
```python
"whisper": {
    "model": "small",
    "device": "cuda",
    "compute_type": "int8_float16",
    "language": "es",
    "beam_size": 5,
    "vad_filter": True,
    "vad_min_silence_duration_ms": 500,
    "initial_prompt": "Comandos de voz en español rioplatense.",
    "hotwords": "Chrome VSCode opencode",
    "condition_on_previous_text": False,
}
```

2. **Actualizar `test_transcribe_success`** — verificar que `model.transcribe()` recibe los nuevos kwargs:
   - `vad_filter=True`
   - `vad_parameters=dict(min_silence_duration_ms=500)`
   - `initial_prompt="Comandos de voz en español rioplatense."`
   - `hotwords="Chrome VSCode opencode"`
   - `condition_on_previous_text=False`

3. **Actualizar `test_transcribe_uses_config_language`** — además de verificar `language="es"`, verificar que `compute_type` se pasa al `WhisperModel()` (no a `transcribe()`).

4. **Agregar test nuevo: `test_transcribe_vad_filter_passed`** — mock con `vad_filter=True` → verificar que `transcribe()` recibe `vad_filter=True` y `vad_parameters` con `min_silence_duration_ms=500`.

5. **Agregar test nuevo: `test_transcribe_vad_filter_disabled`** — mock con `vad_filter=False` → verificar que `transcribe()` recibe `vad_filter=False` y `vad_parameters=None`.

6. **Agregar test nuevo: `test_transcribe_initial_prompt_passed`** — verificar que `initial_prompt` del config llega a `model.transcribe()`.

7. **Agregar test nuevo: `test_transcribe_hotwords_passed`** — verificar que `hotwords` del config llega a `model.transcribe()`.

8. **Agregar test nuevo: `test_transcribe_condition_on_previous_text_false`** — verificar que `condition_on_previous_text=False` llega a `model.transcribe()`.

9. **Agregar test nuevo: `test_transcribe_defaults_when_config_missing`** — settings sin los campos nuevos (settings.json viejo) → `transcribe()` usa defaults: `vad_filter=False`, `vad_parameters=None`, `initial_prompt=None`, `hotwords=None`, `condition_on_previous_text=True`. Verificar que no crashea.

10. **Mantener `test_no_secrets_logged`** — seguir verificando que no se loguean paths sensibles.

## Micro-Spec D: `README.md`

**Complejidad:** Trivial

**Agente:** `@dev`

**Rama:** `feature/stt-accuracy-improvements`

**Cambio:** Actualizar la sección de STT en el README para documentar los nuevos parámetros:
- `compute_type: int8_float16` (mejor precisión, misma VRAM)
- `vad_filter: true` (filtra silencios, reduce alucinaciones)
- `initial_prompt` (contexto de dominio, jerga técnica)
- `hotwords` (términos técnicos prioritarios)
- `condition_on_previous_text: false` (evita loops en comandos cortos)

## Flujo de trabajo (Git)

1. Rama `feature/stt-accuracy-improvements` ya creada desde `master`. ✅
2. Trabajo local (sin remote, sin push).
3. Delegar Micro-Specs A+B a `@dev_senior` (whisper_stt_client.py + settings.json).
4. Delegar Micro-Spec C a `@tester` (tests) — puede ser paralelo si se usa worktree, pero los tests dependen del código, así que es secuencial: primero @dev_senior, luego @tester.
5. Delegar Micro-Spec D a `@dev` (README) — paralelo con @tester.
6. Auditoría `@security` sobre los cambios.
7. Verificación final `@tester` (UAT).
8. Merge a `master` previa confirmación del usuario.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `int8_float16` no soportado por ctranslate2 viejo | faster-whisper 1.2.1 ya instalado, soporta int8_float16. Verificado en README oficial. |
| VAD corta audio válido si `min_silence_duration_ms` muy bajo | 500ms es conservador. Si causa problemas, subir a 1000-2000ms. Es configurable. |
| `initial_prompt` demasiado largo (>244 tokens) | El prompt propuesto es ~30 tokens. Seguro. |
| `hotwords` no tiene efecto si `prefix` está set | No usamos `prefix`. `hotwords` funciona. |
| `condition_on_previous_text=False` hace texto inconsistente entre segmentos | Para comandos cortos (1-10s) no hay múltiples segmentos. No aplica. |
| Tests fallan por nuevos kwargs | @tester actualiza mocks y agrega tests nuevos. |