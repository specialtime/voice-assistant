# Spec: Selector de motor TTS primario (primary_engine)

**Fecha:** 2026-07-19
**Rama futura:** `feature/tts-primary-engine`
**Tipo:** Feature normal (cambio chico en config + main.py)
**Clasificación:** Normal — lógica lineal, sin algoritmos complejos
**Estado:** Pendiente de implementación (deuda técnica del fix `fix/streaming-tts-fallback`)

## 1. Motivación

Actualmente `config/settings.json` solo permite elegir entre motores TTS **locales** (`local.tts_engine`: `"piper"` | `"kokoro"`). No hay forma de saltear el TTS local y usar Gemini o Azure como primario.

**Casos de uso:**
- Usuario quiere usar Gemini TTS porque suena mejor, sin depender de modelos locales descargados.
- Usuario quiere usar Azure TTS para voces específicas no disponibles en local.
- Testing/debug: forzar un motor específico sin borrar/renombrar modelos del filesystem (hack actual).

**Workaround actual (hacky):** renombrar `models/kokoro/kokoro-v1.0.onnx` para que `_local_tts.synthesize` levante `RuntimeError` y caiga al fallback. Frágil, depende de estado del filesystem, loguea warnings en cada oración.

## 2. Solución

Nuevo campo `tts.primary_engine` en `config/settings.json` que controla el orden de la cadena de fallback.

### 2.1 Config

```json
"tts": {
  "primary_engine": "local"
}
```

Valores válidos:
- `"local"` (default) — orden: local → Gemini → Azure (comportamiento actual).
- `"gemini"` — orden: Gemini → Azure (sin local).
- `"azure"` — orden: Azure solo (sin local ni Gemini).

Si el valor es inválido o falta, default a `"local"` con warning en log.

### 2.2 Cambios en `src/main.py`

#### 2.2.1 `__init__` — leer `primary_engine` y condicionar `_local_tts`

```python
# TTS primario: ¿usar local o saltar directo a cloud?
primary_engine = self._settings.get("tts", {}).get("primary_engine", "local")
if primary_engine not in ("local", "gemini", "azure"):
    logger.warning("tts.primary_engine='%s' inválido, usando 'local' por defecto", primary_engine)
    primary_engine = "local"

self._tts_primary_engine = primary_engine

# TTS local: solo instanciar si primary_engine == "local"
if primary_engine == "local":
    tts_engine = self._settings.get("local", {}).get("tts_engine", "piper")
    if tts_engine not in ("piper", "kokoro"):
        logger.warning("tts_engine='%s' inválido, usando 'piper' por defecto", tts_engine)
        tts_engine = "piper"
    if tts_engine == "kokoro":
        self._local_tts = KokoroTTSClient(self._settings)
        logger.info("TTS local: Kokoro (selector)")
    else:
        self._local_tts = PiperTTSClient(self._settings)
        logger.info("TTS local: Piper (selector)")
else:
    self._local_tts = None
    logger.info("TTS local deshabilitado — primary_engine=%s", primary_engine)
```

#### 2.2.2 `_synthesize_one_sentence_with_fallback` — respetar `primary_engine`

El helper actual asume que local es siempre el primario. Hay que hacerlo consciente de `_tts_primary_engine`:

```python
def _synthesize_one_sentence_with_fallback(self, sentence: str) -> Optional[bytes]:
    """Sintetiza una oración con cadena según _tts_primary_engine.

    - 'local':  local → Gemini → Azure
    - 'gemini': Gemini → Azure
    - 'azure':  Azure solo

    Retorna PCM bytes si algún TTS funciona, None si todos fallan.
    No lanza excepciones.
    """
    # 1. TTS local (solo si primary_engine == "local")
    if self._tts_primary_engine == "local" and self._local_tts is not None:
        try:
            return self._local_tts.synthesize(sentence, style_hint="")
        except Exception as e:
            logger.warning(
                "TTS local falló para oración (%s: %s), intentando Gemini",
                type(e).__name__, e,
            )

    # 2. Gemini TTS (fallback para 'local' y 'gemini'; primario para 'gemini')
    if self._tts_primary_engine in ("local", "gemini") and self._gemini_tts is not None:
        if self._gemini_tts.is_available():
            try:
                return self._gemini_tts.synthesize(sentence, style_hint="")
            except Exception as e:
                logger.warning(
                    "Gemini TTS falló (%s: %s), intentando Azure",
                    type(e).__name__, e,
                )
        else:
            logger.warning("Gemini TTS circuit breaker abierto — saltando a Azure")

    # 3. Azure TTS (fallback para 'local' y 'gemini'; primario para 'azure')
    if self._tts_primary_engine in ("local", "gemini", "azure") and self._azure_tts is not None:
        try:
            return b"".join(self._azure_tts.synthesize_stream(sentence, style_hint=""))
        except Exception as e:
            logger.error(
                "Azure TTS falló (%s: %s) — sin más fallbacks para esta oración",
                type(e).__name__, e,
            )

    logger.error("Todos los TTS fallaron para oración: '%s'", sentence[:80])
    return None
```

#### 2.2.3 Flujo síncrono `_run_sync_pipeline` — también respetar `primary_engine`

El flujo síncrono (`main.py:360-382`) también asume local como primario. Hay que aplicar la misma lógica. **Opción:** extraer un helper `_synthesize_with_fallback(text, style_hint)` reutilizable por ambos flujos (síncrono y streaming). Esto reduce duplicación.

```python
def _synthesize_with_fallback(self, text: str, style_hint: str) -> Optional[bytes]:
    """Sintetiza texto completo con cadena según _tts_primary_engine.
    
    Usado por _run_sync_pipeline. Para streaming, usar
    _synthesize_sentence_stream_with_fallback (que llama a este por oración).
    """
    # Misma lógica que _synthesize_one_sentence_with_fallback pero con
    # text + style_hint completos (no style_hint="").
    ...
```

**Nota de diseño:** `_synthesize_one_sentence_with_fallback` podría delegar a `_synthesize_with_fallback(sentence, style_hint="")`. Pero como el flujo streaming siempre pasa `style_hint=""`, se puede dejar separado para evitar overhead. **Decisión:** mantener separados, documentar la diferencia.

### 2.3 Cambios en `config/settings.json`

Agregar sección `tts` (nueva):

```json
"tts": {
  "primary_engine": "local"
}
```

**Migración prod:** `scripts/deploy.ps1` NO copia `config/settings.json`. Agregar manualmente la sección `tts` a prod con `"primary_engine": "local"` (default seguro).

## 3. Contratos estrictos

### 3.1 `VoiceAssistant._tts_primary_engine`

- **Tipo:** `str`
- **Valores:** `"local"` | `"gemini"` | `"azure"`
- **Default:** `"local"` si falta el campo o es inválido.
- **Inmutable:** se setea en `__init__` y no cambia durante la vida de la instancia.

### 3.2 Orden de fallback por `primary_engine`

| `primary_engine` | Orden de intento |
|---|---|
| `"local"` | local → Gemini → Azure |
| `"gemini"` | Gemini → Azure |
| `"azure"` | Azure (sin fallback) |

### 3.3 Comportamiento cuando un motor no está configurado

- `primary_engine="gemini"` pero `GEMINI_API_KEY` no seteada → `_gemini_tts is None` → cae a Azure. Loguear warning en `__init__`.
- `primary_engine="azure"` pero `AZURE_SPEECH_KEY` no seteada → `_azure_tts is None` → todos fallan, helper retorna `None`. Loguear error en `__init__`.
- `primary_engine="local"` pero modelos locales no descargados → `_local_tts.synthesize` levanta `RuntimeError` → cae a Gemini. (Comportamiento actual, sin cambios.)

## 4. Tests requeridos

### 4.1 `tests/test_state_machine.py`

- **Test:** `primary_engine="local"` (default) → helper llama local primero (regresión, ya cubierto).
- **Test:** `primary_engine="gemini"` → helper NO llama `_local_tts.synthesize`, llama `_gemini_tts.synthesize` primero.
- **Test:** `primary_engine="azure"` → helper NO llama local ni Gemini, llama `_azure_tts.synthesize_stream` directamente.
- **Test:** `primary_engine="gemini"` + Gemini no configurado (`_gemini_tts is None`) → cae a Azure.
- **Test:** `primary_engine="azure"` + Azure no configurado → retorna None sin excepción.
- **Test:** `primary_engine` inválido (ej. `"foo"`) → default a `"local"` con warning.
- **Test:** `__init__` con `primary_engine="gemini"` → `_local_tts is None`.

### 4.2 `tests/test_state_machine.py` — flujo síncrono

- **Test:** `_run_sync_pipeline` con `primary_engine="gemini"` → usa Gemini, no local.
- **Test:** `_run_sync_pipeline` con `primary_engine="azure"` → usa Azure streaming, no local ni Gemini.

## 5. Out of scope

- No modificar `GeminiTTSClient`, `AzureTTSClient`, `PiperTTSClient`, `KokoroTTSClient`.
- No modificar `SentenceBuffer` ni `response_parser`.
- No agregar `synthesize_sentence_stream` a Gemini/Azure (no se necesita).
- No modificar el system prompt del agente.

## 6. Dependencias

- Ninguna nueva. `Optional` ya importado en `main.py:15`.
- Helper `_synthesize_one_sentence_with_fallback` ya existe (del fix `fix/streaming-tts-fallback`).

## 7. Verificación

```powershell
.venv\Scripts\pytest tests/test_state_machine.py -v -m unit -k "primary_engine or fallback"
.venv\Scripts\pytest tests/ --cov=main -v -m unit
```

## 8. Notas

- Esta feature surge como deuda técnica del fix `fix/streaming-tts-fallback` (especificada en conversación con el usuario el 2026-07-19).
- El hallazgo MINOR-01 de @security (None-check defensivo en `_synthesize_one_sentence_with_fallback` para `self._local_tts`) queda automáticamente cubierto por esta feature cuando `primary_engine != "local"`, ya que `_local_tts` será `None` por diseño.
- El helper actual ya maneja `_local_tts is None` correctamente (ver `test_fallback_local_none_returns_none` en `tests/test_state_machine.py`).