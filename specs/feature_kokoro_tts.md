# Spec: TTS Local alternativo con Kokoro (selector Piper/Kokoro)

## Resumen

Agregar **Kokoro-ONNX** como segunda opción de TTS local, seleccionable via `config/settings.json`. Piper se mantiene intacto (sin perder compatibilidad). Un campo `local.tts_engine` decide cuál motor es el primario; el otro queda como código disponible pero inactivo. La cadena de failover cloud (Gemini → Azure) no se modifica.

**Hardware target:** CPU para TTS (igual que Piper). Kokoro también puede usar GPU via `onnxruntime-gpu`, pero el default es CPU para no competir por VRAM con Whisper.

## Decisiones de diseño

| Aspecto | Decisión | Justificación |
|---|---|---|
| Motor TTS local primario | Selector `local.tts_engine: "piper" \| "kokoro"` (default `"piper"` para no romper comportamiento existente) | El usuario elige sin tocar código. Backward-compatible: si el campo falta, default = piper. |
| Motor TTS local alternativo | El no seleccionado queda importado pero NO instanciado | Ahorra RAM y evita cargar modelo innecesario. |
| TTS fallback 1 (cloud) | Gemini TTS (sin cambios) | Si el motor local seleccionado falla, cae a Gemini cloud. |
| TTS fallback 2 (cloud) | Azure TTS streaming (sin cambios) | Si Gemini también falla, streaming de Azure. |
| Librería Kokoro | `kokoro-onnx==0.5.0` (MIT) + modelo `kokoro-v1.0.onnx` (Apache 2.0) | Mejor licencia que Piper (GPL-3.0). ONNX Runtime ya presente en el repo. |
| Voz Kokoro default | `em_alex` (español, masculina, grade implícito) | Única voz masculina española "principal" (la otra es `em_santa`, temática navideña). Consistencia con Piper (`es_AR-daniela-high` es femenina, pero el usuario pidió masculina para Kokoro). |
| Idioma Kokoro default | `"es"` (lang param de `kokoro.create()`) | Coincide con `local.whisper.language: "es"`. |
| Sample rate Kokoro | 24kHz mono s16le (igual que Piper) | Compatibilidad con `AudioManager.play_audio(pcm_bytes)` que usa `np.frombuffer(pcm_bytes, dtype=np.int16)`. Kokoro produce float32 24kHz → hay que convertir a int16 PCM. |
| Descarga de modelos Kokoro | MANUAL (no auto-download) | A diferencia de Piper (`ensure_voice_exists`), Kokoro requiere descargar 2 archivos (`kokoro-v1.0.onnx` ~300MB + `voices-v1.0.bin`) desde GitHub releases. Si faltan, el handler lanza `RuntimeError` con instrucciones. No se implementa auto-download por tamaño y complejidad. |

## Arquitectura

### Pipeline TTS (nuevo orden con selector)

```
run_pipeline(text, style_hint)
  └─> [selector: settings.local.tts_engine]
        ├─ "piper"  → PiperTTSClient.synthesize(text)        [PRIMARIO — local, CPU]
        └─ "kokoro" → KokoroTTSClient.synthesize(text)       [PRIMARIO — local, CPU]
              └─> si falla: GeminiTTSClient.synthesize(text, style_hint)  [FALLBACK 1 — cloud]
                    └─> si falla: AzureTTSClient.synthesize_stream(...)   [FALLBACK 2 — cloud streaming]
```

**Nota:** Solo se instancia el cliente seleccionado. El otro NO se inicializa (ahorra memoria y evita cargar modelo).

### Configuración en settings.json

Nuevos campos en sección `local`:

```json
{
  "local": {
    "tts_engine": "piper",
    "whisper": { ... sin cambios ... },
    "piper": { ... sin cambios ... },
    "kokoro": {
      "model_path": "models/kokoro/kokoro-v1.0.onnx",
      "voices_path": "models/kokoro/voices-v1.0.bin",
      "voice": "em_alex",
      "lang": "es",
      "speed": 1.0
    }
  }
}
```

**Notas:**
- `tts_engine: "piper"` — default para mantener comportamiento existente. Valores válidos: `"piper"`, `"kokoro"`. Si el campo falta o es inválido, se loguea warning y se cae a `"piper"`.
- `model_path` / `voices_path` — rutas a los archivos de modelo Kokoro. NO se commitean (agregar `models/kokoro/` a `.gitignore`). El usuario debe descargarlos manualmente (ver README).
- `voice: "em_alex"` — voz masculina española. Alternativas: `em_santa` (español masc), `ef_dora` (español fem).
- `lang: "es"` — código de idioma para `kokoro.create()`. Kokoro usa códigos espeak-ng: `"es"`, `"en-us"`, etc.
- `speed: 1.0` — controla velocidad (1.0 = normal, <1.0 = más lento, >1.0 = más rápido). Equivalente a `length_scale` de Piper pero invertido (Piper: >1.0 = más lento).

### Dependencias (requirements.txt)

Agregar:
```
kokoro-onnx==0.5.0
misaki==0.2.0
```

**Notas:**
- `kokoro-onnx` requiere `onnxruntime` (ya presente en el repo, v1.27.0).
- `misaki` es el paquete G2P recomendado por kokoro-onnx v1.0+ para grafema-a-fonema. Se importa automáticamente.
- `kokoro-onnx` requiere Python <3.14, >=3.10. Mismo constraint que el entorno de tests actual (que ya stubea `piper` por incompatibilidad de wheels en 3.14).
- `soundfile` es dependencia de kokoro-onnx pero NO se usa directamente en el handler (Kokoro retorna `np.ndarray` float32, no escribe WAV a disco).

### Estructura de archivos

```
src/handlers/
  kokoro_tts_client.py    [NUEVO] — KokoroTTSClient
  piper_tts_client.py     [SIN CAMBIOS]
  whisper_stt_client.py   [SIN CAMBIOS]
  gemini_tts_client.py    [SIN CAMBIOS]
  azure_tts_client.py     [SIN CAMBIOS]

src/main.py                [MODIFICAR] — selector de motor TTS en __init__ y run_pipeline

config/settings.json       [MODIFICAR] — agregar "tts_engine" y sección "kokoro"
requirements.txt            [MODIFICAR] — agregar kokoro-onnx + misaki
.gitignore                 [MODIFICAR] — agregar models/kokoro/

tests/
  test_kokoro_tts_client.py    [NUEVO] — tests unitarios KokoroTTSClient
  test_local_integration.py    [MODIFICAR] — stub de kokoro_onnx + tests de selector
  test_state_machine.py         [MODIFICAR] — mockear KokoroTTSClient cuando tts_engine=kokoro
```

---

## Micro-Specs de implementación

### Micro-Spec A: `src/handlers/kokoro_tts_client.py`

**Complejidad:** Media (integración simple, librería pip, sin lógica de negocio crítica, pero maneja conversión float32→int16)

**Agente:** `@dev_senior`

**Archivo:** `src/handlers/kokoro_tts_client.py` (nuevo)

**Diseño estricto:**

```python
"""Cliente de síntesis de voz usando Kokoro-ONNX (local, CPU).

Implementa KokoroTTSClient que usa kokoro-onnx para generar audio PCM
a partir de texto. No requiere API key ni conexión a internet (tras
descarga manual de los archivos de modelo).
"""

import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

# Kokoro produce float32 24kHz. AudioManager.play_audio espera PCM crudo
# s16le (int16 little-endian). Hay que convertir.
_SAMPLE_RATE = 24000  # Kokoro siempre produce 24kHz


class KokoroTTSClient:
    """Cliente para síntesis de voz con Kokoro local.

    Attributes:
        settings: Dict con configuración local.kokoro (model_path, voices_path, voice, lang, speed).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente TTS local Kokoro.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['kokoro']).
        """
        self.settings = settings
        self._kokoro: Kokoro | None = None  # lazy-load

        cfg = settings["local"]["kokoro"]
        logger.debug(
            "KokoroTTSClient inicializado — voice=%s, lang=%s, model=%s",
            cfg["voice"], cfg["lang"], os.path.basename(cfg["model_path"]),
        )

    def _ensure_model_loaded(self) -> None:
        """Carga el modelo Kokoro si aún no está cargado (lazy-load).

        A diferencia de Piper, NO hay auto-download. Si los archivos
        model_path o voices_path no existen, lanza RuntimeError con
        instrucciones de descarga.

        Raises:
            RuntimeError: Si los archivos de modelo no existen.
        """
        if self._kokoro is not None:
            return
        cfg = self.settings["local"]["kokoro"]
        model_path = Path(cfg["model_path"])
        voices_path = Path(cfg["voices_path"])

        if not model_path.exists():
            raise RuntimeError(
                f"Modelo Kokoro no encontrado en {model_path}. "
                f"Descargar kokoro-v1.0.onnx desde "
                f"https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )
        if not voices_path.exists():
            raise RuntimeError(
                f"Voces Kokoro no encontradas en {voices_path}. "
                f"Descargar voices-v1.0.bin desde "
                f"https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )

        logger.info("Cargando modelo Kokoro local — model=%s...", model_path.name)
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        logger.info("Modelo Kokoro cargado OK")

    def synthesize(self, text: str, style_hint: str = "") -> bytes:
        """Sintetiza texto a voz usando Kokoro local.

        Genera audio PCM crudo 24kHz mono s16le a partir de texto.
        El style_hint se ignora (Kokoro no soporta estilos SSML).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado (compatibilidad de firma con GeminiTTSClient).

        Returns:
            Bytes PCM crudo s16le (sin cabecera WAV) — compatible con AudioManager.play_audio().

        Raises:
            RuntimeError: Si la síntesis falla o el modelo no está descargado.
        """
        self._ensure_model_loaded()

        cfg = self.settings["local"]["kokoro"]
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=cfg["voice"],
                speed=cfg["speed"],
                lang=cfg["lang"],
            )
        except Exception as exc:
            logger.error("Kokoro TTS falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Kokoro TTS falló: {exc}") from exc

        # Kokoro retorna np.ndarray float32 en [-1.0, 1.0].
        # Convertir a int16 PCM little-endian para AudioManager.play_audio().
        samples_clipped = np.clip(samples, -1.0, 1.0)
        samples_int16 = (samples_clipped * 32767).astype(np.int16)
        pcm_bytes = samples_int16.tobytes()  # little-endian en x86/x64

        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Kokoro TTS OK — texto='%s', %d bytes PCM", truncated, len(pcm_bytes))
        return pcm_bytes

    def synthesize_stream(self, text: str, style_hint: str = "") -> Iterator[bytes]:
        """Versión streaming: sintetiza y hace yield de chunks PCM.

        Kokoro no soporta streaming nativo, así que sintetiza todo y
        divide en chunks de 4096 bytes (compatible con play_audio_stream).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado.

        Yields:
            Bytes PCM crudo (sin cabecera WAV) en chunks de hasta 4096 bytes.
        """
        pcm_bytes = self.synthesize(text, style_hint)
        chunk_size = 4096
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i:i + chunk_size]
```

**Reglas:**
- NO refactorizar `piper_tts_client.py`, `gemini_tts_client.py` ni `azure_tts_client.py`. Handler nuevo independiente.
- Lazy-load del modelo en `_ensure_model_loaded()` (no en `__init__`) para no bloquear startup.
- `synthesize()` retorna PCM crudo s16le (sin cabecera WAV) — mismo contrato que `PiperTTSClient.synthesize()`.
- `synthesize_stream()` hace yield de PCM crudo en chunks (compatible con `play_audio_stream`).
- `style_hint` se acepta pero se ignora (Kokoro no tiene estilos).
- NO loguear paths absolutos del usuario (usar `os.path.basename` o `Path.name` en logs).
- NO auto-download de modelos (distinto a Piper). Lanzar RuntimeError con URL de descarga si faltan.
- Importar `kokoro_onnx` top-level (no lazy import) — si falta la dep, falla al importar el módulo, lo cual es el comportamiento esperado y consistente con Piper.
- La conversión float32→int16 debe clipar a [-1.0, 1.0] antes de escalar para evitar overflow.

**Tests unitarios (`tests/test_kokoro_tts_client.py`):**

Mockear `kokoro_onnx.Kokoro` con `unittest.mock.patch`. Sin red, sin disco, sin modelo real.

Casos a cubrir:
1. `test_synthesize_success` — mock Kokoro.create retorna (np.array float32, 24000) → retorna PCM crudo int16 (2 bytes por sample).
2. `test_synthesize_lazy_load` — el modelo NO se carga en `__init__`, sí en la 1ra llamada a `synthesize()`.
3. `test_synthesize_model_not_found` — mock Path.exists=False para model_path → RuntimeError con mensaje de descarga.
4. `test_synthesize_voices_not_found` — mock Path.exists=True para model, False para voices → RuntimeError con mensaje de descarga.
5. `test_synthesize_failure` — mock Kokoro.create lanza excepción → RuntimeError("Kokoro TTS falló").
6. `test_synthesize_stream_chunks` — synthesize_stream() yields chunks de hasta 4096 bytes PCM.
7. `test_style_hint_ignored` — synthesize(text, "cheerful") funciona igual que synthesize(text, "").
8. `test_returns_pcm_not_wav` — el resultado NO empieza con "RIFF" (cabecera WAV), es PCM crudo.
9. `test_float32_to_int16_conversion` — mock Kokoro.create retorna array con valores fuera de [-1,1] → el resultado se clipa correctamente (no overflow).
10. `test_no_secrets_logged` — paths absolutos del usuario NO aparecen en logs (usar sentinel `SECRET_USER_DO_NOT_LEAK_999` en model_path).

---

### Micro-Spec B: Integración en `src/main.py`

**Complejidad:** Media (cableado del selector, no lógica nueva)

**Agente:** `@dev`

**Archivo:** `src/main.py` (modificar)

**Cambios:**

1. **Imports** (línea ~28, agregar después del import de PiperTTSClient):
```python
from handlers.kokoro_tts_client import KokoroTTSClient
```

2. **`__init__`** (líneas 81-84, modificar para instanciar solo el motor seleccionado):

Reemplazar el bloque de inicialización de TTS local:

```python
# TTS local: selector de motor (piper | kokoro)
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

# TTS cloud fallback (sin cambios)
self._gemini_tts = GeminiTTSClient(self._settings, gemini_key) if gemini_key else None
self._azure_tts = AzureTTSClient(self._settings, azure_key, azure_region) if azure_key else None
```

**Notas:**
- `self._piper_tts` se renombra a `self._local_tts` (atributo polimórfico). Esto afecta a `run_pipeline` y a los tests que referencian `_piper_tts`.
- NO se mantienen ambos clientes instanciados — solo el seleccionado.

3. **`run_pipeline`** — paso 5+6 TTS (líneas 239-261, modificar referencias de `self._piper_tts` a `self._local_tts`):

```python
# 5+6. TTS — local (selector) → Gemini (fallback 1) → Azure streaming (fallback 2)
pcm_bytes = None
try:
    pcm_bytes = self._local_tts.synthesize(clean_text, style_hint)
    logger.debug("TTS local OK — %d bytes", len(pcm_bytes))
except Exception as e:
    logger.warning("TTS local falló (%s), intentando Gemini fallback", e)
    try:
        if self._gemini_tts is None:
            raise RuntimeError("Gemini TTS no configurado")
        if not self._gemini_tts.is_available():
            raise RuntimeError("Gemini TTS circuit breaker abierto")
        pcm_bytes = self._gemini_tts.synthesize(clean_text, style_hint)
        logger.debug("TTS Gemini fallback OK — %d bytes", len(pcm_bytes))
    except Exception as e2:
        logger.warning("Gemini TTS falló (%s), intentando Azure streaming fallback", e2)
        if self._azure_tts is None:
            logger.error("Azure TTS no configurado (AZURE_SPEECH_KEY faltante)")
            return
        self._audio.play_audio_stream(self._azure_tts.synthesize_stream(clean_text, style_hint))
        logger.debug("TTS Azure streaming OK")
        # pcm_bytes se queda en None → paso 7 se salta (ya reproducido)

# 7. Playback (local o Gemini — no streaming)
if pcm_bytes:
    self._audio.play_audio(pcm_bytes)
    logger.debug("Playback completado")
```

**Reglas:**
- NO refactorizar la máquina de estados (toggle, states, lock, generation).
- NO refactorizar AudioManager, OpenCodeClient, response_parser, overlay.
- Mantener el patrón de cancelación cooperativa (check `self._pipeline_generation != generation`).
- El atributo `self._piper_tts` se RENOMBRA a `self._local_tts` en TODO el archivo. Usar find-and-replace con cuidado.
- El failover es secuencial: local → cloud1 → cloud2. Si local funciona, no se llama a cloud.
- El selector lee `settings["local"]["tts_engine"]` con default `"piper"` si falta el campo (backward-compatible).

---

### Micro-Spec C: Configuración y dependencias

**Complejidad:** Simple / Trivial

**Agente:** `@dev`

**Archivos a modificar:**

1. **`config/settings.json`** — agregar campo `tts_engine` y sección `kokoro` dentro de `local`:

```json
"local": {
    "tts_engine": "piper",
    "whisper": { ... sin cambios ... },
    "piper": { ... sin cambios ... },
    "kokoro": {
      "model_path": "models/kokoro/kokoro-v1.0.onnx",
      "voices_path": "models/kokoro/voices-v1.0.bin",
      "voice": "em_alex",
      "lang": "es",
      "speed": 1.0
    }
}
```

2. **`requirements.txt`** — agregar al final:
```
kokoro-onnx==0.5.0
misaki==0.2.0
```

3. **`.gitignore`** — agregar línea después de `models/piper-voices/`:
```
models/kokoro/
```

**Reglas:**
- NO modificar las secciones existentes de settings.json (whisper, piper, audio, hotkey, logging, etc.).
- Mantener el JSON válido (sin trailing commas).
- El default `tts_engine: "piper"` preserva el comportamiento actual.

---

### Micro-Spec D: Tests de integración

**Complejidad:** Media (actualizar fixtures y mocks existentes)

**Agente:** `@tester`

**Archivos a modificar/crear:**

1. **`tests/test_local_integration.py`** (modificar):
   - Agregar stub de `kokoro_onnx` en `_ensure_stub_modules()` (igual que piper):
     ```python
     if "kokoro_onnx" not in sys.modules:
         kokoro_stub = types.ModuleType("kokoro_onnx")
         kokoro_stub.Kokoro = MagicMock(name="Kokoro")
         sys.modules["kokoro_onnx"] = kokoro_stub
     ```
   - Agregar `numpy` import (ya disponible en el entorno de tests).
   - En fixture `local_settings`, agregar campo `tts_engine: "piper"` (default) y sección `kokoro` completa.
   - En fixture `patched_assistant`, mockear `KokoroTTSClient` igual que `PiperTTSClient`:
     ```python
     mp.setattr("main.KokoroTTSClient", MagicMock(name="KokoroTTSClient"))
     ```
   - Renombrar referencias `_piper_tts` a `_local_tts` en TODOS los tests existentes (los tests actuales usan `patched_assistant._piper_tts.synthesize...` → cambiar a `patched_assistant._local_tts.synthesize...`).
   - Agregar tests nuevos en clase `TestTTSEngineSelector`:
     - `test_selector_piper_default` — `tts_engine` ausente o `"piper"` → `_local_tts` es instancia de PiperTTSClient mock, KokoroTTSClient NO se instancia.
     - `test_selector_kokoro` — `tts_engine: "kokoro"` → `_local_tts` es instancia de KokoroTTSClient mock, PiperTTSClient NO se instancia.
     - `test_selector_invalid_falls_back_to_piper` — `tts_engine: "invalid"` → warning logueado, cae a Piper.
     - `test_tts_kokoro_primary_no_cloud_called` — Kokoro OK → Gemini y Azure NO se invocan.
     - `test_tts_kokoro_fails_gemini_fallback` — Kokoro falla → Gemini OK.
     - `test_tts_kokoro_gemini_fail_azure_streaming` — Kokoro y Gemini fallan → Azure streaming.

2. **`tests/test_state_machine.py`** (modificar):
   - Si la fixture `patched_assistant` referencia `PiperTTSClient`, agregar mock de `KokoroTTSClient` también.
   - Renombrar `_piper_tts` a `_local_tts` en aserciones si las hay.

**Reglas:**
- NO eliminar tests existentes — solo renombrar referencias `_piper_tts` → `_local_tts`.
- Los tests de selector deben verificar que SOLO el motor seleccionado se instancia (el otro mock no se llama).
- Mantener marker `@pytest.mark.unit` en todos los tests nuevos.
- NO usar red ni disco real.

---

## Flujo de trabajo (Git)

1. Rama `feature/kokoro-tts` ya creada desde `master`.
2. Trabajo SECUENCIAL (no paralelizable por dependencias):
   - Paso 1: `@dev_senior` implementa `kokoro_tts_client.py` + `test_kokoro_tts_client.py`.
   - Paso 2: `@dev` integra selector en `main.py` + settings + requirements + .gitignore.
   - Paso 3: `@tester` actualiza tests de integración.
3. Auditoría `@security` sobre todos los cambios.
4. Tests `@tester` sobre la rama integrada (suite completa).
5. Merge a `master` previa confirmación del usuario.

**Nota sobre paralelización:** Los pasos 1 y 2 NO son paralelizables porque `main.py` importa `KokoroTTSClient` (debe existir el archivo). El paso 3 depende de 1 y 2. Se hace secuencial.

## Setup post-implementación

El usuario debe ejecutar una vez (no automatizado en código):

```powershell
# Instalar nuevas deps
pip install -r requirements.txt

# Descargar modelo Kokoro (manual, ~300MB + voices)
mkdir models\kokoro
curl -L -o models\kokoro\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models\kokoro\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Cambiar el motor en config/settings.json:
# "local": { "tts_engine": "kokoro", ... }

# Arrancar el orquestador (igual que antes)
python src\main.py
```

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Modelo Kokoro no descargado → RuntimeError en 1er uso | Mensaje de error con URL de descarga. Documentar en README. NO hay auto-download (300MB es demasiado para auto). |
| `kokoro-onnx` incompatible con Python 3.14 | Mismo constraint que piper. Los tests ya stubean piper por esto. Stubear kokoro_onnx igual. |
| `misaki` (G2P) falla en español | Kokoro usa espeak-ng fallback para español. Si misaki falla, kokoro-onnx cae a espeak. Verificar en tests E2E. |
| Conversión float32→int16 con overflow | Clipar a [-1.0, 1.0] antes de escalar por 32767. Test dedicado `test_float32_to_int16_conversion`. |
| `kokoro-onnx` GPL vs MIT | kokoro-onnx es MIT, modelo es Apache 2.0. Sin problema de licencia (mejor que Piper GPL-3.0). |
| Renombrar `_piper_tts` → `_local_tts` rompe tests | Micro-Spec D cubre la migración de referencias en tests. |
| Voz `em_alex` calidad desconocida | Es la única voz masc española "normal" (la otra es `em_santa`). Si suena mal, el usuario puede cambiar a `ef_dora` (fem) o probar voces inglés con lang="en-us". Documentar alternativas en README. |