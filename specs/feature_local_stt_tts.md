# Spec: STT/TTS Local con Whisper + Piper

## Resumen

Agregar modelos locales como **primario** para STT y TTS, manteniendo Gemini/Azure como fallback en la cadena de failover. Objetivo: reducir dependencia de APIs cloud, eliminar costos recurrentes, y permitir operación offline parcial.

**Hardware target:** 4GB VRAM (GPU), CPU para TTS.

## Decisiones de diseño

| Aspecto | Decisión | Justificación |
|---|---|---|
| STT primario | `faster-whisper` modelo `small` (244M params) | ~2GB VRAM, WER multilingüe ~7%, 4x real-time. Cabe holgado en 4GB. |
| STT fallback | Gemini STT (existente, sin cambios) | Si Whisper local falla (modelo no descargado, GPU OOM, etc.), cae a Gemini cloud. |
| TTS primario | `piper-tts` voz `es_AR-daniela-high` (114MB, ONNX, CPU) | TTS neural local, corre en CPU (no compite por VRAM), voz argentina disponible. |
| TTS fallback 1 | Gemini TTS (existente, sin cambios) | Si Piper falla, cae a Gemini cloud. |
| TTS fallback 2 | Azure TTS streaming (existente, sin cambios) | Si Gemini también falla, streaming de Azure. |

## Arquitectura

### Pipeline STT (nuevo orden de failover)

```
run_pipeline(wav_path)
  └─> WhisperSTTClient.transcribe(wav_path)      [PRIMARIO — local, GPU]
        └─> si falla: GeminiSTTClient.transcribe(wav_path)  [FALLBACK — cloud]
              └─> si falla: RuntimeError("STT falló: local y cloud no respondieron")
```

### Pipeline TTS (nuevo orden de failover)

```
run_pipeline(text, style_hint)
  └─> PiperTTSClient.synthesize(text)           [PRIMARIO — local, CPU]
        └─> si falla: GeminiTTSClient.synthesize(text, style_hint)  [FALLBACK 1 — cloud]
              └─> si falla: AzureTTSClient.synthesize_stream(text, style_hint)  [FALLBACK 2 — cloud streaming]
```

### Configuración en settings.json

Nueva sección `local` en `config/settings.json`:

```json
{
  "local": {
    "whisper": {
      "model": "small",
      "device": "cuda",
      "compute_type": "int8",
      "language": "es",
      "beam_size": 5
    },
    "piper": {
      "voice_model": "es_AR-daniela-high",
      "voices_dir": "models/piper-voices",
      "download_url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0",
      "length_scale": 1.0
    }
  }
}
```

**Notas:**
- `device: "cuda"` — usa GPU (4GB VRAM). Si no hay CUDA, `faster-whisper` cae a CPU automáticamente.
- `compute_type: "int8"` — cuantización para reducir VRAM (~1GB en lugar de ~2GB).
- `voices_dir: "models/piper-voices"` — directorio local donde se descargan las voces de Piper. NO se commitean (agregar a .gitignore).
- `length_scale: 1.0` — controla velocidad de habla Piper (1.0 = normal, <1.0 = más rápido, >1.0 = más lento).

### Dependencias (requirements.txt)

Agregar:
```
faster-whisper==1.0.3
piper-tts==1.4.2
onnxruntime>=1.16
```

**Nota:** `faster-whisper` depende de `ctranslate2` (incluye soporte CUDA). `piper-tts` depende de `onnxruntime`. Ambos se instalan via pip.

### Estructura de archivos

```
src/handlers/
  whisper_stt_client.py    [NUEVO] — WhisperSTTClient
  piper_tts_client.py      [NUEVO] — PiperTTSClient
  gemini_stt_client.py     [SIN CAMBIOS]
  gemini_tts_client.py     [SIN CAMBIOS]
  azure_tts_client.py      [SIN CAMBIOS]

src/main.py                [MODIFICAR] — integrar nuevos clientes en __init__ y run_pipeline

config/settings.json       [MODIFICAR] — agregar sección "local"
requirements.txt            [MODIFICAR] — agregar deps
.gitignore                 [MODIFICAR] — agregar models/piper-voices/

tests/
  test_whisper_stt_client.py   [NUEVO] — tests unitarios WhisperSTTClient
  test_piper_tts_client.py     [NUEVO] — tests unitarios PiperTTSClient
  test_state_machine.py        [MODIFICAR] — actualizar mocks para nuevos clientes
```

---

## Micro-Specs de implementación

### Micro-Spec A: `src/handlers/whisper_stt_client.py`

**Complejidad:** Compleja / Core (involucra GPU, modelo ML, manejo de audio)

**Agente:** `@dev_senior`

**Archivo:** `src/handlers/whisper_stt_client.py` (nuevo)

**Diseño estricto:**

```python
"""Cliente de transcripción de voz (STT) usando faster-whisper (local, GPU).

Implementa WhisperSTTClient que carga un modelo Whisper localmente
(faster-whisper / CTranslate2) y transcribe archivos .wav a texto.
No requiere API key ni conexión a internet (tras descarga inicial del modelo).
"""

import logging
import os
import wave

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperSTTClient:
    """Cliente para transcripción de audio con Whisper local.

    Carga el modelo una vez en __init__ (lazy-load diferido al primer
    transcribe() para no bloquear el startup si no hay GPU).

    Attributes:
        settings: Dict con configuración local.whisper (model, device, compute_type, language, beam_size).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente STT local.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['whisper']).
        """
        self.settings = settings
        self._model: WhisperModel | None = None  # lazy-load

        cfg = settings["local"]["whisper"]
        logger.debug(
            "WhisperSTTClient inicializado — model=%s, device=%s, compute_type=%s",
            cfg["model"], cfg["device"], cfg["compute_type"],
        )

    def _ensure_model_loaded(self) -> None:
        """Carga el modelo Whisper si aún no está cargado (lazy-load)."""
        if self._model is not None:
            return
        cfg = self.settings["local"]["whisper"]
        logger.info("Cargando modelo Whisper local — model=%s, device=%s...", cfg["model"], cfg["device"])
        self._model = WhisperModel(
            cfg["model"],
            device=cfg["device"],
            compute_type=cfg["compute_type"],
        )
        logger.info("Modelo Whisper cargado OK")

    def transcribe(self, wav_path: str) -> str:
        """Transcribe un archivo .wav a texto usando Whisper local.

        Lee el WAV, lo pasa al modelo Whisper y retorna el texto transcrito.
        Aplica limpieza: strip() de espacios. No aplica el stt_prompt de
        Gemini (Whisper tiene su propio manejo de idioma via parámetro language).

        Args:
            wav_path: Ruta absoluta al archivo .wav a transcribir.

        Returns:
            Texto transcrito (str), limpio y sin espacios extra.

        Raises:
            FileNotFoundError: Si wav_path no existe.
            RuntimeError: Si el modelo falla al transcribir.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio no encontrado: {wav_path}")

        self._ensure_model_loaded()

        cfg = self.settings["local"]["whisper"]
        try:
            segments, _info = self._model.transcribe(
                wav_path,
                language=cfg["language"],
                beam_size=cfg["beam_size"],
            )
            # segments es un generator — consumir y concatenar
            text = "".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error("Whisper STT falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Whisper STT falló: {exc}") from exc

        truncated = text[:100] + "..." if len(text) > 100 else text
        logger.debug("Whisper STT OK — texto='%s'", truncated)
        return text
```

**Reglas:**
- NO refactorizar `gemini_stt_client.py`. Es un handler nuevo independiente.
- Lazy-load del modelo en `_ensure_model_loaded()` (no en `__init__`) para no bloquear startup.
- El método `transcribe()` debe tener la misma firma que `GeminiSTTClient.transcribe(wav_path) -> str` para que sean intercambiables en el pipeline.
- NO loguear el contenido completo del audio ni paths sensibles.
- Importar `faster_whisper` top-level (no lazy import) — si falta la dep, falla al importar el módulo, lo cual es el comportamiento esperado.

**Tests unitarios (`tests/test_whisper_stt_client.py`):**

Mockear `faster_whisper.WhisperModel` con `unittest.mock.patch`. Sin GPU, sin red.

Casos a cubrir:
1. `test_transcribe_success` — mock WhisperModel.transcribe retorna segments con texto → retorna texto limpio (stripped).
2. `test_transcribe_lazy_load` — el modelo NO se carga en `__init__`, sí en la 1ra llamada a `transcribe()`.
3. `test_transcribe_file_not_found` — wav_path inexistente → FileNotFoundError.
4. `test_transcribe_model_failure` — mock transcribe lanza excepción → RuntimeError("Whisper STT falló").
5. `test_transcribe_uses_config_language` — verificar que `model.transcribe()` recibe `language="es"` del config.
6. `test_no_secrets_logged` — no hay API keys, pero verificar que paths absolutos del usuario no se loguean en DEBUG.

---

### Micro-Spec B: `src/handlers/piper_tts_client.py`

**Complejidad:** Media (integración simple, librería pip, sin lógica de negocio crítica)

**Agente:** `@dev_senior`

**Archivo:** `src/handlers/piper_tts_client.py` (nuevo)

**Diseño estricto:**

```python
"""Cliente de síntesis de voz usando Piper TTS (local, CPU).

Implementa PiperTTSClient que usa piper-tts (ONNX Runtime) para generar
audio WAV a partir de texto. No requiere API key ni conexión a internet
(tras descarga inicial de la voz).
"""

import logging
import os
import wave
import io
from typing import Iterator

from piper.download import ensure_voice_exists
import piper

logger = logging.getLogger(__name__)


class PiperTTSClient:
    """Cliente para síntesis de voz con Piper local.

    Attributes:
        settings: Dict con configuración local.piper (voice_model, voices_dir, length_scale).
    """

    def __init__(self, settings: dict) -> None:
        """Inicializa el cliente TTS local.

        Args:
            settings: Dict completo de settings.json (usa settings['local']['piper']).
        """
        self.settings = settings
        self._voice: piper.PiperVoice | None = None  # lazy-load

        cfg = settings["local"]["piper"]
        logger.debug(
            "PiperTTSClient inicializado — voice=%s, voices_dir=%s",
            cfg["voice_model"], cfg["voices_dir"],
        )

    def _ensure_voice_loaded(self) -> None:
        """Descarga y carga la voz de Piper si aún no está cargada (lazy-load)."""
        if self._voice is not None:
            return
        cfg = self.settings["local"]["piper"]
        voices_dir = cfg["voices_dir"]
        voice_model = cfg["voice_model"]

        # Construir rutas esperadas
        # Estructura: voices_dir/es/es_AR/<speaker>/<quality>/<voice>.onnx
        # voice_model = "es_AR-daniela-high" → lang=es, locale=es_AR, speaker=daniela, quality=high
        parts = voice_model.split("-")  # ["es_AR", "daniela", "high"]
        lang_locale = parts[0]  # es_AR
        lang = lang_locale.split("_")[0]  # es
        speaker = parts[1]  # daniela
        quality = parts[2]  # high

        onnx_path = os.path.join(voices_dir, lang, lang_locale, speaker, quality, f"{voice_model}.onnx")
        json_path = onnx_path + ".json"

        # Descargar si no existe
        if not os.path.exists(onnx_path):
            logger.info("Descargando voz Piper — voice=%s...", voice_model)
            ensure_voice_exists(
                voices_dir,
                download_url_base=cfg["download_url_base"],
                lang=lang,
                lang_locale=lang_locale,
                speaker=speaker,
                quality=quality,
            )
            logger.info("Voz Piper descargada OK")

        # Cargar modelo
        logger.info("Cargando voz Piper local — voice=%s...", voice_model)
        self._voice = piper.PiperVoice.load(onnx_path, config_path=json_path)
        logger.info("Voz Piper cargada OK")

    def synthesize(self, text: str, style_hint: str = "") -> bytes:
        """Sintetiza texto a voz usando Piper local.

        Genera audio WAV 24kHz mono s16le a partir de texto. El style_hint
        se ignora (Piper no soporta estilos — es TTS neural simple).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado (compatibilidad de firma con GeminiTTSClient).

        Returns:
            Bytes de audio WAV (con cabecera) 24kHz mono s16le.

        Raises:
            RuntimeError: Si la síntesis falla.
        """
        self._ensure_voice_loaded()

        cfg = self.settings["local"]["piper"]
        try:
            # Piper sintetiza a un stream de bytes WAV
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self._voice.synthesize(wav_file, [text], length_scale=cfg["length_scale"])
            wav_bytes = buffer.getvalue()
        except Exception as exc:
            logger.error("Piper TTS falló — %s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"Piper TTS falló: {exc}") from exc

        truncated = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Piper TTS OK — texto='%s', %d bytes WAV", truncated, len(wav_bytes))
        return wav_bytes

    def synthesize_stream(self, text: str, style_hint: str = "") -> Iterator[bytes]:
        """Versión streaming: sintetiza y hace yield de chunks PCM.

        Piper no soporta streaming nativo, así que sintetiza todo y
        divide en chunks de 4096 bytes (compatible con play_audio_stream).

        Args:
            text: Texto limpio a sintetizar.
            style_hint: Ignorado.

        Yields:
            Bytes PCM crudo (sin cabecera WAV) en chunks de hasta 4096 bytes.
        """
        wav_bytes = self.synthesize(text, style_hint)
        # Extraer PCM crudo del WAV (saltar cabecera de 44 bytes)
        # Cabecera WAV estándar = 44 bytes
        pcm_bytes = wav_bytes[44:]
        chunk_size = 4096
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i:i + chunk_size]
```

**Reglas:**
- NO refactorizar `gemini_tts_client.py` ni `azure_tts_client.py`. Handler nuevo.
- Lazy-load de la voz en `_ensure_voice_loaded()`.
- `synthesize()` retorna WAV con cabecera (24kHz mono s16le). El pipeline actual espera PCM crudo de Gemini TTS. **Verificar compatibilidad con `AudioManager.play_audio()`** — si play_audio espera PCM crudo, extraer PCM antes de retornar. **Decisión: `synthesize()` retorna PCM crudo (sin cabecera WAV) para ser compatible con `play_audio(pcm_bytes)` existente.** Ajustar el código: extraer PCM del buffer WAV antes de retornar.
- `synthesize_stream()` hace yield de PCM crudo en chunks (compatible con `play_audio_stream`).
- `style_hint` se acepta pero se ignora (Piper no tiene estilos).
- NO loguear paths absolutos del usuario.

**Corrección al diseño (importante):** `synthesize()` debe retornar **PCM crudo s16le** (sin cabecera WAV) para ser compatible con `AudioManager.play_audio(pcm_bytes)` que usa `np.frombuffer(pcm_bytes, dtype=np.int16)`. El handler debe extraer el PCM del WAV internamente:

```python
# Dentro de synthesize(), después de generar wav_bytes:
# Extraer PCM crudo (saltar cabecera WAV de 44 bytes)
pcm_bytes = wav_bytes[44:]
return pcm_bytes
```

**Tests unitarios (`tests/test_piper_tts_client.py`):**

Mockear `piper.PiperVoice` y `piper.download.ensure_voice_exists`. Sin red, sin disco.

Casos a cubrir:
1. `test_synthesize_success` — mock PiperVoice.synthesize escribe WAV al buffer → retorna PCM crudo (sin cabecera).
2. `test_synthesize_lazy_load` — la voz NO se carga en `__init__`, sí en la 1ra llamada.
3. `test_synthesize_downloads_voice_if_missing` — mock os.path.exists=False → ensure_voice_exists llamado.
4. `test_synthesize_no_download_if_exists` — mock os.path.exists=True → ensure_voice_exists NO llamado.
5. `test_synthesize_failure` — mock PiperVoice.synthesize lanza excepción → RuntimeError("Piper TTS falló").
6. `test_synthesize_stream_chunks` — synthesize_stream() yields chunks de hasta 4096 bytes PCM.
7. `test_style_hint_ignored` — synthesize(text, "cheerful") funciona igual que synthesize(text, "").
8. `test_returns_pcm_not_wav` — el resultado NO empieza con "RIFF" (cabecera WAV), es PCM crudo.

---

### Micro-Spec C: Integración en `src/main.py`

**Complejidad:** Media (cableado, no lógica nueva)

**Agente:** `@dev`

**Archivo:** `src/main.py` (modificar)

**Cambios:**

1. **Imports** (línea ~20-26, agregar):
```python
from handlers.whisper_stt_client import WhisperSTTClient
from handlers.piper_tts_client import PiperTTSClient
```

2. **`__init__`** (líneas 68-86, modificar para crear clientes locales):

Reemplazar la inicialización de STT y TTS. Los clientes locales se inicializan SIEMPRE (no requieren API key). Los cloud quedan como fallback.

```python
# Clientes STT (local primario + cloud fallback)
gemini_key = os.getenv("GEMINI_API_KEY")
azure_key = os.getenv("AZURE_SPEECH_KEY")
azure_region = os.getenv("AZURE_SPEECH_REGION", "southamericaeast")
opencode_password = os.getenv("OPENCODE_SERVER_PASSWORD")
opencode_base_url = os.getenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")

# STT: Whisper local (primario), Gemini (fallback)
self._whisper_stt = WhisperSTTClient(self._settings)
self._stt = GeminiSTTClient(self._settings, gemini_key) if gemini_key else None

# TTS: Piper local (primario), Gemini (fallback 1), Azure (fallback 2)
self._piper_tts = PiperTTSClient(self._settings)
self._gemini_tts = GeminiTTSClient(self._settings, gemini_key) if gemini_key else None
self._azure_tts = AzureTTSClient(self._settings, azure_key, azure_region) if azure_key else None

# OpenCode (sin cambios)
self._opencode = OpenCodeClient(self._settings, opencode_password or "", opencode_base_url) if opencode_base_url else None
```

3. **`run_pipeline`** — paso 1 STT (líneas 177-185, modificar):

```python
# 1. STT — Whisper local (primario) → Gemini (fallback)
if self._pipeline_generation != generation:
    logger.info("Pipeline (gen=%d) cancelado antes de STT", generation)
    return

text = None
try:
    text = self._whisper_stt.transcribe(wav_path)
    logger.debug("STT Whisper OK: %s", text[:100])
except Exception as e:
    logger.warning("Whisper STT falló (%s), intentando Gemini fallback", e)
    if self._stt is None:
        logger.error("Gemini STT no configurado (GEMINI_API_KEY faltante)")
        return
    text = self._stt.transcribe(wav_path)
    logger.debug("STT Gemini fallback OK: %s", text[:100])

if not text:
    logger.error("STT retornó texto vacío")
    return
```

4. **`run_pipeline`** — pasos 5+6 TTS (líneas 219-241, modificar):

```python
# 5+6. TTS — Piper local (primario) → Gemini (fallback 1) → Azure streaming (fallback 2)
pcm_bytes = None
try:
    pcm_bytes = self._piper_tts.synthesize(clean_text, style_hint)
    logger.debug("TTS Piper OK — %d bytes", len(pcm_bytes))
except Exception as e:
    logger.warning("Piper TTS falló (%s), intentando Gemini fallback", e)
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

# 7. Playback (Piper o Gemini — no streaming)
if pcm_bytes:
    self._audio.play_audio(pcm_bytes)
    logger.debug("Playback completado")
```

**Reglas:**
- NO refactorizar la máquina de estados (toggle, states, lock, generation).
- NO refactorizar AudioManager, OpenCodeClient, response_parser, overlay.
- Mantener el patrón de cancelación cooperativa (check `self._pipeline_generation != generation`).
- Los clientes locales (`_whisper_stt`, `_piper_tts`) se inicializan SIEMPRE, sin condicional de API key.
- El failover es secuencial: local → cloud1 → cloud2. Si local funciona, no se llama a cloud.

**Tests (`tests/test_state_machine.py`):**

Modificar la fixture `patched_assistant` para mockear los 2 nuevos clientes:
- Agregar `patch("main.WhisperSTTClient")` y `patch("main.PiperTTSClient")` al context manager.
- Los mocks retornan MagicMocks.
- Actualizar tests de pipeline para reflejar el nuevo orden de failover:
  - `test_pipeline_stt_none_returns` → ahora verifica `_whisper_stt` None (no `_stt`).
  - `test_pipeline_tts_fallback` → ahora Piper falla → Gemini falla → Azure streaming.
  - `test_pipeline_both_tts_none` → ahora los 3 TTS son None.
  - Tests de cancelación: ajustar side_effects para mockear `_whisper_stt.transcribe` en lugar de `_stt.transcribe`.
- Agregar tests nuevos:
  - `test_pipeline_stt_whisper_success` — Whisper OK, Gemini NO se llama.
  - `test_pipeline_stt_whisper_fails_gemini_fallback` — Whisper falla → Gemini OK.
  - `test_pipeline_tts_piper_success` — Piper OK, Gemini y Azure NO se llaman.
  - `test_pipeline_tts_piper_fails_gemini_fallback` — Piper falla → Gemini OK.
  - `test_pipeline_tts_all_fail` — Piper, Gemini y Azure todos fallan → return sin playback.

---

## Flujo de trabajo (Git)

1. Rama `feature/local-stt-tts` creada desde `master`.
2. Worktrees aislados para trabajo paralelo:
   - `.worktrees/whisper-stt/` — WhisperSTTClient + tests
   - `.worktrees/piper-tts/` — PiperTTSClient + tests
3. Merge de ambos worktrees a `feature/local-stt-tts`.
4. Integración en `main.py` sobre `feature/local-stt-tts` (no paralelizable con los handlers).
5. Auditoría `@security` sobre los cambios.
6. Tests `@tester` sobre la rama integrada.
7. Merge a `master` previa confirmación del usuario.

## Setup post-implementación

El usuario debe ejecutar una vez (no automatizado en código):

```powershell
# Instalar nuevas deps
pip install -r requirements.txt

# El modelo Whisper se descarga automáticamente en la 1ra transcribe()
# La voz Piper se descarga automáticamente en la 1ra synthesize()

# Opcional: pre-descargar para evitar latencia en el primer uso
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='int8')"
```

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| GPU sin CUDA → Whisper cae a CPU (lento) | `device="cuda"` con fallback automático de faster-whisper. Log warning si cae a CPU. |
| Modelo Whisper no descargado → latencia en 1er uso | Lazy-load + log info al cargar. Documentar pre-descarga opcional. |
| Voz Piper no descargada → latencia en 1er uso | `ensure_voice_exists()` descarga automáticamente. Log info. |
| Piper calidad < Azure Neural | Es esperado. Azure queda como fallback 2. El usuario eligió local como primario. |
| `faster-whisper` incompatible con Python 3.x | Verificar compatibilidad. faster-whisper 1.0.3 soporta Python 3.8+. |
| `piper-tts` GPL-3.0 vs repo MIT/Apache | Piper es GPL-3.0. Verificar licencia del proyecto. Si es problema, usar edge-tts (MIT) como alternativa. |