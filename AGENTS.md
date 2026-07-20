# AGENTS.md — Dev-Cortex

Asistente de voz para Windows. Orquestador Python (`src/main.py`) + servidor OpenCode (`opencode serve`) como "cerebro". Pipeline: `Alt+V` → STT (Whisper local primario, Gemini fallback) → OpenCode → TTS (Piper/Kokoro local primario, Gemini fallback 1, Azure fallback 2).

## Arquitectura

Dos procesos separados que se comunican por HTTP:
- **Orquestador**: `pythonw.exe src\main.py` — hotkey, grabación, STT, TTS, overlay tkinter. Logs en `logs/cortex.log`.
- **Cerebro**: `opencode serve` — agente `asistente_voz` con bash + memoria. Logs en `logs/opencode-wrapper.log`.

`src/main.py` es la máquina de estados (`idle`/`recording`/`processing`/`speaking`) y el entrypoint. `src/handlers/` contiene los handlers del pipeline: `audio_manager`, `whisper_stt_client`, `gemini_stt_client`, `opencode_client`, `piper_tts_client`, `kokoro_tts_client`, `gemini_tts_client`, `azure_tts_client`, `response_parser`, `sentence_buffer`, `overlay`.

`specs/` contiene documentos de diseño por feature/bug. Si trabajás en código relacionado con streaming, Kokoro, overlay, o TTS, revisá los specs relevantes primero.

## Requisito crítico: Python 3.10

`kokoro-onnx` y `piper-tts` no soportan Python 3.13+. **Siempre usar Python 3.10** para el venv:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r tests\requirements-test.txt
```

## Configuración

- **`config/settings.json`** es la fuente de verdad para modelos, voces, timeouts, audio, logging y selector de TTS local (`local.tts_engine`: `"piper"` | `"kokoro"`). NO usar env vars para eso.
- **`tts.primary_engine`** (`"local"` | `"gemini"` | `"azure"`) controla la cadena de fallback TTS. `"local"` → local → Gemini → Azure. `"gemini"` → Gemini → Azure. `"azure"` → solo Azure. Si falta o es inválido, usa `"local"`.
- **`.env`** solo contiene secrets: `GEMINI_API_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `OPENCODE_SERVER_PASSWORD`, `OPENCODE_BASE_URL`. Nunca leer ni commitear `.env`.
- **Puerto del servidor OpenCode**: `OPENCODE_BASE_URL` env var gana. El default en código (`main.py:77`) es `4096`; `.env.example`, README y scripts usan `57214`. Setear `OPENCODE_BASE_URL` explícitamente para evitar ambigüedad.

## Comandos

```powershell
# Instalar deps (siempre dentro del venv Python 3.10)
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r tests\requirements-test.txt

# Levantar cerebro (otra terminal)
$env:OPENCODE_SERVER_PASSWORD = "<pass>"
opencode serve --port 57214 --hostname 127.0.0.1

# Arrancar orquestador (dev)
.venv\Scripts\python src\main.py

# Arranque completo (opencode + orquestador en background)
scripts\start_cortex.bat
scripts\start-dev.bat        # puerto 57215, dir aislado .cortex-dev

# Detener
powershell -ExecutionPolicy Bypass -File scripts\stop-cortex.ps1
```

## Tests

```powershell
pytest tests/ -m unit -v          # sin red, rápidos
pytest tests/ -m e2e -v           # requiere .env + opencode serve levantado
pytest tests/ --cov=handlers --cov=main -v
pytest tests/test_state_machine.py -v   # un solo archivo
```

- **Correr siempre desde la raíz del repo** — `conftest.py` resuelve paths relativos a `config/settings.json`.
- **Imports top-level**: `conftest.py` inserta `src/` en `sys.path`. Los módulos se importan como `from handlers.audio_manager import ...` y `from main import VoiceAssistant`, NO como `src.handlers...`. Mantener este estilo al agregar código.
- **Markers**: `unit` (sin red, mocks), `e2e` (API keys reales + servidor), `integration` (reservado, sin uso actual).
- **E2E auto-skip**: si faltan env vars o el servidor no responde a `/global/health`, los tests E2E hacen `pytest.skip()`. Son seguros en CI.
- **Tests de secrets**: `test_*_no_*_logged` verifican con `caplog` que las API keys no aparezcan en logs. Usan strings sentinel tipo `SECRET_*_DO_NOT_LEAK_777`. Preservar al tocar logging de clientes.
- **Tests desde `git worktree` aislados**: cuando un subagente trabaja en `.worktrees/<rama>/`, el worktree solo contiene archivos trackeados — `.venv/`, `.env` y `models/` están gitignored y NO se copian al worktree. Patrones:
  - **Venv**: NO recrear `.venv` dentro del worktree. Invocar pytest con ruta absoluta al venv del repo principal: `C:\Users\crist\repos\dev-cortex\.venv\Scripts\python.exe -m pytest tests/ -m unit -v` (correr desde el directorio del worktree con `workdir`).
  - **`.env`**: `conftest.py` carga `_PROJECT_ROOT / ".env"` explícitamente, donde `_PROJECT_ROOT` se resuelve desde `__file__` (no desde CWD). Funciona automáticamente desde cualquier worktree — no hace falta symlink ni copiar `.env`.
  - **`config/settings.json`**: la fixture `settings` y el `__init__` de `VoiceAssistant` lo resuelven vía `_PROJECT_ROOT / "config" / "settings.json"`. Funciona desde cualquier worktree.
  - **`models/`** (Whisper/Kokoro/Piper): no se necesitan para unit tests (todo se mockea). Los E2E que requieren modelos reales se auto-skippean si no los encuentran. No hace falta symlink.

## Pipeline de streaming

Cuando `opencode.streaming_enabled` es `true` (default), el pipeline usa streaming end-to-end:
1. `POST /session/:id/prompt_async` (retorna 204 inmediatamente).
2. Stream SSE `GET /event` para recibir tokens en tiempo real.
3. `SentenceBuffer` acumula deltas y emite oraciones completas (split por `. ! ? ;`).
4. Cada oración se sintetiza con Kokoro/Piper y se reproduce vía `play_audio_stream`.

**Filtrado de reasoning**: el cliente trackea `partID → part.type` via `message.part.updated` y solo emite al TTS los deltas de parts tipo `text`. Los tipos `reasoning`, `tool`, `file`, `step-start`, `step-finish`, `compaction`, `subtask` se descartan.

**Fallback automático**: si `prompt_async` falla antes de enviar, cae al flujo síncrono. Si falla después, no se reenvía (el agente ya recibió el comando).

## Contrato de respuesta del agente

El agente `asistente_voz` (definido en el dir de config aislado de OpenCode, NO en este repo) responde con formato `[STYLE: <estilo>] <texto>`. Estilos válidos: `cheerful`, `sad`, `friendly`, `excited`, `calm`, `serious`, `whisper`, `apologetic`, `confident`.

`src/handlers/response_parser.py` parsea ese prefijo y limpia sintaxis markdown del texto. `_strip_markdown` se importa desde `response_parser` en `main.py` para el pipeline de streaming. **Si se cambia el system prompt del agente, validar que siga devolviendo el formato `[STYLE: ...]`** — el parser fallará silenciosamente si el formato cambia.

## Deploy dev → prod

`scripts\deploy.ps1` copia `src\`, `requirements.txt` y `scripts` a `$env:CORTEX_PROD_DIR` (default `C:\Users\crist\voice-assistant`).

- **NO copia `config/settings.json`** — dev usa `logging.level=DEBUG`, prod usa `INFO`. Si el código necesita un nuevo campo de config, agregarlo manualmente a prod.
- **NO copia `.env` ni `.gitignore`**.
- **NO copia modelos locales** (Whisper, Piper, Kokoro) — están en `.gitignore`. Si prod no los tiene, descargarlos manualmente.

## Gotchas

- **Cancelación cooperativa**: `VoiceAssistant._pipeline_generation` es un contador que se incrementa en cada `toggle()`. El pipeline chequea `self._pipeline_generation != generation` en cada etapa y aborta si el usuario interrumpió. Al agregar código al pipeline, mantener estos chequeos.
- **`_send_lock`**: `threading.Lock` que serializa los envíos HTTP a OpenCode. Evita que dos pipelines concurrentes envíen comandos simultáneos. Siempre adquirir antes de `send_command` o `send_command_stream`.
- **`_strip_markdown` es privado pero se importa en `main.py`**: `from handlers.response_parser import _strip_markdown`. Si se renombra o mueve, actualizar `main.py:25`.
- **`start_opencode_hidden.py`** vive en `scripts/` de este repo. Es referenciado por `start_cortex.bat`, `start-dev.bat` y `deploy.ps1`. Usa Python global (no el venv).
- **`logs/`** está en `.gitignore` — no commitear logs ni `comando.wav`.
- **Hotkey real**: `Alt+V` (toggle grabación), definido en `config/settings.json` (`"hotkey": "alt+v"`).
- **Circuit breaker de TTS**: Gemini TTS tiene cooldown configurable (`gemini.tts_circuit_breaker_cooldown_seconds`). Si el TTS local falla, cae a Gemini; si Gemini está en cooldown, cae a Azure streaming.
- **Auto-restart asimétrico**: `start_opencode_hidden.py` reinicia opencode serve automáticamente si crashea (cada 5 segundos). El orquestador Python NO tiene auto-restart — si crashea, hay que reiniciarlo manualmente.
- **Selector de TTS local**: `config/settings.json` → `local.tts_engine` (`"piper"` | `"kokoro"`). Si falta o es inválido, usa `"piper"` por defecto. Kokoro requiere descarga manual de modelos (~300MB), ver README.
- **Modelos Whisper/Kokoro/Piper**: se descargan en `models/` (gitignored). Whisper se auto-descarga; Piper se auto-descarga; Kokoro requiere descarga manual.
