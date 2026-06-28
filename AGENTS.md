# AGENTS.md — Dev-Cortex

Asistente de voz para Windows. Orquestador Python (`src/main.py`) + servidor OpenCode (`opencode serve`) como "cerebro". Pipeline: `Alt+V` → STT (Gemini) → OpenCode → TTS (Azure primario, Gemini fallback).

## Arquitectura

Dos procesos separados que se comunican por HTTP:
- **Orquestador**: `pythonw.exe src\main.py` — hotkey, grabación, STT, TTS, overlay tkinter. Logs en `logs/cortex.log`.
- **Cerebro**: `opencode serve` — agente `asistente_voz` con bash + memoria. Logs en `logs/opencode-wrapper.log`.

`src/main.py` es la máquina de estados (`idle`/`recording`/`processing`/`speaking`) y el entrypoint. `src/handlers/` contiene las 6 etapas del pipeline (`audio_manager`, `gemini_stt_client`, `opencode_client`, `azure_tts_client`, `gemini_tts_client`, `response_parser`, `overlay`).

## Configuración

- **`config/settings.json`** es la fuente de verdad para modelos, voces, timeouts, audio y logging. NO usar env vars para eso.
- **`.env`** solo contiene secrets: `GEMINI_API_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `OPENCODE_SERVER_PASSWORD`, `OPENCODE_BASE_URL`. Nunca leer ni commitear `.env`.
- **Puerto del servidor OpenCode**: `OPENCODE_BASE_URL` env var gana. Defaults conflictivos en el repo: el código y los tests usan `4096`; `.env.example` y el README dicen `57214`. Setear `OPENCODE_BASE_URL` explícitamente para evitar ambigüedad.

## Comandos

```powershell
# Instalar deps
pip install -r requirements.txt
pip install -r tests\requirements-test.txt   # testing

# Levantar cerebro (otra terminal)
$env:OPENCODE_SERVER_PASSWORD = "<pass>"
opencode serve --port 57214 --hostname 127.0.0.1

# Arrancar orquestador (dev)
python src\main.py

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

## Deploy dev → prod

`scripts\deploy.ps1` copia `src\`, `requirements.txt` y scripts a `$env:CORTEX_PROD_DIR` (default `C:\Users\crist\voice-assistant`).

- **NO copia `config\settings.json`** — dev usa `logging.level=DEBUG`, prod usa `INFO`. Si el código necesita un nuevo campo de config, agregarlo manualmente a prod.
- **NO copia `.env` ni `.gitignore`**.

## Contrato SSML del agente

El agente `asistente_voz` (definido en el dir de config aislado de OpenCode, NO en este repo) tiene un system prompt que **obliga** a responder con un bloque SSML válido para Azure TTS, usando `<mstts:express-as style="...">` con estilos apropiados (`cheerful`, `sad`, `friendly`). **No escribe texto fuera del bloque XML/SSML.**

`src/handlers/response_parser.py` parsea ese SSML para alimentar Azure TTS. **Si se cambia el system prompt del agente, validar que siga devolviendo SSML válido** — el parser fallará silenciosamente o producirá audio vacío si el formato cambia.

## Gotchas

- **`start_opencode_hidden.py`** es referenciado por `scripts\start_cortex.bat`, `start-dev.bat` y `deploy.ps1`, pero no vive en este repo (está en el dir de prod). Los scripts fallan si no existe en el destino.
- **`logs/`** está en `.gitignore` — no commitear logs ni `comando.wav`.
- **Hotkey real**: `Alt+V` (toggle grabación), definido en `config/settings.json` (`"hotkey": "alt+v"`).
- **Circuit breaker de TTS**: Gemini TTS tiene cooldown configurable (`gemini.tts_circuit_breaker_cooldown_seconds`). Si Azure falla, cae a Gemini; si Gemini también falla, se silencia.
- **Auto-restart asimétrico**: el wrapper `start_opencode_hidden.py` reinicia opencode serve automáticamente si crashea (cada 5 segundos). El orquestador Python NO tiene auto-restart — si crashea, hay que reiniciarlo manualmente o reiniciar sesión.