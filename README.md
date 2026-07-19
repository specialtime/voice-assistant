# Dev-Cortex — Asistente de Voz para Windows

Asistente de voz en segundo plano para Windows que interactúa con el sistema operativo mediante lenguaje natural. Arquitectura "Dos Cerebros" con aislamiento estricto: el orquestador Python captura audio, transcribe con Whisper (local) o Gemini (cloud fallback), razona con OpenCode y responde con TTS local (Piper o Kokoro, seleccionable) o Azure/Gemini (cloud fallback).

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│  PROCESO USUARIO (Session 1 — tu desktop)            │
│                                                       │
│  pythonw.exe src\main.py                              │
│    └─ Orquestador (hotkey Alt+V, STT, TTS, overlay)   │
│       Logs: logs/cortex.log                           │
│                                                       │
│  opencode serve --port 57214 --hostname 127.0.0.1    │
│    └─ Cerebro (agente asistente_voz, bash + memoria)  │
│       Logs: logs/opencode-wrapper.log                │
└─────────────────────────────────────────────────────┘
```

**Pipeline de un comando:**
1. **Trigger:** `Alt+V` activa grabación.
2. **STT:** Audio → Whisper local (`small`, GPU) con fallback a Gemini (`gemini-3.1-flash-lite`).
3. **Razonamiento (streaming):** Texto → OpenCode vía `prompt_async` + stream SSE `/event`. Los tokens del agente se reciben en tiempo real sin esperar la respuesta completa.
4. **TTS (streaming por oración):** Los deltas de texto se acumulan en un `SentenceBuffer` que emite oraciones completas. Cada oración se sintetiza con Kokoro/Piper y se reproduce vía `play_audio_stream` en tiempo real. Fallback a flujo síncrono (respuesta completa → sintetizar todo → reproducir) si el streaming falla antes de enviar el prompt.

---

## Estructura del Proyecto

```
dev-cortex/
├── .env.example               # Template de credenciales
├── .gitignore
├── AGENTS.md                  # Guía para agentes que trabajan en el código
├── README.md                  # Este archivo
├── requirements.txt            # Dependencias Python
├── config/
│   └── settings.json           # Configuración de modelos, audio, logging
├── specs/                      # Especificaciones por feature/bug
├── src/                        # Código fuente
│   ├── __init__.py
│   ├── main.py                 # Punto de entrada + máquina de estados
│   └── handlers/               # Handlers del pipeline
│       ├── audio_manager.py     #   Grabación y reproducción de audio
│       ├── whisper_stt_client.py #  STT local (Whisper, GPU, primario)
│       ├── gemini_stt_client.py #   STT cloud (Gemini, fallback)
│       ├── opencode_client.py   #   Cliente del cerebro (OpenCode serve, streaming SSE + síncrono)
│       ├── piper_tts_client.py  #   TTS local (Piper, CPU, seleccionable)
│       ├── kokoro_tts_client.py #   TTS local (Kokoro, CPU, seleccionable, streaming por oración)
│       ├── gemini_tts_client.py #   TTS cloud (Gemini, fallback 1)
│       ├── azure_tts_client.py  #   TTS cloud (Azure, fallback 2 streaming)
│       ├── response_parser.py  #   Parser de respuestas SSML
│       ├── sentence_buffer.py  #   Buffer de oraciones para streaming (acumula deltas SSE)
│       └── overlay.py           #   Overlay visual (chip tkinter)
└── tests/                      # Suite de tests
    ├── conftest.py             # Fixtures compartidas + markers
    ├── requirements-test.txt   # Dependencias de testing
    ├── test_audio_manager.py
    ├── test_e2e_scenarios.py
    ├── test_gemini_stt_client.py
    ├── test_local_integration.py #  Tests de failover chain local→cloud
    ├── test_logging.py
    ├── test_opencode_client.py
    ├── test_opencode_wrapper.py
    ├── test_overlay.py
    ├── test_piper_tts_client.py #  Tests Piper TTS local
    ├── test_kokoro_tts_client.py #  Tests Kokoro TTS local
    ├── test_response_parser.py
    ├── test_state_machine.py
    ├── test_tts_clients.py
    └── test_whisper_stt_client.py # Tests Whisper STT local
```

---

## Requisitos Previos

- **Python 3.10** instalado (requerido por `kokoro-onnx` y `piper-tts`, que no soportan 3.13+).
- **opencode** instalado globalmente.
- **Archivo `.env`** configurado (copiar de `.env.example`).
- **GPU NVIDIA con CUDA** (opcional, para Whisper en GPU — si no hay GPU, cae a CPU automáticamente).

---

## Inicialización

### 1. Clonar repo y crear venv

```powershell
cd dev-cortex

# Crear venv con Python 3.10 (obligatorio — kokoro-onnx/piper-tts no soportan 3.13+)
py -3.10 -m venv .venv

# Instalar dependencias
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r tests\requirements-test.txt
```

### 2. Configurar credenciales

Copiar `.env.example` a `.env` y completar:

```env
GEMINI_API_KEY=<tu_key_de_google_ai_studio>
AZURE_SPEECH_KEY=<tu_key_de_azure_speech>
AZURE_SPEECH_REGION=southamericaeast
OPENCODE_SERVER_PASSWORD=<password_opencode>
OPENCODE_BASE_URL=http://127.0.0.1:57214
```

### 3. Verificar configuración

El archivo `config/settings.json` define modelos, voces, timeouts y logging. Settings clave:

| Campo | Default | Descripción |
|---|---|---|
| `tts.primary_engine` | `"local"` | Cadena de fallback TTS: `"local"` → local → Gemini → Azure; `"gemini"` → Gemini → Azure; `"azure"` → solo Azure |
| `local.tts_engine` | `"piper"` | Motor TTS local: `"piper"` o `"kokoro"` |
| `opencode.streaming_enabled` | `true` | Streaming end-to-end (SSE) |
| `logging.level` | `"DEBUG"` | Nivel de log (prod usa `"INFO"`)

### 4. Modelos locales (STT y TTS)

El pipeline usa modelos locales como **primario** para STT y TTS, con fallback automático a APIs cloud (Gemini/Azure) si los modelos locales fallan.

**STT — Whisper local (faster-whisper):**
- Modelo: `small` (244M params, ~1GB VRAM con `int8_float16`), GPU CUDA (cae a CPU si no hay).
- Se descarga automáticamente en la primera transcripción (~466MB).
- Parámetros clave en `config/settings.json` → `local.whisper`: `vad_filter`, `initial_prompt`, `hotwords`, `condition_on_previous_text`.

**TTS — Piper local (piper-tts):**
- Voz: `es_AR-daniela-high` (114MB, ONNX Runtime, CPU). Se descarga automáticamente.

**TTS — Kokoro local (kokoro-onnx, alternativa):**
- Mayor naturalidad que Piper (MOS 4.3-4.5). Voces: `em_alex` (default), `em_santa`, `ef_dora`.
- Descarga manual (~300MB):
  ```powershell
  mkdir models\kokoro
  curl -L -o models\kokoro\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  curl -L -o models\kokoro\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
  ```
- Activar: `config/settings.json` → `"local": { "tts_engine": "kokoro" }`.

> **Hardware mínimo:** 4GB VRAM para Whisper small en GPU. Piper/Kokoro corren en CPU.

### Streaming TTS (latencia reducida)

El pipeline soporta **streaming end-to-end** (activado por defecto, `opencode.streaming_enabled: true`):

1. `POST /session/:id/prompt_async` → 204 inmediato.
2. Stream SSE `GET /event` → tokens en tiempo real.
3. `SentenceBuffer` acumula deltas y emite oraciones completas (split por `. ! ? ;`).
4. Cada oración se sintetiza con Kokoro/Piper y se reproduce vía `play_audio_stream`.

**Fallback automático:** si `prompt_async` falla antes de enviar, cae al flujo síncrono. Si falla después, no se reenvía (el agente ya recibió el comando).

El cliente filtra deltas de reasoning/tool/compaction para que solo el texto de respuesta llegue al TTS. Ver `specs/feature_streaming_tts_kokoro.md` para el diseño completo.

### 5. Levantar el servidor OpenCode

```powershell
# opencode serve usa el Python global (no necesita venv)
$env:OPENCODE_SERVER_PASSWORD = "<tu_pass>"
opencode serve --port 57214 --hostname 127.0.0.1
```

Verificar salud:

```powershell
curl http://127.0.0.1:57214/global/health
# → {"healthy":true}
```

### 6. Iniciar el orquestador

```powershell
.venv\Scripts\python src\main.py
```

### 7. Verificar funcionamiento

1. Presionar `Alt+V` → debe aparecer chip "● Grabando..." (rojo, pulsante) abajo al centro.
2. Hablar un comando (ej: "abrí Chrome").
3. Presionar `Alt+V` de nuevo → chip cambia a "● Procesando..." (amarillo). Permanece en amarillo durante toda la fase de STT + razonamiento del agente + síntesis TTS.
4. Cuando el audio empieza a sonar realmente → chip cambia a "● Hablando..." (verde). La transición a verde ocurre **solo al primer chunk PCM real** del playback, no antes.
5. Al terminar la respuesta → chip desaparece (vuelve a idle).

> **Nota:** Si el streaming falla y cae al flujo síncrono (fallback), el comportamiento del overlay es equivalente: amarillo durante procesamiento, verde al iniciar playback.

---

## Tests

```powershell
# Tests unitarios (sin red)
pytest tests/ -m unit -v

# Tests E2E (requieren .env + opencode serve levantado)
pytest tests/ -m e2e -v

# Cobertura
pytest tests/ --cov=handlers --cov=main -v

# Un solo archivo
pytest tests/test_state_machine.py -v
```

- **Correr siempre desde la raíz del repo**.
- **Markers:** `unit` (sin red, mocks), `e2e` (API keys reales + servidor), `integration` (reservado).
- **E2E auto-skip:** si faltan env vars o el servidor no responde, los tests E2E hacen `pytest.skip()`. Son seguros en CI.

---

## Troubleshooting

| Problema | Solución |
|---|---|
| El chip no aparece | Ejecutar `python src\main.py` en terminal (con `python.exe`, no `pythonw.exe`) para ver errores en consola. Revisar `logs\cortex.log`. |
| opencode no responde (WinError 10061) | Verificar `opencode serve` levantado y puerto 57214 libre (`netstat -ano \| findstr 57214`). |
| Hotkey no funciona | La librería `keyboard` requiere permisos de administrador para hooks globales. |
| No hay respuesta de voz | Verificar `.env` y `curl http://127.0.0.1:57214/global/health` → `{"healthy":true}`. |
| `cublas64_12.dll is not found` | Las deps de CUDA no están instaladas en el venv. Ejecutá `.venv\Scripts\pip install -r requirements.txt`. |
| `kokoro-onnx`/`piper-tts` no instala (Python 3.13+) | Crear el venv con `py -3.10 -m venv .venv`. |
| Modelo Kokoro no encontrado | Descargar manualmente (ver sección Modelos locales). |

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Orquestador | Python 3.10 (venv) (`keyboard`, `httpx`, `sounddevice`, `numpy`) |
| STT primario | Whisper local (`faster-whisper`, modelo `small`, GPU CUDA, int8) |
| STT fallback | Google AI Studio — `gemini-3.1-flash-lite` (fallback: `gemini-2.5-flash-lite`) |
| Cerebro | OpenCode serve (agente `asistente_voz`, bash + memoria) |
| TTS primario | Local seleccionable: Piper (`piper-tts`, voz `es_AR-daniela-high`) o Kokoro (`kokoro-onnx`, voz `em_alex`), ONNX Runtime, CPU |
| TTS fallback 1 | Gemini (`gemini-3.1-flash-tts-preview`, voz `Charon`) |
| TTS fallback 2 | Azure Cognitive Services (SSML, `es-MX-JorgeNeural`, streaming) |
| Memoria | Plugin `opencode-mem` (BD vectorial local) |
| Overlay | tkinter (chip visual durante grabación/procesamiento) |