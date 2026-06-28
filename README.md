# Dev-Cortex — Asistente de Voz para Windows

Asistente de voz en segundo plano para Windows que interactúa con el sistema operativo mediante lenguaje natural. Arquitectura "Dos Cerebros" con aislamiento estricto: el orquestador Python captura audio, transcribe con Gemini, razona con OpenCode y responde con TTS (Azure/Gemini).

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
2. **STT:** Audio → Gemini (`gemini-3.1-flash-lite` con fallback).
3. **Razonamiento:** Texto → OpenCode (agente `asistente_voz`).
4. **TTS:** Respuesta SSML → Azure/Gemini → altavoces.

**Por qué proceso usuario (no servicio):** un asistente de voz necesita desktop (overlay tkinter), audio (micrófono/altavoces) y hotkey global. Los servicios de Windows corren en Session 0 (aislada, sin desktop desde Vista) — no pueden mostrar ventanas ni lanzar programas visibles. Correr todo en Session 1 es más simple y funciona correctamente.

---

## Estructura del Proyecto

```
dev-cortex/
├── .env / .env.example        # Credenciales (NO commitear .env)
├── .gitignore
├── AGENTS.md                  # Guía para agentes que trabajan en el código
├── README.md                  # Este archivo
├── requirements.txt            # Dependencias Python
├── config/
│   └── settings.json           # Configuración de modelos, audio, logging
├── scripts/                    # Scripts de operaciones
│   ├── start_cortex.bat        # Arranque completo (opencode + orquestador)
│   ├── start-dev.bat           # Arranque en modo desarrollo
│   ├── stop-cortex.ps1         # Detener procesos
│   ├── deploy.ps1              # Deploy / instalación
│   └── cleanup_voz_sessions.py # Limpieza de sesiones de voz
├── specs/                      # Especificaciones por feature/bug
├── src/                        # Código fuente
│   ├── __init__.py
│   ├── main.py                 # Punto de entrada + máquina de estados
│   └── handlers/               # Handlers del pipeline
│       ├── audio_manager.py     #   Grabación y reproducción de audio
│       ├── gemini_stt_client.py #   STT (Speech-to-Text) con Gemini
│       ├── opencode_client.py   #   Cliente del cerebro (OpenCode serve)
│       ├── gemini_tts_client.py #   TTS con Gemini (fallback)
│       ├── azure_tts_client.py  #   TTS con Azure Cognitive Services
│       ├── response_parser.py  #   Parser de respuestas SSML
│       └── overlay.py           #   Overlay visual (chip tkinter)
└── tests/                      # Suite de tests
    ├── conftest.py             # Fixtures compartidas + markers
    ├── requirements-test.txt   # Dependencias de testing
    ├── test_audio_manager.py
    ├── test_e2e_scenarios.py
    ├── test_gemini_stt_client.py
    ├── test_logging.py
    ├── test_opencode_client.py
    ├── test_opencode_wrapper.py
    ├── test_overlay.py
    ├── test_response_parser.py
    ├── test_state_machine.py
    └── test_tts_clients.py
```

---

## Requisitos Previos

- **Python 3.10+** en el PATH (`python --version`). Verificar `pythonw.exe` (`where pythonw.exe`).
- **opencode** instalado (`C:\Users\<usuario>\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe`).
- **LM Studio** con modelo **Qwen3.5-2b** cargado (para opencode-mem).
- **Archivo `.env`** configurado (ver sección de configuración).

---

## Inicialización

### 1. Clonar e instalar dependencias

```powershell
cd C:\Users\<usuario>\repos\dev-cortex
pip install -r requirements.txt
pip install -r tests\requirements-test.txt   # solo para tests
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

> **⚠️ NUNCA commitear `.env`.** Está en `.gitignore`.

### 3. Verificar configuración

El archivo `config/settings.json` define modelos, voces, timeouts y logging. Revisar que los valores sean correctos para tu entorno.

### 4. Levantar el servidor OpenCode

```powershell
$env:OPENCODE_SERVER_PASSWORD = "<tu_pass>"
opencode serve --port 57214 --hostname 127.0.0.1
```

Verificar salud:

```powershell
curl http://127.0.0.1:57214/global/health
# → {"healthy":true}
```

### 5. Iniciar el orquestador

```powershell
python src\main.py
```

O usar el script de arranque completo:

```powershell
scripts\start_cortex.bat
```

### 6. Verificar funcionamiento

1. Presionar `Alt+V` → debe aparecer chip "● Grabando..." abajo al centro.
2. Hablar un comando (ej: "abrí Chrome").
3. Presionar `Alt+V` de nuevo → chip cambia a "● Procesando...".
4. Escuchar la respuesta de voz.

---

## Uso

| Acción | Comando |
|---|---|
| Iniciar todo (opencode + orquestador) | `scripts\start_cortex.bat` |
| Iniciar en modo desarrollo | `scripts\start-dev.bat` |
| Detener procesos | `powershell -ExecutionPolicy Bypass -File scripts\stop-cortex.ps1` |
| Detener manualmente | `taskkill /f /im pythonw.exe` |
| Activar grabación | `Alt+V` (toggle) |

---

## Autoarranque (inicio automático con Task Scheduler)

Configura el Programador de Tareas para que ejecute `start_cortex.bat` al iniciar sesión. El bat lanza automáticamente:
1. `pythonw.exe start_opencode_hidden.py` — opencode serve en background (invisible, con auto-restart).
2. `pythonw.exe src\main.py` — orquestador con overlay visual.

### Crear la tarea

1. Presionar `Win + R`, escribir `taskschd.msc` y presionar Enter.
2. Hacer clic en **"Crear tarea..."** en el panel derecho.

**Pestaña "General":**
- **Nombre:** `Cortex`
- ✅ **Ejecutar solo cuando el usuario haya iniciado sesión**
- ✅ **Ejecutar con los privilegios más altos** (necesario para el hotkey `keyboard`)

**Pestaña "Desencadenadores:**
- **Iniciar la tarea:** `Al iniciar sesión` (At log on)
- **Estado:** `Habilitado`

**Pestaña "Acciones":**
- **Acción:** `Iniciar un programa`
- **Programa o script:** ruta a `start_cortex.bat`
- **Iniciar en:** ruta al directorio del proyecto

**Pestaña "Condiciones":**
- Desmarcar **"Iniciar la tarea solo si el equipo está conectado a la alimentación de CA"** (para notebooks).

3. Guardar (puede pedir contraseña).

### Verificación

1. **Reiniciar la PC** (o cerrar sesión y volver a iniciar).
2. **Esperar 10 segundos** después de iniciar sesión.
3. **Presionar `Alt+V`:** debe aparecer el chip "● Grabando...".
4. **Verificar procesos:** `tasklist /FI "IMAGENAME eq pythonw.exe"` debe mostrar 2 procesos pythonw.exe.
5. **Verificar logs:** `logs\cortex.log` y `logs\opencode-wrapper.log` deben tener entradas.

### Desinstalación

1. Abrir Programador de Tareas → buscar "Cortex" → clic derecho → "Eliminar".
2. `taskkill /f /im pythonw.exe` para detener procesos.

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

- **Correr siempre desde la raíz del repo** — `conftest.py` resuelve paths relativos a `config/settings.json`.
- **Markers:** `unit` (sin red, mocks), `e2e` (API keys reales + servidor), `integration` (reservado).
- **E2E auto-skip:** si faltan env vars o el servidor no responde, los tests E2E hacen `pytest.skip()`. Son seguros en CI.

Detalle técnico de la suite (cobertura por módulo, tests de secrets) en [`AGENTS.md`](AGENTS.md).

---

## Troubleshooting

| Problema | Solución |
|---|---|
| El chip no aparece | Ejecutar `python src\main.py` en terminal (con `python.exe`, no `pythonw.exe`) para ver errores en consola. Revisar `logs\cortex.log`. |
| opencode no responde (WinError 10061) | Verificar `opencode serve` levantado y puerto 57214 libre (`netstat -ano \| findstr 57214`). Revisar `logs\opencode-wrapper.log`. |
| Hotkey no funciona | El Programador de Tareas necesita "privilegios más altos" — la librería `keyboard` requiere admin para hooks globales. |
| No hay respuesta de voz | Verificar `.env`, LM Studio (Qwen3.5-2b) y `logs\cortex.log`. Verificar `curl http://127.0.0.1:57214/global/health` → `{"healthy":true}`. |
| El agente no abre programas | Usar esquema URI para apps de Store (`whatsapp:`, `spotify:`). Probar `start chrome`, `start notepad`, `start explorer`. |
| "pythonw.exe no se reconoce" | `where pythonw.exe` — si no aparece, reinstalar Python marcando "Add Python to PATH". |
| Logs crecen demasiado | Rotan automáticamente cada 5 MB con 3 backups. Editar `config/settings.json` → `logging.max_bytes` y `backup_count`. |

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Orquestador | Python 3.10+ (`keyboard`, `httpx`, `sounddevice`, `numpy`) |
| STT | Google AI Studio — `gemini-3.1-flash-lite` (fallback: `gemini-2.5-flash-lite`) |
| Cerebro | OpenCode serve (agente `asistente_voz`, bash + memoria) |
| TTS primario | Azure Cognitive Services (SSML, `es-MX-JorgeNeural`) |
| TTS fallback | Gemini (`gemini-3.1-flash-tts-preview`, voz `Charon`) |
| Memoria | Plugin `opencode-mem` (BD vectorial local) |
| Overlay | tkinter (chip visual durante grabación/procesamiento) |