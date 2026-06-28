# Autoarranque del Asistente de Voz Córtex

Guía para configurar el inicio automático del asistente al iniciar sesión, usando **Programador de Tareas** de Windows. Tanto opencode serve como el orquestador Python corren como procesos del usuario en Session 1 (mismo desktop), sin servicio de Windows ni NSSM.

---

## Arquitectura — Proceso Usuario Session 1

```
┌─────────────────────────────────────────────────────┐
│  PROCESO USUARIO (Session 1 — tu desktop)            │
│                                                       │
│  pythonw.exe start_opencode_hidden.py                 │
│    └─ opencode serve --port 57214 --hostname 127.0.0.1│
│       (CREATE_NO_WINDOW, auto-restart si crashea)     │
│       Logs: logs/opencode-wrapper.log                 │
│                                                       │
│  pythonw.exe src\main.py                              │
│    └─ Orquestador (hotkey Alt+V, STT, TTS, overlay)   │
│       Logs: logs/cortex.log                           │
│                                                       │
│  Agente usa bash con `start` directamente:            │
│    start chrome → Chrome aparece en desktop ✅         │
│    start whatsapp: → WhatsApp abre (URI scheme) ✅     │
│                                                       │
│  Auto-start: Programador de Tareas "Al iniciar        │
│  sesión" via start_cortex.bat                         │
└─────────────────────────────────────────────────────┘
```

**Por qué proceso usuario (no servicio):** un asistente de voz necesita desktop (overlay tkinter), audio (micrófono/altavoces) y hotkey global. Los servicios de Windows corren en Session 0 (aislada, sin desktop desde Vista) — no pueden mostrar ventanas ni lanzar programas visibles. Correr todo en Session 1 es más simple y funciona correctamente.

> **Historial:** la arquitectura anterior usaba NSSM (servicio Session 0) + Launch Bridge HTTP para delegar lanzamiento de programas a Session 1. El bridge no funcionaba (`chrome` no en PATH, `whatsapp` no en App Paths del registry). Fue reemplazada por esta arquitectura más simple en junio 2026.

---

## Requisitos previos

- **Python 3.10+** instalado y en el PATH (`python --version` desde cualquier cmd). Verificá que `pythonw.exe` también esté disponible (`where pythonw.exe`).
- **opencode** instalado en `C:\Users\crist\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe`.
- **LM Studio** corriendo con el modelo **Qwen3.5-2b** cargado (necesario para opencode-mem).
- **Archivo `.env`** configurado con las claves (`GEMINI_API_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `OPENCODE_BASE_URL=http://127.0.0.1:57214`).
- **Dependencias Python** instaladas: `pip install -r requirements.txt`.

---

## Paso 1 — Configurar el Programador de Tareas

El Programador de Tareas ejecuta `start_cortex.bat` al iniciar sesión. Este bat lanza automáticamente:
1. `pythonw.exe start_opencode_hidden.py` — opencode serve en background (invisible, con auto-restart)
2. `pythonw.exe src\main.py` — orquestador con overlay visual

### Crear la tarea

1. Presionar `Win + R`, escribir `taskschd.msc` y presionar Enter.
2. Hacer clic en **"Crear tarea..."** en el panel derecho.

### Pestaña "General"
- **Nombre:** `Cortex`
- **Descripción:** Inicia opencode serve + orquestador Python al iniciar sesión.
- ✅ **Ejecutar solo cuando el usuario haya iniciado sesión**
- ✅ **Ejecutar con los privilegios más altos** (necesario para el hotkey `keyboard` — ver nota abajo)

### Pestaña "Desencadenadores"
- **Nuevo...**
- **Iniciar la tarea:** `Al iniciar sesión` (At log on)
- **Estado:** `Habilitado`

### Pestaña "Acciones"
- **Nuevo...**
- **Acción:** `Iniciar un programa`
- **Programa o script:** `C:\Users\crist\Documents\proyectos\agentes\voice-assistant\start_cortex.bat`
- **Iniciar en:** `C:\Users\crist\Documents\proyectos\agentes\voice-assistant\`

### Pestaña "Condiciones"
- Desmarcar **"Iniciar la tarea solo si el equipo está conectado a la alimentación de CA"** (para notebooks).

### Pestaña "Configuración"
- ✅ **Permitir la ejecución de la tarea a petición**
- ✅ **Ejecutar la tarea lo antes posible después de que se inicie el desencadenador programado**

3. Guardar (puede pedir contraseña).

---

## Nota sobre privilegios de admin y `keyboard`

La librería `keyboard` (hotkey global Alt+V) requiere privilegios de admin en Windows. Por eso el Programador de Tareas debe tener **"Ejecutar con los privilegios más altos"** marcado.

- Con admin: el hotkey funciona, el overlay tkinter funciona, opencode lanza programas visibles.
- Sin admin: el hotkey `keyboard` puede no funcionar. Alternativa: migrar a `pynput` (no requiere admin) — pero requiere modificación de código.

> **Recomendado:** marcar "privilegios más altos" en el Programador de Tareas.

---

## Verificación

1. **Reiniciar la PC** (o cerrar sesión y volver a iniciar).
2. **Esperar 10 segundos** después de iniciar sesión (opencode + orquestador arrancan automáticamente).
3. **Presionar `Alt+V`:**
   - Debe aparecer un chip abajo al centro de la pantalla con "● Grabando..." (rojo pulsante).
4. **Hablar** un comando (ej: "abrí Chrome").
5. **Presionar `Alt+V` de nuevo:**
   - El chip cambia a "● Procesando..." (amarillo).
6. **Esperar** la respuesta de voz.
7. El chip desaparece con fade-out al terminar.
8. **Verificar logs:**
   - `logs\cortex.log` — log del orquestador (debe tener entradas de logging).
   - `logs\opencode-wrapper.log` — log del wrapper de opencode (debe mostrar "opencode serve lanzado").
9. **Verificar procesos:** `tasklist /FI "IMAGENAME eq pythonw.exe"` debe mostrar 2 procesos pythonw.exe.

---

## Arranque manual

Si querés iniciar el asistente sin reiniciar la PC:

```
cd C:\Users\crist\Documents\proyectos\agentes\voice-assistant
start_cortex.bat
```

El bat lanza ambos procesos (opencode + orquestador) en background y verifica que arrancaron.

---

## Detener el asistente

```
taskkill /f /im pythonw.exe
```

Esto mata ambos procesos pythonw.exe (opencode wrapper + orquestador).

---

## Troubleshooting

### ❌ El orquestador no arranca (chip no aparece)

1. Abrir una terminal y ejecutar manualmente:
   ```
   cd C:\Users\crist\Documents\proyectos\agentes\voice-assistant
   python src\main.py
   ```
   (con `python.exe`, no `pythonw.exe` — para ver errores en consola).
2. Si hay error de import o sintaxis, se verá en la terminal.
3. Revisar `logs\cortex.log` — si está vacío, el crash fue antes de que el logging se configurara.
4. Verificar que `config/settings.json` tiene la sección `logging`.
5. Verificar que `src\handlers\overlay.py` existe y no tiene errores de sintaxis.

### ❌ opencode no responde (WinError 10061 — conexión rechazada)

1. Revisar `logs\opencode-wrapper.log` — debe mostrar "opencode serve lanzado (pid=...)".
2. Si el log muestra "opencode serve terminó (exit_code=...)" repetidamente, opencode está crasheando. El wrapper lo reinicia automáticamente cada 5 segundos.
3. Verificar que el binario existe: `Test-Path "C:\Users\crist\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe"`.
4. Probar arrancar opencode manualmente para ver el error:
   ```
   cd C:\Users\crist\.cortex
   "C:\Users\crist\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe" serve --port 57214 --hostname 127.0.0.1
   ```
5. Verificar que el puerto 57214 no está en uso: `netstat -ano | findstr 57214`.

### ❌ El chip aparece pero el hotkey no funciona

1. Verificar que el Programador de Tareas tiene **"Ejecutar con los privilegios más altos"** marcado.
2. La librería `keyboard` requiere admin para hooks globales.
3. Si no querés admin, considerar migrar a `pynput` (requiere cambio de código).

### ❌ El chip aparece pero no hay respuesta de voz

1. Verificar que opencode está corriendo: `curl http://127.0.0.1:57214/global/health` → debe dar `{"healthy":true}`.
2. Verificar `.env` — todas las claves deben estar completas.
3. Revisar `logs\cortex.log` — buscar errores de STT, OpenCode o TTS.
4. Verificar que LM Studio está corriendo con Qwen3.5-2b (para opencode-mem).

### ❌ El agente no abre programas (Chrome, WhatsApp, etc.)

1. Revisar `logs\cortex.log` — buscar si el agente respondió con `[STYLE: ...]`.
2. Si el agente respondió pero el programa no abrió, puede ser que el nombre no sea reconocido por `start`. Probar:
   - `start chrome` (navegador)
   - `start whatsapp:` (app de Store — usa esquema URI con dos puntos)
   - `start notepad` (bloc de notas)
   - `start explorer` (explorador de archivos)
3. Si es una app de Microsoft Store, usar el esquema URI (`whatsapp:`, `spotify:`, etc.).
4. Si no funciona, abrir manualmente el programa para confirmar que está instalado.

### ❌ "pythonw.exe no se reconoce como un comando"

1. Verificar: `where pythonw.exe` en una terminal.
2. Si no aparece, reinstalar Python marcando **"Add Python to PATH"**.
3. `pythonw.exe` viene con Python — si `python.exe` funciona, `pythonw.exe` debería estar en el mismo directorio.

### ❌ Logs crecen demasiado

- El logging rota automáticamente cada 5 MB, manteniendo 3 backups (20 MB máximo).
- Para cambiar: editar `config/settings.json` sección `logging` → `max_bytes` y `backup_count`.
- Para debug temporal: cambiar `level` de `"INFO"` a `"DEBUG"` (¡ojo: DEBUG loguea transcripciones de voz!).

---

## Desinstalación

### Quitar el Programador de Tareas
1. Abrir Programador de Tareas → buscar "Cortex" → clic derecho → "Eliminar".

### Detener los procesos manualmente
```
taskkill /f /im pythonw.exe
```

### Si tenés el servicio NSSM instalado (arquitectura anterior)

Si migraste desde la arquitectura anterior con NSSM, desinstalar el servicio:
```
cd C:\Users\crist\Documents\proyectos\agentes\voice-assistant
nssm\nssm.exe stop CortexServer
nssm\nssm.exe remove CortexServer confirm
```

Los archivos `install_service.bat`, `start_opencode_service.bat`, `nssm/`, `query` y `__validate_json.ps1` fueron eliminados tras la migración a la arquitectura de proceso usuario.

---

## Notas adicionales

- **Auto-restart:** el wrapper `start_opencode_hidden.py` reinicia opencode serve automáticamente si crashea (cada 5 segundos). El orquestador Python NO tiene auto-restart — si crashea, hay que reiniciarlo manualmente o reiniciar sesión.
- **Logs separados:** `logs/cortex.log` (orquestador) y `logs/opencode-wrapper.log` (opencode). Ambos rotan a 5 MB con 3 backups.
- **Procesos invisibles:** ambos corren con `pythonw.exe` (sin ventana de consola). El único feedback visual es el overlay chip tkinter al presionar Alt+V.
- **El chip visual** solo aparece durante RECORDING y PROCESSING. En IDLE no se muestra.
- **Si cerrás los procesos** (`taskkill /f /im pythonw.exe`), ambos se detienen (opencode + orquestador). Para reiniciar, ejecutar `start_cortex.bat`.

---

*Documentación actualizada para Fase 10 — Junio 2026*
