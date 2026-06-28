# ESPECIFICACIÓN TÉCNICA Y PLAN DE IMPLEMENTACIÓN: ASISTENTE DE VOZ WINDOWS

## 1. Visión General del Proyecto
Creación de un asistente de voz en segundo plano para Windows, capaz de interactuar con el sistema operativo (abrir programas, navegar, gestionar automatizaciones) mediante lenguaje natural. El sistema opera de manera completamente independiente al entorno de desarrollo del usuario, garantizando cero contaminación cruzada entre sesiones de programación y comandos cotidianos.

---

## 2. Decisiones de Arquitectura y Patrones

### 2.1. Arquitectura "Dos Cerebros" (Aislamiento Estricto)
El principio fundamental es la separación del contexto. 
- **Entorno Default:** OpenCode estándar sin plugins intrusivos, usado manualmente en la terminal para escribir código.
- **Entorno de Voz:** Instancia separada de OpenCode corriendo como un servicio REST (`opencode serve`) en un puerto específico (ej. 4096), apuntando a un directorio de configuración aislado (`~/.opencode-voz/`).

### 2.2. Patrones de Diseño del Orquestador (Python)
- **Event-Driven (Orientado a Eventos):** El ciclo de vida de la grabación se activa/desactiva asíncronamente mediante un listener global de teclado (patrón Toggle/Interruptor).
- **Facade (Fachada):** El script de Python actúa como un enrutador central que oculta la complejidad de interactuar con tres APIs distintas (Gemini, OpenCode, Azure).
- **Circuit Breaker & Fallback:** Si el proveedor primario de STT falla o alcanza el límite de *rate-limiting*, el sistema hace un "failover" automático a un modelo secundario.

---

## 3. Stack Tecnológico y Proveedores

| Componente | Tecnología / Servicio | Configuración Específica / Tier |
| :--- | :--- | :--- |
| **Orquestador** | Python 3.10+ | Librería `keyboard` (Toggle), `requests`, `azure-cognitiveservices-speech`. |
| **Trigger (Atajo)** | Teclado Global | `Ctrl + Alt + V` (Interruptor On/Off). Evita conflictos con IDEs y juegos. |
| **STT (Voz a Texto)** | Google AI Studio API | **Principal:** `gemini-3.1-flash-lite` (15 RPM). **Fallback:** `gemini-2.5-flash-lite` (10 RPM). Elimina balbuceos y aplica puntuación. |
| **Agente (Cerebro)** | OpenCode API | Modo `serve`. Acceso a shell local y webhooks para ejecución en Windows. |
| **TTS (Texto a Voz)**| Azure Cognitive Services | Free Tier (F0). Procesamiento exclusivo mediante **SSML** para inyectar expresividad y pausas naturales. |
| **Memoria** | Plugin `opencode-mem` | BD Vectorial local (USearch/SQLite). Indexación semántica automática. |

---

## 4. Gestión de Memoria y Contexto

- **Memoria a Corto Plazo (Sesión Activa):** Gestionada en RAM por el script de Python reteniendo el `Thread_ID` devuelto por OpenCode. Si se reinicia la PC, el script muere, la variable se vacía y la sesión de voz se reinicia limpia (sin necesidad de programar rutinas de borrado).
- **Memoria a Largo Plazo:** Gestionada por el plugin `tickernelz/opencode-mem`. Al estar instalado en la carpeta aislada `.opencode-voz`, construye una base de datos vectorial exclusiva para los hábitos, gustos (ej. playlists de Spotify) y comandos del asistente, sin indexar repositorios de código.

---

## 5. El Flujo de Ejecución (Ciclo de Vida de un Comando)

1. **Trigger:** El usuario presiona `Ctrl + Alt + V`. Python cambia el estado a `grabando = True`.
2. **Grabación:** El usuario habla. Al presionar nuevamente `Ctrl + Alt + V`, se guarda un `comando.wav`.
3. **STT (Limpieza):** Python envía el audio a `gemini-3.1-flash-lite` con el prompt: *"Transcribe eliminando balbuceos y puntuando correctamente"*.
4. **Razonamiento & Ejecución:** Python hace un POST a `localhost:4096` enviando el texto. OpenCode interpreta, usa herramientas de bash/webhooks (n8n/PowerAutomate) para abrir Chrome/Spotify en Windows.
5. **Generación SSML:** Por instrucción de su System Prompt, OpenCode responde un XML válido definiendo la emoción del resultado (ej. `<mstts:express-as style="cheerful">`).
6. **TTS (Voz):** Python envía el XML a Azure. Azure devuelve un stream de audio hiperrealista. Python lo reproduce en los altavoces.

---

## 6. Estructura de Archivos (Workspace Python)

```text
/assistant-voz/
├── main.py                     # Punto de entrada; contiene el loop principal y el listener de 'keyboard'.
├── requirements.txt            # Dependencias (keyboard, requests, azure-cognitiveservices-speech, sounddevice).
├── config/
│   └── settings.json           # Claves de API (Azure, Google) y configuraciones de puertos.
├── handlers/
│   ├── audio_manager.py        # Controla el micrófono y la reproducción de audio (I/O de hardware).
│   ├── gemini_client.py        # Construye el payload multimodar para Google AI Studio.
│   ├── opencode_client.py      # Mantiene el Thread_ID en memoria y hace POST a localhost.
│   └── azure_tts_client.py     # Recibe el XML de OpenCode y reproduce el resultado con Azure.
└── .opencode-voz/              # DIRECTORIO AISLADO PARA EL CEREBRO DE VOZ
    ├── config.json             # Define el uso del plugin ["opencode-mem"].
    ├── agents/
    │   └── asistente_voz.md    # System prompt crítico (Obliga la salida a SSML).
    └── (db_vectorial)          # Archivos auto-generados por el plugin de memoria.
```

---

## 7. Configuración Crítica (El System Prompt)
Archivo: `.opencode-voz/agents/asistente_voz.md`

```markdown
---
name: asistente_voz
description: Agente headless para control de SO por voz. Obliga salida SSML.
tools: [bash, write, read]
---
Eres el asistente principal de mi sistema operativo Windows. Responde a mis peticiones y ejecuta acciones mediante consola o webhooks. 
REGLA ABSOLUTA: Tu respuesta final DEBE OBLIGATORIAMENTE ser un bloque SSML válido para Azure TTS. Dependiendo de si la tarea tuvo éxito, falló o es informativa, usa la etiqueta `<mstts:express-as>` con el 'style' apropiado ('cheerful', 'sad', 'friendly').
NO escribas texto fuera del bloque XML/SSML.
```

---

## 8. Plan de Implementación (Fases)

### Fase 1: Preparación del Entorno Aislado
1. Crear el directorio `~/.opencode-voz`.
2. Crear `config.json` e instalar/habilitar el plugin `opencode-mem`.
3. Crear el archivo del agente `asistente_voz.md`.
4. Probar levantar el servidor aislado: `opencode serve --config-dir ~/.opencode-voz --port 4096`.

### Fase 2: Captura y Hardware (Python)
1. Iniciar proyecto Python y estructurar carpetas.
2. Implementar `audio_manager.py` para grabar micrófono hacia `.wav`.
3. Implementar en `main.py` la máquina de estados con `keyboard.add_hotkey('ctrl+alt+v', toggle_recording)`.

### Fase 3: Integración STT (Gemini)
1. Implementar `gemini_client.py`.
2. Crear la función que envía el archivo `.wav` junto al prompt restrictivo para limpieza de balbuceos.
3. Testear hablando de forma natural y verificar que retorna un `String` limpio.

### Fase 4: Integración del Cerebro (OpenCode)
1. Implementar `opencode_client.py`.
2. Conectar el texto devuelto por Gemini y hacer el POST a `localhost:4096`.
3. Implementar la lógica para guardar el `Thread_ID` en una variable para mantener el contexto.
4. Validar que la respuesta sea el bloque XML/SSML esperado.

### Fase 5: Integración TTS (Azure) y Salida
1. Implementar `azure_tts_client.py`.
2. Enviar el string XML recibido de OpenCode hacia los servicios cognitivos de Azure.
3. Reproducir el audio resultante.

### Fase 6: Pruebas de Estrés y Automatización de Windows
1. Testear ejecución cruzada: pedirle al sistema que abra un programa en Windows.
2. Configurar un archivo `.bat` o un script en el Programador de Tareas de Windows para que, al encender la PC, se ejecuten automáticamente:
   - El proceso de OpenCode (`opencode serve ...`).
   - El script orquestador de Python (`python main.py`).
