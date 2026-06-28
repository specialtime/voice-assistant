# Bug: Rutas rotas en scripts/ tras reestructuración del proyecto

## Contexto

En el commit `062d00e` ("reorganizar estructura del proyecto") los scripts de operaciones (`start_cortex.bat`, `start-dev.bat`, `stop-cortex.ps1`, `deploy.ps1`, `cleanup_voz_sessions.py`) fueron movidos a la carpeta `scripts/`. Sin embargo, los `.bat` y `deploy.ps1` siguen asumiendo que su CWD es la raíz del repo, por lo que ahora resuelven rutas incorrectas.

## Síntomas

- `start_cortex.bat` y `start-dev.bat` hacen `cd /d "%~dp0"` (→ `scripts\`) y luego invocan `pythonw.exe src\main.py` (→ `scripts\src\main.py`, inexistente) y `pythonw.exe start_opencode_hidden.py` (→ `scripts\start_opencode_hidden.py`, inexistente en dev).
- `deploy.ps1` usa `$DEV_DIR = $PSScriptRoot` (→ `scripts\`), por lo que `Test-Path "$DEV_DIR\src"` falla y aborta.
- `deploy.ps1` copia `start_cortex.bat` y `start_opencode_hidden.py` sueltos a la raíz de prod, rompiendo la simetría con la nueva estructura dev (donde viven en `scripts\`).

## Decisión de diseño (confirmada con el usuario)

1. **Estrategia de rutas:** los `.bat` harán `cd /d "%~dp0\.."` al arrancar, para que el CWD sea la raíz del repo. Se mantienen las rutas relativas `src\main.py`, `start_opencode_hidden.py`, `logs\cortex.log` tal como están.
2. **Estructura en prod:** `deploy.ps1` copiará la carpeta `scripts\` entera a prod, manteniendo simetría dev↔prod. El autoarranque (Task Scheduler) apuntará a `scripts\start_cortex.bat`.

## Alcance

### Archivos a modificar

- `scripts/start_cortex.bat`
- `scripts/start-dev.bat`
- `scripts/deploy.ps1`

### Archivos NO modificados (verificados correctos)

- `scripts/stop-cortex.ps1` — matchea procesos por `CommandLine` regex, independiente del CWD. Sin cambios.
- `scripts/cleanup_voz_sessions.py` — usa path absoluto a la BD. Sin cambios.

## Especificación técnica

### `scripts/start_cortex.bat`

Cambio único en línea 2:
- **Antes:** `cd /d "%~dp0"`
- **Después:** `cd /d "%~dp0\.."`

Resto del archivo sin cambios. Las rutas `start_opencode_hidden.py`, `src\main.py`, `logs\cortex.log` quedan como están (relativas a la raíz del repo).

### `scripts/start-dev.bat`

Cambio único en línea 2:
- **Antes:** `cd /d "%~dp0"`
- **Después:** `cd /d "%~dp0\..""`

Resto del archivo sin cambios.

### `scripts/deploy.ps1`

#### Cambio 1 — `$DEV_DIR` debe apuntar a la raíz del repo, no a `scripts\`

- **Antes (línea 7):** `$DEV_DIR = $PSScriptRoot`
- **Después:** `$DEV_DIR = Split-Path -Parent $PSScriptRoot`

#### Cambio 2 — `$toCopy` debe copiar `scripts\` entero, no scripts sueltos

- **Antes (líneas 31-36):**
  ```powershell
  $toCopy = @(
      "src",
      "requirements.txt",
      "start_cortex.bat",
      "start_opencode_hidden.py"
  )
  ```
- **Después:**
  ```powershell
  $toCopy = @(
      "src",
      "requirements.txt",
      "scripts"
  )
  ```

#### Cambio 3 — Actualizar mensajes informativos

- Línea 71-72: el mensaje final que dice "NO se copió config\settings.json" queda igual. No requiere cambio.
- Verificar que los mensajes `[INFO] Dev:` y `[INFO] Prod:` sigan siendo correctos con el nuevo `$DEV_DIR`.

### Notas

- `start_opencode_hidden.py` **no vive en este repo** (está en el dir de prod). Tras el cambio, `deploy.ps1` ya no lo copia individualmente — se asume que ya existe en prod. Si en el futuro se agrega al repo, vivirá en `scripts\` y se copiará con el resto de la carpeta.
- El README ya documenta los scripts bajo `scripts\` (líneas 44-49). No requiere cambios.
- El AGENTS.md menciona `scripts\start_cortex.bat` y `scripts\start-dev.bat` (sección Comandos). No requiere cambios.

## Verificación

1. **Sintáctica:** los `.bat` no tienen validador, pero el cambio es trivial (un `\..` añadido).
2. **Funcional (manual, post-fix):**
   - Ejecutar `scripts\start_cortex.bat` desde la raíz del repo → debe encontrar `src\main.py` y arrancar el orquestador.
   - Ejecutar `scripts\start-dev.bat` → idem con puerto 57215.
   - Ejecutar `powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1` con `$env:CORTEX_PROD_DIR` apuntando a un dir de prueba → debe copiar `src\`, `requirements.txt` y `scripts\` (entero) al destino.
3. **Tests automatizados:** delegar a `@tester` la verificación de que los scripts resuelvan rutas correctamente.