@echo off
cd /d "%~dp0"

:: Variables de entorno para DEV (override de los defaults de prod)
set CORTEX_OPENCODE_DIR=C:\Users\crist\.cortex-dev
set CORTEX_PORT=57215

:: Detectar pythonw.exe
where pythonw.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pythonw.exe no encontrado en PATH.
    pause
    exit /b 1
)

:: 1. Lanzar opencode serve en background (hidden, puerto 57215, .cortex-dev\)
echo [INFO] Iniciando opencode serve DEV (puerto 57215, .cortex-dev\)...
start "" pythonw.exe start_opencode_hidden.py

:: 2. Esperar 5 segundos a que opencode arranque
timeout /t 5 /nobreak >nul

:: 3. Lanzar orquestador Python (sin consola)
echo [INFO] Iniciando orquestador Cortex DEV (pythonw.exe)...
start "" pythonw.exe src\main.py

:: 4. Dar 3 segundos para que arranque
timeout /t 3 /nobreak >nul

:: 5. Verificar que pythonw.exe está corriendo
tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | find /I "pythonw.exe" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Los procesos no arrancaron. Revisá logs\cortex.log y logs\opencode-wrapper.log.
    pause
    exit /b 1
)

echo [OK] Cortex DEV iniciado en background.
echo [INFO] Presioná Alt+V para activar el asistente (memoria de dev, no contamina prod).
echo [INFO] Logs: logs\cortex.log + logs\opencode-wrapper.log
echo.
echo Para detener: powershell -ExecutionPolicy Bypass -File stop-cortex.ps1
echo.
