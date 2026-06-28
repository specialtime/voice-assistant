@echo off
cd /d "%~dp0\.."

:: Verificar que el venv existe (Python 3.10 requerido por kokoro-onnx/piper-tts)
if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERROR] venv no encontrado en .venv\Scripts\pythonw.exe
    echo [INFO] Crear el venv con Python 3.10:
    echo     py -3.10 -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Detectar pythonw.exe
where pythonw.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pythonw.exe no encontrado en PATH.
    pause
    exit /b 1
)

:: 1. Lanzar opencode serve en background (sin ventana, con auto-restart)
echo [INFO] Iniciando opencode serve (hidden)...
start "" pythonw.exe start_opencode_hidden.py

:: 2. Esperar 5 segundos a que opencode arranque
timeout /t 5 /nobreak >nul

:: 3. Lanzar orquestador Python (sin consola)
echo [INFO] Iniciando orquestador Cortex (pythonw.exe)...
start "" ".venv\Scripts\pythonw.exe" src\main.py

:: 4. Dar 3 segundos para que arranque
timeout /t 3 /nobreak >nul

:: 5. Verificar que pythonw.exe está corriendo
tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | find /I "pythonw.exe" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Los procesos no arrancaron. Revisá logs\cortex.log y logs\opencode-wrapper.log.
    echo [INFO] Verificá que el venv tenga las deps: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Cortex iniciado en background.
echo [INFO] Presioná Alt+V para activar el asistente.
echo [INFO] Logs: logs\cortex.log (orquestador) + logs\opencode-wrapper.log (opencode)
echo.
echo Para detener: ejecuta taskkill /f /im pythonw.exe
echo.
