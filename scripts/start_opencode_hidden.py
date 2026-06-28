"""Wrapper para opencode serve — corre sin ventana visible (pythonw.exe),
monitorea el proceso y lo reinicia si crashea.

Se ejecuta con pythonw.exe (sin consola) desde start_cortex.bat.
opencode serve corre como proceso hijo con CREATE_NO_WINDOW.
"""

import logging
import logging.handlers
import os
import subprocess
import sys
import time
from pathlib import Path

# Configurar logging a archivo (pythonw no tiene consola)
log_path = Path(__file__).parent.parent / "logs" / "opencode-wrapper.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5242880, backupCount=3, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

OPENCODE_EXE = r"C:\Users\crist\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe"
OPENCODE_DIR = os.environ.get("CORTEX_OPENCODE_DIR", r"C:\Users\crist\.cortex")
PORT = int(os.environ.get("CORTEX_PORT", "57214"))
HOSTNAME = "127.0.0.1"
RESTART_DELAY = 5  # segundos entre restarts


def main() -> None:
    """Loop principal: lanza opencode serve, espera, reinicia si crashea."""
    # Setear entorno del usuario (necesario para que opencode encuentre auth.json y config)
    env = os.environ.copy()
    env["USERPROFILE"] = r"C:\Users\crist"
    env["HOME"] = r"C:\Users\crist"
    env["APPDATA"] = r"C:\Users\crist\AppData\Roaming"
    env["LOCALAPPDATA"] = r"C:\Users\crist\AppData\Local"

    logger.info("opencode-wrapper iniciado — puerto %d, dir %s", PORT, OPENCODE_DIR)

    while True:
        try:
            logger.info("Lanzando opencode serve...")
            proc = subprocess.Popen(
                [OPENCODE_EXE, "serve", "--port", str(PORT), "--hostname", HOSTNAME],
                cwd=OPENCODE_DIR,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            logger.info("opencode serve lanzado (pid=%d)", proc.pid)

            # Esperar a que termine (crash o shutdown)
            exit_code = proc.wait()
            logger.warning("opencode serve terminó (exit_code=%d)", exit_code)

            # Leer output para diagnóstico
            if proc.stdout:
                output = proc.stdout.read().decode("utf-8", errors="replace")
                if output.strip():
                    logger.info("opencode output:\n%s", output[:2000])

        except Exception as e:
            logger.exception("Error inesperado en wrapper: %s", e)

        logger.info("Esperando %d segundos antes de reiniciar...", RESTART_DELAY)
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    main()
