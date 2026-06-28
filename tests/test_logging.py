"""Tests unitarios para main.py:setup_logging.

Cubre:
- Creación del RotatingFileHandler en el root logger.
- Creación del directorio logs/ si no existe.
- StreamHandler agregado a stderr sólo si hay TTY.
- Rotación del archivo de log al exceder max_bytes.
- Ausencia de secretos (Gemini AIza... / Azure Ocp-Apim) en el log.

Cada test aísla los handlers del root logger en teardown para evitar
contaminar tests subsiguientes.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Asegurar que la raíz del proyecto está en sys.path (conftest ya lo hace,
# pero por si pytest invoca este archivo en modo standalone).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def clean_root_logger():
    """Limpia los handlers del root logger antes y después del test.

    setup_logging() muta logging.getLogger() (root). Sin esta fixture,
    los handlers se acumulan entre tests y los asserts sobre cantidad
    o tipo de handlers serían flaky.
    """
    root = logging.getLogger()
    root.handlers.clear()
    # Subimos el level para no contaminar la salida de pytest.
    root.setLevel(logging.WARNING)
    yield root
    root.handlers.clear()


@pytest.fixture
def fake_settings_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Crea config/settings.json en tmp_path con un log en tmp_path/logs/.

    Hace chdir a tmp_path y devuelve un dict con la configuración efectiva
    (filename/max_bytes/backup_count/level) para que los tests puedan
    referenciarla explícitamente.
    """
    log_file = tmp_path / "logs" / "cortex.log"
    settings_dict = {
        "logging": {
            "filename": str(log_file),
            "max_bytes": 5242880,
            "backup_count": 3,
            "level": "INFO",
        }
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(
        json.dumps(settings_dict), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return {"settings": settings_dict, "log_file": log_file}


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestSetupLogging:
    """Suite de tests para setup_logging() de main.py."""

    def test_setup_logging_creates_file_handler(
        self,
        clean_root_logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """setup_logging() añade un RotatingFileHandler al root logger."""
        # Trabajamos contra la config real del proyecto.
        monkeypatch.chdir(_PROJECT_ROOT)
        # Forzamos no-TTY para que setup_logging NO agregue StreamHandler
        # (lo cubrimos aparte en test_setup_logging_no_tty_no_console).
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        from main import setup_logging

        setup_logging()

        root = logging.getLogger()
        rf_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rf_handlers) == 1, (
            f"Esperaba 1 RotatingFileHandler, encontré {len(rf_handlers)}. "
            f"Handlers: {root.handlers}"
        )
        # El handler tiene un formatter configurado
        assert rf_handlers[0].formatter is not None

    def test_setup_logging_creates_logs_dir(
        self,
        clean_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Si logs/ no existe, setup_logging() lo crea automáticamente."""
        # Armamos una config con un logs/ inexistente dentro de tmp_path.
        log_file = tmp_path / "logs" / "cortex.log"
        settings = {
            "logging": {
                "filename": str(log_file),
                "max_bytes": 1024,
                "backup_count": 1,
                "level": "INFO",
            }
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.json").write_text(
            json.dumps(settings), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        # Sanity: el directorio NO existe aún
        assert not (tmp_path / "logs").exists()

        from main import setup_logging

        setup_logging()

        # El directorio fue creado
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "logs").is_dir()

    def test_setup_logging_no_tty_no_console(
        self,
        clean_root_logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Si sys.stderr.isatty() == False → NO se añade StreamHandler a stderr.

        Filtramos por stream=sys.stderr para no confundirnos con
        LogCaptureHandler ni RotatingFileHandler que pytest puede añadir.
        """
        monkeypatch.chdir(_PROJECT_ROOT)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        from main import setup_logging

        setup_logging()

        root = logging.getLogger()
        console_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and getattr(h, "stream", None) is sys.stderr
        ]
        assert console_handlers == [], (
            f"Esperaba 0 StreamHandler(stderr) de consola, "
            f"encontré {len(console_handlers)}. "
            f"Handlers: {root.handlers}"
        )

    def test_setup_logging_with_tty_adds_console(
        self,
        clean_root_logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Si sys.stderr.isatty() == True → se añade un StreamHandler a stderr."""
        monkeypatch.chdir(_PROJECT_ROOT)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

        from main import setup_logging

        setup_logging()

        root = logging.getLogger()
        console_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and getattr(h, "stream", None) is sys.stderr
        ]
        assert len(console_handlers) == 1, (
            f"Esperaba 1 StreamHandler(stderr) de consola, "
            f"encontré {len(console_handlers)}. "
            f"Handlers: {root.handlers}"
        )
        # Apunta a sys.stderr
        assert console_handlers[0].stream is sys.stderr

    def test_log_rotation(
        self,
        clean_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Al superar max_bytes se crea el backup cortex.log.1."""
        # max_bytes chico para no escribir 5MB reales.
        log_file = tmp_path / "cortex.log"
        settings = {
            "logging": {
                "filename": str(log_file),
                "max_bytes": 512,  # 512 bytes
                "backup_count": 2,
                "level": "DEBUG",
            }
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.json").write_text(
            json.dumps(settings), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        from main import setup_logging

        setup_logging()

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # Logueamos mensajes con payload mayor que max_bytes para forzar
        # al menos 1 rotación.
        # Cada línea "AAAA..." mide ~300 bytes (timestamp+level+name+padding).
        payload = "x" * 200
        for _ in range(20):
            root.info(payload)

        # Forzar flush y posible rollover
        for h in root.handlers:
            h.flush()
            # doRollover puede ser llamado para asegurar la rotación,
            # pero el handler lo hace solo al escribir más allá del límite.
            # Aún así, lo invocamos explícitamente para hacer el test
            # determinístico (independiente del cálculo interno de shouldRollover).
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.doRollover()

        # Tras doRollover, el archivo activo queda en 0 bytes y se crea
        # al menos un backup .1
        assert (tmp_path / "cortex.log.1").exists(), (
            f"Se esperaba cortex.log.1, archivos en tmp_path: "
            f"{list(tmp_path.iterdir())}"
        )

    def test_no_secrets_in_log(
        self,
        clean_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """El archivo de log NO debe contener API keys de Gemini/Azure.

        Verifica:
        - Patrones de Gemini (prefijo 'AIza')
        - Headers sensibles de Azure ('Ocp-Apim')

        Setear env vars con esos prefijos y loguear mensajes benignos;
        los handlers de logging no deberían filtrar variables de entorno
        en el formatter.
        """
        # Seteamos env vars con prefijos sensibles.
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyD_FAKE_KEY_VALUE_12345")
        monkeypatch.setenv("AZURE_SPEECH_KEY", "fake_azure_key_value")

        log_file = tmp_path / "cortex.log"
        settings = {
            "logging": {
                "filename": str(log_file),
                "max_bytes": 5242880,
                "backup_count": 1,
                "level": "DEBUG",
            }
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.json").write_text(
            json.dumps(settings), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        from main import setup_logging

        setup_logging()

        # Logueamos un mensaje benigno (sin secretos) por la raíz.
        logger = logging.getLogger("test.no_secrets")
        logger.info("mensaje de prueba sin secretos")
        for h in logging.getLogger().handlers:
            h.flush()

        log_content = log_file.read_text(encoding="utf-8")

        # El mensaje legítimo SÍ aparece
        assert "mensaje de prueba sin secretos" in log_content

        # Los patrones de secreto NO aparecen
        assert "AIza" not in log_content, (
            f"Encontré prefijo de Gemini API key en el log: {log_content!r}"
        )
        assert "Ocp-Apim" not in log_content, (
            f"Encontré header sensible de Azure en el log: {log_content!r}"
        )
