"""Configuración global de pytest para la suite de tests de Jarvis.

Define fixtures compartidos (settings, mock_settings, mock_overlay, tmp_wav)
y los marcadores personalizados (unit, e2e, integration). Se ejecuta
automáticamente al invocar `pytest` desde la raíz del proyecto.
"""

import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────
# Path bootstrap: asegura que `src/` y la raíz del proyecto estén en
# sys.path para que `from handlers...` y `from main import ...` funcionen
# sin necesidad de instalar el paquete. El código vive en `src/`
# (alineado con los permisos de edición `**/src/**` de los agentes
# @dev / @dev_senior) y se importa como módulos top-level.
# ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SRC), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────────
# Carga de variables de entorno desde .env — DEBE ejecutarse a nivel
# de módulo (no dentro de fixtures ni de tests) para que las fixtures
# E2E que consultan `os.getenv("GEMINI_API_KEY")`,
# `os.getenv("AZURE_SPEECH_KEY")` y `os.getenv("AZURE_SPEECH_REGION")`
# vean los valores reales del archivo `.env` y NO hagan un
# `pytest.skip()` falso cuando el .env existe pero el shell no
# heredó sus variables. pytest importa `conftest.py` antes de
# recolectar/evaluar cualquier fixture, por lo que cualquier
# `os.getenv()` posterior ya tendrá las variables disponibles.
# ──────────────────────────────────────────────────────────────────
load_dotenv()


# ──────────────────────────────────────────────────────────────────
# Marcadores personalizados — registrados para evitar warnings.
# ──────────────────────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    """Registra los markers personalizados en la sesión de pytest."""
    config.addinivalue_line(
        "markers",
        "unit: tests unitarios (sin red, sin servidores, mocks con unittest.mock).",
    )
    config.addinivalue_line(
        "markers",
        "e2e: tests end-to-end (requieren API keys reales en .env y servidor opencode levantado).",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests de integración (requieren red pero no servidor opencode).",
    )


# ──────────────────────────────────────────────────────────────────
# Fixtures compartidos
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def settings() -> dict:
    """Carga la configuración real desde config/settings.json.

    Retorna el dict completo tal como lo consume el orquestador.
    Útil cuando el test quiere verificar comportamiento con la config
    real del proyecto.
    """
    settings_path = _PROJECT_ROOT / "config" / "settings.json"
    with open(settings_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_settings() -> dict:
    """Settings sintéticos para tests que no quieren depender del archivo.

    Replica la estructura de config/settings.json con valores válidos,
    incluyendo la sección ``logging`` introducida en Fase 9.
    """
    return {
        "gemini": {
            "stt_model_primary": "gemini-3.1-flash-lite",
            "stt_model_fallback": "gemini-2.5-flash-lite",
            "tts_model": "gemini-3.1-flash-tts-preview",
            "tts_voice": "Charon",
            "tts_circuit_breaker_cooldown_seconds": 1800,
            "stt_prompt": "Transcribe el siguiente audio al español rioplatense.",
        },
        "opencode": {
            "agent": "asistente_voz",
            "model_fallback": "opencode/big-pickle",
            "timeout_ms": 120000,
            "max_session_messages": 10,
        },
        "azure": {
            "voice": "es-AR-TomasNeural",
            "locale": "es-AR",
            "output_format": "audio-24khz-48kbitrate-mono-mp3",
        },
        "audio": {
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "recording_filename": "comando.wav",
        },
        "hotkey": "alt+v",
        "logging": {
            "filename": "logs/cortex.log",
            "max_bytes": 5242880,
            "backup_count": 3,
            "level": "INFO",
        },
    }


@pytest.fixture
def mock_overlay():
    """Mockea ``main.OverlayChip`` retornando un MagicMock.

    Evita que ``VoiceAssistant.__init__`` cree un thread real con
    ``tk.Tk()`` durante los tests. La instancia retornada es un
    MagicMock con sub-mocks ``start``, ``show``, ``hide``, ``set_state``,
    ``destroy`` que pueden ser asserted con ``assert_called_once_with``.

    Uso::

        def test_foo(mock_overlay):
            # self._overlay.show será MagicMock y sus llamadas registrables
            ...
            mock_overlay.show.assert_called_once_with("recording")

    Yields:
        La instancia mock de OverlayChip (no la clase).
    """
    with patch("main.OverlayChip") as mock_cls:
        mock_cls.return_value = MagicMock(name="OverlayChipInstance")
        yield mock_cls.return_value


@pytest.fixture
def tmp_wav(tmp_path: Path) -> str:
    """Genera un archivo WAV silencioso válido en tmp_path.

    Formato: 24 kHz, 1 canal, 16-bit (s16le), 1 segundo de silencio.
    Retorna la ruta absoluta (str) — usable por GeminiSTTClient y
    AudioManager.stop_recording.
    """
    wav_path = tmp_path / "test_silence.wav"
    sample_rate = 24000
    n_samples = sample_rate  # 1 segundo

    audio_data = np.zeros(n_samples, dtype=np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return str(wav_path)
