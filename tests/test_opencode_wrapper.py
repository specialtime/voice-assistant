"""Tests unitarios para start_opencode_hidden.py.

El wrapper es un script que mantiene ``opencode serve`` corriendo como
proceso hijo (pythonw.exe) y lo reinicia si crashea, con un delay de
``RESTART_DELAY`` segundos entre intentos.

Estrategia de testing
---------------------
- Se mockean ``subprocess.Popen`` y ``time.sleep`` con
  ``unittest.mock.patch`` (nunca se lanza el binario real de opencode,
  ni se esperan 5s entre iteraciones).
- Para cortar el ``while True`` sin colgar el test, se hace que
  ``mock_sleep.side_effect`` levante ``StopIteration`` en la iteración
  N. Como ``time.sleep()`` está **fuera** del ``try/except Exception``
  del wrapper, la excepción propaga fuera del loop y termina el test.
  (Si la levantáramos en ``mock_popen.side_effect``, el ``except
  Exception`` del wrapper la cazaría y el loop continuaría.)
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Asegurar que la raíz del proyecto está en sys.path (conftest ya lo hace,
# pero por si pytest invoca este archivo en modo standalone).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import start_opencode_hidden  # noqa: E402


def _make_proc(exit_code: int = 0, pid: int = 9999) -> MagicMock:
    """Crea un mock de ``subprocess.Popen`` con ``wait()`` y ``stdout`` listos.

    ``stdout=None`` evita que el wrapper intente leer del pipe — no nos
    interesa el contenido de la salida en estos tests.
    """
    proc = MagicMock(name=f"proc(exit_code={exit_code})")
    proc.pid = pid
    proc.wait.return_value = exit_code
    proc.stdout = None
    return proc


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestStartOpencodeHidden:
    """Suite de tests para el wrapper de ``opencode serve``."""

    @patch("start_opencode_hidden.time.sleep")
    @patch("start_opencode_hidden.subprocess.Popen")
    def test_popen_called_with_correct_args(
        self, mock_popen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """``Popen`` se llama con los args/flags exactos del wrapper.

        Verifica:
        - Comando: ``[OPENCODE_EXE, 'serve', '--port', PORT, '--hostname', HOSTNAME]``
        - ``cwd=OPENCODE_DIR``
        - ``creationflags=subprocess.CREATE_NO_WINDOW`` (oculta ventana en Windows)
        - ``stdout=subprocess.PIPE``
        - ``stderr=subprocess.STDOUT`` (mezcla stderr en stdout para diagnóstico)
        """
        proc_ok = _make_proc(exit_code=0)
        mock_popen.return_value = proc_ok
        # Iter 1: Popen #1 → wait=0 → log → sleep #1 (StopIteration) → sale del loop.
        mock_sleep.side_effect = StopIteration("stop loop after 1 iteration")

        with pytest.raises(StopIteration):
            start_opencode_hidden.main()

        # Exactamente 1 intento de Popen
        assert mock_popen.call_count == 1

        # Args del comando (1 solo arg posicional: la lista del comando)
        call_args, call_kwargs = mock_popen.call_args
        assert call_args[0] == [
            start_opencode_hidden.OPENCODE_EXE,
            "serve",
            "--port",
            str(start_opencode_hidden.PORT),
            "--hostname",
            start_opencode_hidden.HOSTNAME,
        ]

        # Kwargs de flags/contexto
        assert call_kwargs["cwd"] == start_opencode_hidden.OPENCODE_DIR
        assert call_kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        assert call_kwargs["stdout"] == subprocess.PIPE
        assert call_kwargs["stderr"] == subprocess.STDOUT

    @patch("start_opencode_hidden.time.sleep")
    @patch("start_opencode_hidden.subprocess.Popen")
    def test_env_vars_set_on_popen(
        self, mock_popen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """El ``env`` pasado a ``Popen`` tiene las 4 env vars críticas seteadas.

        El wrapper setea ``USERPROFILE``, ``HOME``, ``APPDATA`` y
        ``LOCALAPPDATA`` para
        que ``opencode.exe`` encuentre ``~/.local/share`` y el registro
        de providers de Anthropic/Gemini/OpenCode.

        Verifica presencia de las 4 keys y los valores hardcoded del
        usuario actual.
        """
        proc_ok = _make_proc(exit_code=0)
        mock_popen.return_value = proc_ok
        mock_sleep.side_effect = StopIteration("stop loop after 1 iteration")

        with pytest.raises(StopIteration):
            start_opencode_hidden.main()

        assert mock_popen.call_count == 1
        env = mock_popen.call_args.kwargs["env"]

        # Las 4 env vars críticas deben estar presentes
        assert "USERPROFILE" in env
        assert "HOME" in env
        assert "APPDATA" in env
        assert "LOCALAPPDATA" in env

        # Y con los valores correctos (hardcoded en el wrapper)
        assert env["USERPROFILE"] == r"C:\Users\crist"
        assert env["HOME"] == r"C:\Users\crist"
        assert env["APPDATA"] == r"C:\Users\crist\AppData\Roaming"
        assert env["LOCALAPPDATA"] == r"C:\Users\crist\AppData\Local"

    @patch("start_opencode_hidden.time.sleep")
    @patch("start_opencode_hidden.subprocess.Popen")
    def test_loop_restarts_after_process_exit(
        self, mock_popen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """El loop se reinicia tras un crash de ``opencode serve``.

        Escenario:
        - ``proc1.wait()`` retorna 1 (opencode crasheó con error)
        - ``proc2.wait()`` retorna 0 (segundo intento terminó OK)

        El watchdog del wrapper debe relanzar ``Popen`` tras el crash.
        Verifica que ``Popen`` se llamó 2 veces y ``sleep`` se llamó
        1 vez entre los dos intentos (más 1 vez final que sale del loop).
        """
        proc1 = _make_proc(exit_code=1, pid=1001)  # crash
        proc2 = _make_proc(exit_code=0, pid=1002)  # exit OK
        mock_popen.side_effect = [proc1, proc2]
        # Iter 1: Popen #1 → wait=1 → log → sleep #1 (None) → loop
        # Iter 2: Popen #2 → wait=0 → log → sleep #2 (StopIteration) → sale
        mock_sleep.side_effect = [None, StopIteration("stop loop after 2 iterations")]

        with pytest.raises(StopIteration):
            start_opencode_hidden.main()

        # Popen se llamó 2 veces (watchdog reinició tras el crash)
        assert mock_popen.call_count == 2
        # sleep se llamó 2 veces (1 OK entre Popens + 1 que sale del loop)
        assert mock_sleep.call_count == 2
        # Ambos wait() se ejecutaron
        proc1.wait.assert_called_once()
        proc2.wait.assert_called_once()

    @patch("start_opencode_hidden.time.sleep")
    @patch("start_opencode_hidden.subprocess.Popen")
    def test_loop_continues_on_popen_exception(
        self, mock_popen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Si ``Popen`` lanza ``OSError`` (ej: binario no encontrado), el
        ``except Exception`` del wrapper captura, loguea, y el loop continúa
        con el siguiente intento.

        Verifica que el segundo ``Popen`` se intentó y que ``proc_ok.wait()``
        se ejecutó (prueba que el segundo intento fue completo y exitoso).
        """
        proc_ok = _make_proc(exit_code=0, pid=2002)
        # Iter 1: Popen #1 → OSError → except → log → sleep #1 (None) → loop
        # Iter 2: Popen #2 → proc_ok → wait=0 → log → sleep #2 (StopIteration) → sale
        mock_popen.side_effect = [OSError("opencode.exe not found"), proc_ok]
        mock_sleep.side_effect = [None, StopIteration("stop loop after 2 iterations")]

        with pytest.raises(StopIteration):
            start_opencode_hidden.main()

        # Popen se intentó 2 veces: la primera falló con OSError, la segunda OK
        assert mock_popen.call_count == 2
        # El segundo intento (proc_ok) sí completó: wait() fue llamado
        proc_ok.wait.assert_called_once()
        # sleep se llamó 2 veces (entre los dos Popens + la que sale del loop)
        assert mock_sleep.call_count == 2
