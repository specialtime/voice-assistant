"""Tests unitarios para handlers/overlay.py:OverlayChip.

Mockea `tkinter` y `threading` con `unittest.mock.patch` — nunca crea
ventanas reales. Cubre la API thread-safe (show/hide/set_state/destroy
encolan comandos en queue.Queue) y el dispatcher interno _handle_command
junto con la actualización visual _update_visual.

Cubre:
- Fase 9: API pública, dispatcher, estados recording/processing.
- Fase 12.C: estado speaking (color verde + texto "Hablando...").
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from handlers.overlay import (
    OverlayChip,
    _DOT_PROCESSING,
    _DOT_RECORDING,
    _DOT_SPEAKING,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def overlay() -> OverlayChip:
    """Crea una instancia de OverlayChip sin iniciar el hilo tkinter.

    Evita llamar a .start() para no abrir un tk.Tk() real durante los tests.
    Sólo ejercitamos la API pública (encolado de comandos) y los métodos
    internos (_handle_command, _update_visual) con mocks de los widgets.
    """
    return OverlayChip()


@pytest.fixture
def overlay_with_widgets(overlay: OverlayChip) -> OverlayChip:
    """Overlay con _root/_dot/_label mockeados para tests de _handle_command.

    Los métodos internos del chip (animate, update_visual) tocan los
    widgets de tkinter — para no abrir ventana real, los substituimos
    por MagicMock antes de invocar el dispatcher.
    """
    overlay._root = MagicMock(name="root")
    overlay._dot = MagicMock(name="dot")
    overlay._label = MagicMock(name="label")
    return overlay


# ──────────────────────────────────────────────────────────────────
# 1) API pública — encolado de comandos (thread-safe)
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestOverlayChipCommandEnqueue:
    """Verifica que show/hide/set_state/destroy encolan en la queue."""

    def test_show_enqueues_command(self, overlay: OverlayChip) -> None:
        """show('recording') debe poner ('show', 'recording') en la cola."""
        overlay.show("recording")

        cmd = overlay._queue.get_nowait()
        assert cmd == ("show", "recording")

    def test_hide_enqueues_command(self, overlay: OverlayChip) -> None:
        """hide() debe poner ('hide',) en la cola."""
        overlay.hide()

        cmd = overlay._queue.get_nowait()
        assert cmd == ("hide",)

    def test_set_state_enqueues_command(self, overlay: OverlayChip) -> None:
        """set_state('processing') debe poner ('set_state', 'processing') en la cola."""
        overlay.set_state("processing")

        cmd = overlay._queue.get_nowait()
        assert cmd == ("set_state", "processing")

    def test_destroy_enqueues_command(self, overlay: OverlayChip) -> None:
        """destroy() debe poner ('destroy',) en la cola."""
        overlay.destroy()

        cmd = overlay._queue.get_nowait()
        assert cmd == ("destroy",)


# ──────────────────────────────────────────────────────────────────
# 2) Dispatcher interno _handle_command
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestOverlayChipHandleCommand:
    """Verifica el routing de _handle_command según el primer elemento del tuple."""

    def test_handle_command_show(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """('show', state) → setea _current_state + llama _update_visual y _animate_show."""
        overlay = overlay_with_widgets

        with patch.object(overlay, "_update_visual") as mock_uv, \
             patch.object(overlay, "_animate_show") as mock_as:
            overlay._handle_command(("show", "recording"))

        assert overlay._current_state == "recording"
        mock_uv.assert_called_once()
        mock_as.assert_called_once()

    def test_handle_command_hide(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """('hide',) → llama _animate_hide (sin tocar _current_state)."""
        overlay = overlay_with_widgets
        overlay._current_state = "recording"  # estado previo

        with patch.object(overlay, "_animate_hide") as mock_ah:
            overlay._handle_command(("hide",))

        mock_ah.assert_called_once()
        # _current_state no debe mutar en hide
        assert overlay._current_state == "recording"

    def test_handle_command_set_state(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """('set_state', state) → actualiza _current_state + llama _update_visual."""
        overlay = overlay_with_widgets

        with patch.object(overlay, "_update_visual") as mock_uv:
            overlay._handle_command(("set_state", "processing"))

        assert overlay._current_state == "processing"
        mock_uv.assert_called_once()

    def test_handle_command_destroy(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """('destroy',) → _stop_pulse + cancela _anim_after_id + destruye root."""
        overlay = overlay_with_widgets
        overlay._anim_after_id = "fake_anim_id"
        overlay._pulse_after_id = "fake_pulse_id"

        # Capturamos referencia al root mock ANTES del dispatch, ya que
        # el handler setea self._root = None al final.
        root_mock = overlay._root

        with patch.object(overlay, "_stop_pulse") as mock_sp:
            overlay._handle_command(("destroy",))

        # _stop_pulse fue invocado
        mock_sp.assert_called_once()
        # _anim_after_id fue cancelado vía root.after_cancel
        root_mock.after_cancel.assert_any_call("fake_anim_id")
        # _anim_after_id queda en None
        assert overlay._anim_after_id is None
        # root.destroy fue llamado y self._root quedó en None
        root_mock.destroy.assert_called_once()
        assert overlay._root is None


# ──────────────────────────────────────────────────────────────────
# 3) Actualización visual _update_visual
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestOverlayChipUpdateVisual:
    """Verifica que _update_visual aplica el color del dot y texto correctos."""

    def test_update_visual_recording(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """Estado 'recording' → dot rojo + texto 'Grabando...'."""
        overlay = overlay_with_widgets
        overlay._current_state = "recording"

        overlay._update_visual()

        # El dot debe configurarse con fg rojo de recording
        overlay._dot.configure.assert_called_once_with(fg=_DOT_RECORDING)
        # El label debe configurarse con el texto "Grabando..."
        overlay._label.configure.assert_called_once_with(text="Grabando...")

    def test_update_visual_processing(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """Estado 'processing' → dot amarillo + texto 'Procesando...'."""
        overlay = overlay_with_widgets
        overlay._current_state = "processing"

        overlay._update_visual()

        # El dot debe configurarse con fg amarillo de processing
        overlay._dot.configure.assert_called_once_with(fg=_DOT_PROCESSING)
        # El label debe configurarse con el texto "Procesando..."
        overlay._label.configure.assert_called_once_with(text="Procesando...")

    def test_set_state_speaking_updates_visual(
        self, overlay_with_widgets: OverlayChip
    ) -> None:
        """set_state('speaking') → dot verde + texto 'Hablando...' (Fase 12.C).

        Verifica que el dispatcher _handle_command procesa el comando
        set_state con 'speaking' y aplica el color verde (#2ecc71) y
        el texto 'Hablando...' a los widgets. Se invoca _handle_command
        directamente para saltarse la cola y testear la lógica de
        actualización visual de forma síncrona.
        """
        overlay = overlay_with_widgets

        overlay._handle_command(("set_state", "speaking"))

        # El estado interno se actualizó a "speaking"
        assert overlay._current_state == "speaking"
        # El dot debe configurarse con fg verde de speaking
        overlay._dot.configure.assert_called_once_with(fg=_DOT_SPEAKING)
        # El label debe configurarse con el texto "Hablando..."
        overlay._label.configure.assert_called_once_with(text="Hablando...")


# ──────────────────────────────────────────────────────────────────
# 4) Inicio del hilo daemon
# ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestOverlayChipStart:
    """Verifica que start() lanza un threading.Thread daemon=True con target=_run_tk."""

    def test_start_creates_daemon_thread(self, overlay: OverlayChip) -> None:
        """start() crea un Thread daemon=True con target=self._run_tk y lo inicia."""
        with patch("handlers.overlay.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock(name="ThreadInstance")
            overlay.start()

        # Se instanció Thread una sola vez
        mock_thread_cls.assert_called_once()
        kwargs = mock_thread_cls.call_args.kwargs
        # daemon=True
        assert kwargs.get("daemon") is True
        # target=_run_tk (método del propio overlay)
        assert kwargs.get("target") == overlay._run_tk
        # Se llamó .start() sobre la instancia
        mock_thread_cls.return_value.start.assert_called_once()
        # self._thread queda apuntando a la instancia creada
        assert overlay._thread is mock_thread_cls.return_value
