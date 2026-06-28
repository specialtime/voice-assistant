"""Overlay chip visual para feedback de estado del asistente.

Implementa un chip flotante borderless con tkinter que aparece abajo al centro
de la pantalla (sobre la barra de tareas) cuando el asistente está grabando
o procesando. Animaciones de fade-in/fade-out + slide + pulso del indicador.

Estilo inspirado en Whisper Desktop.
"""

import queue
import threading
import tkinter as tk
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes visuales
# ---------------------------------------------------------------------------
_WINDOW_W = 240
_WINDOW_H = 50
_MARGIN_BOTTOM = 40  # px sobre la barra de tareas
_BG_COLOR = "#1a1a2e"
_BORDER_COLOR = "#2d2d4f"
_TEXT_COLOR = "#e0e0e0"
_DOT_RECORDING = "#e74c3c"  # rojo
_DOT_PROCESSING = "#f1c40f"  # amarillo
_DOT_SPEAKING = "#2ecc71"  # verde — el agente está hablando
_FONT = ("Segoe UI", 11)
_ANIM_DURATION_MS = 200
_ANIM_STEPS = 20
_PULSE_INTERVAL_MS = 800


class OverlayChip:
    """Chip visual flotante con animaciones de entrada/salida y pulso.

    Thread-safe: los métodos públicos (show, hide, set_state) se llaman
    desde el hilo del orquestador y encolan comandos via queue.Queue.
    El hilo interno de tkinter procesa los comandos via root.after(50, poll).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._label: tk.Label | None = None
        self._dot: tk.Label | None = None
        self._visible: bool = False
        self._alpha: float = 0.0
        self._base_y: int = 0
        self._current_state: str = "recording"
        self._pulse_after_id: str | None = None
        self._anim_after_id: str | None = None

    # ------------------------------------------------------------------
    # API pública (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Lanza el hilo daemon con la ventana tkinter."""
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        logger.info("OverlayChip thread started")

    def destroy(self) -> None:
        """Detiene el overlay limpiamente: cancela callbacks y destruye la ventana."""
        self._queue.put(("destroy",))

    def show(self, state: str = "recording") -> None:
        """Muestra el chip con el estado indicado (thread-safe)."""
        self._queue.put(("show", state))

    def hide(self) -> None:
        """Oculta el chip con animación (thread-safe)."""
        self._queue.put(("hide",))

    def set_state(self, state: str) -> None:
        """Cambia el estado visual sin mostrar/ocultar (thread-safe)."""
        self._queue.put(("set_state", state))

    # ------------------------------------------------------------------
    # Hilo interno de tkinter
    # ------------------------------------------------------------------

    def _run_tk(self) -> None:
        """Crea y ejecuta la ventana borderless. Bloqueante (mainloop)."""
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.0)

        # Frame contenedor
        frame = tk.Frame(
            self._root,
            bg=_BG_COLOR,
            highlightbackground=_BORDER_COLOR,
            highlightthickness=1,
        )
        frame.pack_propagate(False)
        frame.pack(fill="both", expand=True)

        # Indicador (punto)
        self._dot = tk.Label(
            frame,
            text="●",
            fg=_DOT_RECORDING,
            font=("Segoe UI", 14),
            bg=_BG_COLOR,
        )
        self._dot.pack(side="left", padx=(12, 4), pady=0)

        # Texto de estado
        self._label = tk.Label(
            frame,
            text="Grabando...",
            fg=_TEXT_COLOR,
            font=_FONT,
            bg=_BG_COLOR,
        )
        self._label.pack(side="left", padx=(4, 12), pady=0)

        self._position_window()
        self._root.after(50, self._poll_queue)
        logger.info("OverlayChip tkinter window created")
        self._root.mainloop()

    def _position_window(self) -> None:
        """Posiciona la ventana abajo al centro de la pantalla."""
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = (screen_w - _WINDOW_W) // 2
        self._base_y = screen_h - _WINDOW_H - _MARGIN_BOTTOM
        self._root.geometry(f"{_WINDOW_W}x{_WINDOW_H}+{x}+{self._base_y}")

    def _poll_queue(self) -> None:
        """Procesa comandos encolados cada 50ms y re-schedule."""
        if self._root is None:
            return
        try:
            while True:
                cmd = self._queue.get_nowait()
                self._handle_command(cmd)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    def _handle_command(self, cmd: tuple) -> None:
        """Despacha un comando recibido por la cola."""
        if cmd[0] == "show":
            self._current_state = cmd[1]
            self._update_visual()
            self._animate_show()
        elif cmd[0] == "hide":
            self._animate_hide()
        elif cmd[0] == "set_state":
            self._current_state = cmd[1]
            self._update_visual()
        elif cmd[0] == "destroy":
            self._stop_pulse()
            if self._anim_after_id is not None and self._root is not None:
                self._root.after_cancel(self._anim_after_id)
                self._anim_after_id = None
            if self._root is not None:
                self._root.destroy()
                self._root = None

    # ------------------------------------------------------------------
    # Actualización visual
    # ------------------------------------------------------------------

    def _update_visual(self) -> None:
        """Actualiza colores y texto según el estado actual."""
        if self._current_state == "recording":
            self._dot.configure(fg=_DOT_RECORDING)
            self._label.configure(text="Grabando...")
        elif self._current_state == "processing":
            self._dot.configure(fg=_DOT_PROCESSING)
            self._label.configure(text="Procesando...")
        elif self._current_state == "speaking":  # NUEVO
            self._dot.configure(fg=_DOT_SPEAKING)
            self._label.configure(text="Hablando...")

    # ------------------------------------------------------------------
    # Animaciones
    # ------------------------------------------------------------------

    def _animate_show(self) -> None:
        """Animación de fade-in + slide-up. Ease-out quad."""
        if self._visible:
            return
        self._visible = True
        self._alpha = 0.0
        self._root.attributes("-alpha", 0.0)
        start_y = self._base_y + 20
        self._root.geometry(
            f"{_WINDOW_W}x{_WINDOW_H}+{self._root.winfo_x()}+{start_y}"
        )

        def step(frame: int) -> None:
            if frame > _ANIM_STEPS:
                self._root.attributes("-alpha", 1.0)
                self._alpha = 1.0
                self._root.geometry(
                    f"{_WINDOW_W}x{_WINDOW_H}+{self._root.winfo_x()}+{self._base_y}"
                )
                self._start_pulse()
                return
            t = frame / _ANIM_STEPS
            eased = 1 - (1 - t) ** 2  # ease-out quad
            self._alpha = eased
            self._root.attributes("-alpha", eased)
            offset = int(20 * (1 - eased))
            self._root.geometry(
                f"{_WINDOW_W}x{_WINDOW_H}+{self._root.winfo_x()}+{self._base_y + offset}"
            )
            self._anim_after_id = self._root.after(
                _ANIM_DURATION_MS // _ANIM_STEPS, step, frame + 1
            )

        step(0)

    def _animate_hide(self) -> None:
        """Animación de fade-out + slide-down. Ease-in quad."""
        if not self._visible:
            return
        self._stop_pulse()
        self._visible = False

        def step(frame: int) -> None:
            if frame > _ANIM_STEPS:
                self._root.attributes("-alpha", 0.0)
                self._alpha = 0.0
                return
            t = frame / _ANIM_STEPS
            eased = t**2  # ease-in quad
            alpha_val = 1.0 - eased
            self._alpha = alpha_val
            self._root.attributes("-alpha", alpha_val)
            offset = int(20 * eased)
            self._root.geometry(
                f"{_WINDOW_W}x{_WINDOW_H}+{self._root.winfo_x()}+{self._base_y + offset}"
            )
            self._anim_after_id = self._root.after(
                _ANIM_DURATION_MS // _ANIM_STEPS, step, frame + 1
            )

        step(0)

    # ------------------------------------------------------------------
    # Pulso del indicador (solo en estado recording)
    # ------------------------------------------------------------------

    def _start_pulse(self) -> None:
        """Inicia el ciclo de pulso si el estado es recording."""
        if self._current_state != "recording":
            return
        self._pulse_step(0)

    def _pulse_step(self, phase: int) -> None:
        """Alterna el color del dot entre rojo normal y oscuro."""
        if not self._visible or self._current_state != "recording":
            return
        if phase == 0:
            self._dot.configure(fg=_DOT_RECORDING)
        elif phase == 1:
            self._dot.configure(fg="#c0392b")  # rojo más oscuro
        self._pulse_after_id = self._root.after(
            _PULSE_INTERVAL_MS // 2, self._pulse_step, 1 - phase
        )

    def _stop_pulse(self) -> None:
        """Detiene el ciclo de pulso."""
        if self._pulse_after_id is not None:
            self._root.after_cancel(self._pulse_after_id)
            self._pulse_after_id = None
