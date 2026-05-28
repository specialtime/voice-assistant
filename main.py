from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from pathlib import Path

from services.azure_service import AzureSpeechService
from services.opencode_client import OpenCodeClient
from utils.audio_recorder import AudioRecorder
from utils.session_manager import SessionManager

ASSISTANT_DIR_NAME = ".opencode-voz"
DEFAULT_PORT = 4096
HOTKEY = "ctrl+alt+v"
AGENT_NAME = "asistente_voz"

ASSISTANT_PROMPT = """# Asistente de voz
Responde con texto claro y natural para la síntesis de voz.
Si necesitas ejecutar comandos, explica la acción de forma breve y natural.
No utilices SSML ni etiquetas XML.
"""


def ensure_isolated_environment(base_dir: Path) -> dict[str, Path]:
    assistant_dir = base_dir / ASSISTANT_DIR_NAME
    agents_dir = assistant_dir / "agents"
    memory_dir = assistant_dir / "memory"

    agents_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    config_path = assistant_dir / "config.json"
    if not config_path.exists():
        config = {
            "server": {"host": "127.0.0.1", "port": DEFAULT_PORT},
            "plugins": {
                "opencode-mem": {
                    "enabled": True,
                    "storage_dir": str(memory_dir),
                }
            },
        }
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    agent_prompt_path = agents_dir / f"{AGENT_NAME}.md"
    if not agent_prompt_path.exists():
        agent_prompt_path.write_text(ASSISTANT_PROMPT, encoding="utf-8")

    return {
        "assistant_dir": assistant_dir,
        "config_path": config_path,
        "agent_prompt_path": agent_prompt_path,
        "memory_dir": memory_dir,
    }


class VoiceAssistantApp:
    def __init__(self, base_dir: Path) -> None:
        paths = ensure_isolated_environment(base_dir)
        self.paths = paths
        self.session_manager = SessionManager(paths["assistant_dir"] / "session.json")
        self.audio_recorder = AudioRecorder(temp_dir=paths["assistant_dir"] / "tmp")
        self.azure_service = AzureSpeechService(
            speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            speech_region=os.getenv("AZURE_SPEECH_REGION", ""),
            voice_name=os.getenv("AZURE_TTS_VOICE", "es-ES-ElviraNeural"),
            voice_style=os.getenv("AZURE_TTS_STYLE"),
        )

        endpoint = os.getenv("OPENCODE_ENDPOINT", "http://127.0.0.1:4096/chat")
        self.opencode_client = OpenCodeClient(endpoint=endpoint, session_manager=self.session_manager)
        self.opencode_client.agent_name = AGENT_NAME
        self.running = True
        self.shutdown_event = threading.Event()

    def handle_toggle(self) -> None:
        if not self.audio_recorder.is_recording:
            self.audio_recorder.start_recording()
            print("[assistant] Grabación iniciada")
            return

        wav_file = self.audio_recorder.stop_recording()
        print(f"[assistant] Grabación finalizada: {wav_file}")
        user_text = self.azure_service.transcribe_wav(wav_file)
        if not user_text:
            print("[assistant] No se detectó voz útil")
            return

        print(f"[assistant] Usuario: {user_text}")
        response_text = self.opencode_client.send_prompt(user_text)
        self.azure_service.speak_text(response_text)

    def _shutdown(self, *_: object) -> None:
        self.running = False
        if self.audio_recorder.is_recording:
            try:
                self.audio_recorder.stop_recording()
            except RuntimeError:
                pass
        self.shutdown_event.set()

    def run(self) -> None:
        try:
            import keyboard  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("keyboard is required to register the global hotkey") from exc

        keyboard.add_hotkey(HOTKEY, self.handle_toggle)
        print(f"[assistant] Listo. Usa {HOTKEY} para iniciar/detener la grabación")

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        while self.running:
            self.shutdown_event.wait(timeout=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asistente de voz con Azure + OpenCode")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directorio donde se creará .opencode-voz (por defecto: actual)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = VoiceAssistantApp(base_dir=Path(args.base_dir).resolve())
    app.run()


if __name__ == "__main__":
    main()
