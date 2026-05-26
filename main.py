from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from services.azure_service import AzureSpeechService
from services.opencode_client import OpenCodeClient
from utils.audio_recorder import AudioRecorder
from utils.session_manager import SessionManager

ASSISTANT_DIR_NAME = ".opencode-voz"
DEFAULT_PORT = 4096
HOTKEY = "ctrl+alt+v"

ASSISTANT_PROMPT = """# Asistente de voz
Siempre responde en SSML válido para Azure Speech.
Si necesitas ejecutar comandos, explica la acción de forma breve y natural.
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

    agent_prompt_path = agents_dir / "asistente_voz.md"
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
        )

        endpoint = os.getenv("OPENCODE_ENDPOINT", "http://127.0.0.1:4096/chat")
        self.opencode_client = OpenCodeClient(endpoint=endpoint, session_manager=self.session_manager)
        self.running = True

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
        ssml = self.opencode_client.send_prompt(user_text)
        self.azure_service.speak_ssml(ssml)

    def _shutdown(self, *_: object) -> None:
        self.running = False
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()

    def run(self) -> None:
        try:
            import keyboard  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("keyboard es obligatorio para registrar el atajo global") from exc

        keyboard.add_hotkey(HOTKEY, self.handle_toggle)
        print(f"[assistant] Listo. Usa {HOTKEY} para iniciar/detener la grabación")

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        while self.running:
            time.sleep(0.1)


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
