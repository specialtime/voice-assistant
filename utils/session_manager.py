from __future__ import annotations

import atexit
import json
from pathlib import Path


class SessionManager:
    def __init__(self, session_file: Path, clear_on_startup: bool = True, clear_on_exit: bool = True) -> None:
        self.session_file = Path(session_file)
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.thread_id: str | None = None

        if clear_on_startup:
            self.reset_session()
        else:
            self._load_session()

        if clear_on_exit:
            atexit.register(self.reset_session)

    def _load_session(self) -> None:
        if not self.session_file.exists():
            return

        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.reset_session()
            return
        self.thread_id = data.get("thread_id")

    def set_thread_id(self, thread_id: str | None) -> None:
        self.thread_id = thread_id
        self._persist()

    def get_thread_id(self) -> str | None:
        return self.thread_id

    def reset_session(self) -> None:
        self.thread_id = None
        self._persist()

    def _persist(self) -> None:
        data = {"thread_id": self.thread_id}
        self.session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
