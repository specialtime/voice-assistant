from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.session_manager import SessionManager


class SessionManagerTests(unittest.TestCase):
    def test_reset_on_startup_clears_previous_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_file = Path(tmp) / "session.json"
            session_file.write_text('{"thread_id": "old-thread"}', encoding="utf-8")

            manager = SessionManager(session_file, clear_on_startup=True, clear_on_exit=False)

            self.assertIsNone(manager.get_thread_id())
            self.assertIn('"thread_id": null', session_file.read_text(encoding="utf-8"))

    def test_set_thread_id_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_file = Path(tmp) / "session.json"
            manager = SessionManager(session_file, clear_on_startup=True, clear_on_exit=False)

            manager.set_thread_id("thread-123")

            self.assertEqual("thread-123", manager.get_thread_id())
            self.assertIn('"thread_id": "thread-123"', session_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
