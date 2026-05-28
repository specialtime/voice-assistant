from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from services.opencode_client import OpenCodeClient
from utils.session_manager import SessionManager


class OpenCodeClientTests(unittest.TestCase):
    def _client(self) -> OpenCodeClient:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        manager = SessionManager(Path(tmp.name) / "session.json", clear_on_startup=True, clear_on_exit=False)
        return OpenCodeClient(endpoint="http://127.0.0.1:4096/chat", session_manager=manager)

    def _mock_session_and_prompt(self, mock_post: MagicMock, prompt_payload: dict, session_id: str = "thread-abc") -> None:
        session_response = MagicMock()
        session_response.json.return_value = {"id": session_id}
        session_response.raise_for_status.return_value = None
        prompt_response = MagicMock()
        prompt_response.json.return_value = prompt_payload
        prompt_response.raise_for_status.return_value = None
        mock_post.side_effect = [session_response, prompt_response]

    @patch("services.opencode_client.requests.post")
    def test_send_prompt_updates_thread_and_returns_text(self, mock_post: MagicMock) -> None:
        client = self._client()
<<<<<<< HEAD
        self._mock_session_and_prompt(
            mock_post,
            {
                "id": "msg-1",
                "ssml": '<speak version="1.0" xml:lang="es-ES"><voice name="es-ES-ElviraNeural">ok</voice></speak>',
            },
        )
=======
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "thread_id": "thread-abc",
            "response": "ok",
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
>>>>>>> origin/main

        response = client.send_prompt("hola")

        self.assertEqual("thread-abc", client.session_manager.get_thread_id())
<<<<<<< HEAD
        self.assertTrue(response.startswith("<speak"))
        self.assertEqual(2, len(mock_post.call_args_list))
        session_call = mock_post.call_args_list[0]
        prompt_call = mock_post.call_args_list[1]
        self.assertEqual("http://127.0.0.1:4096/session", session_call.args[0])
        self.assertEqual({"agent": "asistente_voz"}, session_call.kwargs["json"])
        self.assertEqual("http://127.0.0.1:4096/session/thread-abc/prompt", prompt_call.args[0])
        self.assertEqual({"parts": [{"type": "text", "text": "hola"}]}, prompt_call.kwargs["json"])
=======
        self.assertEqual("ok", response)
>>>>>>> origin/main

    @patch("services.opencode_client.requests.post")
    def test_plain_text_response_is_returned(self, mock_post: MagicMock) -> None:
        client = self._client()
        self._mock_session_and_prompt(
            mock_post,
            {"parts": [{"type": "text", "text": "abre calculadora"}]},
        )

        response = client.send_prompt("hola")

        self.assertEqual("abre calculadora", response)

    @patch("services.opencode_client.requests.post")
    def test_network_errors_raise_descriptive_exception(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_post.side_effect = requests.RequestException("boom")

        with self.assertRaisesRegex(RuntimeError, "Failed to contact OpenCode"):
            client.send_prompt("hola")

    @patch("services.opencode_client.requests.post")
    def test_ssml_response_is_stripped_to_text(self, mock_post: MagicMock) -> None:
        client = self._client()
<<<<<<< HEAD
        self._mock_session_and_prompt(
            mock_post,
            {"parts": [{"type": "text", "text": "<speak><voice>bad"}]},
        )
=======
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '<speak version="1.0"><voice name="es-ES-ElviraNeural">hola mundo</voice></speak>'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
>>>>>>> origin/main

        response = client.send_prompt("hola")

        self.assertEqual("hola mundo", response)

    @patch("services.opencode_client.requests.post")
    def test_malformed_ssml_is_cleaned_when_possible(self, mock_post: MagicMock) -> None:
        client = self._client()
<<<<<<< HEAD
        self._mock_session_and_prompt(
            mock_post,
            {"parts": [{"type": "text", "text": "&lt;ok&gt;"}]},
        )
=======
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "<speak><voice>bad"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
>>>>>>> origin/main

        response = client.send_prompt("hola")

        self.assertEqual("bad", response)


if __name__ == "__main__":
    unittest.main()
