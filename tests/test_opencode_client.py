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

    @patch("services.opencode_client.requests.post")
    def test_send_prompt_updates_thread_and_returns_ssml(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "thread_id": "thread-abc",
            "ssml": '<speak version="1.0" xml:lang="es-ES"><voice name="es-ES-ElviraNeural">ok</voice></speak>',
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = client.send_prompt("hola")

        mock_post.assert_called_once_with(
            "http://127.0.0.1:4096/agents/asistente_voz/chat",
            json={"input": "hola", "thread_id": None},
            timeout=client.timeout,
        )
        self.assertEqual("thread-abc", client.session_manager.get_thread_id())
        self.assertTrue(response.startswith("<speak"))

    @patch("services.opencode_client.requests.post")
    def test_plain_text_response_is_wrapped_as_ssml(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "abre calculadora"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = client.send_prompt("hola")

        self.assertIn("<speak", response)
        self.assertIn("abre calculadora", response)

    @patch("services.opencode_client.requests.post")
    def test_network_errors_raise_descriptive_exception(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_post.side_effect = requests.RequestException("boom")

        with self.assertRaisesRegex(RuntimeError, "Failed to contact OpenCode"):
            client.send_prompt("hola")

    @patch("services.opencode_client.requests.post")
    def test_malformed_ssml_is_escaped_and_wrapped(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"ssml": "<speak><voice>bad"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = client.send_prompt("hola")

        self.assertIn("&lt;speak&gt;&lt;voice&gt;bad", response)

    @patch("services.opencode_client.requests.post")
    def test_preescaped_entities_are_not_double_escaped(self, mock_post: MagicMock) -> None:
        client = self._client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "&lt;ok&gt;"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = client.send_prompt("hola")

        self.assertIn("&lt;ok&gt;", response)
        self.assertNotIn("&amp;lt;ok&amp;gt;", response)


if __name__ == "__main__":
    unittest.main()
