from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import requests

from utils.session_manager import SessionManager


@dataclass
class OpenCodeClient:
    endpoint: str
    session_manager: SessionManager
    timeout: int = 30
    agent_name: str = "asistente_voz"
    ssml_lang: str = "es-ES"
    ssml_voice_name: str = "es-ES-ElviraNeural"

    def _to_ssml(self, text: str) -> str:
        if text.lstrip().startswith("<speak"):
            try:
                root = ElementTree.fromstring(text)
                if root.tag.endswith("speak"):
                    return text
            except ElementTree.ParseError:
                pass

        escaped = escape(unescape(text), {'"': "&quot;", "'": "&apos;"})
        return (
            f'<speak version="1.0" xml:lang="{self.ssml_lang}">'
            f"<voice name=\"{self.ssml_voice_name}\">{escaped}</voice>"
            "</speak>"
        )

    def _extract_response(self, data: dict) -> str:
        candidates = [
            data.get("ssml"),
            data.get("response_ssml"),
            data.get("response"),
            data.get("message"),
        ]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            candidates.append(message.get("ssml"))
            candidates.append(message.get("content"))

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return self._to_ssml(candidate.strip())

        raise RuntimeError("OpenCode returned an empty response")

    def send_prompt(self, prompt: str) -> str:
        payload = {
            "agent": self.agent_name,
            "input": prompt,
            "thread_id": self.session_manager.get_thread_id(),
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to contact OpenCode at {self.endpoint}") from exc

        new_thread_id = data.get("thread_id") or data.get("id")
        if isinstance(new_thread_id, str) and new_thread_id:
            self.session_manager.set_thread_id(new_thread_id)

        return self._extract_response(data)
