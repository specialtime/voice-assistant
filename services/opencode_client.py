from __future__ import annotations

from dataclasses import dataclass
import re
from xml.etree import ElementTree

import requests

from utils.session_manager import SessionManager


@dataclass
class OpenCodeClient:
    endpoint: str
    session_manager: SessionManager
    timeout: int = 30
    agent_name: str = "asistente_voz"

    @staticmethod
    def _strip_tags(text: str) -> str:
        stripped = re.sub(r"<[^>]+>", "", text).strip()
        return stripped or text

    def _strip_ssml(self, text: str) -> str:
        if not text.lstrip().startswith("<speak"):
            return text

        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return self._strip_tags(text)

        if root.tag.endswith("speak"):
            stripped = "".join(root.itertext()).strip()
            return stripped or self._strip_tags(text)

        return text

    def _extract_response(self, data: dict) -> str:
        candidates = [
            data.get("response"),
            data.get("message"),
        ]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            candidates.append(message.get("content"))

        candidates.extend([data.get("ssml"), data.get("response_ssml")])

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return self._strip_ssml(candidate.strip())

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
