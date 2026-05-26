from __future__ import annotations

from dataclasses import dataclass

import requests

from utils.session_manager import SessionManager


@dataclass
class OpenCodeClient:
    endpoint: str
    session_manager: SessionManager
    timeout: int = 30

    def _to_ssml(self, text: str) -> str:
        if text.lstrip().startswith("<speak"):
            return text

        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
        return (
            '<speak version="1.0" xml:lang="es-ES">'
            f"<voice name=\"es-ES-ElviraNeural\">{escaped}</voice>"
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

        raise RuntimeError("OpenCode devolvió una respuesta vacía")

    def send_prompt(self, prompt: str) -> str:
        payload = {
            "agent": "asistente_voz",
            "input": prompt,
            "thread_id": self.session_manager.get_thread_id(),
        }

        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        new_thread_id = data.get("thread_id") or data.get("id")
        if isinstance(new_thread_id, str) and new_thread_id:
            self.session_manager.set_thread_id(new_thread_id)

        return self._extract_response(data)
