from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from urllib.parse import urlsplit, urlunsplit
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

    def _base_url(self) -> str:
        raw = self.endpoint.rstrip("/")
        parsed = urlsplit(raw)
        if not parsed.scheme:
            return raw
        path = parsed.path or ""
        for suffix in ("/session/chat", "/chat", "/session"):
            if path.endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                break
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def _session_url(self) -> str:
        return f"{self._base_url()}/session"

    def _prompt_url(self, thread_id: str) -> str:
        return f"{self._base_url()}/session/{thread_id}/prompt"

    def _post_json(self, url: str, payload: dict) -> dict:
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to contact OpenCode at {url}") from exc

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

    def _extract_parts_text(self, parts: list[dict[str, object]] | None) -> str | None:
        if not isinstance(parts, list):
            return None
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            content = part.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None

    def _extract_response(self, data: dict) -> str:
        parts_text = self._extract_parts_text(data.get("parts"))
        if parts_text:
            return self._to_ssml(parts_text)

        message = data.get("message")
        if isinstance(message, dict):
            parts_text = self._extract_parts_text(message.get("parts"))
            if parts_text:
                return self._to_ssml(parts_text)
            for key in ("ssml", "response_ssml", "response", "content", "text"):
                candidate = message.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return self._to_ssml(candidate.strip())

        candidates = [
            data.get("ssml"),
            data.get("response_ssml"),
            data.get("response"),
            data.get("message"),
        ]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                parts_text = self._extract_parts_text(message.get("parts"))
                if parts_text:
                    return self._to_ssml(parts_text)
                candidates.append(message.get("ssml"))
                candidates.append(message.get("content"))

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return self._to_ssml(candidate.strip())

        raise RuntimeError("OpenCode returned an empty response")

    def send_prompt(self, prompt: str) -> str:
        thread_id = self.session_manager.get_thread_id()
        if not thread_id:
            session_payload = {"agent": self.agent_name}
            session_data = self._post_json(self._session_url(), session_payload)
            thread_id = session_data.get("id")
            if not isinstance(thread_id, str) or not thread_id:
                thread_id = session_data.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                self.session_manager.set_thread_id(thread_id)
            else:
                raise RuntimeError("OpenCode did not return a session id")

        payload = {"parts": [{"type": "text", "text": prompt}]}
        data = self._post_json(self._prompt_url(thread_id), payload)
        return self._extract_response(data)
