"""Tests unitarios para handlers/opencode_client.py.

Mockea `httpx.Client` con `unittest.mock.patch` — sin red.
Cubre el contrato de `OpenCodeClient` definido en IMPLEMENTATION.md §4.5,
incluyendo caching de sesión y failover big-pickle → deepseek-v4-flash.
"""

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from handlers.opencode_client import OpenCodeClient


def _ok_response(json_data: dict) -> MagicMock:
    """Crea un mock de httpx.Response con status 200 y JSON arbitrario."""
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.raise_for_status = MagicMock()
    response.json.return_value = json_data
    return response


def _err_response(status_code: int, text: str = "error") -> MagicMock:
    """Crea un mock de httpx.Response que lanza HTTPStatusError en raise_for_status."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} error",
        request=MagicMock(),
        response=response,
    )
    return response


def _message_payload(text: str) -> dict:
    """Payload JSON esperado de POST /session/{id}/message."""
    return {
        "info": {"role": "assistant"},
        "parts": [{"type": "text", "text": text}],
    }


_BASE_URL = "http://127.0.0.1:4096"


@pytest.mark.unit
class TestOpenCodeClient:
    """Suite de tests para OpenCodeClient con httpx mockeado."""

    @patch("handlers.opencode_client.httpx.Client")
    def test_ensure_session_creates_and_caches(self, mock_client_cls, mock_settings):
        """1ª ensure_session → POST /session. 2ª → cache (sin nuevo POST)."""
        mock_client = mock_client_cls.return_value
        mock_client.post.return_value = _ok_response({"id": "sess_abc123"})

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        # Primer call: crea sesión
        sid1 = client.ensure_session()
        assert sid1 == "sess_abc123"
        assert client.session_id == "sess_abc123"
        assert mock_client.post.call_count == 1

        # Segundo call: usa cache, NO hace POST
        sid2 = client.ensure_session()
        assert sid2 == "sess_abc123"
        assert mock_client.post.call_count == 1  # sigue siendo 1

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_success(self, mock_client_cls, mock_settings):
        """POST /session/{id}/message 200 → retorna texto del primer part type=text."""
        mock_client = mock_client_cls.return_value

        # Secuencia de calls:
        # 1) ensure_session → POST /session
        # 2) send_command → POST /session/{id}/message (primario)
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _ok_response(_message_payload("[STYLE: cheerful] Hola, abrí Chrome")),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        result = client.send_command("abrí chrome")

        assert result == "[STYLE: cheerful] Hola, abrí Chrome"
        assert mock_client.post.call_count == 2

        # El body del mensaje al primario NO debe incluir `model`: el server
        # opencode usa el `model:` del frontmatter del agente. Solo el fallback
        # override de `model` en el body (ver test_send_command_failover).
        msg_body = mock_client.post.call_args_list[1].kwargs["json"]
        assert "model" not in msg_body
        assert msg_body.get("model") is None
        assert msg_body["agent"] == "asistente_voz"
        assert msg_body["parts"] == [{"type": "text", "text": "abrí chrome"}]

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_failover(self, mock_client_cls, mock_settings):
        """Primario error → fallback 200 → usa el modelo fallback del settings."""
        mock_client = mock_client_cls.return_value

        # Flujo tras Parche C (reset_session + ensure_session antes del fallback):
        # 1) ensure_session OK (sesión inicial)
        # 2) primario falla (500)
        # 3) ensure_session OK (nueva sesión tras reset)
        # 4) fallback OK con respuesta del modelo fallback
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _err_response(500, "Server error"),
            _ok_response({"id": "sess_2"}),
            _ok_response(_message_payload("[STYLE: friendly] Listo, OK")),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        result = client.send_command("hola")

        assert result == "[STYLE: friendly] Listo, OK"
        assert mock_client.post.call_count == 4

        # Verificar que el body del fallback usa el modelo fallback del settings
        fallback_body = mock_client.post.call_args_list[3].kwargs["json"]
        fallback_provider, fallback_model = (
            mock_settings["opencode"]["model_fallback"].split("/", 1)
        )
        assert fallback_body["model"]["providerID"] == fallback_provider
        assert fallback_body["model"]["modelID"] == fallback_model

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_both_fail(self, mock_client_cls, mock_settings):
        """Primario y fallback fallan → RuntimeError."""
        mock_client = mock_client_cls.return_value

        # Flujo tras Parche C:
        # 1) ensure_session OK
        # 2) primario falla (500)
        # 3) ensure_session OK (nueva sesión tras reset)
        # 4) fallback también falla (500) → RuntimeError
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _err_response(500),
            _ok_response({"id": "sess_2"}),
            _err_response(500),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        with pytest.raises(RuntimeError, match=r"agente|fallaron"):
            client.send_command("hola")

        assert mock_client.post.call_count == 4

    @patch("handlers.opencode_client.httpx.Client")
    def test_reset_session(self, mock_client_cls, mock_settings):
        """reset_session() setea session_id a None y permite re-crear."""
        mock_client = mock_client_cls.return_value
        # Primer POST crea sesión "sess_abc", segundo POST crea "sess_def"
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_abc"}),
            _ok_response({"id": "sess_def"}),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        sid1 = client.ensure_session()
        assert sid1 == "sess_abc"
        assert client.session_id == "sess_abc"

        client.reset_session()
        assert client.session_id is None

        # Tras reset, ensure_session debe crear una nueva sesión
        sid2 = client.ensure_session()
        assert sid2 == "sess_def"
        assert mock_client.post.call_count == 2

    @patch("handlers.opencode_client.httpx.Client")
    def test_no_password_logged(
        self, mock_client_cls, mock_settings, caplog
    ):
        """La password de opencode NO debe aparecer en ningún log record."""
        secret_password = "SECRET_OPENCODE_PASSWORD_DO_NOT_LEAK_999"

        mock_client = mock_client_cls.return_value
        # Forzar error para que se logueen warnings/errors
        mock_client.post.return_value = _err_response(500)

        with caplog.at_level(logging.DEBUG, logger="handlers.opencode_client"):
            client = OpenCodeClient(mock_settings, secret_password, _BASE_URL)
            with pytest.raises(RuntimeError):
                client.ensure_session()  # fallará → loguea error

        all_logs = "\n".join(record.getMessage() for record in caplog.records)
        assert secret_password not in all_logs, (
            f"Password filtrada en logs: {[r.getMessage() for r in caplog.records]}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec B: Swap de modelo primario
    # ──────────────────────────────────────────────────────────────────
    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_primary_has_no_model_in_body(
        self, mock_client_cls, mock_settings
    ):
        """El body del POST al primario NO incluye `model`: el server opencode
        resuelve el modelo desde el frontmatter del agente. Si el primario
        quisiera override de modelo, se pasaría por body — pero ese contrato
        ya no se usa (model_primary fue eliminado de settings.json)."""
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _ok_response(_message_payload("[STYLE: cheerful] OK")),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        client.send_command("test")

        primary_body = mock_client.post.call_args_list[1].kwargs["json"]
        assert "model" not in primary_body
        # El cliente solo persiste el fallback como atributo; no debe exponer
        # ya un `_model_primary` (el setting ya no existe).
        assert not hasattr(client, "_model_primary") or client._model_primary is None

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_fallback_has_model_in_body(
        self, mock_client_cls, mock_settings
    ):
        """El body del POST al fallback SÍ incluye `model` con el
        model_fallback del settings (override explícito tras failover)."""
        mock_client = mock_client_cls.return_value
        # Flujo tras Parche C: 1) session 2) primary fail 3) new session 4) fallback OK
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _err_response(500, "primary failed"),
            _ok_response({"id": "sess_2"}),
            _ok_response(_message_payload("[STYLE: friendly] fallback OK")),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        client.send_command("test")

        fallback_body = mock_client.post.call_args_list[3].kwargs["json"]
        fallback_provider, fallback_model = (
            mock_settings["opencode"]["model_fallback"].split("/", 1)
        )
        assert "model" in fallback_body
        assert fallback_body["model"]["providerID"] == fallback_provider
        assert fallback_body["model"]["modelID"] == fallback_model

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_failover_uses_model_fallback(
        self, mock_client_cls, mock_settings
    ):
        """Si el primario falla, el body del fallback usa model_fallback del settings."""
        mock_client = mock_client_cls.return_value
        # Flujo tras Parche C: 1) session 2) primary fail 3) new session 4) fallback OK
        mock_client.post.side_effect = [
            _ok_response({"id": "sess_1"}),
            _err_response(500, "primary failed"),
            _ok_response({"id": "sess_2"}),
            _ok_response(_message_payload("[STYLE: friendly] fallback OK")),
        ]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        client.send_command("test")

        fallback_body = mock_client.post.call_args_list[3].kwargs["json"]
        fallback_provider, fallback_model = (
            mock_settings["opencode"]["model_fallback"].split("/", 1)
        )
        assert fallback_body["model"]["providerID"] == fallback_provider
        assert fallback_body["model"]["modelID"] == fallback_model

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec B: _parse_model() helper
    # ──────────────────────────────────────────────────────────────────
    def test_parse_model_valid(self):
        """'opencode-go/deepseek-v4-flash' → {providerID, modelID}."""
        result = OpenCodeClient._parse_model("opencode-go/deepseek-v4-flash")
        assert result == {"providerID": "opencode-go", "modelID": "deepseek-v4-flash"}

    def test_parse_model_valid_opencode_big_pickle(self):
        """'opencode/big-pickle' → {providerID, modelID}."""
        result = OpenCodeClient._parse_model("opencode/big-pickle")
        assert result == {"providerID": "opencode", "modelID": "big-pickle"}

    def test_parse_model_invalid_no_slash(self):
        """Sin '/' → ValueError."""
        with pytest.raises(ValueError, match=r"inválido|providerID/modelID"):
            OpenCodeClient._parse_model("invalid")

    def test_parse_model_invalid_empty(self):
        """String vacío → ValueError."""
        with pytest.raises(ValueError, match=r"inválido|providerID/modelID"):
            OpenCodeClient._parse_model("")

    def test_parse_model_invalid_only_slash(self):
        """Solo '/' → ValueError (providerID y modelID vacíos)."""
        with pytest.raises(ValueError, match=r"inválido|providerID/modelID"):
            OpenCodeClient._parse_model("/")

    def test_parse_model_invalid_empty_provider(self):
        """'/modelID' → ValueError (providerID vacío)."""
        with pytest.raises(ValueError, match=r"inválido|providerID/modelID"):
            OpenCodeClient._parse_model("/deepseek-v4-flash")

    def test_parse_model_multiple_slashes(self):
        """Múltiples '/' → split solo en el primero (modelID puede contener '/')."""
        result = OpenCodeClient._parse_model("opencode-go/foo/bar")
        assert result == {"providerID": "opencode-go", "modelID": "foo/bar"}

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec D: Compactación de sesión
    # ──────────────────────────────────────────────────────────────────
    @patch("handlers.opencode_client.httpx.Client")
    def test_session_compaction_after_max_messages(
        self, mock_client_cls, mock_settings
    ):
        """Tras max_session_messages comandos exitosos, session_id vuelve a None
        y _message_count vuelve a 0."""
        mock_client = mock_client_cls.return_value

        # Asegurar max_session_messages=10 en mock_settings para este test
        max_msgs = mock_settings["opencode"]["max_session_messages"]

        # 1) Primer ensure_session crea sesión
        # 2..N+1) send_command: una llamada a POST /session y (N) a /session/{id}/message
        # Tras el N-ésimo send_command exitoso, debe dispararse reset_session()
        # y la siguiente send_command debe crear una nueva sesión.
        # Construimos side_effect con suficientes respuestas:
        # - 1 sesión inicial
        # - max_msgs respuestas de message (primario OK)
        # - 1 nueva sesión tras compactación
        # - 1 respuesta de message (post-compactación, primario OK)
        responses = [_ok_response({"id": "sess_initial"})]
        for i in range(max_msgs + 1):
            responses.append(
                _ok_response(_message_payload(f"[STYLE: cheerful] msg {i}"))
            )

        mock_client.post.side_effect = responses

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        # Ejecutar max_msgs send_command — el último debe disparar compactación
        for i in range(max_msgs):
            text = client.send_command(f"comando {i}")
            assert text.startswith("[STYLE: cheerful]")

        # Tras el último send_command, el session_id debe haberse reseteado a None
        assert client.session_id is None
        assert client._message_count == 0

    @patch("handlers.opencode_client.httpx.Client")
    def test_session_compaction_creates_new_session(
        self, mock_client_cls, mock_settings
    ):
        """Tras compactación, el siguiente send_command crea sesión nueva (verifica
        que session_id cambió y que se llamó a POST /session de nuevo)."""
        mock_client = mock_client_cls.return_value

        max_msgs = mock_settings["opencode"]["max_session_messages"]

        # Sesión 1 (ensure_session inicial) + N mensajes (primario OK)
        # + Sesión 2 (nueva sesión tras compactación) + 1 mensaje post-compactación
        responses = [
            _ok_response({"id": "sess_old"}),  # 1ª sesión
        ]
        for i in range(max_msgs):
            responses.append(
                _ok_response(_message_payload(f"[STYLE: cheerful] resp {i}"))
            )
        responses.append(_ok_response({"id": "sess_new"}))  # sesión tras compactar
        responses.append(
            _ok_response(_message_payload("[STYLE: friendly] post-compact"))
        )

        mock_client.post.side_effect = responses

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        # Verificar estado inicial
        assert client.session_id is None
        assert client._message_count == 0

        # Saturar el contador
        for i in range(max_msgs):
            client.send_command(f"cmd {i}")

        # Compactación: sesión reseteada, contador a 0
        assert client.session_id is None
        assert client._message_count == 0

        # El próximo send_command debe crear una nueva sesión
        result = client.send_command("post-compact-cmd")
        assert result == "[STYLE: friendly] post-compact"
        assert client.session_id == "sess_new"
        assert client._message_count == 1  # se incrementó en el nuevo send_command

    @patch("handlers.opencode_client.httpx.Client")
    def test_session_compaction_only_on_success(
        self, mock_client_cls, mock_settings
    ):
        """El contador solo se incrementa en respuestas exitosas (no en fallos)."""
        mock_client = mock_client_cls.return_value

        max_msgs = mock_settings["opencode"]["max_session_messages"]

        # Flujo tras Parche C — 2 send_command, cada uno con failover:
        #  send_command 1:
        #    a) ensure_session OK → sess_1
        #    b) primario fail
        #    c) ensure_session OK → sess_2 (post-reset)
        #    d) fallback OK
        #  send_command 2:
        #    a) ensure_session (cached) — no POST
        #    b) primario fail
        #    c) ensure_session OK → sess_3 (post-reset)
        #    d) fallback OK
        # El contador NO debe incrementarse en los fallos.
        responses = [
            _ok_response({"id": "sess_1"}),  # ensure_session inicial
            _err_response(500, "primary fail"),  # 1
            _ok_response({"id": "sess_2"}),  # ensure_session post-reset (1)
            _ok_response(_message_payload("[STYLE: cheerful] fallback 1")),
            _err_response(500, "primary fail 2"),  # 2
            _ok_response({"id": "sess_3"}),  # ensure_session post-reset (2)
            _ok_response(_message_payload("[STYLE: cheerful] fallback 2")),
        ]
        # Tras Parche C, reset_session() (que resetea _message_count a 0) se llama
        # antes del fallback. Así que cada send_command con failover deja el contador
        # en 1 (reset → 0, fallback OK → +1). Los errores primarios NO incrementan.
        mock_client.post.side_effect = responses

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        client.send_command("cmd 1")
        assert client._message_count == 1
        client.send_command("cmd 2")
        assert client._message_count == 1  # reset por Parche C → 0, fallback OK → 1
