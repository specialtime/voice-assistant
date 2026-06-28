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

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec Streaming TTS: send_command_stream (T6)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_sse_stream(lines: list[bytes]) -> MagicMock:
        """Construye un mock de respuesta SSE consumible vía ``with stream() as r``.

        Args:
            lines: Lista de líneas (bytes) que ``iter_lines()`` retornará en orden.

        Returns:
            MagicMock cuyo ``__enter__`` / ``__exit__`` son context manager
            y ``iter_lines()`` retorna ``iter(lines)``.
        """
        mock_stream = MagicMock(name="StreamResponse")
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = False
        mock_stream.raise_for_status = MagicMock()
        mock_stream.iter_lines.return_value = iter(lines)
        return mock_stream

    @staticmethod
    def _ok_post(status_code: int = 204) -> MagicMock:
        """Mock de respuesta POST exitosa (status arbitrario, sin raise_for_status)."""
        mock = MagicMock(name="PostResponse")
        mock.status_code = status_code
        mock.raise_for_status = MagicMock()
        return mock

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_success(self, mock_client_cls, mock_settings):
        """Stream SSE feliz: yield de deltas en orden, termina en session.idle.

        Mockea ``prompt_async`` con 204 OK y ``stream("GET", "/event")`` con una
        secuencia de eventos que incluye 2 deltas de la sesión ses_1 + session.idle.
        Verifica que los deltas se yield en orden, que el filtro por sessionID
        deja pasar los de la sesión actual y que el generador termina limpio.
        """
        mock_client = mock_client_cls.return_value

        # 1) ensure_session → POST /session
        # 2) prompt_async → POST /session/{id}/prompt_async (204)
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_1"}),
            self._ok_post(204),
        ]

        # Stream SSE con 2 deltas de ses_1 + session.idle + un server.connected ignorable
        sse_lines = [
            b'data: {"id":"evt_1","type":"server.connected","properties":{}}',
            b'',
            b'data: {"id":"evt_2","type":"session.next.text.delta","properties":{"sessionID":"ses_1","delta":"Hola"}}',
            b'',
            b'data: {"id":"evt_3","type":"session.next.text.delta","properties":{"sessionID":"ses_1","delta":" mundo"}}',
            b'',
            b'data: {"id":"evt_4","type":"session.idle","properties":{"sessionID":"ses_1"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        deltas = list(client.send_command_stream("abrí chrome"))

        # Yield en orden: "Hola" + " mundo"
        assert deltas == ["Hola", " mundo"]
        # 2 POST (ensure_session + prompt_async)
        assert mock_client.post.call_count == 2
        # 1 GET stream
        assert mock_client.stream.call_count == 1
        # Argumentos del prompt_async
        prompt_body = mock_client.post.call_args_list[1].kwargs["json"]
        assert prompt_body["agent"] == "asistente_voz"
        assert prompt_body["parts"] == [{"type": "text", "text": "abrí chrome"}]

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_filters_other_sessions(self, mock_client_cls, mock_settings):
        """Eventos de OTRAS sesiones NO se yield-ean al consumidor.

        El stream SSE es global (todas las sesiones pasan por el mismo /event).
        El handler debe filtrar por ``properties.sessionID == self.session_id``.
        """
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_self"}),
            self._ok_post(204),
        ]

        # Solo el último delta pertenece a ses_self; los anteriores son de otras sesiones
        sse_lines = [
            b'data: {"id":"e1","type":"session.next.text.delta","properties":{"sessionID":"ses_other_a","delta":"ignorame"}}',
            b'',
            b'data: {"id":"e2","type":"session.next.text.delta","properties":{"sessionID":"ses_other_b","delta":"tambien_ignorame"}}',
            b'',
            b'data: {"id":"e3","type":"session.next.text.delta","properties":{"sessionID":"ses_self","delta":"para_mi"}}',
            b'',
            b'data: {"id":"e4","type":"session.idle","properties":{"sessionID":"ses_self"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        deltas = list(client.send_command_stream("hola"))

        # Solo 1 delta (el de ses_self); los de otras sesiones se filtran
        assert deltas == ["para_mi"]

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_session_error(self, mock_client_cls, mock_settings):
        """Evento ``session.error`` → ``RuntimeError``.

        El handler propaga el error al caller, que decidirá si hace fallback
        a ``send_command()`` síncrono.
        """
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_err"}),
            self._ok_post(204),
        ]

        sse_lines = [
            b'data: {"id":"e1","type":"session.next.text.delta","properties":{"sessionID":"ses_err","delta":"arranc"}}',
            b'',
            b'data: {"id":"e2","type":"session.error","properties":{"sessionID":"ses_err","error":"rate limit"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        with pytest.raises(RuntimeError, match=r"Agente error"):
            list(client.send_command_stream("hola"))

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_prompt_async_fails(self, mock_client_cls, mock_settings):
        """``prompt_async`` con HTTP 500 → ``RuntimeError`` DESPUÉS de abrir el SSE.

        Tras el fix de la micro-spec ``fix/streaming-sse-zero-deltas``, el handler
        SUSCRIBE al stream ``GET /event`` ANTES de enviar ``prompt_async`` para
        cerrar la race window. Por lo tanto:

        - El SSE SÍ se abre (el stream ya estaba listo cuando llega el fallo).
        - El primer evento del stream (típicamente ``server.connected``) debe
          haberse consumido para confirmar la suscripción activa.
        - El ``prompt_async`` posterior falla con 500 → ``RuntimeError``.
        - El ``with`` cierra el stream al propagar la excepción.
        """
        mock_client = mock_client_cls.return_value
        # ensure_session OK → prompt_async 500
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_x"}),
            _err_response(500, "internal error"),
        ]

        # El stream entrega al menos un primer evento válido (server.connected
        # de una sesión arbitraria — el handler acepta cualquier primer evento
        # para confirmar la suscripción). Tras eso el iterador se agota y el
        # código sigue a ``prompt_async`` que falla con 500.
        sse_lines = [
            b'data: {"id":"evt_conn","type":"server.connected","properties":{}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        with pytest.raises(RuntimeError, match=r"prompt_async"):
            list(client.send_command_stream("hola"))

        # El SSE SÍ se abrió (subscribe antes de prompt_async)
        assert mock_client.stream.call_count == 1
        # ensure_session + prompt_async (fallido) = 2 POSTs
        assert mock_client.post.call_count == 2

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_increments_message_count(self, mock_client_cls, mock_settings):
        """Al recibir ``session.idle`` el ``_message_count`` se incrementa en 1.

        La compactación del contador se hace al FINAL del stream (no durante),
        para reflejar el contrato de la spec 6.1.
        """
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_inc"}),
            self._ok_post(204),
        ]

        sse_lines = [
            b'data: {"id":"e1","type":"session.next.text.delta","properties":{"sessionID":"ses_inc","delta":"x"}}',
            b'',
            b'data: {"id":"e2","type":"session.idle","properties":{"sessionID":"ses_inc"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        assert client._message_count == 0
        list(client.send_command_stream("hola"))

        assert client._message_count == 1

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_compaction(self, mock_client_cls, mock_settings):
        """Tras ``max_session_messages`` streams exitosos, la sesión se resetea.

        Mismo comportamiento que ``send_command`` síncrono: al alcanzar
        ``_max_messages`` se invoca ``reset_session()`` y ``_message_count``
        vuelve a 0.
        """
        mock_client = mock_client_cls.return_value

        max_msgs = mock_settings["opencode"]["max_session_messages"]

        # Para cada stream se necesita:
        #   - 1 ensure_session (la 1ª vez; el resto reutiliza session cache)
        #   - 1 prompt_async (204)
        #   - 1 SSE con delta + session.idle
        # Construimos las respuestas con side_effect:
        #   1) ensure_session inicial → sess_init   [POST #1]
        #   2..max_msgs+1) prompt_async 204 (max_msgs veces)  [POSTs #2..#11]
        #   max_msgs+2) ensure_session tras compactación → sess_after_compact  [POST #12]
        #   max_msgs+3) prompt_async 204 para stream post-compact  [POST #13]
        post_responses = [_ok_response({"id": "sess_init"})]
        for _ in range(max_msgs):
            post_responses.append(self._ok_post(204))
        # Respuestas post-compactación
        post_responses.append(_ok_response({"id": "sess_after_compact"}))
        post_responses.append(self._ok_post(204))
        mock_client.post.side_effect = post_responses

        # side_effect que retorna un NUEVO iterador por cada llamada a stream()
        # (necesario porque el iter se agota tras la primera lectura)
        sse_idle_lines = [
            b'data: {"id":"e1","type":"session.next.text.delta","properties":{"sessionID":"sess_init","delta":"a"}}',
            b'',
            b'data: {"id":"e2","type":"session.idle","properties":{"sessionID":"sess_init"}}',
            b'',
        ]
        sse_after_compact = [
            b'data: {"id":"e1","type":"session.next.text.delta","properties":{"sessionID":"sess_after_compact","delta":"b"}}',
            b'',
            b'data: {"id":"e2","type":"session.idle","properties":{"sessionID":"sess_after_compact"}}',
            b'',
        ]

        def stream_factory(*args, **kwargs):
            # Devuelve un mock nuevo con iter_lines que retorna un iter NUEVO
            return self._make_sse_stream(sse_idle_lines)

        # Reemplazamos el mock de stream con un side_effect que devuelve mocks nuevos
        # cuyo iter_lines() retorna un iter fresco cada vez
        from unittest.mock import MagicMock as _MM

        def make_fresh_stream_mock(lines):
            m = _MM(name="StreamResponse")
            m.__enter__.return_value = m
            m.__exit__.return_value = False
            m.raise_for_status = _MM()
            m.iter_lines.return_value = iter(list(lines))
            return m

        mock_client.stream.side_effect = [
            make_fresh_stream_mock(sse_idle_lines) for _ in range(max_msgs)
        ] + [make_fresh_stream_mock(sse_after_compact)]

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        # Consumir max_msgs streams
        for i in range(max_msgs):
            deltas = list(client.send_command_stream(f"cmd {i}"))
            assert deltas == ["a"], (
                f"Iter {i}: se esperaba ['a'], se obtuvo {deltas}"
            )

        # El último stream disparó compactación: session_id=None, count=0
        assert client.session_id is None, (
            f"Esperaba session_id=None tras compactación, se obtuvo {client.session_id}"
        )
        assert client._message_count == 0

        # El siguiente stream recrea sesión: ensure_session llamado de nuevo
        deltas = list(client.send_command_stream("post-compact"))
        assert deltas == ["b"]
        assert client.session_id == "sess_after_compact"
        assert client._message_count == 1

    # ──────────────────────────────────────────────────────────────────
    # Micro-Spec fix/streaming-sse-zero-deltas
    # ──────────────────────────────────────────────────────────────────

    def test_process_sse_event_anti_stale_idle_ignored(self):
        """``session.idle`` con ``delta_count == 0`` se IGNORA (no cierra stream).

        Regla anti-stale: si llega un ``session.idle`` antes que cualquier delta,
        se interpreta como un evento huérfano de un prompt previo cuya sesión
        coincidió por casualidad. El handler loguea WARNING y retorna
        ``([], False)`` — el consumidor sigue leyendo el stream.

        Verifica directamente el helper ``_process_sse_event`` con la nueva firma
        de 3 argumentos.
        """
        client = OpenCodeClient.__new__(OpenCodeClient)  # bypass __init__
        client.settings = {}

        event_json = (
            '{"type":"session.idle","properties":{"sessionID":"ses_x"}}'
        )
        deltas, done = client._process_sse_event(event_json, "ses_x", delta_count=0)

        # NO cierra el stream
        assert deltas == []
        assert done is False

    def test_process_sse_event_idle_after_delta_closes(self):
        """``session.idle`` con ``delta_count > 0`` cierra el stream.

        Caso opuesto: ya recibimos al menos un delta en este stream, por lo
        tanto el ``session.idle`` es válido y debe señalizar ``done=True``
        para que el handler haga ``break`` del loop de ``iter_lines``.
        """
        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        event_json = (
            '{"type":"session.idle","properties":{"sessionID":"ses_x"}}'
        )
        deltas, done = client._process_sse_event(event_json, "ses_x", delta_count=3)

        assert deltas == []
        assert done is True

    def test_process_sse_event_v2_format(self):
        """Formato v2: delta en ``data.sessionID`` y ``data.field.text``.

        El server opencode puede emitir el evento con el campo ``data.*`` en
        lugar de ``properties.*``. El parser debe aceptar ambos formatos y
        extraer el delta correctamente.
        """
        client = OpenCodeClient.__new__(OpenCodeClient)
        client.settings = {}

        event_json = (
            '{"type":"session.next.text.delta",'
            '"data":{"sessionID":"ses_v2","field":{"text":"hola v2"}}}'
        )
        deltas, done = client._process_sse_event(event_json, "ses_v2", delta_count=0)

        assert deltas == ["hola v2"]
        assert done is False

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_subscribes_before_prompt(
        self, mock_client_cls, mock_settings
    ):
        """El stream ``GET /event`` se abre ANTES del ``POST .../prompt_async``.

        El fix invierte el orden para cerrar la race window entre el POST del
        prompt y el primer delta del SSE. Verificamos el orden en
        ``mock_client.method_calls`` para detectar regresiones si alguien
        revierte el orden en el futuro.
        """
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_order"}),  # ensure_session
            self._ok_post(204),                 # prompt_async
        ]

        sse_lines = [
            b'data: {"id":"c","type":"server.connected","properties":{}}',
            b'',
            b'data: {"id":"d1","type":"session.next.text.delta","properties":{"sessionID":"ses_order","delta":"x"}}',
            b'',
            b'data: {"id":"d2","type":"session.idle","properties":{"sessionID":"ses_order"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)
        list(client.send_command_stream("orden"))

        # Localizar el call al stream y al prompt_async dentro de method_calls
        method_calls = mock_client.method_calls
        stream_call_idx = next(
            (
                i
                for i, call in enumerate(method_calls)
                if call[0] == "stream"
            ),
            None,
        )
        prompt_async_idx = next(
            (
                i
                for i, call in enumerate(method_calls)
                if call[0] == "post"
                and len(call[1]) >= 1
                and call[1][0].endswith("/prompt_async")
            ),
            None,
        )

        assert stream_call_idx is not None, (
            "No se encontró ninguna llamada a mock_client.stream"
        )
        assert prompt_async_idx is not None, (
            "No se encontró ninguna llamada a mock_client.post(.../prompt_async)"
        )
        assert stream_call_idx < prompt_async_idx, (
            f"Orden incorrecto: stream_call_idx={stream_call_idx}, "
            f"prompt_async_idx={prompt_async_idx}. El stream DEBE abrirse "
            f"ANTES que prompt_async para cerrar la race window."
        )

    # ──────────────────────────────────────────────────────────────────
    # Regression: httpx.StreamConsumed (bug fix/stream-consumed-sse)
    # ──────────────────────────────────────────────────────────────────
    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_iter_lines_called_once(
        self, mock_client_cls, mock_settings
    ):
        """Regression: ``send_command_stream`` debe invocar ``iter_lines()`` UNA SOLA VEZ.

        Bug original: el handler llamaba ``stream_response.iter_lines()`` dos veces
        sobre el mismo ``httpx.Response``:

        - Primera llamada: dentro de ``while not connected_seen`` para consumir
          el evento ``server.connected`` y confirmar la suscripción.
        - Segunda llamada: dentro de ``for line_bytes in stream_response.iter_lines()``
          para procesar el resto de deltas y ``session.idle``.

        ``httpx.Response.iter_raw()`` setea ``is_stream_consumed = True`` antes
        de la primera iteración, por lo que la segunda llamada a ``iter_lines()``
        (que pasa por ``iter_text()`` → ``iter_bytes()`` → ``iter_raw()``) levanta
        ``httpx.StreamConsumed`` y aborta el stream con cero deltas emitidos.

        Este test monta un mock que reproduce fielmente ese comportamiento de
        httpx: la segunda invocación de ``iter_lines()`` levanta
        ``StreamConsumed``. Con el código viejo, el test falla. Con el fix
        (un único ``for line_bytes in iter_lines()``), el test pasa.
        """
        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_reg"}),  # ensure_session
            self._ok_post(204),               # prompt_async
        ]

        # Stream SSE con un server.connected + 2 deltas + session.idle
        sse_lines = [
            b'data: {"id":"c","type":"server.connected","properties":{}}',
            b'',
            b'data: {"id":"d1","type":"session.next.text.delta",'
            b'"properties":{"sessionID":"ses_reg","delta":"Hola"}}',
            b'',
            b'data: {"id":"d2","type":"session.next.text.delta",'
            b'"properties":{"sessionID":"ses_reg","delta":" mundo"}}',
            b'',
            b'data: {"id":"d3","type":"session.idle",'
            b'"properties":{"sessionID":"ses_reg"}}',
            b'',
        ]

        # Mock que reproduce el comportamiento de httpx real:
        # iter_lines() devuelve un iter NUEVO cada vez, pero la segunda
        # invocación levanta StreamConsumed (porque iter_raw agotó el stream).
        iter_for_lines = iter(sse_lines)

        def iter_lines_side_effect():
            """Primera llamada OK, segunda llamada → StreamConsumed."""
            iter_lines_side_effect.call_count += 1
            if iter_lines_side_effect.call_count == 1:
                return iter(iter_for_lines)
            # Segunda invocación: simula lo que hace httpx real
            raise httpx.StreamConsumed()

        iter_lines_side_effect.call_count = 0

        def iter_lines_stream_consumed():
            """Segunda invocación → StreamConsumed (httpx 0.27.x)."""
            raise httpx.StreamConsumed()

        iter_lines_side_effect.consumed = iter_lines_stream_consumed

        mock_stream = MagicMock(name="StreamResponse")
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = False
        mock_stream.raise_for_status = MagicMock()
        mock_stream.iter_lines.side_effect = iter_lines_side_effect
        mock_client.stream.return_value = mock_stream

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        # Antes del fix: esto lanzaba httpx.StreamConsumed al re-iterar.
        # Después del fix: un solo iter_lines() consume todo el stream → OK.
        deltas = list(client.send_command_stream("hola"))

        assert deltas == ["Hola", " mundo"], (
            f"Esperaba ['Hola', ' mundo'], obtuve {deltas}. "
            "Si el test falla con StreamConsumed, alguien revirtió el fix "
            "que unifica los dos iter_lines() en uno solo."
        )
        # Verificar explícitamente que iter_lines() se invoca UNA SOLA VEZ
        assert iter_lines_side_effect.call_count == 1, (
            f"iter_lines() fue invocado {iter_lines_side_effect.call_count} "
            "veces — debe ser exactamente 1 (bug StreamConsumed si es >1)."
        )

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_raises_if_closed_before_any_event(
        self, mock_client_cls, mock_settings
    ):
        """Edge case: stream se cierra sin emitir ningún evento (ni server.connected).

        Antes del fix unificado, el ``while not connected_seen`` con ``next(..., None)``
        ya lanzaba ``RuntimeError("Stream SSE cerrado prematuramente")`` al primer
        ``None`` retornado por ``iter_lines``. Después del fix unificado, ese caso
        se detecta como ``iter_lines()`` agotado sin haber visto ningún evento
        completo: el código debe seguir lanzando el mismo ``RuntimeError`` y,
        crucialmente, NO debe haber enviado ``prompt_async``.

        Regresión objetivo: si alguien revierte el comportamiento o mueve la guarda
        al lugar equivocado, este test detecta que ``prompt_async`` se envía antes
        de confirmar la suscripción (reabre la race window).
        """
        mock_client = mock_client_cls.return_value
        # ensure_session crea la sesión; prompt_async NO debe invocarse.
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_premature"}),  # ensure_session
        ]

        # Stream SSE que se cierra sin emitir NADA (ni server.connected).
        sse_lines: list[bytes] = []

        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        with pytest.raises(RuntimeError, match="cerrado prematuramente"):
            list(client.send_command_stream("hola"))

        # Verificar el invariante crítico: NO se envió prompt_async.
        # Si el handler enviara prompt_async antes de confirmar la suscripción,
        # reabriría la race window (commit 9790971) y este test fallaría.
        assert mock_client.post.call_count == 1, (
            f"Solo debe haber 1 POST (ensure_session); se hicieron "
            f"{mock_client.post.call_count}. prompt_async NO debe enviarse si "
            "el stream se cerró antes de server.connected."
        )

    @patch("handlers.opencode_client.httpx.Client")
    def test_send_command_stream_respects_absolute_timeout(
        self, mock_client_cls, mock_settings
    ):
        """Edge case: timeout absoluto del stream corta el loop con WARNING.

        El fix unificado mantiene el chequeo ``time.monotonic() - start_time >
        streaming_timeout`` dentro del único ``for line_bytes in iter_lines()``.
        Si el stream está produciendo líneas pero el agente no termina, el handler
        debe cortar limpio cuando se excede el timeout, loguear WARNING con la
        cantidad de deltas emitidos y retornar sin propagar excepción (el caller
        puede hacer fallback síncrono).

        Mockeamos ``time.monotonic`` para forzar el timeout inmediatamente,
        agregando un evento ``server.connected`` válido antes de que el delta
        hipotético dispare la guarda. ``prompt_async`` SÍ debe enviarse (porque
        el primer evento completo sí se vio), pero el loop debe cortarse antes
        de procesar más deltas.
        """
        # Forzar timeout corto vía settings
        mock_settings["opencode"]["streaming_timeout_seconds"] = 1

        mock_client = mock_client_cls.return_value
        mock_client.post.side_effect = [
            _ok_response({"id": "ses_to"}),  # ensure_session
            self._ok_post(204),              # prompt_async
        ]

        # Stream con server.connected + 1 delta + 1 session.idle.
        # El timeout se fuerza vía mock de time.monotonic, así que el handler
        # debería cortar DESPUÉS del server.connected (que ya envió prompt_async)
        # y ANTES de procesar el delta.
        sse_lines = [
            b'data: {"id":"c","type":"server.connected","properties":{}}',
            b'',
            b'data: {"id":"d1","type":"session.next.text.delta",'
            b'"properties":{"sessionID":"ses_to","delta":"nunca"}}',
            b'',
            b'data: {"id":"d2","type":"session.idle",'
            b'"properties":{"sessionID":"ses_to"}}',
            b'',
        ]
        mock_client.stream.return_value = self._make_sse_stream(sse_lines)

        # Forzar que la PRIMERA llamada a monotonic devuelva un valor que ya
        # excede el timeout (start_time=0, check=0+1e9 > 1 → corta inmediato).
        # Pero el handler guarda start_time DENTRO de la función, así que el
        # truco es que monotonic devuelva un número creciente que rápidamente
        # supere el timeout. Lo más simple: que cada llamada devuelva un valor
        # que crece linealmente con un step enorme.
        monotonic_values = iter([0.0, 0.0, 0.0, 1e9, 1e9, 1e9, 1e9, 1e9])

        def fake_monotonic():
            return next(monotonic_values, 1e9)

        client = OpenCodeClient(mock_settings, "fake_pass", _BASE_URL)

        with patch(
            "handlers.opencode_client.time.monotonic", side_effect=fake_monotonic
        ):
            # No debe lanzar excepción: el corte por timeout es un cierre limpio.
            deltas = list(client.send_command_stream("hola"))

        # El delta nunca se yield-ó porque el timeout cortó antes de procesarlo.
        assert deltas == [], (
            f"Esperaba [] (timeout cortó antes del delta), obtuve {deltas}"
        )

        # prompt_async SÍ se envió (porque el primer evento completo, server.connected,
        # sí se vio antes del check de timeout).
        prompt_async_calls = []
        for c in mock_client.post.call_args_list:
            # `c` es un unittest.mock.call; sus args posicionales están en c.args
            args = getattr(c, "args", ())
            if args and isinstance(args[0], str) and args[0].endswith("/prompt_async"):
                prompt_async_calls.append(c)
        assert len(prompt_async_calls) == 1, (
            f"prompt_async debió invocarse 1 vez tras server.connected; "
            f"se invocó {len(prompt_async_calls)} veces."
        )
