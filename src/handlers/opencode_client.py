"""Cliente REST para OpenCode server.

Implementa OpenCodeClient con httpx + HTTP Basic Auth para comunicarse con
el servidor opencode serve (modo subagente oculto). Gestiona sesión en RAM
y failover entre el modelo default del agente (frontmatter) y un modelo fallback.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OpenCodeClient:
    """Cliente HTTP para interactuar con el agente opencode vía REST API.

    Gestiona sesiones (POST /session), envía comandos al subagente oculto
    (POST /session/:id/message) y aplica failover automático entre modelos.

    Attributes:
        settings: Dict con configuración opencode (agent, timeout_ms, etc.).
        base_url: URL base del servidor opencode (ej: http://127.0.0.1:4096).
        session_id: ID de sesión cacheado en RAM (None si no hay sesión activa).
    """

    def __init__(self, settings: dict, password: str, base_url: str) -> None:
        """Configura cliente httpx con auth básica y base_url.

        Args:
            settings: Dict completo de settings.json (usa settings['opencode']).
            password: Password configurada para opencode serve.
            base_url: URL base del servidor opencode (ej: http://127.0.0.1:4096).
        """
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

        timeout_seconds = settings["opencode"]["timeout_ms"] / 1000.0

        client_kwargs = {
            "base_url": self.base_url,
            "timeout": httpx.Timeout(timeout_seconds),
            "headers": {"Content-Type": "application/json"},
        }
        if password:  # password no vacía → auth básica
            client_kwargs["auth"] = httpx.BasicAuth("opencode", password)
            logger.info("OpenCodeClient: auth básica configurada")
        else:  # password vacía o None → servidor sin auth
            logger.info("OpenCodeClient: servidor sin auth (password vacía)")

        self._client = httpx.Client(**client_kwargs)

        self._model_fallback = self._parse_model(settings["opencode"]["model_fallback"])
        self._message_count: int = 0
        self._max_messages: int = settings["opencode"].get("max_session_messages", 10)

        logger.debug(
            "OpenCodeClient inicializado — base_url=%s, timeout=%.0fs, fallback=%s/%s",
            self.base_url,
            timeout_seconds,
            self._model_fallback["providerID"],
            self._model_fallback["modelID"],
        )

    @staticmethod
    def _parse_model(model_str: str) -> dict:
        """Convierte 'opencode-go/deepseek-v4-flash' → {'providerID':'opencode-go','modelID':'deepseek-v4-flash'}.

        Lanza ValueError si no contiene exactamente un '/'.
        """
        parts = model_str.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Formato de modelo inválido: '{model_str}' (esperado 'providerID/modelID')"
            )
        return {"providerID": parts[0], "modelID": parts[1]}

    def ensure_session(self) -> str:
        """Obtiene o crea una sesión en el servidor opencode.

        Si ya existe un session_id cacheado en RAM, lo retorna directamente.
        En caso contrario, crea una nueva sesión vía POST /session con title "voz".

        Returns:
            El session_id (str) de la sesión activa.

        Raises:
            RuntimeError: Si el servidor no responde o falla la creación de sesión.
        """
        if self.session_id is not None:
            return self.session_id

        try:
            response = self._client.post("/session", json={"title": "voz"})
            response.raise_for_status()
            data = response.json()
            self.session_id = data["id"]
            logger.debug("Sesión creada/obtenida: %s", self.session_id)
            return self.session_id
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error("No se pudo crear sesión opencode — %s: %s", type(exc).__name__, exc)
            raise RuntimeError("No se pudo crear sesión opencode") from exc
        except (KeyError, ValueError) as exc:
            logger.error(
                "Respuesta inesperada al crear sesión opencode: %s",
                exc,
            )
            raise RuntimeError("No se pudo crear sesión opencode") from exc

    def _post_message(self, session_id: str, body: dict) -> httpx.Response:
        """POST a /session/:id/message con auto-reset de sesión en 404.

        Si el servidor devuelve 404 (sesión no encontrada), resetea la sesión
        cacheada, crea una nueva, y reintenta una vez. Si el segundo intento
        también falla, propaga la excepción.

        Args:
            session_id: ID de sesión actual (puede ser stale).
            body: Body JSON del POST.

        Returns:
            Response del POST exitoso.

        Raises:
            httpx.HTTPStatusError: Si el segundo intento también falla.
        """
        response = self._client.post(
            f"/session/{session_id}/message", json=body
        )
        # Parche B1: Telemetría de errores del server (no rompe nada, solo loguea)
        if response.status_code >= 400:
            logger.warning(
                "OpenCode server error body: status=%d, session=%s, body=%s",
                response.status_code,
                session_id,
                response.text[:500],
            )
        if response.status_code == 404:
            logger.warning(
                "Sesión no encontrada (404) — reseteando sesión y reintentando..."
            )
            self.reset_session()
            session_id = self.ensure_session()
            response = self._client.post(
                f"/session/{session_id}/message", json=body
            )
            # Parche B1: Telemetría también en el reintento
            if response.status_code >= 400:
                logger.warning(
                    "OpenCode server error body (retry): status=%d, session=%s, body=%s",
                    response.status_code,
                    session_id,
                    response.text[:500],
                )
        response.raise_for_status()
        return response

    def send_command(self, text: str) -> str:
        """Envía un comando de texto al agente y retorna la respuesta.

        Flujo:
        1. Asegura sesión activa (ensure_session).
        2. Intenta con el modelo default del agente (frontmatter, sin override).
        3. Si falla (HTTP error, timeout, etc.), reintenta con fallback
           explícito usando model_fallback de settings.json.
        4. Extrae el texto del primer Part de tipo "text" en la respuesta.

        Args:
            text: Texto del comando a enviar al agente (transcripción del usuario).

        Returns:
            Respuesta del agente en texto plano (str), formato [STYLE: ...] texto.

        Raises:
            RuntimeError: Si ambos intentos (default y fallback) fallan.
        """
        session_id = self.ensure_session()

        # --- Intento con modelo default del agente ---
        primary_body = {
            "agent": self.settings["opencode"]["agent"],
            "parts": [{"type": "text", "text": text}],
        }

        try:
            response = self._post_message(session_id, primary_body)
            response_text = self._extract_text(response.json())
            truncated = response_text[:100] if len(response_text) > 100 else response_text
            logger.debug(
                "Respuesta primaria (model default del agente): %s",
                truncated,
            )
            self._message_count += 1
            if self._message_count >= self._max_messages:
                logger.info(
                    "Sesión compactada tras %d mensajes — reset_session()",
                    self._message_count,
                )
                self.reset_session()
                self._message_count = 0
            return response_text
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning(
                "Error en modelo default del agente — %s: %s. Iniciando failover...",
                type(exc).__name__,
                exc,
            )

        # --- Failover: modelo fallback ---
        logger.warning(
            "Failover a %s/%s tras error en modelo default del agente",
            self._model_fallback["providerID"],
            self._model_fallback["modelID"],
        )

        # Parche C: Reset sesión antes del fallback para evitar 404 stale
        # Cuando el primary falla (e.g., 500), la sesión puede haber quedado
        # en estado inconsistente en el server. Reseteamos y creamos una nueva.
        self.reset_session()
        session_id = self.ensure_session()

        fallback_body = {
            "agent": self.settings["opencode"]["agent"],
            "model": self._model_fallback,
            "parts": [{"type": "text", "text": text}],
        }

        try:
            response = self._post_message(session_id, fallback_body)
            response_text = self._extract_text(response.json())
            truncated = response_text[:100] if len(response_text) > 100 else response_text
            logger.debug(
                "Respuesta %s/%s: %s",
                self._model_fallback["providerID"],
                self._model_fallback["modelID"],
                truncated,
            )
            self._message_count += 1
            if self._message_count >= self._max_messages:
                logger.info(
                    "Sesión compactada tras %d mensajes — reset_session()",
                    self._message_count,
                )
                self.reset_session()
                self._message_count = 0
            return response_text
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error(
                "Error en modelo fallback (%s/%s) — %s: %s",
                self._model_fallback["providerID"],
                self._model_fallback["modelID"],
                type(exc).__name__,
                exc,
            )
            raise RuntimeError(
                "No se pudo obtener respuesta del agente (ambos intentos fallaron)"
            ) from exc
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "Respuesta inesperada del modelo fallback (%s/%s): %s",
                self._model_fallback["providerID"],
                self._model_fallback["modelID"],
                exc,
            )
            raise RuntimeError(
                "No se pudo obtener respuesta del agente (ambos intentos fallaron)"
            ) from exc

    def reset_session(self) -> None:
        """Reinicia la sesión limpiando el session_id cacheado en RAM.

        La próxima llamada a ensure_session() o send_command() creará una
        nueva sesión, reiniciando efectivamente el contexto de voz.
        """
        self.session_id = None
        self._message_count = 0
        logger.info("Sesión reiniciada")

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _extract_text(self, data: dict) -> str:
        """Extrae el texto del primer Part de tipo 'text' en la respuesta JSON.

        La estructura esperada es:
            {"info": {...}, "parts": [{"type": "text", "text": "..."}, ...]}

        Args:
            data: Dict decodificado del JSON de respuesta.

        Returns:
            Texto extraído del primer Part de tipo text.

        Raises:
            KeyError: Si 'parts' no existe o está vacío.
            ValueError: Si ningún Part tiene type == "text".
        """
        parts = data.get("parts", [])
        if not parts:
            raise KeyError("La respuesta no contiene 'parts' o está vacía")

        for part in parts:
            if part.get("type") == "text":
                return part.get("text", "")

        raise ValueError("Ningún Part de tipo 'text' encontrado en la respuesta")
