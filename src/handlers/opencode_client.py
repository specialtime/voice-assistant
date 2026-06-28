"""Cliente REST para OpenCode server.

Implementa OpenCodeClient con httpx + HTTP Basic Auth para comunicarse con
el servidor opencode serve (modo subagente oculto). Gestiona sesión en RAM
y failover entre el modelo default del agente (frontmatter) y un modelo fallback.
"""

import json
import logging
import time
from typing import Iterator, List, Optional, Tuple

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

    def send_command_stream(self, text: str) -> Iterator[str]:
        """Envía prompt_async y hace yield de deltas de texto del stream SSE.

        Flujo:
        1. ensure_session()
        2. **Abrir** GET /event con httpx.Client.stream("GET", "/event", ...) — SSE stream
           - El handler del server registra el listener ANTES de retornar 200 (ver
             event.ts en opencode: ``yield* events.all()``). Hay que suscribirse
             ANTES de enviar ``prompt_async`` para no perder deltas emitidos entre
             el POST y el GET (race condition documentada en micro-spec
             fix/streaming-sse-zero-deltas).
        3. **Esperar el primer evento del stream** (``server.connected``) para
           confirmar que la suscripción está activa antes de enviar el prompt.
           Si el stream falla antes de eso → raise (no leak, el ``with`` cierra).
        4. POST /session/:id/prompt_async con body {agent, parts: [{type:"text", text}]}
           - Si falla con HTTP error → cerrar el stream y raise RuntimeError.
        5. Parsear el resto de eventos SSE (formato `data: {json}\\n\\n`).
        6. Filtrar: solo procesar eventos donde sessionID (en ``properties`` o
           ``data``) == self.session_id.
        7. Yield ``delta`` (en ``properties.delta`` o ``data.field.text``) de
           eventos type=="session.next.text.delta".
        8. Terminar al recibir ``session.idle`` válido (ver regla anti-stale abajo).
        9. Al terminar: incrementar ``_message_count`` y compactar si llega a max.

        **Regla anti-stale de session.idle**: el stream ``/event`` es GLOBAL
        (todas las sesiones del servidor pasan por el mismo endpoint) y LIVE-ONLY
        (no replay histórico). Si el handler se suscribe tarde puede recibir un
        ``session.idle`` residual de un prompt previo cuyo ``sessionID`` coincide
        por casualidad con el actual. Para evitar cierres prematuros con 0
        deltas, se exige: si llega ``session.idle`` y ``delta_count == 0``, se
        IGNORA con WARNING y se sigue leyendo el stream.

        Yields:
            str: cada delta de texto del agente.

        Raises:
            RuntimeError: si el subscribe al stream falla, si prompt_async falla,
                o si llega session.error.
        """
        session_id = self.ensure_session()

        # --- Paso 2-3: Suscribirse al stream ANTES de enviar prompt_async ---
        streaming_timeout = self.settings.get("opencode", {}).get("streaming_timeout_seconds", 120)
        start_time = time.monotonic()
        delta_count = 0
        first_delta_logged = False
        stream_opened = False

        body = {
            "agent": self.settings["opencode"]["agent"],
            "parts": [{"type": "text", "text": text}],
        }

        try:
            with self._client.stream("GET", "/event") as stream_response:
                stream_response.raise_for_status()
                stream_opened = True
                data_buffer = ""

                # FIX StreamConsumed (commit de fix/stream-consumed-sse):
                # El bug original llamaba ``stream_response.iter_lines()`` dos
                # veces sobre el mismo ``httpx.Response``. La primera llamada
                # (en el ``while not connected_seen``) marca el stream interno
                # como consumido (``iter_raw`` setea ``is_stream_consumed=True``
                # antes de la primera iteración) y la segunda llamada
                # (``for line_bytes in stream_response.iter_lines()``)
                # levantaba ``httpx.StreamConsumed``.
                #
                # Fix: un único ``for line_bytes in iter_lines()`` que consume
                # TODO el stream. ``prompt_async`` se envía en cuanto se ve el
                # primer evento completo (típicamente ``server.connected``),
                # preservando el orden subscribe→prompt_async del commit
                # ``9790971`` y el filtro anti-stale (que vive en
                # ``_process_sse_event``, commit ``43c506e``).
                connected_seen = False
                prompt_async_sent = False
                saw_done = False

                # Consumir TODO el stream con un único iter_lines().
                for line_bytes in stream_response.iter_lines():
                    if time.monotonic() - start_time > streaming_timeout:
                        logger.warning(
                            "Stream SSE excedió timeout de %ds — cerrando (%d deltas)",
                            streaming_timeout,
                            delta_count,
                        )
                        break

                    line = line_bytes.decode("utf-8") if isinstance(line_bytes, bytes) else line_bytes

                    if line.startswith("data: "):
                        data_buffer += line[6:]  # acumular JSON sin prefijo
                        continue

                    if line == "" and data_buffer:
                        # Fin de un evento SSE: procesarlo.
                        deltas, done = self._process_sse_event(
                            data_buffer, session_id, delta_count
                        )
                        for delta in deltas:
                            if not first_delta_logged:
                                logger.debug("Primer delta recibido: %r", delta)
                                first_delta_logged = True
                            delta_count += 1
                            yield delta
                        if done:
                            if not connected_seen:
                                # session.idle ANTES de prompt_async: caso raro
                                # (no debería pasar — server.connected siempre
                                # es el primer evento), pero por las dudas.
                                logger.debug(
                                    "session.idle recibido antes de prompt_async — cerrando"
                                )
                                saw_done = True
                                break
                            logger.debug(
                                "session.idle válido recibido — cerrando stream SSE (%d deltas)",
                                delta_count,
                            )
                            saw_done = True
                            break
                        data_buffer = ""

                        # --- Paso 3→4: enviar prompt_async tras confirmar
                        #     suscripción (primer evento completo visto).
                        if not connected_seen:
                            connected_seen = True
                            try:
                                response = self._client.post(
                                    f"/session/{session_id}/prompt_async",
                                    json=body,
                                )
                                response.raise_for_status()
                                prompt_async_sent = True
                                logger.debug(
                                    "prompt_async enviado y stream /event ya suscrito"
                                    " — race window cerrado"
                                )
                            except (
                                httpx.HTTPStatusError,
                                httpx.TimeoutException,
                                httpx.RequestError,
                            ) as exc:
                                logger.error(
                                    "prompt_async falló (post-subscribe) — %s: %s",
                                    type(exc).__name__,
                                    exc,
                                )
                                raise RuntimeError("prompt_async falló") from exc

                    # line == "" sin data_buffer (separador entre eventos): ignorar
                    # Otras líneas (event:, id:, comentarios): ignorar

                # Fin del stream sin haber visto server.connected ni enviado prompt_async
                if not connected_seen and not saw_done:
                    logger.warning(
                        "Stream SSE cerrado por server antes de cualquier evento — "
                        "subscription no confirmada"
                    )
                    raise RuntimeError("Stream SSE cerrado prematuramente")

                # Fin del stream: procesar cualquier evento residual en data_buffer
                if data_buffer and not saw_done:
                    deltas, _ = self._process_sse_event(
                        data_buffer, session_id, delta_count
                    )
                    for delta in deltas:
                        if not first_delta_logged:
                            logger.debug("Primer delta recibido: %r", delta)
                            first_delta_logged = True
                        delta_count += 1
                        yield delta

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error(
                "Error en stream SSE — %s: %s",
                type(exc).__name__,
                exc,
            )
            if stream_opened:
                # El stream se abrió pero falló la lectura — no propagar
                # RuntimeError adicional, el caller puede hacer fallback.
                raise RuntimeError("Error en stream SSE") from exc
            # El stream ni siquiera se pudo abrir → propagar para fallback síncrono.
            raise RuntimeError("Error en stream SSE") from exc

        # --- Paso 9: Compactación ---
        self._message_count += 1
        if self._message_count >= self._max_messages:
            logger.info(
                "Sesión compactada tras %d mensajes (streaming) — reset_session()",
                self._message_count,
            )
            self.reset_session()
            self._message_count = 0

        logger.debug(
            "Stream SSE finalizado — %d deltas emitidos",
            delta_count,
        )

    def _process_sse_event(
        self, data_str: str, session_id: str, delta_count: int
    ) -> Tuple[List[str], bool]:
        """Procesa un evento SSE completo: parsea JSON, filtra por sessionID,
        y retorna tupla (deltas, done) o lanza error según el tipo de evento.

        Tolerante a dos formatos de payload (v1 ``properties.*`` y v2 ``data.*``).
        Loguea a DEBUG el ``type`` y el sessionID resuelto de cada evento para
        diagnóstico en producción.

        Args:
            data_str: String JSON del campo `data:` del evento SSE.
            session_id: ID de sesión esperado para filtrar eventos.
            delta_count: Cantidad de deltas recibidos hasta ahora en este stream.
                Se usa para descartar ``session.idle`` espurios (sin deltas).

        Returns:
            Tupla (deltas, done): deltas es lista de strings, done es True si
            el evento señaliza fin del stream (session.idle) y es válido.

        Raises:
            RuntimeError: si el evento es session.error.
        """
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError as exc:
            logger.warning("Evento SSE con JSON inválido: %s", exc)
            return [], False

        event_type = event.get("type", "")
        properties = event.get("properties", {})
        data_field = event.get("data", {})

        # Telemetría de diagnóstico: loguear type y sessionID de CADA evento
        # antes de cualquier filtro, para entender qué emite el server realmente.
        resolved_session_id = (
            properties.get("sessionID")
            or data_field.get("sessionID")
            or ""
        )
        logger.debug(
            "SSE event received: type=%r, sessionID=%r (expected=%r)",
            event_type,
            resolved_session_id,
            session_id,
        )

        # Filtrar por sessionID (acepta tanto properties.sessionID como data.sessionID)
        if resolved_session_id != session_id:
            return [], False

        if event_type == "session.next.text.delta":
            # Acepta delta en properties.delta (v1) o data.field.text (v2)
            delta = properties.get("delta") or data_field.get("field", {}).get("text", "")
            if delta:
                return [delta], False
            return [], False
        elif event_type == "session.idle":
            # Regla anti-stale: ignorar session.idle si aún no llegó ningún delta.
            # El server emite session.idle por sesión al cerrarse; si la suscripción
            # se hace tarde o el stream arrastra eventos previos, podríamos recibir
            # un session.idle "huérfano" antes que los deltas reales. Requerir al
            # menos un delta garantiza que el cierre corresponde a ESTE prompt.
            if delta_count == 0:
                logger.warning(
                    "session.idle recibido ANTES de cualquier delta — ignorando "
                    "(probable evento stale de un prompt previo). sessionID=%s",
                    session_id,
                )
                return [], False  # NO cerrar el stream, seguir leyendo
            return [], True
        elif event_type == "session.error":
            error_msg = properties.get("error") or data_field.get("error") or "error desconocido"
            logger.error("Agente reportó error: %s", error_msg)
            raise RuntimeError(f"Agente error: {error_msg}")

        return [], False

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
