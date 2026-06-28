"""Buffer que acumula deltas de texto del stream SSE y emite oraciones completas.

Maneja el prefijo [STYLE: ...] al inicio del stream del agente asistente_voz.
Split por puntuación fuerte (. ! ? ;) preservando el signo en la oración.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Patrón para extraer [STYLE: <estilo>] al inicio del texto
_STYLE_PATTERN = re.compile(r"^\[STYLE:\s*(\w+)\]\s*")

# Split después de puntuación fuerte (preserva signo en chunk anterior)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")

# Límite de seguridad: si el buffer excede este tamaño, se fuerza flush por espacios
MAX_BUFFER_CHARS = 10000


class SentenceBuffer:
    """Acumula deltas de texto y emite oraciones completas.

    Attributes:
        _buffer: Texto acumulado pendiente de emitir.
        _style_hint: Style extraído del prefijo [STYLE: ...] (vacío si no hay).
        _style_parsed: True si ya se intentó parsear el prefijo.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._style_hint: str = ""
        self._style_parsed: bool = False

    def add(self, delta: str) -> List[str]:
        """Agrega un delta y retorna lista de oraciones completas.

        1. Concatena delta al buffer.
        2. Si no se parseó el style y el buffer tiene >=10 chars o contiene ']',
           intentar extraer [STYLE: ...].
        3. Split por puntuación fuerte. Las oraciones completas (todas menos la última)
           se retornan. La última (posiblemente incompleta) queda en el buffer.

        Returns:
            Lista de oraciones completas (puede ser vacía).
        """
        self._buffer += delta

        # Safety net: si el buffer excede el límite, forzar flush por espacios
        if len(self._buffer) > MAX_BUFFER_CHARS:
            logger.warning(
                "SentenceBuffer excedió %d chars — forzando flush por espacios",
                MAX_BUFFER_CHARS,
            )
            words = self._buffer.split(" ")
            sentences = []
            current = ""
            for w in words:
                if current and len(current) + len(w) + 1 > 500:
                    sentences.append(current.strip())
                    current = w
                else:
                    current = (current + " " + w).strip() if current else w
            if current:
                sentences.append(current.strip())
            self._buffer = ""
            self._style_parsed = True  # ya no intentar parsear style
            return sentences

        # Parsear [STYLE: ...] del inicio (una sola vez)
        if not self._style_parsed:
            if "]" in self._buffer or len(self._buffer) >= 30:
                match = _STYLE_PATTERN.match(self._buffer)
                if match:
                    self._style_hint = match.group(1)
                    self._buffer = self._buffer[match.end():]
                self._style_parsed = True

        # Split por puntuación fuerte
        parts = _SENTENCE_SPLIT.split(self._buffer)
        if len(parts) > 1:
            # Todas menos la última son oraciones completas
            sentences = [p.strip() for p in parts[:-1] if p.strip()]
            self._buffer = parts[-1]  # la última queda pendiente
            return sentences
        return []

    def flush(self) -> List[str]:
        """Retorna oraciones parciales restantes (para el final del stream).

        Returns:
            Lista con el contenido restante del buffer (puede ser vacía).
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            return [remaining]
        return []

    @property
    def style_hint(self) -> str:
        """Style hint extraído del prefijo [STYLE: ...]. Vacío si no hay."""
        return self._style_hint
