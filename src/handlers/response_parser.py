"""Parseador de respuesta del agente.

Extrae el prefijo opcional '[STYLE: <estilo>]' del inicio de la respuesta
del agente, limpia sintaxis markdown y retorna (style_hint, clean_text).
Si no hay prefijo, style_hint es cadena vacía y clean_text es la respuesta
completa con el markdown eliminado.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patrón para extraer [STYLE: <estilo>] al inicio del texto
_STYLE_PATTERN = re.compile(r"^\[STYLE:\s*(\w+)\]\s*(.*)$", re.DOTALL)

# Estilos válidos (solo documentación, no se validan estrictamente)
_VALID_STYLES = frozenset({
    "cheerful", "sad", "friendly", "excited", "calm",
    "serious", "whisper", "apologetic", "confident",
})


def _strip_markdown(text: str) -> str:
    """Limpia sintaxis markdown de un texto para que TTS no lea caracteres literales.

    Elimina: bloques de código, separadores, encabezados, viñetas, citas,
    enlaces, negrita, cursiva, inline code. Preserva el contenido textual.
    """
    # 1. Bloques de código (fences): eliminar completos (fences + contenido)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Separadores horizontales: eliminar líneas ---, ***, ___ (3+ chars)
    text = re.sub(r"^[\s]*[-*_]{3,}[\s]*$", "", text, flags=re.MULTILINE)

    # 3. Encabezados: quitar #, ##, ###, etc. al inicio de línea
    text = re.sub(r"^[\s]*#{1,6}[\s]+", "", text, flags=re.MULTILINE)

    # 4. Viñetas de listas: quitar -, *, + al inicio de línea
    text = re.sub(r"^[\s]*[-*+][\s]+", "", text, flags=re.MULTILINE)

    # 5. Citas (blockquote): quitar > al inicio de línea
    text = re.sub(r"^[\s]*>[\s]*", "", text, flags=re.MULTILINE)

    # 6. Enlaces: [texto](url) → texto
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 7. Negrita: **texto** → texto
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # 8. Cursiva con asterisco: *texto* → texto (DESPUÉS de negrita)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # 9. Cursiva con underscore: _texto_ → texto (solo word boundaries)
    text = re.sub(r"\b_([^_]+)_\b", r"\1", text)

    # 10. Inline code: `texto` → texto
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 11. Normalizar espacios
    text = re.sub(r"[ \t]+", " ", text)                        # múltiples espacios → uno
    text = re.sub(r"\n{2,}", "\n", text)                       # múltiples newlines → uno
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)  # espacios al inicio/final de línea
    text = text.strip()                                        # strip final

    return text


def parse_response(response: str) -> tuple[str, str]:
    """Parsea la salida del agente: '[STYLE: <estilo>] <texto>'.

    Retorna (style_hint, clean_text). Si no hay prefijo,
    style_hint='' y clean_text=response completa.

    Args:
        response: Texto de respuesta del agente.

    Returns:
        Tupla (style_hint, clean_text). style_hint es el estilo
        extraído (ej. "cheerful") o cadena vacía.
    """
    match = _STYLE_PATTERN.match(response)

    if match:
        style_hint = match.group(1)
        clean_text = match.group(2).strip()
        clean_text = _strip_markdown(clean_text)

        # Log debug truncado (máximo 120 caracteres del texto limpio)
        truncated = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
        logger.debug(
            "parse_response — estilo='%s', texto='%s' (%d chars)",
            style_hint,
            truncated,
            len(clean_text),
        )
    else:
        style_hint = ""
        clean_text = response.strip()
        clean_text = _strip_markdown(clean_text)

        # Log debug truncado
        truncated = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
        logger.debug(
            "parse_response — sin prefijo, texto='%s' (%d chars)",
            truncated,
            len(clean_text),
        )

    return style_hint, clean_text
