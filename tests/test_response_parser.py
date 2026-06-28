"""Tests unitarios para handlers/response_parser.py.

Sin red, sin mocks — solo lógica pura de regex sobre strings.
Cubre el contrato de `parse_response(response: str) -> tuple[str, str]`
definido en IMPLEMENTATION.md §4.8.
"""

import pytest

from handlers.response_parser import parse_response


@pytest.mark.unit
class TestParseResponse:
    """Suite de tests para parse_response()."""

    def test_parse_with_style(self):
        """[STYLE: cheerful] Listo, abrí Chrome → ('cheerful', 'Listo, abrí Chrome')."""
        style, text = parse_response("[STYLE: cheerful] Listo, abrí Chrome")
        assert style == "cheerful"
        assert text == "Listo, abrí Chrome"

    def test_parse_without_style(self):
        """Sin prefijo de estilo → ('', texto completo limpio)."""
        style, text = parse_response("Listo, abrí Chrome")
        assert style == ""
        assert text == "Listo, abrí Chrome"

    def test_parse_empty_style(self):
        """[STYLE:] sin nombre de estilo → edge case: regex no matchea, texto pasa tal cual.

        El regex actual requiere \\w+ (al menos un word char) en el grupo de estilo,
        por lo que '[STYLE:]' (vacío) NO matchea. La función debe manejar esto
        sin crashear: retorna ('', texto limpio).
        """
        style, text = parse_response("[STYLE:] texto")
        assert style == ""
        assert text == "[STYLE:] texto"

    def test_parse_multiline_text(self):
        """[STYLE: calm] Línea 1\\nLínea 2 → multilinea preservado gracias a re.DOTALL."""
        style, text = parse_response("[STYLE: calm] Línea 1\nLínea 2")
        assert style == "calm"
        assert text == "Línea 1\nLínea 2"

    def test_parse_empty_response(self):
        """String vacío → ('', '')."""
        style, text = parse_response("")
        assert style == ""
        assert text == ""

    def test_parse_style_with_extra_spaces(self):
        """[STYLE:  cheerful]  texto → espacios extra alrededor del estilo y del texto.

        Caso documentado en §4.8: el parser debe tolerar espacios extra
        entre '[STYLE:' y el nombre del estilo, y entre ']' y el texto.
        Resultado esperado: ('cheerful', 'texto').

        NOTA: el regex actual del parser es `^\\[STYLE:\\s*(\\w+)\\]\\s*(.*)$`,
        que SÍ tolera espacios antes del estilo (\\s* inicial) y después de ']'
        (\\s* final), pero NO tolera espacios DENTRO de los corchetes entre el
        nombre del estilo y ']'. Este test documenta el comportamiento esperado;
        si el regex se fortalece para tolerar espacios internos, debe seguir
        pasando con el resultado esperado.
        """
        style, text = parse_response("[STYLE:  cheerful]  texto")
        assert style == "cheerful"
        assert text == "texto"


@pytest.mark.unit
class TestStripMarkdown:
    """Tests para la limpieza de markdown en parse_response()."""

    def test_parse_strips_bold(self):
        """**Listo** → Listo (sin asteriscos)."""
        style, text = parse_response("**Listo**")
        assert style == ""
        assert text == "Listo"

    def test_parse_strips_italic(self):
        """*Listo* → Listo (sin asteriscos)."""
        style, text = parse_response("*Listo*")
        assert style == ""
        assert text == "Listo"

    def test_parse_strips_heading(self):
        """# Título → Título (sin hashtag)."""
        style, text = parse_response("# Título")
        assert style == ""
        assert text == "Título"

    def test_parse_strips_heading_multiple(self):
        """## Sub → Sub (múltiples hashtags)."""
        style, text = parse_response("## Sub")
        assert style == ""
        assert text == "Sub"

    def test_parse_strips_list_bullet(self):
        """- item → item (sin viñeta guion)."""
        style, text = parse_response("- item")
        assert style == ""
        assert text == "item"

    def test_parse_strips_list_asterisk(self):
        """* item → item (sin viñeta asterisco)."""
        style, text = parse_response("* item")
        assert style == ""
        assert text == "item"

    def test_parse_strips_inline_code(self):
        """`Chrome` → Chrome (sin backticks, preserva contenido)."""
        style, text = parse_response("`Chrome`")
        assert style == ""
        assert text == "Chrome"

    def test_parse_strips_code_block(self):
        """Bloque ```bash\\nls\\n``` → eliminado completo."""
        style, text = parse_response("Antes\n```bash\nls\n```\nDespués")
        assert style == ""
        # El bloque se elimina, queda "Antes" y "Después" (normalizado)
        assert "```" not in text
        assert "bash" not in text
        assert "ls" not in text
        assert "Antes" in text
        assert "Después" in text

    def test_parse_strips_link(self):
        """[texto](http://url) → texto (sin URL)."""
        style, text = parse_response("[texto](http://url.com)")
        assert style == ""
        assert text == "texto"
        assert "http" not in text

    def test_parse_strips_blockquote(self):
        """> cita → cita (sin >)."""
        style, text = parse_response("> cita")
        assert style == ""
        assert text == "cita"

    def test_parse_strips_separator(self):
        """--- → eliminado (separador horizontal)."""
        style, text = parse_response("Antes\n---\nDespués")
        assert style == ""
        assert "---" not in text
        assert "Antes" in text
        assert "Después" in text

    def test_parse_strips_combined(self):
        """# **Hola** *mundo* → Hola mundo (múltiples markdown combinados)."""
        style, text = parse_response("# **Hola** *mundo*")
        assert style == ""
        assert text == "Hola mundo"

    def test_parse_preserves_plain_text(self):
        """Texto sin markdown → sin cambios."""
        style, text = parse_response("Listo, abrí Chrome")
        assert style == ""
        assert text == "Listo, abrí Chrome"

    def test_parse_strips_underscore_italic(self):
        """_hola_ → hola (cursiva con underscore en word boundaries)."""
        style, text = parse_response("_hola_")
        assert style == ""
        assert text == "hola"

    def test_parse_preserves_underscore_in_words(self):
        """hola_mundo → hola_mundo (underscore en medio de palabra se preserva)."""
        style, text = parse_response("hola_mundo")
        assert style == ""
        assert text == "hola_mundo"

    def test_parse_style_with_markdown(self):
        """[STYLE: cheerful] **Listo** → ('cheerful', 'Listo')."""
        style, text = parse_response("[STYLE: cheerful] **Listo**")
        assert style == "cheerful"
        assert text == "Listo"

    def test_parse_style_with_markdown_heading(self):
        """[STYLE: calm] # Título → ('calm', 'Título')."""
        style, text = parse_response("[STYLE: calm] # Título")
        assert style == "calm"
        assert text == "Título"
