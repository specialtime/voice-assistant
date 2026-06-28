"""Tests unitarios para ``handlers/sentence_buffer.py``.

Verifica el contrato del buffer de oraciones:
- Acumula deltas de texto del stream SSE del agente.
- Emite oraciones completas al recibir puntuación fuerte (``.!?;``)
  **seguida de whitespace** (split con lookbehind).
- Parsea y descarta el prefijo ``[STYLE: <estilo>]`` del inicio del stream.
- ``flush()`` retorna el remanente del buffer al final del stream —
  necesario para emitir la última oración que no tiene whitespace post-puntuación
  (caso típico del último delta antes de ``session.idle``).

Nota sobre el split:
    El regex interno es ``r"(?<=[.!?;])\\s+"`` (lookbehind por puntuación
    + consume whitespace). Por lo tanto, ``"Hola mundo."`` (sin trailing
    space) NO se emite en ``add()`` — queda pendiente hasta que llegue un
    nuevo delta con whitespace (caso multi-oración) o hasta que el llamador
    invoque ``flush()``. Este es el contrato real del handler y está
    deliberadamente alineado con el uso en ``main.py.run_pipeline`` que
    llama a ``flush()`` tras consumir todo el stream.

Todos los tests son ``@pytest.mark.unit`` — sin red, sin disco.
"""

from __future__ import annotations

import pytest

from handlers.sentence_buffer import SentenceBuffer


@pytest.mark.unit
class TestSentenceBufferAdd:
    """Suite: SentenceBuffer.add() — acumulación y split por puntuación."""

    def test_add_partial_no_sentence(self):
        """Delta SIN puntuación → retorna lista vacía (acumula en buffer)."""
        buf = SentenceBuffer()
        result = buf.add("Hola")
        assert result == []
        # El contenido queda pendiente en el buffer
        assert buf._buffer == "Hola"

    def test_add_complete_sentence_with_trailing_space(self):
        """Delta con ``.\\s`` (puntuación + whitespace) → retorna la oración completa.

        Caso natural multi-delta del streaming: el agente emite ``"Hola. "`` y
        luego otra oración; el split ocurre en el whitespace post-puntuación.
        """
        buf = SentenceBuffer()
        result = buf.add("Hola mundo. ")
        assert result == ["Hola mundo."]

    def test_add_multiple_sentences(self):
        """Delta con VARIAS oraciones separadas por puntuación → lista con todas."""
        buf = SentenceBuffer()
        result = buf.add("Primera oración. Segunda oración. Tercera. ")
        assert result == ["Primera oración.", "Segunda oración.", "Tercera."]

    def test_add_accumulates_then_emits(self):
        """Delta partido en 2 → primera parte acumula, segunda emite la oración.

        Simula el caso real de streaming: el agente emite ``"Hola mun"``,
        luego ``"do. "``; solo al recibir el ``.\\s`` se emite la oración.
        """
        buf = SentenceBuffer()
        assert buf.add("Hola mun") == []
        assert buf.add("do. ") == ["Hola mundo."]

    def test_add_incremental_with_partial_sentence_kept(self):
        """Tras emitir una oración completa, el resto queda en el buffer.

        Patrón del streaming: ``"Hola. Mun"`` → emite ``["Hola."]`` y deja
        ``"Mun"`` pendiente para la próxima llamada. La oración completa del
        primer delta requiere que haya whitespace tras el ``.``.
        """
        buf = SentenceBuffer()
        result = buf.add("Hola. Mun")
        assert result == ["Hola."]
        assert buf._buffer == "Mun"

    def test_add_final_sentence_without_trailing_space_needs_flush(self):
        """Última oración sin whitespace post-puntuación NO se emite en add().

        Caso real del último delta del stream (justo antes de ``session.idle``):
        el agente cierra con ``"Final."`` sin trailing space → ``add()``
        retorna ``[]`` y la oración queda pendiente. El llamador debe invocar
        ``flush()`` para emitirla.
        """
        buf = SentenceBuffer()
        result = buf.add("Hola mundo.")
        # Sin whitespace trailing, el split no ocurre
        assert result == []
        assert buf._buffer == "Hola mundo."


@pytest.mark.unit
class TestSentenceBufferStylePrefix:
    """Suite: parsing del prefijo ``[STYLE: <estilo>]``."""

    def test_style_prefix_parsed(self):
        """Delta ``"[STYLE: cheerful] Hola. "`` → style_hint extraído + oración sin prefijo."""
        buf = SentenceBuffer()
        result = buf.add("[STYLE: cheerful] Hola. ")
        # La oración NO contiene el prefijo [STYLE:]
        assert result == ["Hola."]
        assert buf.style_hint == "cheerful"

    def test_no_style_prefix(self):
        """Delta SIN ``[STYLE:...]`` → style_hint queda vacío."""
        buf = SentenceBuffer()
        buf.add("Hola mundo. ")
        assert buf.style_hint == ""

    def test_style_split_preserved(self):
        """Prefijo + 2 oraciones → ambas oraciones se emiten, style_hint OK."""
        buf = SentenceBuffer()
        result = buf.add("[STYLE: friendly] Hola. Chau. ")
        assert result == ["Hola.", "Chau."]
        assert buf.style_hint == "friendly"

    def test_style_parsed_once(self):
        """El flag de parsing se setea una vez; deltas posteriores NO lo sobrescriben.

        Caso real del streaming: el primer delta trae el ``[STYLE: ...]`` y los
        siguientes NO (el prefijo ya fue removido del buffer). El handler setea
        ``_style_parsed = True`` para no re-parsear (defensa contra prefijos
        espurios en el medio del texto, que serían falsos positivos).
        """
        buf = SentenceBuffer()
        # Primer delta con prefijo (lo parsea)
        buf.add("[STYLE: cheerful] Primera. ")
        assert buf.style_hint == "cheerful"
        # Segundo delta SIN prefijo (caso normal del streaming)
        result = buf.add("Segunda. ")
        assert result == ["Segunda."]
        # style_hint NO cambia
        assert buf.style_hint == "cheerful"

    def test_style_parsed_from_split_delta(self):
        """Prefijo partido en 2 deltas → se parsea al completarse.

        El buffer acumula hasta que aparece el ``]`` (o llega a 30 chars).
        Caso real: el agente emite ``"[STYLE: cheer"`` y luego ``"ful] Hola. "``.
        """
        buf = SentenceBuffer()
        assert buf.add("[STYLE: cheer") == []
        assert buf.style_hint == ""
        result = buf.add("ful] Hola. ")
        assert result == ["Hola."]
        assert buf.style_hint == "cheerful"


@pytest.mark.unit
class TestSentenceBufferFlush:
    """Suite: SentenceBuffer.flush() — emisión del remanente.

    ``flush()`` es crítico para el pipeline streaming: tras consumir todos
    los deltas (al recibir ``session.idle``), el llamador invoca ``flush()``
    para emitir la ÚLTIMA oración que no tenía whitespace post-puntuación.
    """

    def test_flush_remaining(self):
        """Acumular sin puntuación → ``flush()`` retorna el resto."""
        buf = SentenceBuffer()
        buf.add("Texto sin punto final")
        assert buf._buffer == "Texto sin punto final"

        result = buf.flush()
        assert result == ["Texto sin punto final"]
        # Tras flush, el buffer queda vacío
        assert buf._buffer == ""

    def test_flush_empty_buffer(self):
        """Buffer vacío → ``flush()`` retorna lista vacía."""
        buf = SentenceBuffer()
        assert buf.flush() == []

    def test_flush_after_complete_sentences(self):
        """Tras emitir oraciones completas, ``flush()`` retorna solo lo pendiente."""
        buf = SentenceBuffer()
        buf.add("Primera. ")
        # Tras esta llamada, queda pendiente "Segunda parcial"
        buf.add("Segunda parcial")
        result = buf.flush()
        # Solo el remanente, NO la primera oración (ya fue emitida en add())
        assert result == ["Segunda parcial"]

    def test_flush_emits_final_sentence_without_trailing_space(self):
        """Caso estrella del pipeline streaming: última oración sin whitespace → flush().

        Simula el patrón real del último delta antes de ``session.idle``:
        el agente emite ``"Hola. Chau."`` (sin whitespace al final). El
        ``add()`` emite solo ``"Hola."`` y deja ``"Chau."`` pendiente; el
        llamador invoca ``flush()`` para emitir el resto.
        """
        buf = SentenceBuffer()
        result = buf.add("Hola. Chau.")
        # Solo "Hola." se emite (split por el whitespace entre ambas)
        assert result == ["Hola."]
        # "Chau." queda pendiente
        assert buf._buffer == "Chau."

        # El flush emite la oración final
        flushed = buf.flush()
        assert flushed == ["Chau."]


@pytest.mark.unit
class TestSentenceBufferPunctuation:
    """Suite: split por cada tipo de puntuación fuerte (cada uno seguido de whitespace)."""

    def test_split_by_dot(self):
        """Split por ``.`` preservando el signo en el chunk anterior."""
        buf = SentenceBuffer()
        result = buf.add("Hola. Chau. ")
        assert result == ["Hola.", "Chau."]

    def test_split_by_exclamation(self):
        """Split por ``!`` preservando el signo."""
        buf = SentenceBuffer()
        result = buf.add("Hola! Chau! ")
        assert result == ["Hola!", "Chau!"]

    def test_split_by_question(self):
        """Split por ``?`` preservando el signo."""
        buf = SentenceBuffer()
        result = buf.add("Hola? Chau? ")
        assert result == ["Hola?", "Chau?"]

    def test_split_by_semicolon(self):
        """Split por ``;`` preservando el signo.

        El agente ``asistente_voz`` puede usar ``;`` como separador de
        cláusulas dentro de una misma línea (no es fin de oración estricto,
        pero el buffer lo trata como puntuación fuerte).
        """
        buf = SentenceBuffer()
        result = buf.add("Hola; Chau; ")
        assert result == ["Hola;", "Chau;"]

    def test_mixed_punctuation(self):
        """Mezcla de ``.!?;`` en el mismo delta → todos se respetan.

        Cada signo seguido de whitespace dispara un split. ``Fin;`` al final
        requiere que haya whitespace después del ``;`` para emitirse en
        ``add()``; de lo contrario, queda pendiente para ``flush()``.
        """
        buf = SentenceBuffer()
        result = buf.add("Hola! Chau? Chau. Fin; ")
        assert result == ["Hola!", "Chau?", "Chau.", "Fin;"]
