# TODO — fix/kokoro-phonemizer-mismatch

## Tareas

- [ ] **@debugger** — Investigar y fixear el bug en `src/handlers/kokoro_tts_client.py`:
  - Reproducir el WARNING "words count mismatch" con el texto exacto del log.
  - Implementar Cambio 1 (normalización de whitespace con `re.sub(r"\s+", " ", text)`).
  - Evaluar Cambio 2 (`trim=False`) si el corte persiste.
  - Agregar log DEBUG del texto normalizado.
  - Verificar que el audio no se corta al final.
  - **Rama:** `fix/kokoro-phonemizer-mismatch` (worktree principal, no hay trabajo paralelo).
  - **No refactorizar** fuera de `kokoro_tts_client.py`.

- [ ] **@tester** — Escribir/actualizar tests automatizados en `tests/test_kokoro_tts_client.py`:
  - Test: texto multilinea con newlines se normaliza antes de pasar a `Kokoro.create()`.
  - Test: em-dash y números con dos puntos no rompen la síntesis.
  - Test: `trim` se pasa como `False` (si el debugger confirma Cambio 2) o se mantiene default.
  - Preservar `test_no_secrets_logged`.
  - Ejecutar `pytest tests/test_kokoro_tts_client.py -v` y reportar resultado.
  - **Rama:** `fix/kokoro-phonemizer-mismatch`.

- [ ] **@architect** — Auditar diff contra spec, merge a master previa confirmación del usuario.