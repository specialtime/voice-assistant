# TODO — fix/kokoro-chunking-510-phonemes

## Tareas

- [ ] **@debugger** — Fixear IndexError 510 phonemes en `src/handlers/kokoro_tts_client.py`:
  - Implementar `_split_text()` con split por puntuación (`.,;:!?—–/:`) + safety net (MAX_CHARS=1500).
  - Modificar `synthesize()` para iterar chunks y concatenar audio con `np.concatenate`.
  - Logging DEBUG de chunks.
  - **Rama:** `fix/kokoro-chunking-510-phonemes` (working directory principal, sin trabajo paralelo).
  - **No refactorizar** fuera de `kokoro_tts_client.py`.

- [ ] **@tester** — Tests automatizados en `tests/test_kokoro_tts_client.py`:
  - Texto largo (678 chars del log de prod) → múltiples chunks, sin IndexError.
  - Split por em-dash, en-dash, dos puntos, slash.
  - Texto corto → 1 chunk.
  - Safety net: chunk sin puntuación > 1500 chars → split por espacios.
  - Preservar tests existentes (actualizar mocks si cambia el número de llamadas a `create`).
  - Ejecutar `pytest tests/test_kokoro_tts_client.py -v`.
  - **Rama:** `fix/kokoro-chunking-510-phonemes`.

- [ ] **@architect** — Auditar diff, merge a master, deploy, actualizar README.