# TODO — Streaming TTS Kokoro + SSE OpenCode

## Tareas

- [ ] **T1** `@dev_senior` — Implementar `OpenCodeClient.send_command_stream()` en `src/handlers/opencode_client.py`
- [ ] **T2** `@dev_senior` — Crear `src/handlers/sentence_buffer.py` (SentenceBuffer + parse [STYLE])
- [ ] **T3** `@dev` — Agregar `KokoroTTSClient.synthesize_sentence_stream()` en `src/handlers/kokoro_tts_client.py`
- [ ] **T4** `@dev` — Agregar campos `streaming_enabled` y `streaming_timeout_seconds` en `config/settings.json`
- [ ] **T5** `@dev_senior` — Modificar `run_pipeline()` en `src/main.py` para flujo streaming + fallback síncrono
- [ ] **T6** `@tester` — Tests de `send_command_stream` (mock SSE) en `tests/test_opencode_client.py`
- [ ] **T7** `@tester` — Crear `tests/test_sentence_buffer.py`
- [ ] **T8** `@tester` — Tests de `synthesize_sentence_stream` en `tests/test_kokoro_tts_client.py`
- [ ] **T9** `@tester` — Tests del pipeline streaming + fallback en `tests/test_state_machine.py`
- [ ] **T10** `@security` — Auditoría del streaming SSE (no filtrar secrets en logs, auth correcta)
- [ ] **T11** `architect` — Commit de documentación + merge a master