# TODO — fix/overlay-speaking-premature

## Tareas
- [ ] **Implementación** (@dev_senior): Mover transición a SPEAKING al primer chunk PCM real en `src/main.py` flujo streaming. Micro-Spec en `specs/bug_overlay_speaking_premature.md`. Branch `fix/overlay-speaking-premature`.
- [ ] **Verificación + Tests** (@tester): UAT con comando.wav real + test unitario del wrapper generador. Validar que overlay salta a verde solo al primer PCM real.
- [ ] **Commit docs** (@architect): Commitear spec + TODO.
- [ ] **Merge a master** (@architect): Tras validación de @tester y aprobación del usuario.