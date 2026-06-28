# Tests — Jarvis Voice Assistant

Suite de tests E2E + unitarios para el asistente de voz (`voice-assistant/`).
Cubre los 6 handlers y la máquina de estados del orquestador (`main.py`).

## Instalacion

Desde la raiz del proyecto `voice-assistant/`:

```bash
pip install -r requirements.txt -r tests/requirements-test.txt
```

## Estructura

```
tests/
├── __init__.py                  # package marker
├── conftest.py                  # fixtures compartidos + markers pytest
├── test_response_parser.py      # UNIT — sin red, logica pura de regex
├── test_audio_manager.py        # UNIT — sounddevice mockeado
├── test_gemini_stt_client.py    # UNIT — httpx mockeado, failover 429
├── test_opencode_client.py      # UNIT — httpx mockeado, failover cerebro
├── test_tts_clients.py          # UNIT — httpx mockeado, Gemini + Azure
├── test_state_machine.py        # UNIT — VoiceAssistant mockeado
├── test_e2e_scenarios.py        # E2E   — API keys + servidor reales
├── requirements-test.txt
└── README.md (este archivo)
```

## Ejecucion

> **Importante:** correr siempre desde la raiz del proyecto (`voice-assistant/`),
> para que `conftest.py` encuentre `config/settings.json` y los imports resuelvan.

### Tests unitarios (rapidos, sin red)

```bash
pytest tests/ -m unit -v
```

### Tests E2E (requieren `.env` con API keys + servidor `opencode serve`)

Antes de correr los E2E, levantar el servidor en otra terminal:

```bash
set OPENCODE_SERVER_PASSWORD=<tu_pass>
opencode serve --port 4096
```

Y verificar la salud:

```bash
curl -u opencode:<tu_pass> http://127.0.0.1:4096/global/health
```

Luego ejecutar:

```bash
pytest tests/ -m e2e -v
```

> Los tests E2E hacen `pytest.skip()` automatico si las env vars no estan
> configuradas o el servidor no responde, asi que son seguros de correr
> en cualquier entorno (CI incluido).

### Todos los tests

```bash
pytest tests/ -v
```

### Cobertura de codigo

```bash
pip install pytest-cov
pytest tests/ --cov=handlers --cov=main -v
```

## Markers pytest

| Marker        | Descripcion                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `unit`        | Tests unitarios. Sin red, mocks con `unittest.mock.patch`.                  |
| `e2e`         | Tests end-to-end. Requieren `.env` con API keys y servidor opencode.        |
| `integration` | Tests de integracion. Requieren red pero no servidor opencode. (reservado) |

## Cobertura esperada

| Modulo / Clase                  | Tests que lo cubren                                  |
| ------------------------------- | ---------------------------------------------------- |
| `handlers/response_parser.py`   | test_response_parser.py (6 tests)                    |
| `handlers/audio_manager.py`     | test_audio_manager.py (4 tests)                      |
| `handlers/gemini_stt_client.py` | test_gemini_stt_client.py (5 tests)                  |
| `handlers/opencode_client.py`   | test_opencode_client.py (6 tests)                    |
| `handlers/gemini_tts_client.py` | test_tts_clients.py (4 tests)                        |
| `handlers/azure_tts_client.py`  | test_tts_clients.py (4 tests)                        |
| `main.py` (VoiceAssistant)      | test_state_machine.py (6 tests)                      |
| E2E (todos los handlers)        | test_e2e_scenarios.py (5 tests, marcados `@e2e`)     |

**Totales:** 35 tests unitarios + 5 tests E2E = **40 tests**.

## Seguridad: verificacion de no-log de secrets

Los siguientes tests verifican explicitamente que las API keys y passwords
**no aparecen en ningun log**, usando `caplog` para capturar los mensajes:

- `test_gemini_stt_client.py::TestGeminiSTTClient::test_no_api_key_logged`
- `test_opencode_client.py::TestOpenCodeClient::test_no_password_logged`
- `test_tts_clients.py::test_no_api_keys_logged`

Usan strings unicos tipo `SECRET_GEMINI_KEY_DO_NOT_LEAK_777` que no
colisionan con ninguna otra logica.
