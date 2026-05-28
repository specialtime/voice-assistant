# voice-assistant

Asistente de voz para Windows con Azure Speech y OpenCode, ejecutado en un entorno aislado (`.opencode-voz/`).

## Requisitos

- Python 3.10+
- OpenCode ejecutándose localmente (`opencode serve --config-dir .opencode-voz`)
- Azure Speech (clave y región)

## Instalación

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Variables de entorno

```bash
export AZURE_SPEECH_KEY="tu_clave"
export AZURE_SPEECH_REGION="tu_region"
export AZURE_TTS_VOICE="es-ES-ElviraNeural" # opcional
export AZURE_TTS_STYLE="friendly" # opcional, depende de la voz
export OPENCODE_ENDPOINT="http://127.0.0.1:4096/chat" # opcional
```

## Ejecución

```bash
python main.py --base-dir .
```

Atajo global: `Ctrl+Alt+V` (toggle iniciar/detener grabación).

## Estructura aislada

En el primer arranque se crean:

- `.opencode-voz/config.json` (plugin `opencode-mem` habilitado en puerto 4096)
- `.opencode-voz/agents/asistente_voz.md` (prompt del agente para respuestas en texto plano)
- `.opencode-voz/memory/` (memoria vectorial persistente)
- `.opencode-voz/session.json` (thread_id efímero; se limpia al inicio/fin)

## Pruebas

```bash
python -m unittest discover -s tests -v
```
