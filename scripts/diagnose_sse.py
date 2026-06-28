#!/usr/bin/env python3
"""Script de diagnostico para el stream SSE de opencode serve.

Se suscribe a GET /event, crea una sesion, envia prompt_async, y loguea
TODOS los eventos crudos que recibe del server. Util para confirmar:

- Si el stream emite eventos live-only o replay historico.
- Que tipos de eventos envia el server realmente (type y properties/data).
- Si session.idle llega antes que los deltas (session.idle stale).
- Latencia entre prompt_async y primer delta.

Uso:
    python scripts/diagnose_sse.py
    python scripts/diagnose_sse.py --prompt "hola, que hora es?"
    OPENCODE_BASE_URL=http://127.0.0.1:4096 python scripts/diagnose_sse.py
    OPENCODE_SERVER_PASSWORD=miclave python scripts/diagnose_sse.py

Lee la URL base de OPENCODE_BASE_URL (env) o usa 127.0.0.1:4096 por defecto.
Lee OPENCODE_SERVER_PASSWORD para auth basica (vacio = sin auth).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx


def _log_event(label: str, payload: dict | str) -> None:
    """Loguea un evento SSE de manera legible."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            print(f"[{label}] (no JSON) {payload!r}")
            return
    evt_type = payload.get("type", "?")
    props = payload.get("properties") or payload.get("data") or {}
    sid = props.get("sessionID", "?")
    extra = ""
    if "delta" in props:
        extra = f" delta={props['delta']!r}"
    elif "field" in props and isinstance(props["field"], dict):
        extra = f" field={props['field']!r}"
    print(f"[{label}] t={time.monotonic():.3f} type={evt_type!r} sessionID={sid!r}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="Decime un chiste corto en una sola oracion",
        help="Prompt a enviar al agente",
    )
    parser.add_argument(
        "--agent",
        default="asistente_voz",
        help="Nombre del agente a invocar (default: asistente_voz)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Segundos maximos de espera total (default: 30)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_url = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096").rstrip("/")
    password = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
    auth = httpx.BasicAuth("opencode", password) if password else None

    print(f"=== Diagnostico SSE contra {base_url} ===")
    print(f"    auth: {'basica' if auth else 'sin auth'}")
    print(f"    prompt: {args.prompt!r}")
    print(f"    timeout: {args.timeout}s")
    print()

    t_start = time.monotonic()
    client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(args.timeout), auth=auth)

    try:
        # 1) Health check
        try:
            r = client.get("/global/health")
            r.raise_for_status()
            print(f"[health] OK ({r.elapsed.total_seconds():.3f}s)")
        except Exception as e:
            print(f"[health] FAIL: {e}")
            return 1

        # 2) Crear sesion
        r = client.post("/session", json={"title": "diag-sse"})
        r.raise_for_status()
        session_id = r.json()["id"]
        print(f"[session] creada: {session_id}")

        # 3) Suscribirse al stream /event ANTES de enviar prompt_async
        print(f"[subscribe] abriendo GET /event (t={time.monotonic()-t_start:.3f})...")
        delta_count = 0
        idle_seen = False
        try:
            with client.stream("GET", "/event") as stream:
                stream.raise_for_status()
                print(f"[subscribe] headers 200 OK (t={time.monotonic()-t_start:.3f})")

                # Leer primer evento (server.connected) — bloquea hasta que llegue
                line_buf: list[str] = []
                first_event = True
                pending_post = True

                for line_bytes in stream.iter_lines():
                    line = line_bytes.decode("utf-8") if isinstance(line_bytes, bytes) else line_bytes

                    if line.startswith("data: "):
                        line_buf.append(line[6:])
                        continue

                    if line == "" and line_buf:
                        data_str = "\n".join(line_buf)
                        line_buf = []

                        try:
                            payload = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            print(f"[evento] JSON invalido: {e}")
                            continue

                        _log_event("evento  ", payload)

                        evt_type = payload.get("type", "")
                        props = payload.get("properties") or payload.get("data") or {}
                        sid = props.get("sessionID")

                        if first_event:
                            first_event = False
                            # 4) Una vez recibido server.connected (o cualquier primer evento
                            #    que confirme suscripcion activa), enviar prompt_async.
                            if pending_post:
                                t_post = time.monotonic()
                                r = client.post(
                                    f"/session/{session_id}/prompt_async",
                                    json={
                                        "agent": args.agent,
                                        "parts": [{"type": "text", "text": args.prompt}],
                                    },
                                )
                                r.raise_for_status()
                                pending_post = False
                                print(
                                    f"[prompt_async] 204 OK (t={time.monotonic()-t_start:.3f}, "
                                    f"Δ={time.monotonic()-t_post:.3f}s)"
                                )

                        if evt_type == "session.next.text.delta" and sid == session_id:
                            delta_count += 1
                        elif evt_type == "session.idle" and sid == session_id:
                            print(
                                f"[session.idle] recibido tras {delta_count} deltas "
                                f"(t={time.monotonic()-t_start:.3f})"
                            )
                            if delta_count == 0:
                                print(
                                    "    ⚠️  session.idle SIN DELTAS — probable evento stale "
                                    "(hipotesis #2 confirmada)"
                                )
                            else:
                                print("    ✅ session.idle valido — cerrando stream")
                                idle_seen = True
                                break

                    if time.monotonic() - t_start > args.timeout:
                        print(f"[timeout] excedido ({args.timeout}s)")
                        break

        except httpx.HTTPStatusError as e:
            print(f"[subscribe] HTTP error: {e}")
            return 1
        except httpx.RequestError as e:
            print(f"[subscribe] Request error: {e}")
            return 1

        print()
        print("=== Resumen ===")
        print(f"  Deltas recibidos:   {delta_count}")
        print(f"  session.idle valido: {'SI' if idle_seen else 'NO'}")
        print(f"  Tiempo total:       {time.monotonic()-t_start:.3f}s")
        if delta_count == 0:
            print()
            print("⚠️  0 DELTAS — bug de race condition o session.idle stale confirmado.")
            print("    Si el server emite muchos eventos NO-delta antes del primer delta,")
            print("    la suscripcion tardia es la causa.")
            return 2  # codigo de salida no-cero para CI
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
