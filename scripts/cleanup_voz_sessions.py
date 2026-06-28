#!/usr/bin/env python3
"""Limpia todas las sesiones de opencode con titulo 'voz' de la base de datos SQLite.

Crea un backup automatico antes de eliminar. Usa --dry-run para previsualizar.

Uso:
    python cleanup_voz_sessions.py --dry-run    # Previsualizar (no elimina)
    python cleanup_voz_sessions.py              # Eliminar con backup previo
"""

import argparse
import datetime
import os
import shutil
import sqlite3
import sys

DB_PATH = r"C:\Users\crist\.local\share\opencode\opencode.db"


def find_column(columns, *candidates):
    """Busca una columna de manera insensible a mayusculas/minusculas."""
    col_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Limpiar sesiones 'voz' de la base de datos de opencode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar lo que se eliminaria, sin tocar la BD",
    )
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: No se encontro la base de datos: {DB_PATH}")
        sys.exit(1)

    # --- Backup ---
    if not args.dry_run:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{DB_PATH}.backup_{ts}"
        shutil.copy2(DB_PATH, backup)
        print(f"Backup creado: {backup}\n")

    # --- Conectar ---
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    cursor = conn.cursor()

    # --- Listar tablas ---
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tablas encontradas: {tables}\n")

    # --- Descubrir esquema de cada tabla ---
    table_cols = {}
    for t in tables:
        cursor.execute(f"PRAGMA table_info('{t}')")
        table_cols[t] = [r[1] for r in cursor.fetchall()]

    # --- Buscar tabla de sesiones (tiene columna 'title' + 'id') ---
    session_table = None
    id_col = None
    title_col = None
    for t, cols in table_cols.items():
        tc = find_column(cols, "title")
        ic = find_column(cols, "id", "session_id")
        if tc and ic:
            session_table = t
            title_col = tc
            id_col = ic
            print(f"Tabla de sesiones: '{t}'")
            print(f"  Columna ID:    '{id_col}'")
            print(f"  Columna title: '{title_col}'")
            print(f"  Columnas:      {cols}\n")
            break

    if not session_table:
        print("ERROR: No se encontro tabla de sesiones con columna 'title'.")
        for t, cols in table_cols.items():
            print(f"  {t}: {cols}")
        conn.close()
        sys.exit(1)

    # --- Buscar tablas con FK a sesiones ---
    fk_tables = []
    for t, cols in table_cols.items():
        if t == session_table:
            continue
        fk_col = find_column(cols, "session_id", "sessionID", "session_id_id")
        if fk_col:
            fk_tables.append((t, fk_col))
    if fk_tables:
        print(f"Tablas con FK a sesiones (se limpiaran en cascada): "
              f"{[(t, c) for t, c in fk_tables]}\n")

    # --- Buscar sesiones con title = 'voz' ---
    cursor.execute(
        f"SELECT {id_col}, {title_col} FROM {session_table} WHERE {title_col} = 'voz'"
    )
    sessions = cursor.fetchall()
    print(f"Sesiones con titulo 'voz': {len(sessions)}\n")

    if not sessions:
        print("No hay sesiones para eliminar. Todo limpio.")
        conn.close()
        return

    for sid, title in sessions:
        print(f"  - {sid}")

    if args.dry_run:
        print(f"\n[DRY RUN] Se eliminararian {len(sessions)} sesiones "
              f"(+ sus mensajes/archivos asociados).")
        print("Ejecuta sin --dry-run para eliminar.")
        conn.close()
        return

    # --- Eliminar ---
    deleted = 0
    errors = 0
    for sid, _ in sessions:
        try:
            # Eliminar dependencias primero (por si no hay CASCADE)
            for fk_t, fk_c in fk_tables:
                cursor.execute(f"DELETE FROM {fk_t} WHERE {fk_c} = ?", (sid,))
            # Eliminar la sesion
            cursor.execute(
                f"DELETE FROM {session_table} WHERE {id_col} = ?", (sid,)
            )
            deleted += 1
        except sqlite3.Error as e:
            print(f"  ERROR al eliminar {sid}: {e}")
            errors += 1
            conn.rollback()

    conn.commit()
    print(f"\nResultado: {deleted}/{len(sessions)} sesiones eliminadas, "
          f"{errors} errores.")

    # --- Verificar ---
    cursor.execute(
        f"SELECT COUNT(*) FROM {session_table} WHERE {title_col} = 'voz'"
    )
    remaining = cursor.fetchone()[0]
    print(f"Sesiones 'voz' restantes: {remaining}")

    conn.close()
    print("\nListo.")


if __name__ == "__main__":
    main()
