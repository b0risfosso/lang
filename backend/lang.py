#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
from flask import Flask, g, request, jsonify, abort

DB_DIR = "/var/www/site/data"
LANG_DB_PATH = os.path.join(DB_DIR, "lang.db")

ADMIN_KEY_ENV = "LANG_ADMIN_KEY"
ADMIN_KEY_DEFAULT = "your-secret"

app = Flask(__name__)

def get_admin_key() -> str:
    return os.environ.get(ADMIN_KEY_ENV, ADMIN_KEY_DEFAULT)

def check_admin() -> bool:
    key = request.headers.get("X-Admin-Key")
    return bool(key) and key == get_admin_key()

def require_admin() -> None:
    if not check_admin():
        abort(401, description="Admin key required.")

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        os.makedirs(DB_DIR, exist_ok=True)
        db = sqlite3.connect(LANG_DB_PATH)
        db.row_factory = sqlite3.Row
        g.db = db
        ensure_schema(db)
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def ensure_schema(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA foreign_keys = ON;")
    db.execute("""
      CREATE TABLE IF NOT EXISTS lang (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_lang_updated_at ON lang(updated_at);")
    db.execute("""
      CREATE TRIGGER IF NOT EXISTS trg_lang_updated_at
      AFTER UPDATE ON lang
      FOR EACH ROW
      BEGIN
        UPDATE lang SET updated_at = datetime('now') WHERE id = NEW.id;
      END;
    """)
    db.commit()

# --- Admin ---
@app.get("/api/admin/ping")
def admin_ping():
    require_admin()
    return jsonify(ok=True)

# --- Lang ---
@app.get("/api/lang")
def list_lang():
    db = get_db()
    limit = request.args.get("limit", "200")
    try:
        limit_n = max(1, min(1000, int(limit)))
    except Exception:
        limit_n = 200

    rows = db.execute(
        "SELECT id, text, created_at, updated_at FROM lang ORDER BY id DESC LIMIT ?",
        (limit_n,),
    ).fetchall()

    return jsonify(entries=[dict(r) for r in rows])

@app.post("/api/lang")
def create_lang():
    require_admin()
    db = get_db()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        abort(400, description="Missing 'text'.")

    cur = db.execute("INSERT INTO lang (text) VALUES (?)", (text,))
    db.commit()
    return jsonify(ok=True, id=int(cur.lastrowid)), 201

@app.delete("/api/lang/<int:lang_id>")
def delete_lang(lang_id: int):
    require_admin()
    db = get_db()
    db.execute("DELETE FROM lang WHERE id=?", (lang_id,))
    db.commit()
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
