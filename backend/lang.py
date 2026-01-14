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

    db.execute("""
      CREATE TABLE IF NOT EXISTS star_stuff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lang_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (lang_id) REFERENCES lang(id) ON DELETE CASCADE
      );
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_lang_id ON star_stuff(lang_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_type ON star_stuff(type);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_updated_at ON star_stuff(updated_at);")
    db.execute("""
      CREATE TRIGGER IF NOT EXISTS trg_star_stuff_updated_at
      AFTER UPDATE ON star_stuff
      FOR EACH ROW
      BEGIN
        UPDATE star_stuff SET updated_at = datetime('now') WHERE id = NEW.id;
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
    include_stars = request.args.get("include_stars", "1")  # default yes

    try:
        limit_n = max(1, min(1000, int(limit)))
    except Exception:
        limit_n = 200

    rows = db.execute(
        "SELECT id, text, created_at, updated_at FROM lang ORDER BY id DESC LIMIT ?",
        (limit_n,),
    ).fetchall()

    entries = [dict(r) for r in rows]

    if include_stars != "0" and entries:
        lang_ids = [e["id"] for e in entries]
        qmarks = ",".join(["?"] * len(lang_ids))
        star_rows = db.execute(
            f"""
            SELECT id, lang_id, type, text, created_at, updated_at
            FROM star_stuff
            WHERE lang_id IN ({qmarks})
            ORDER BY id DESC
            """,
            tuple(lang_ids),
        ).fetchall()

        stars_by_lang = {}
        for r in star_rows:
            d = dict(r)
            stars_by_lang.setdefault(d["lang_id"], []).append(d)

        for e in entries:
            e["stars"] = stars_by_lang.get(e["id"], [])
    else:
        for e in entries:
            e["stars"] = []

    return jsonify(entries=entries)

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

# --- Stars ---
@app.get("/api/lang/<int:lang_id>/stars")
def list_stars(lang_id: int):
    db = get_db()
    rows = db.execute(
        """
        SELECT id, lang_id, type, text, created_at, updated_at
        FROM star_stuff
        WHERE lang_id=?
        ORDER BY id DESC
        """,
        (lang_id,),
    ).fetchall()
    return jsonify(stars=[dict(r) for r in rows])

@app.post("/api/lang/<int:lang_id>/stars")
def create_star(lang_id: int):
    require_admin()
    db = get_db()

    # Ensure parent exists (clean error)
    parent = db.execute("SELECT id FROM lang WHERE id=?", (lang_id,)).fetchone()
    if not parent:
        abort(404, description="Lang entry not found.")

    data = request.get_json(silent=True) or {}
    star_type = (data.get("type") or "").strip()
    star_text = (data.get("text") or "").strip()

    if not star_type:
        abort(400, description="Missing 'type'.")
    if not star_text:
        abort(400, description="Missing 'text'.")

    cur = db.execute(
        "INSERT INTO star_stuff (lang_id, type, text) VALUES (?, ?, ?)",
        (lang_id, star_type, star_text),
    )
    db.commit()
    return jsonify(ok=True, id=int(cur.lastrowid)), 201

@app.delete("/api/stars/<int:star_id>")
def delete_star(star_id: int):
    require_admin()
    db = get_db()
    db.execute("DELETE FROM star_stuff WHERE id=?", (star_id,))
    db.commit()
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
