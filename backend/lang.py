#!/usr/bin/env python3
import os
import sqlite3
from flask import Flask, g, jsonify, request, abort
from werkzeug.exceptions import HTTPException

DB_DIR = "/var/www/site/data"
DB_PATH = os.path.join(DB_DIR, "lang.db")

# Set this env var in your service config:
#   export LANG_ADMIN_KEY="your-secret"
ADMIN_KEY_ENV = "LANG_ADMIN_KEY"

app = Flask(__name__)

def ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)

def get_db():
    if "db" not in g:
        ensure_db_dir()
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db

@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_words (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          word TEXT NOT NULL UNIQUE,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.commit()

@app.before_request
def _init_once():
    # Safe to run repeatedly; SQLite will no-op on existing table
    init_db()

def require_admin_key():
    expected = os.environ.get(ADMIN_KEY_ENV)
    if not expected:
        abort(500, description=f"{ADMIN_KEY_ENV} is not set on the server.")
    supplied = request.headers.get("X-Admin-Key", "")
    if not supplied or supplied != expected:
        abort(401, description="Invalid or missing admin key.")

@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    return jsonify(error="Internal server error"), 500

@app.get("/api/lang_words")
def list_lang_words():
    db = get_db()
    rows = db.execute(
        "SELECT id, word FROM lang_words ORDER BY created_at DESC, word ASC"
    ).fetchall()
    return jsonify(words=[{"id": r["id"], "word": r["word"]} for r in rows])

@app.post("/api/lang_words")
def create_lang_word():
    require_admin_key()
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")

    db = get_db()
    try:
        db.execute("INSERT INTO lang_words (word) VALUES (?)", (word,))
        db.commit()
    except sqlite3.IntegrityError:
        abort(409, description="Word already exists.")
    new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify(ok=True, id=new_id, word=word), 201


@app.get("/api/lang_words/<int:word_id>")
def get_lang_word(word_id):
    db = get_db()
    row = db.execute(
        "SELECT id, word FROM lang_words WHERE id = ?",
        (word_id,)
    ).fetchone()
    if row is None:
        abort(404, description="Lang word not found.")

    kids = db.execute(
        "SELECT id, word FROM child_words WHERE lang_word_id = ? ORDER BY id ASC",
        (word_id,)
    ).fetchall()

    return jsonify({
        "id": row["id"],
        "word": row["word"],
        "child_words": [{"id": k["id"], "word": k["word"]} for k in kids]
    })


@app.post("/api/lang_words/<int:word_id>/child_words")
def create_child_word(word_id):
    require_admin_key()
    db = get_db()

    # ensure parent exists
    parent = db.execute("SELECT 1 FROM lang_words WHERE id = ?", (word_id,)).fetchone()
    if parent is None:
        abort(404, description="Lang word not found.")

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")

    cur = db.execute(
        "INSERT INTO child_words (lang_word_id, word) VALUES (?, ?)",
        (word_id, word)
    )
    db.commit()

    return jsonify(ok=True, id=cur.lastrowid, word=word), 201


@app.put("/api/child_words/<int:child_id>")
def update_child_word(child_id):
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")

    cur = db.execute(
        "UPDATE child_words SET word = ?, updated_at = datetime('now') WHERE id = ?",
        (word, child_id)
    )
    db.commit()
    if cur.rowcount == 0:
        abort(404, description="Child word not found.")

    return jsonify(ok=True, id=child_id, word=word)


@app.delete("/api/child_words/<int:child_id>")
def delete_child_word(child_id):
    require_admin_key()
    db = get_db()

    cur = db.execute("DELETE FROM child_words WHERE id = ?", (child_id,))
    db.commit()
    if cur.rowcount == 0:
        abort(404, description="Child word not found.")

    return jsonify(ok=True, id=child_id)

