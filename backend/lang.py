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

    return jsonify({"id": row["id"], "word": row["word"]})


