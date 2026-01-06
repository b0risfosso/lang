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
        """
        SELECT w.id AS lang_word_id, w.word,
               v.id AS version_id, v.version
        FROM lang_words w
        JOIN lang_word_versions v
          ON v.lang_word_id = w.id
        WHERE v.version = (
          SELECT MAX(v2.version) FROM lang_word_versions v2 WHERE v2.lang_word_id = w.id
        )
        ORDER BY w.word ASC
        """
    ).fetchall()

    return jsonify(words=[
        {"lang_word_id": r["lang_word_id"], "word": r["word"], "version_id": r["version_id"], "version": r["version"]}
        for r in rows
    ])



@app.post("/api/lang_words")
def create_lang_word():
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")

    try:
        cur = db.execute("INSERT INTO lang_words (word) VALUES (?)", (word,))
        lang_word_id = cur.lastrowid
        cur2 = db.execute(
            "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, ?)",
            (lang_word_id, 1)
        )
        version_id = cur2.lastrowid
        db.commit()
    except sqlite3.IntegrityError:
        abort(409, description="Word already exists.")

    return jsonify(ok=True, lang_word_id=lang_word_id, version_id=version_id, word=word), 201


@app.post("/api/lang_words/<int:lang_word_id>/versions")
def create_version(lang_word_id: int):
    require_admin_key()
    db = get_db()

    exists = db.execute("SELECT 1 FROM lang_words WHERE id = ?", (lang_word_id,)).fetchone()
    if exists is None:
        abort(404, description="Lang word not found.")

    row = db.execute(
        "SELECT COALESCE(MAX(version), 0) AS mv FROM lang_word_versions WHERE lang_word_id = ?",
        (lang_word_id,)
    ).fetchone()
    next_ver = int(row["mv"]) + 1

    cur = db.execute(
        "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, ?)",
        (lang_word_id, next_ver)
    )
    db.commit()
    return jsonify(ok=True, version_id=cur.lastrowid, lang_word_id=lang_word_id, version=next_ver), 201



@app.get("/api/lang_words/<int:version_id>")
def get_lang_word_by_version(version_id: int):
    db = get_db()

    v = db.execute(
        """
        SELECT v.id AS version_id, v.version, v.lang_word_id, w.word
        FROM lang_word_versions v
        JOIN lang_words w ON w.id = v.lang_word_id
        WHERE v.id = ?
        """,
        (version_id,)
    ).fetchone()
    if v is None:
        abort(404, description="Lang word version not found.")

    versions = db.execute(
        """
        SELECT id AS version_id, version
        FROM lang_word_versions
        WHERE lang_word_id = ?
        ORDER BY version ASC
        """,
        (v["lang_word_id"],)
    ).fetchall()

    out_versions = []
    for vr in versions:
        kids = db.execute(
            "SELECT id, word FROM child_words WHERE lang_word_version_id = ? ORDER BY id ASC",
            (vr["version_id"],)
        ).fetchall()

        out_versions.append({
            "version_id": vr["version_id"],
            "version": vr["version"],
            "child_words": [{"id": k["id"], "word": k["word"]} for k in kids]
        })

    return jsonify({
        "lang_word_id": v["lang_word_id"],
        "word": v["word"],
        "current_version_id": v["version_id"],
        "current_version": v["version"],
        "versions": out_versions
    })



@app.post("/api/lang_word_versions/<int:version_id>/child_words")
def create_child_word(version_id: int):
    require_admin_key()
    db = get_db()

    parent = db.execute("SELECT 1 FROM lang_word_versions WHERE id = ?", (version_id,)).fetchone()
    if parent is None:
        abort(404, description="Version not found.")

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")

    cur = db.execute(
        "INSERT INTO child_words (lang_word_version_id, word) VALUES (?, ?)",
        (version_id, word)
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


@app.get("/api/admin/ping")
def admin_ping():
    require_admin_key()
    return jsonify(ok=True)


@app.put("/api/child_words/<int:child_id>/move")
def move_child_word(child_id: int):
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    target_version_id = data.get("lang_word_version_id")
    if not isinstance(target_version_id, int):
        abort(400, description="Missing/invalid 'lang_word_version_id'.")

    parent = db.execute("SELECT 1 FROM lang_word_versions WHERE id = ?", (target_version_id,)).fetchone()
    if parent is None:
        abort(404, description="Target version not found.")

    cur = db.execute(
        "UPDATE child_words SET lang_word_version_id = ?, updated_at = datetime('now') WHERE id = ?",
        (target_version_id, child_id)
    )
    db.commit()
    if cur.rowcount == 0:
        abort(404, description="Child word not found.")

    return jsonify(ok=True, id=child_id, lang_word_version_id=target_version_id)

