#!/usr/bin/env python3
import os
import sqlite3
import json
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

def _table_has_column(db, table: str, col: str) -> bool:
    rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == col for r in rows)

def init_db():
    """
    Idempotent: create missing tables/indexes and run lightweight migrations.
    Your existing DB already has most tables; this keeps dev + prod consistent.
    """
    db = get_db()

    # Core tables
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_words (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          word TEXT NOT NULL UNIQUE,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_word_versions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_word_id INTEGER NOT NULL,
          version INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE(lang_word_id, version),
          FOREIGN KEY (lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_versions_lang_word_id
          ON lang_word_versions(lang_word_id);
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS child_words (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_word_version_id INTEGER NOT NULL,
          word TEXT NOT NULL,
          link TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (lang_word_version_id) REFERENCES lang_word_versions(id) ON DELETE CASCADE
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_words_version_id
          ON child_words(lang_word_version_id);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_words_link
          ON child_words(link);
        """
    )

    # Sentences table
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_sentences (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_word_ids TEXT NOT NULL,  -- JSON array of ints
          sentence TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_lang_sentences_updated
          ON lang_sentences(updated_at);
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS child_sentences (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_sentence_id INTEGER NOT NULL,
          child_word_ids TEXT NOT NULL,  -- JSON array of ints
          sentence TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (lang_sentence_id) REFERENCES lang_sentences(id) ON DELETE CASCADE
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_sentences_lang_sentence_id
          ON child_sentences(lang_sentence_id);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_sentences_updated
          ON child_sentences(updated_at);
        """
    )


    # Lightweight migration if an older DB created child_words without 'link'
    # (Your production DB currently does not have it.)
    if not _table_has_column(db, "child_words", "link"):
        db.execute("ALTER TABLE child_words ADD COLUMN link TEXT;")

    db.commit()

@app.before_request
def _init_once():
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


# =========================
# Words + Versions (existing)
# =========================

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
            "SELECT id, word, link FROM child_words WHERE lang_word_version_id = ? ORDER BY id ASC",
            (vr["version_id"],)
        ).fetchall()

        out_versions.append({
            "version_id": vr["version_id"],
            "version": vr["version"],
            "child_words": [{"id": k["id"], "word": k["word"], "link": k["link"]} for k in kids]
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
    link = (data.get("link") or "").strip() or None

    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")
    if link and len(link) > 2000:
        abort(400, description="Link too long (max 2000 chars).")

    cur = db.execute(
        "INSERT INTO child_words (lang_word_version_id, word, link) VALUES (?, ?, ?)",
        (version_id, word, link)
    )
    db.commit()
    return jsonify(ok=True, id=cur.lastrowid, word=word, link=link), 201


@app.put("/api/child_words/<int:child_id>")
def update_child_word(child_id: int):
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    link = (data.get("link") or "").strip()
    link = link if link else None

    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")
    if link and len(link) > 2000:
        abort(400, description="Link too long (max 2000 chars).")

    cur = db.execute(
        "UPDATE child_words SET word = ?, link = ?, updated_at = datetime('now') WHERE id = ?",
        (word, link, child_id)
    )
    db.commit()
    if cur.rowcount == 0:
        abort(404, description="Child word not found.")

    return jsonify(ok=True, id=child_id, word=word, link=link)


@app.delete("/api/child_words/<int:child_id>")
def delete_child_word(child_id: int):
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


# =========================
# Sentences (new)
# =========================

def _parse_ids(value):
    """
    Accept JSON array string, comma-separated string, list of ints, etc.
    Return a de-duplicated list[int] preserving order.
    """
    ids = []
    if value is None:
        return ids

    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raw = []
        else:
            # try JSON first
            if s.startswith("["):
                try:
                    raw = json.loads(s)
                except Exception:
                    raw = []
            else:
                raw = [x.strip() for x in s.split(",")]
    else:
        raw = []

    seen = set()
    for x in raw:
        try:
            n = int(x)
        except Exception:
            continue
        if n > 0 and n not in seen:
            seen.add(n)
            ids.append(n)
    return ids

@app.get("/api/lang_sentences")
def list_lang_sentences():
    db = get_db()

    # Optional filter: ?lang_word_ids=1,2,3  (means: sentences containing ALL selected ids)
    filter_ids = _parse_ids(request.args.get("lang_word_ids"))

    rows = db.execute(
        """
        SELECT id, lang_word_ids, sentence, created_at, updated_at
        FROM lang_sentences
        ORDER BY updated_at DESC, id DESC
        """
    ).fetchall()

    out = []
    for r in rows:
        try:
            ids = json.loads(r["lang_word_ids"])
            if not isinstance(ids, list):
                ids = []
            ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
        except Exception:
            ids = []

        if filter_ids:
            s = set(ids)
            if any(fid not in s for fid in filter_ids):
                continue

        out.append({
            "id": r["id"],
            "lang_word_ids": ids,
            "sentence": r["sentence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"]
        })

    return jsonify(sentences=out)


@app.post("/api/lang_sentences")
def create_lang_sentence():
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    sentence = (data.get("sentence") or "").strip()
    ids = _parse_ids(data.get("lang_word_ids"))

    if not sentence:
        abort(400, description="Missing 'sentence'.")
    if len(sentence) > 4000:
        abort(400, description="Sentence too long (max 4000 chars).")
    if not ids:
        abort(400, description="Select at least one lang_word_id.")

    # Ensure referenced words exist
    qmarks = ",".join("?" for _ in ids)
    found = db.execute(f"SELECT id FROM lang_words WHERE id IN ({qmarks})", ids).fetchall()
    found_ids = {int(r["id"]) for r in found}
    missing = [i for i in ids if i not in found_ids]
    if missing:
        abort(400, description=f"Unknown lang_word_id(s): {', '.join(map(str, missing))}")

    cur = db.execute(
        "INSERT INTO lang_sentences (lang_word_ids, sentence) VALUES (?, ?)",
        (json.dumps(ids), sentence)
    )
    db.commit()
    return jsonify(ok=True, id=cur.lastrowid, lang_word_ids=ids, sentence=sentence), 201


@app.get("/api/lang_sentences/<int:sentence_id>")
def get_lang_sentence(sentence_id: int):
    db = get_db()

    r = db.execute(
        """
        SELECT id, lang_word_ids, sentence, created_at, updated_at
        FROM lang_sentences
        WHERE id = ?
        """,
        (sentence_id,)
    ).fetchone()
    if r is None:
        abort(404, description="Sentence not found.")

    try:
        ids = json.loads(r["lang_word_ids"])
        if not isinstance(ids, list):
            ids = []
        ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
    except Exception:
        ids = []

    # For each lang_word_id: resolve latest version and its child words (with link)
    words_out = []
    for wid in ids:
        wrow = db.execute("SELECT id, word FROM lang_words WHERE id = ?", (wid,)).fetchone()
        if wrow is None:
            continue

        vrow = db.execute(
            """
            SELECT id AS version_id, version
            FROM lang_word_versions
            WHERE lang_word_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (wid,)
        ).fetchone()
        if vrow is None:
            continue

        kids = db.execute(
            "SELECT id, word, link FROM child_words WHERE lang_word_version_id = ? ORDER BY id ASC",
            (vrow["version_id"],)
        ).fetchall()

        words_out.append({
            "lang_word_id": int(wrow["id"]),
            "word": wrow["word"],
            "version_id": int(vrow["version_id"]),
            "version": int(vrow["version"]),
            "child_words": [{"id": k["id"], "word": k["word"], "link": k["link"]} for k in kids]
        })

    return jsonify({
        "id": r["id"],
        "lang_word_ids": ids,
        "sentence": r["sentence"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "words": words_out
    })

@app.get("/api/lang_sentences/<int:sentence_id>/child_sentences")
def list_child_sentences(sentence_id: int):
    db = get_db()

    # ensure parent sentence exists
    parent = db.execute("SELECT 1 FROM lang_sentences WHERE id = ?", (sentence_id,)).fetchone()
    if parent is None:
        abort(404, description="Sentence not found.")

    filter_ids = _parse_ids(request.args.get("child_word_ids"))

    rows = db.execute(
        """
        SELECT id, child_word_ids, sentence, created_at, updated_at
        FROM child_sentences
        WHERE lang_sentence_id = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (sentence_id,)
    ).fetchall()

    out = []
    for r in rows:
        try:
            ids = json.loads(r["child_word_ids"])
            if not isinstance(ids, list):
                ids = []
            ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
        except Exception:
            ids = []

        if filter_ids:
            s = set(ids)
            if any(fid not in s for fid in filter_ids):
                continue

        out.append({
            "id": r["id"],
            "child_word_ids": ids,
            "sentence": r["sentence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"]
        })

    return jsonify(child_sentences=out)


@app.post("/api/lang_sentences/<int:sentence_id>/child_sentences")
def create_child_sentence(sentence_id: int):
    require_admin_key()
    db = get_db()

    parent = db.execute("SELECT 1 FROM lang_sentences WHERE id = ?", (sentence_id,)).fetchone()
    if parent is None:
        abort(404, description="Sentence not found.")

    data = request.get_json(silent=True) or {}
    sentence = (data.get("sentence") or "").strip()
    ids = _parse_ids(data.get("child_word_ids"))

    if not sentence:
        abort(400, description="Missing 'sentence'.")
    if len(sentence) > 4000:
        abort(400, description="Sentence too long (max 4000 chars).")
    if not ids:
        abort(400, description="Select at least one child_word_id.")

    # ensure child_word ids exist
    qmarks = ",".join("?" for _ in ids)
    found = db.execute(f"SELECT id FROM child_words WHERE id IN ({qmarks})", ids).fetchall()
    found_ids = {int(r["id"]) for r in found}
    missing = [i for i in ids if i not in found_ids]
    if missing:
        abort(400, description=f"Unknown child_word_id(s): {', '.join(map(str, missing))}")

    cur = db.execute(
        "INSERT INTO child_sentences (lang_sentence_id, child_word_ids, sentence) VALUES (?, ?, ?)",
        (sentence_id, json.dumps(ids), sentence)
    )
    db.commit()

    return jsonify(ok=True, id=cur.lastrowid, child_word_ids=ids, sentence=sentence), 201


@app.get("/api/child_sentences/<int:child_sentence_id>")
def get_child_sentence(child_sentence_id: int):
    db = get_db()

    r = db.execute(
        """
        SELECT id, lang_sentence_id, child_word_ids, sentence, created_at, updated_at
        FROM child_sentences
        WHERE id = ?
        """,
        (child_sentence_id,)
    ).fetchone()
    if r is None:
        abort(404, description="Child sentence not found.")

    # parse child_word_ids
    try:
        ids = json.loads(r["child_word_ids"])
        if not isinstance(ids, list):
            ids = []
        ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
    except Exception:
        ids = []

    # load child word rows (with link) + their parent word/version info (optional but useful)
    child_words_out = []
    for cid in ids:
        crow = db.execute(
            """
            SELECT cw.id AS child_word_id, cw.word AS child_word, cw.link,
                   v.id AS version_id, v.version,
                   w.id AS lang_word_id, w.word AS lang_word
            FROM child_words cw
            JOIN lang_word_versions v ON v.id = cw.lang_word_version_id
            JOIN lang_words w ON w.id = v.lang_word_id
            WHERE cw.id = ?
            """,
            (cid,)
        ).fetchone()

        if crow is None:
            continue

        child_words_out.append({
            "child_word_id": int(crow["child_word_id"]),
            "word": crow["child_word"],
            "link": crow["link"],
            "lang_word_id": int(crow["lang_word_id"]),
            "lang_word": crow["lang_word"],
            "version_id": int(crow["version_id"]),
            "version": int(crow["version"]),
        })

    return jsonify({
        "id": int(r["id"]),
        "lang_sentence_id": int(r["lang_sentence_id"]),
        "child_word_ids": ids,
        "sentence": r["sentence"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "child_words": child_words_out
    })



if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
