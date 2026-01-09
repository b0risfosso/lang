#!/usr/bin/env python3
"""
lang.py (updated)

New data model:
- No `langs` table.
- `lang_words` are nodes with version history in `lang_word_versions`.
- Each version can have:
    * child_words (orbiting phrases) in `child_words`
    * child lang words (themes/subthemes) via `lang_word_children` edges from parent version -> child lang_word_id
- Sentences live in `lang_sentences` (JSON list of lang_word_ids).
- Optional child sentences live in `child_sentences` (JSON list of child_word_ids).
- LLM pipeline tables are keyed by `parent_lang_word_id` (a root/parent word), not `lang_id`.

This file is intended to fully replace your previous lang.py.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback
from typing import Any, Optional

from flask import Flask, abort, g, jsonify, request

# Optional: OpenAI for the built-in LLM worker.
# If you don't want the in-process worker, you can remove these imports and the worker functions.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DB_DIR = "/var/www/site/data"
LANG_DB_PATH = os.path.join(DB_DIR, "lang.db")

ADMIN_KEY_ENV = "LANG_ADMIN_KEY"
ADMIN_KEY_DEFAULT = "your-secret"  # fallback if env not set

# LLM config
LLM_MODEL_CREATE_LANG_WORDS = os.environ.get("LLM_MODEL_CREATE_LANG_WORDS", "gpt-5-mini-2025-08-07")


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

app = Flask(__name__)

# -------------------------------------
# Star Stuff prompts (PHRASE is the child word; CONTEXT is lang_name + " / " + lang_word)
# -------------------------------------
STAR_STUFF_PROMPTS = {
    "create_star_stuff_words": """Role:
You are an expert technical writer and domain specialist.
Task:
Build a clear, accurate, and well-structured description of the following Phrase, using the provided Context to frame its domain, purpose, and relevance. If a Modifier is provided, use it to guide emphasis, scope, or perspective.
Inputs
Phrase:
{PHRASE}
Context:
{CONTEXT}
Modifier (optional):
{MODIFIER}
Instructions
Define and situate the phrase
Clearly explain what the phrase refers to
Place it within the scientific, technical, organizational, or conceptual domain implied by the context
Explain structure, function, or mechanism
Describe how the system/entity/idea works
Highlight key components, processes, or relationships
Use domain-appropriate terminology with clarity
Describe role and significance
Explain why it matters within the given context
Connect it to broader workflows, lifecycles, or ecosystems where applicable
Apply the modifier (if provided)
Adjust tone, depth, and emphasis according to the modifier
Examples: technical, conceptual, user-focused, regulatory, educational, comparative, computational
Maintain clarity and structure
Use short sections with informative headings
Avoid unnecessary jargon unless the modifier specifies an expert audience
Output Requirements
Length: ~200–400 words
Style: Clear, professional, and explanatory
Structure:
Title or opening definition
2–4 short thematic sections
Do not ask follow-up questions
Do not reference this prompt or the input format
""",
    "create_star_stuff_images": """Role
You are an expert research assistant skilled in visual curation for education, analysis, and conceptual communication.
Inputs
Phase:
{PHRASE}
Context:
{CONTEXT}
Optional Modifier (if provided):
{MODIFIER}
Task
Collect and curate representative images that visually explain and contextualize the Phase, interpreted within the provided Context and guided by the Optional Modifier.
Output Format
Use the following structure:
## [Conceptual Group Name]

Image 1:
- Title:
- Description:
- Relevance:
- Source:
""",
    "create_star_stuff_data": """You are an expert analyst and technical explainer.

TASK:
Describe the DATA associated with the following phase, focusing on:
- What data exist
- How the data are generated or measured
- How the data are structured and used
- Where the data come from (data sources)
- What assumptions or limitations apply

PHASE:
"{PHRASE}"

CONTEXT:
{CONTEXT}

OPTIONAL MODIFIER:
{MODIFIER}
""",
    "create_star_stuff_synthetic_data": """SYSTEM / ROLE
You are an expert synthetic data architect and domain modeler.
You generate fictional but internally consistent datasets suitable for analysis, simulation, ML, and education.

TASK INSTRUCTION
Build a synthetic dataset describing the following:
Phase: {PHRASE}
Context: {CONTEXT}
Modifier (if provided): {MODIFIER}
""",
    "create_star_stuff_code": """You are a senior software engineer and technical educator.

PHRASE:
{PHRASE}

CONTEXT:
{CONTEXT}

OPTIONAL MODIFIER:
{MODIFIER}

DELIVERABLE:
Output ONLY the code in a single fenced code block with the language specified.
""",
}

_worker_started = False
_worker_lock = threading.Lock()


# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------

def ensure_db_dir() -> None:
    os.makedirs(DB_DIR, exist_ok=True)


def get_db() -> sqlite3.Connection:
    """
    Flask request-scoped DB connection.
    """
    if "db" not in g:
        ensure_db_dir()
        g.db = sqlite3.connect(LANG_DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
        # WAL is fine; we use it for better concurrency on a single file DB.
        g.db.execute("PRAGMA journal_mode = WAL;")
    return g.db


@app.teardown_appcontext
def close_db(_exc: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def require_admin_key() -> None:
    expected = os.environ.get(ADMIN_KEY_ENV, ADMIN_KEY_DEFAULT)
    got = request.headers.get("X-Admin-Key", "")
    if not got or got != expected:
        abort(401, description="Missing/invalid admin key.")


def _json_loads_safe(s: Any, default: Any) -> Any:
    try:
        out = json.loads(s) if isinstance(s, str) else default
        return out
    except Exception:
        return default


def _parse_int_list_json(s: str) -> list[int]:
    arr = _json_loads_safe(s, [])
    if not isinstance(arr, list):
        return []
    out: list[int] = []
    for x in arr:
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, str) and x.isdigit():
            out.append(int(x))
    return out


def ensure_word_has_v1(db: sqlite3.Connection, lang_word_id: int) -> int:
    """
    Ensure there is at least version 1 for this lang_word.
    Returns the newest version_id (v1 if newly created).
    """
    v = db.execute(
        "SELECT id, version FROM lang_word_versions WHERE lang_word_id=? ORDER BY version DESC LIMIT 1",
        (lang_word_id,),
    ).fetchone()
    if v is not None:
        return int(v["id"])

    cur = db.execute(
        "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, 1)",
        (lang_word_id,),
    )
    return int(cur.lastrowid)


def create_next_version(db: sqlite3.Connection, lang_word_id: int) -> int:
    """
    Create and return a new version row for a lang_word (v+1).
    If no versions exist yet, creates v1.
    """
    last = db.execute(
        "SELECT version FROM lang_word_versions WHERE lang_word_id=? ORDER BY version DESC LIMIT 1",
        (lang_word_id,),
    ).fetchone()
    next_version = int(last["version"]) + 1 if last else 1
    cur = db.execute(
        "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, ?)",
        (lang_word_id, next_version),
    )
    return int(cur.lastrowid)


    cur = db.execute(
        "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, 1)",
        (lang_word_id,),
    )
    return int(cur.lastrowid)


# ---------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------

def ensure_schema(db: sqlite3.Connection) -> None:
    """
    Create tables if missing. Safe to call many times.
    """
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
    db.execute("CREATE INDEX IF NOT EXISTS idx_versions_lang_word_id ON lang_word_versions(lang_word_id);")

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
    db.execute("CREATE INDEX IF NOT EXISTS idx_child_words_version_id ON child_words(lang_word_version_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_child_words_link ON child_words(link);")

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_word_children (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          parent_lang_word_version_id INTEGER NOT NULL,
          child_lang_word_id INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE(parent_lang_word_version_id, child_lang_word_id),
          FOREIGN KEY (parent_lang_word_version_id) REFERENCES lang_word_versions(id) ON DELETE CASCADE,
          FOREIGN KEY (child_lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
        );
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_lwc_parent_version ON lang_word_children(parent_lang_word_version_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_lwc_child_word ON lang_word_children(child_lang_word_id);")

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS lang_sentences (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_word_ids TEXT NOT NULL,   -- JSON array of ints (lang_words.id)
          sentence TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_lang_sentences_updated ON lang_sentences(updated_at);")

    # Optional child sentences (keep backend support even if the page is deleted)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS child_sentences (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_sentence_id INTEGER NOT NULL,
          child_word_ids TEXT NOT NULL,     -- JSON array of ints (child_words.id)
          sentence TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (lang_sentence_id) REFERENCES lang_sentences(id) ON DELETE CASCADE
        );
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_child_sentences_lang_sentence_id ON child_sentences(lang_sentence_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_child_sentences_updated ON child_sentences(updated_at);")

    # LLM pipeline (keyed by parent_lang_word_id)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          parent_lang_word_id INTEGER NOT NULL,
          task_type TEXT NOT NULL,         -- e.g. "create_lang_words"
          identifier TEXT NOT NULL,        -- JSON {"parent_lang_word_id": ...}
          payload TEXT NOT NULL,           -- JSON {modifier, ...}
          status TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|error
          error TEXT,
          result_writing_id INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (parent_lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
        );
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_llm_tasks_status ON llm_tasks(status);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_llm_tasks_parent_id ON llm_tasks(parent_lang_word_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_llm_tasks_parent_status ON llm_tasks(parent_lang_word_id, status);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_llm_tasks_created_at ON llm_tasks(created_at);")

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS temporary_writings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          parent_lang_word_id INTEGER NOT NULL,
          identifier TEXT NOT NULL,        -- JSON {"parent_lang_word_id": ...}
          prompt_type TEXT NOT NULL,       -- e.g. "create_lang_words"
          text TEXT NOT NULL,              -- JSON output from LLM
          model TEXT,
          modifier TEXT,
          task_id INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (parent_lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
        );
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_temp_parent_id ON temporary_writings(parent_lang_word_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_temp_parent_created ON temporary_writings(parent_lang_word_id, created_at);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_temp_prompt_type ON temporary_writings(prompt_type);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_temp_created_at ON temporary_writings(created_at);")

    # updated_at triggers
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_child_words_updated_at
        AFTER UPDATE ON child_words
        FOR EACH ROW
        BEGIN
          UPDATE child_words SET updated_at = datetime('now') WHERE id = NEW.id;
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_lang_sentences_updated_at
        AFTER UPDATE ON lang_sentences
        FOR EACH ROW
        BEGIN
          UPDATE lang_sentences SET updated_at = datetime('now') WHERE id = NEW.id;
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_child_sentences_updated_at
        AFTER UPDATE ON child_sentences
        FOR EACH ROW
        BEGIN
          UPDATE child_sentences SET updated_at = datetime('now') WHERE id = NEW.id;
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_llm_tasks_updated_at
        AFTER UPDATE ON llm_tasks
        FOR EACH ROW
        BEGIN
          UPDATE llm_tasks SET updated_at = datetime('now') WHERE id = NEW.id;
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_temporary_writings_updated_at
        AFTER UPDATE ON temporary_writings
        FOR EACH ROW
        BEGIN
          UPDATE temporary_writings SET updated_at = datetime('now') WHERE id = NEW.id;
        END;
        """
    )

    
    # -------------------------------------
    # star_stuff: LLM outputs for a child_word
    # -------------------------------------
    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS star_stuff (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          child_word_id INTEGER NOT NULL,
          prompt_type TEXT NOT NULL,
          text TEXT NOT NULL,
          model TEXT,
          modifier TEXT,
          task_id INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (child_word_id) REFERENCES child_words(id) ON DELETE CASCADE
        );
        '''
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_child_word_id ON star_stuff(child_word_id);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_prompt_type ON star_stuff(prompt_type);")
    db.execute("CREATE INDEX IF NOT EXISTS idx_star_stuff_created_at ON star_stuff(created_at);")

db.commit()


@app.before_request
def _ensure_schema_and_worker() -> None:
    db = get_db()
    ensure_schema(db)
    _ensure_worker_started()


# ---------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------

@app.get("/api/admin/ping")
def admin_ping():
    """
    Returns ok:true if the provided X-Admin-Key matches.
    """
    require_admin_key()
    return jsonify(ok=True)


def _resolve_star_context(db: sqlite3.Connection, child_word_id: int) -> dict:
    row = db.execute(
        """
        SELECT
          cw.id AS child_word_id,
          cw.word AS child_word,
          cw.link AS child_link,
          v.id AS version_id,
          v.version AS version,
          lw.id AS lang_word_id,
          lw.word AS lang_word
        FROM child_words cw
        JOIN lang_word_versions v ON v.id = cw.lang_word_version_id
        JOIN lang_words lw ON lw.id = v.lang_word_id
        WHERE cw.id = ?
        """,
        (child_word_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"child_word_id {child_word_id} not found")

    parent = db.execute(
        """
        SELECT plw.id AS parent_lang_word_id, plw.word AS parent_lang_word
        FROM lang_word_children lwc
        JOIN lang_word_versions pv ON pv.id = lwc.parent_lang_word_version_id
        JOIN lang_words plw ON plw.id = pv.lang_word_id
        WHERE lwc.child_lang_word_id = ?
        ORDER BY pv.created_at DESC
        LIMIT 1
        """,
        (int(row["lang_word_id"]),),
    ).fetchone()

    return {
        "child_word": {"id": int(row["child_word_id"]), "word": row["child_word"], "link": row["child_link"]},
        "lang_word": {
            "id": int(row["lang_word_id"]),
            "word": row["lang_word"],
            "version_id": int(row["version_id"]),
            "version": int(row["version"]),
        },
        "lang_name": (
            {"id": int(parent["parent_lang_word_id"]), "word": parent["parent_lang_word"]}
            if parent is not None
            else {"id": None, "word": None}
        ),
    }


@app.get("/api/stars")
def api_stars():
    child_word_id = request.args.get("child_word_id", type=int)
    if not child_word_id:
        abort(400, description="child_word_id is required")

    db = get_db()
    ctx = _resolve_star_context(db, child_word_id)

    stuff = db.execute(
        """
        SELECT id, child_word_id, prompt_type, text, model, modifier, task_id, created_at, updated_at
        FROM star_stuff
        WHERE child_word_id=?
        ORDER BY created_at DESC
        """,
        (child_word_id,),
    ).fetchall()

    return jsonify(
        {
            **ctx,
            "star_stuff": [
                {
                    "id": int(r["id"]),
                    "child_word_id": int(r["child_word_id"]),
                    "prompt_type": r["prompt_type"],
                    "text": r["text"],
                    "model": r["model"],
                    "modifier": r["modifier"],
                    "task_id": r["task_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in stuff
            ],
        }
    )


@app.post("/api/star_stuff/tasks")
@require_admin
def api_star_stuff_tasks():
    data = request.get_json(force=True) or {}
    child_word_id = int(data.get("child_word_id") or 0)
    prompts = data.get("prompts") or []
    modifier = (data.get("modifier") or "").strip()

    if not child_word_id:
        abort(400, description="child_word_id is required")
    if not isinstance(prompts, list) or not prompts:
        abort(400, description="prompts must be a non-empty list")

    allowed = set(STAR_STUFF_PROMPTS.keys())
    for p in prompts:
        if p not in allowed:
            abort(400, description=f"invalid prompt: {p}")

    db = get_db()
    ctx = _resolve_star_context(db, child_word_id)
    parent_id = ctx["lang_name"]["id"] or ctx["lang_word"]["id"]

    task_ids = []
    for p in prompts:
        identifier = json.dumps(
            {
                "child_word_id": child_word_id,
                "lang_word_id": ctx["lang_word"]["id"],
                "lang_name_id": ctx["lang_name"]["id"],
            }
        )
        payload = json.dumps({"modifier": modifier})
        cur = db.execute(
            """
            INSERT INTO llm_tasks (parent_lang_word_id, task_type, identifier, payload, status)
            VALUES (?, ?, ?, ?, 'queued')
            """,
            (int(parent_id), p, identifier, payload),
        )
        task_ids.append(int(cur.lastrowid))

    db.commit()
    return jsonify({"task_ids": task_ids})


# ---------------------------------------------------------------------
# Words + versions + children
# ---------------------------------------------------------------------

@app.get("/api/lang_words")
def list_lang_words():
    db = get_db()
    rows = db.execute("SELECT id, word, created_at FROM lang_words ORDER BY word ASC, id ASC").fetchall()

    out = []
    for r in rows:
        v = db.execute(
            "SELECT id AS version_id, version FROM lang_word_versions WHERE lang_word_id=? ORDER BY version DESC LIMIT 1",
            (r["id"],),
        ).fetchone()
        out.append({
            "lang_word_id": int(r["id"]),
            "word": r["word"],
            "created_at": r["created_at"],
            "latest_version_id": int(v["version_id"]) if v else None,
            "latest_version": int(v["version"]) if v else None,
        })
    return jsonify(words=out)


@app.post("/api/lang_words")
def create_lang_word_generic():
    """
    Generic lang word creation (admin). Use /api/lang_words/parents for explicit "root" creation in the UI,
    but this endpoint remains useful for scripts.
    """
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")

    cur = db.execute("INSERT INTO lang_words (word) VALUES (?)", (word,))
    lang_word_id = int(cur.lastrowid)
    version_id = ensure_word_has_v1(db, lang_word_id)
    db.commit()

    return jsonify(ok=True, lang_word_id=lang_word_id, version_id=version_id, word=word), 201


@app.post("/api/lang_words/parents")
def create_parent_lang_word():
    """
    Create a parent/root lang word (admin). Root-ness is defined by "has no parent edge in lang_word_children".
    """
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        abort(400, description="Missing 'word'.")
    if len(word) > 200:
        abort(400, description="Word too long (max 200 chars).")

    cur = db.execute("INSERT INTO lang_words (word) VALUES (?)", (word,))
    lang_word_id = int(cur.lastrowid)
    version_id = ensure_word_has_v1(db, lang_word_id)
    db.commit()
    return jsonify(ok=True, lang_word_id=lang_word_id, version_id=version_id, word=word), 201


@app.post("/api/lang_words/<int:lang_word_id>/versions")
def create_lang_word_version(lang_word_id: int):
    require_admin_key()
    db = get_db()

    w = db.execute("SELECT id FROM lang_words WHERE id=?", (lang_word_id,)).fetchone()
    if w is None:
        abort(404, description="Lang word not found.")

    last = db.execute(
        "SELECT version FROM lang_word_versions WHERE lang_word_id=? ORDER BY version DESC LIMIT 1",
        (lang_word_id,),
    ).fetchone()
    next_version = int(last["version"]) + 1 if last else 1

    cur = db.execute(
        "INSERT INTO lang_word_versions (lang_word_id, version) VALUES (?, ?)",
        (lang_word_id, next_version),
    )
    db.commit()
    return jsonify(ok=True, version_id=int(cur.lastrowid), version=next_version), 201


@app.get("/api/lang_words/<int:version_id>")
def get_lang_word_version(version_id: int):
    """
    Get a specific lang_word_version, including:
      - child_words
      - child_lang_words (linked via lang_word_children)
    """
    db = get_db()

    vr = db.execute(
        """
        SELECT v.id AS version_id, v.version, v.created_at, w.id AS lang_word_id, w.word
        FROM lang_word_versions v
        JOIN lang_words w ON w.id = v.lang_word_id
        WHERE v.id=?
        """,
        (version_id,),
    ).fetchone()
    if vr is None:
        abort(404, description="Version not found.")

    crows = db.execute(
        """
        SELECT id, word, link, created_at, updated_at
        FROM child_words
        WHERE lang_word_version_id=?
        ORDER BY created_at ASC, id ASC
        """,
        (version_id,),
    ).fetchall()

    child_words = [{
        "id": int(r["id"]),
        "word": r["word"],
        "link": r["link"] or "",
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    } for r in crows]

    grows = db.execute(
        """
        SELECT w.id, w.word
        FROM lang_word_children lwc
        JOIN lang_words w ON w.id = lwc.child_lang_word_id
        WHERE lwc.parent_lang_word_version_id=?
        ORDER BY w.word ASC, w.id ASC
        """,
        (version_id,),
    ).fetchall()

    child_lang_words = [{"lang_word_id": int(r["id"]), "word": r["word"]} for r in grows]

    return jsonify({
        "version_id": int(vr["version_id"]),
        "version": int(vr["version"]),
        "created_at": vr["created_at"],
        "lang_word_id": int(vr["lang_word_id"]),
        "word": vr["word"],
        "child_words": child_words,
        "child_lang_words": child_lang_words,
    })


@app.delete("/api/lang_words/<int:lang_word_id>")
def delete_lang_word(lang_word_id: int):
    require_admin_key()
    db = get_db()
    db.execute("DELETE FROM lang_words WHERE id=?", (lang_word_id,))
    db.commit()
    return jsonify(ok=True)


@app.delete("/api/lang_word_versions/<int:version_id>")
def delete_lang_word_version(version_id: int):
    require_admin_key()
    db = get_db()
    db.execute("DELETE FROM lang_word_versions WHERE id=?", (version_id,))
    db.commit()
    return jsonify(ok=True)


@app.post("/api/lang_word_versions/<int:version_id>/child_words")
def create_child_word(version_id: int):
    require_admin_key()
    db = get_db()

    vr = db.execute("SELECT id FROM lang_word_versions WHERE id=?", (version_id,)).fetchone()
    if vr is None:
        abort(404, description="Version not found.")

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    link = (data.get("link") or "").strip() or None
    if not word:
        abort(400, description="Missing 'word'.")

    cur = db.execute(
        "INSERT INTO child_words (lang_word_version_id, word, link) VALUES (?, ?, ?)",
        (version_id, word, link),
    )
    db.commit()
    return jsonify(ok=True, child_id=int(cur.lastrowid)), 201


@app.post("/api/lang_word_versions/<int:version_id>/child_lang_words")
def add_child_lang_word(version_id: int):
    require_admin_key()
    db = get_db()

    parent = db.execute("SELECT id FROM lang_word_versions WHERE id=?", (version_id,)).fetchone()
    if parent is None:
        abort(404, description="Version not found.")

    data = request.get_json(silent=True) or {}
    child_lang_word_id = data.get("child_lang_word_id", None)
    try:
        child_lang_word_id = int(child_lang_word_id)
    except Exception:
        abort(400, description="Missing/invalid child_lang_word_id.")

    exists = db.execute("SELECT id FROM lang_words WHERE id=?", (child_lang_word_id,)).fetchone()
    if exists is None:
        abort(404, description="Child lang word not found.")

    db.execute(
        """
        INSERT OR IGNORE INTO lang_word_children (parent_lang_word_version_id, child_lang_word_id)
        VALUES (?, ?)
        """,
        (version_id, child_lang_word_id),
    )
    db.commit()
    return jsonify(ok=True, parent_version_id=version_id, child_lang_word_id=child_lang_word_id)


@app.put("/api/child_words/<int:child_id>")
def update_child_word(child_id: int):
    require_admin_key()
    db = get_db()

    row = db.execute("SELECT id FROM child_words WHERE id=?", (child_id,)).fetchone()
    if row is None:
        abort(404, description="Child word not found.")

    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    link = (data.get("link") or "").strip()

    if not word:
        abort(400, description="Missing 'word'.")

    db.execute(
        "UPDATE child_words SET word=?, link=? WHERE id=?",
        (word, link or None, child_id),
    )
    db.commit()
    return jsonify(ok=True)


@app.delete("/api/child_words/<int:child_id>")
def delete_child_word(child_id: int):
    require_admin_key()
    db = get_db()
    db.execute("DELETE FROM child_words WHERE id=?", (child_id,))
    db.commit()
    return jsonify(ok=True)


@app.put("/api/child_words/<int:child_id>/move")
def move_child_word(child_id: int):
    """
    Move a child_word to a different lang_word_version.
    Body: { "to_version_id": 123 }
    """
    require_admin_key()
    db = get_db()

    row = db.execute("SELECT id FROM child_words WHERE id=?", (child_id,)).fetchone()
    if row is None:
        abort(404, description="Child word not found.")

    data = request.get_json(silent=True) or {}
    to_version_id = data.get("to_version_id", None)
    try:
        to_version_id = int(to_version_id)
    except Exception:
        abort(400, description="Missing/invalid to_version_id.")

    vr = db.execute("SELECT id FROM lang_word_versions WHERE id=?", (to_version_id,)).fetchone()
    if vr is None:
        abort(404, description="Target version not found.")

    db.execute("UPDATE child_words SET lang_word_version_id=? WHERE id=?", (to_version_id, child_id))
    db.commit()
    return jsonify(ok=True)


# ---------------------------------------------------------------------
# Write tree (for write.html read panel)
# ---------------------------------------------------------------------

@app.get("/api/write")
def write_tree():
    """
    Return root/parent lang words (words that are NOT a child of any parent version),
    with their versions, and (for each version) both child_words and child_lang_words.

    Shape:
    {
      "roots": [
        {
          "lang_word_id": ...,
          "word": ...,
          "versions": [
            {
              "version_id": ...,
              "version": ...,
              "child_words": [...],
              "child_lang_words": [...]
            }, ...
          ],
          "current_child_lang_words": [...]  # children of latest version
        }
      ]
    }
    """
    db = get_db()

    root_rows = db.execute(
        """
        SELECT w.id, w.word, w.created_at
        FROM lang_words w
        WHERE NOT EXISTS (
          SELECT 1
          FROM lang_word_children lwc
          WHERE lwc.child_lang_word_id = w.id
        )
        ORDER BY w.word ASC, w.id ASC
        """
    ).fetchall()

    def load_versions(lang_word_id: int):
        vrows = db.execute(
            """
            SELECT id AS version_id, version, created_at
            FROM lang_word_versions
            WHERE lang_word_id=?
            ORDER BY version DESC
            """,
            (lang_word_id,),
        ).fetchall()

        versions_out = []
        for vr in vrows:
            version_id = int(vr["version_id"])

            crows = db.execute(
                """
                SELECT id, word, link, created_at, updated_at
                FROM child_words
                WHERE lang_word_version_id=?
                ORDER BY created_at ASC, id ASC
                """,
                (version_id,),
            ).fetchall()
            child_words = [{
                "id": int(cr["id"]),
                "word": cr["word"],
                "link": cr["link"] or "",
                "created_at": cr["created_at"],
                "updated_at": cr["updated_at"],
            } for cr in crows]

            grows = db.execute(
                """
                SELECT w.id, w.word
                FROM lang_word_children lwc
                JOIN lang_words w ON w.id = lwc.child_lang_word_id
                WHERE lwc.parent_lang_word_version_id=?
                ORDER BY w.word ASC, w.id ASC
                """,
                (version_id,),
            ).fetchall()
            child_lang_words = [{"lang_word_id": int(r["id"]), "word": r["word"]} for r in grows]

            versions_out.append({
                "version_id": version_id,
                "version": int(vr["version"]),
                "created_at": vr["created_at"],
                "child_words": child_words,
                "child_word_count": len(child_words),
                "child_lang_words": child_lang_words,
                "child_lang_word_count": len(child_lang_words),
            })
        return versions_out

    roots_out = []
    for rr in root_rows:
        lang_word_id = int(rr["id"])
        versions = load_versions(lang_word_id)
        latest = versions[0] if versions else None
        current_children = latest["child_lang_words"] if latest else []
        roots_out.append({
            "lang_word_id": lang_word_id,
            "word": rr["word"],
            "created_at": rr["created_at"],
            "versions": versions,
            "version_count": len(versions),
            "current_child_lang_words": current_children,
            "current_child_lang_word_count": len(current_children),
        })

    return jsonify(roots=roots_out)


# ---------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------

@app.get("/api/lang_sentences")
def list_lang_sentences():
    db = get_db()
    rows = db.execute(
        """
        SELECT id, lang_word_ids, sentence, created_at, updated_at
        FROM lang_sentences
        ORDER BY updated_at DESC, id DESC
        LIMIT 1000
        """
    ).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "lang_word_ids": _parse_int_list_json(r["lang_word_ids"]),
            "sentence": r["sentence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return jsonify(sentences=out)


@app.post("/api/lang_sentences")
def create_lang_sentence():
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    ids = data.get("lang_word_ids", [])
    sentence = (data.get("sentence") or "").strip()

    if not isinstance(ids, list):
        abort(400, description="lang_word_ids must be a list.")
    ids_int: list[int] = []
    for x in ids:
        try:
            ids_int.append(int(x))
        except Exception:
            pass

    if not sentence:
        abort(400, description="Missing 'sentence'.")
    if not ids_int:
        abort(400, description="lang_word_ids cannot be empty.")

    # Ensure all words exist
    for wid in ids_int:
        w = db.execute("SELECT id FROM lang_words WHERE id=?", (wid,)).fetchone()
        if w is None:
            abort(400, description=f"lang_word_id {wid} does not exist.")

    cur = db.execute(
        "INSERT INTO lang_sentences (lang_word_ids, sentence) VALUES (?, ?)",
        (json.dumps(ids_int), sentence),
    )
    db.commit()
    return jsonify(ok=True, id=int(cur.lastrowid)), 201


@app.get("/api/lang_sentences/<int:sentence_id>")
def get_lang_sentence(sentence_id: int):
    db = get_db()
    r = db.execute(
        "SELECT id, lang_word_ids, sentence, created_at, updated_at FROM lang_sentences WHERE id=?",
        (sentence_id,),
    ).fetchone()
    if r is None:
        abort(404, description="Sentence not found.")
    return jsonify({
        "id": int(r["id"]),
        "lang_word_ids": _parse_int_list_json(r["lang_word_ids"]),
        "sentence": r["sentence"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    })


# Optional child-sentence endpoints
@app.get("/api/lang_sentences/<int:sentence_id>/child_sentences")
def list_child_sentences(sentence_id: int):
    db = get_db()
    sr = db.execute("SELECT id FROM lang_sentences WHERE id=?", (sentence_id,)).fetchone()
    if sr is None:
        abort(404, description="Sentence not found.")

    rows = db.execute(
        """
        SELECT id, lang_sentence_id, child_word_ids, sentence, created_at, updated_at
        FROM child_sentences
        WHERE lang_sentence_id=?
        ORDER BY updated_at DESC, id DESC
        """,
        (sentence_id,),
    ).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "lang_sentence_id": int(r["lang_sentence_id"]),
            "child_word_ids": _parse_int_list_json(r["child_word_ids"]),
            "sentence": r["sentence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return jsonify(child_sentences=out)


@app.post("/api/lang_sentences/<int:sentence_id>/child_sentences")
def create_child_sentence(sentence_id: int):
    require_admin_key()
    db = get_db()

    sr = db.execute("SELECT id FROM lang_sentences WHERE id=?", (sentence_id,)).fetchone()
    if sr is None:
        abort(404, description="Sentence not found.")

    data = request.get_json(silent=True) or {}
    ids = data.get("child_word_ids", [])
    sentence = (data.get("sentence") or "").strip()

    if not isinstance(ids, list):
        abort(400, description="child_word_ids must be a list.")
    ids_int: list[int] = []
    for x in ids:
        try:
            ids_int.append(int(x))
        except Exception:
            pass

    if not sentence:
        abort(400, description="Missing 'sentence'.")
    if not ids_int:
        abort(400, description="child_word_ids cannot be empty.")

    # Ensure child_words exist
    for cid in ids_int:
        w = db.execute("SELECT id FROM child_words WHERE id=?", (cid,)).fetchone()
        if w is None:
            abort(400, description=f"child_word_id {cid} does not exist.")

    cur = db.execute(
        "INSERT INTO child_sentences (lang_sentence_id, child_word_ids, sentence) VALUES (?, ?, ?)",
        (sentence_id, json.dumps(ids_int), sentence),
    )
    db.commit()
    return jsonify(ok=True, id=int(cur.lastrowid)), 201


@app.get("/api/child_sentences/<int:child_sentence_id>")
def get_child_sentence(child_sentence_id: int):
    db = get_db()
    r = db.execute(
        """
        SELECT id, lang_sentence_id, child_word_ids, sentence, created_at, updated_at
        FROM child_sentences
        WHERE id=?
        """,
        (child_sentence_id,),
    ).fetchone()
    if r is None:
        abort(404, description="Child sentence not found.")
    return jsonify({
        "id": int(r["id"]),
        "lang_sentence_id": int(r["lang_sentence_id"]),
        "child_word_ids": _parse_int_list_json(r["child_word_ids"]),
        "sentence": r["sentence"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    })


# ---------------------------------------------------------------------
# LLM task queue + temporary writings
# ---------------------------------------------------------------------

@app.post("/api/write/create_lang_words")
def enqueue_create_lang_words():
    """
    Queue an LLM job to create child lang words ("themes") under a selected parent lang word.
    Body: { parent_lang_word_id: int, modifier: str? }
    """
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    parent_lang_word_id = data.get("parent_lang_word_id", None)
    modifier = (data.get("modifier") or "").strip()

    if parent_lang_word_id is None:
        abort(400, description="Missing parent_lang_word_id.")
    try:
        parent_lang_word_id = int(parent_lang_word_id)
    except Exception:
        abort(400, description="Invalid parent_lang_word_id.")

    pr = db.execute("SELECT id FROM lang_words WHERE id=?", (parent_lang_word_id,)).fetchone()
    if pr is None:
        abort(404, description="Parent lang word not found.")

    existing = db.execute(
        """
        SELECT id, status
        FROM llm_tasks
        WHERE parent_lang_word_id=? AND task_type=? AND status IN ('queued','running')
        ORDER BY id DESC
        LIMIT 1
        """,
        (parent_lang_word_id, "create_lang_words"),
    ).fetchone()

    if existing is not None:
        return jsonify(ok=True, task_id=int(existing["id"]), status=existing["status"], deduped=True), 202

    identifier = {"parent_lang_word_id": parent_lang_word_id}
    payload = {"modifier": modifier}

    cur = db.execute(
        """
        INSERT INTO llm_tasks (parent_lang_word_id, task_type, identifier, payload, status)
        VALUES (?, ?, ?, ?, 'queued')
        """,
        (parent_lang_word_id, "create_lang_words", json.dumps(identifier), json.dumps(payload)),
    )
    db.commit()
    return jsonify(ok=True, task_id=int(cur.lastrowid)), 202



@app.get("/api/tasks/<int:task_id>")
def api_task_status(task_id: int):
    db = get_db()
    t = db.execute(
        """
        SELECT id, parent_lang_word_id, task_type, identifier, payload, status, error, result_writing_id, created_at, updated_at
        FROM llm_tasks
        WHERE id=?
        """,
        (task_id,),
    ).fetchone()
    if t is None:
        raise NotFound(f"task {task_id} not found")
    return jsonify(
        {
            "id": int(t["id"]),
            "parent_lang_word_id": int(t["parent_lang_word_id"]),
            "task_type": t["task_type"],
            "identifier": t["identifier"],
            "payload": t["payload"],
            "status": t["status"],
            "error": t["error"],
            "result_writing_id": t["result_writing_id"],
            "created_at": t["created_at"],
            "updated_at": t["updated_at"],
        }
    )

@app.get("/api/write/tasks/<int:task_id>")
def get_write_task(task_id: int):
    db = get_db()
    r = db.execute(
        """
        SELECT id, parent_lang_word_id, task_type, identifier, payload, status, error, result_writing_id, created_at, updated_at
        FROM llm_tasks
        WHERE id=?
        """,
        (task_id,),
    ).fetchone()
    if r is None:
        abort(404, description="Task not found.")
    return jsonify({
        "id": int(r["id"]),
        "parent_lang_word_id": int(r["parent_lang_word_id"]),
        "task_type": r["task_type"],
        "identifier": _json_loads_safe(r["identifier"], {}),
        "payload": _json_loads_safe(r["payload"], {}),
        "status": r["status"],
        "error": r["error"] or "",
        "result_writing_id": int(r["result_writing_id"]) if r["result_writing_id"] is not None else None,
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    })


@app.get("/api/temporary_writings")
def list_temporary_writings():
    """
    Query: ?parent_lang_word_id=...
    """
    require_admin_key()
    db = get_db()

    parent_id = (request.args.get("parent_lang_word_id") or "").strip()
    if not parent_id:
        abort(400, description="Missing parent_lang_word_id.")
    try:
        parent_id = int(parent_id)
    except Exception:
        abort(400, description="Invalid parent_lang_word_id.")

    rows = db.execute(
        """
        SELECT id, parent_lang_word_id, identifier, prompt_type, text, model, modifier, task_id, created_at, updated_at
        FROM temporary_writings
        WHERE parent_lang_word_id=? AND prompt_type='create_lang_words'
        ORDER BY id DESC
        LIMIT 50
        """,
        (parent_id,),
    ).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "parent_lang_word_id": int(r["parent_lang_word_id"]),
            "identifier": _json_loads_safe(r["identifier"], {}),
            "prompt_type": r["prompt_type"],
            "text": _json_loads_safe(r["text"], {}),
            "model": r["model"] or "",
            "modifier": r["modifier"] or "",
            "task_id": int(r["task_id"]) if r["task_id"] is not None else None,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })

    return jsonify(writings=out)


@app.post("/api/temporary_writings/<int:writing_id>/apply_create_lang_words")
def apply_create_lang_words(writing_id: int):
    """
    Apply a temporary writing:
      - create a NEW version of the selected parent word and link ONLY the newly generated themes to that new parent version.
      - create child_words (orbiting phrases) under each created theme's version (v1).

    Expected writing JSON:
      { "themes": [ { "theme": "...", "orbiting_phrases": ["...", ...] }, ... ] }
    """
    require_admin_key()
    db = get_db()

    wr = db.execute(
        "SELECT id, parent_lang_word_id, prompt_type, text FROM temporary_writings WHERE id=?",
        (writing_id,),
    ).fetchone()
    if wr is None:
        abort(404, description="Temporary writing not found.")
    if wr["prompt_type"] != "create_lang_words":
        abort(400, description="Wrong prompt_type for this apply endpoint.")

    parent_lang_word_id = int(wr["parent_lang_word_id"])
    parent_version_id = create_next_version(db, parent_lang_word_id)

    try:
        payload = json.loads(wr["text"] or "{}")
    except Exception:
        abort(400, description="Temporary writing JSON is invalid.")

    themes = payload.get("themes", [])
    if not isinstance(themes, list):
        themes = []

    created: list[dict[str, Any]] = []

    for t in themes:
        if not isinstance(t, dict):
            continue
        theme_word = (t.get("theme") or "").strip()
        if not theme_word:
            continue

        # Create or reuse child lang word (lang_words.word is UNIQUE)
        existing = db.execute(
            "SELECT id FROM lang_words WHERE word=?",
            (theme_word,),
        ).fetchone()

        if existing is not None:
            child_lang_word_id = int(existing["id"])
        else:
            cur = db.execute("INSERT INTO lang_words (word) VALUES (?)", (theme_word,))
            child_lang_word_id = int(cur.lastrowid)

        # Create a NEW version for an existing theme word so orbiting phrases don't mutate old versions
        child_version_id = create_next_version(db, child_lang_word_id) if existing is not None else ensure_word_has_v1(db, child_lang_word_id)

        # Link parent version -> child lang word
        db.execute(
            """
            INSERT OR IGNORE INTO lang_word_children (parent_lang_word_version_id, child_lang_word_id)
            VALUES (?, ?)
            """,
            (parent_version_id, child_lang_word_id),
        )

        # Create orbiting phrases as child_words under the child theme's version
        orbit = t.get("orbiting_phrases", [])
        if isinstance(orbit, list):
            for phrase in orbit:
                phrase = (str(phrase) or "").strip()
                if not phrase:
                    continue
                db.execute(
                    "INSERT INTO child_words (lang_word_version_id, word, link) VALUES (?, ?, NULL)",
                    (child_version_id, phrase),
                )

        created.append({
            "child_lang_word_id": child_lang_word_id,
            "word": theme_word,
            "child_version_id": child_version_id,
        })

    db.execute("DELETE FROM temporary_writings WHERE id=?", (writing_id,))
    db.commit()

    return jsonify(ok=True, parent_lang_word_id=parent_lang_word_id, created=created)


# ---------------------------------------------------------------------
# In-process LLM worker (optional)
# ---------------------------------------------------------------------

def _ensure_worker_started() -> None:
    """
    Starts a single background worker thread per process to drain llm_tasks.
    """
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_llm_worker_loop, name="llm-worker", daemon=True)
        t.start()


def _llm_worker_loop() -> None:
    while True:
        try:
            _process_one_task()
            time.sleep(1.0)
        except Exception:
            traceback.print_exc()
            time.sleep(2.0)


def _process_one_task() -> None:
    """
    Claim one queued task and run it.
    """
    ensure_db_dir()
    db = sqlite3.connect(LANG_DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON;")
    db.execute("PRAGMA journal_mode = WAL;")
    ensure_schema(db)

    task = db.execute(
        """
        SELECT id, parent_lang_word_id, task_type, identifier, payload
        FROM llm_tasks
        WHERE status = 'queued'
        ORDER BY id ASC
        LIMIT 1
        """
    ).fetchone()

    if task is None:
        db.close()
        return

    task_id = int(task["id"])
    db.execute("UPDATE llm_tasks SET status='running' WHERE id=?", (task_id,))
    db.commit()

    try:
        task_type = task["task_type"]
        parent_id = int(task["parent_lang_word_id"])
        payload = _json_loads_safe(task["payload"], {})

        if task_type != "create_lang_words":
            raise ValueError(f"Unknown task_type: {task_type}")

        modifier = (payload.get("modifier") or "").strip() or None

        
        if task_type in STAR_STUFF_PROMPTS:
            ident = _json_loads_safe(task["identifier"], {})
            child_word_id = int(ident.get("child_word_id") or 0)
            if not child_word_id:
                raise ValueError("child_word_id missing in identifier for star task")

            ctx = _resolve_star_context(db, child_word_id)
            phrase = ctx["child_word"]["word"]
            context = ""
            if ctx["lang_name"]["word"]:
                context += str(ctx["lang_name"]["word"]).strip()
            if ctx["lang_word"]["word"]:
                if context:
                    context += " / "
                context += str(ctx["lang_word"]["word"]).strip()

            prompt = STAR_STUFF_PROMPTS[task_type].format(
                PHRASE=phrase,
                CONTEXT=context,
                MODIFIER=modifier or "none",
            )

            result_text, used_model = call_openai_text(prompt)

            db.execute(
                """
                INSERT INTO star_stuff (child_word_id, prompt_type, text, model, modifier, task_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (child_word_id, task_type, result_text, used_model, modifier or "", task_id),
            )
            db.execute("UPDATE llm_tasks SET status='done', error=NULL, updated_at=datetime('now') WHERE id=?", (task_id,))
            db.commit()
            return

        if task_type != "create_lang_words":
            raise ValueError(f"Unknown task_type: {task_type}")

pr = db.execute("SELECT word FROM lang_words WHERE id=?", (parent_id,)).fetchone()
        if pr is None:
            raise ValueError("Parent lang word not found.")
        primary_phrase = pr["word"]

        # Run LLM and store result
        plan = _run_create_lang_words_llm(primary_phrase=primary_phrase, optional_modifier=modifier)

        cur = db.execute(
            """
            INSERT INTO temporary_writings (parent_lang_word_id, identifier, prompt_type, text, model, modifier, task_id)
            VALUES (?, ?, 'create_lang_words', ?, ?, ?, ?)
            """,
            (
                parent_id,
                json.dumps({"parent_lang_word_id": parent_id}),
                json.dumps(plan),
                LLM_MODEL_CREATE_LANG_WORDS,
                modifier or "",
                task_id,
            ),
        )
        writing_id = int(cur.lastrowid)

        db.execute(
            """
            UPDATE llm_tasks
            SET status='done', result_writing_id=?
            WHERE id=?
            """,
            (writing_id, task_id),
        )
        db.commit()

    except Exception as e:
        db.execute(
            "UPDATE llm_tasks SET status='error', error=? WHERE id=?",
            (str(e), task_id),
        )
        db.commit()
    finally:
        db.close()


def _run_create_lang_words_llm(primary_phrase: str, optional_modifier: Optional[str]) -> dict[str, Any]:
    """
    Produce a plan:
      { "themes": [ { "theme": "...", "orbiting_phrases": ["...", ...] }, ... ] }

    Uses OpenAI Responses API if openai package is installed and OPENAI_API_KEY is set.
    If not available, raises an error.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed in this environment.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    modifier_txt = (optional_modifier or "").strip()
    system = (
        "You create concise hierarchical theme plans for a phrase.\n"
        "Return ONLY valid JSON with the schema:\n"
        "{ \"themes\": [ { \"theme\": string, \"orbiting_phrases\": [string, ...] }, ... ] }\n"
        "Do not include markdown or extra keys."
    )

    user = f"Primary phrase: {primary_phrase}\n"
    if modifier_txt:
        user += f"Modifier: {modifier_txt}\n"
    user += (
        "Generate 5-12 themes. Each theme should be a short phrase.\n"
        "For each theme, provide 3-12 orbiting_phrases (short phrases) that relate to that theme.\n"
        "Keep everything lower-case unless a proper noun.\n"
    )

    client = OpenAI()
    resp = client.responses.create(
        model=LLM_MODEL_CREATE_LANG_WORDS,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # Attempt to parse JSON from the output text
    text_out = ""
    try:
        # responses.create returns output text in resp.output_text in newer SDKs
        text_out = getattr(resp, "output_text", "") or ""
    except Exception:
        text_out = ""

    text_out = text_out.strip()
    if not text_out:
        # fallback: try to dig through raw structure
        try:
            text_out = json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception:
            raise RuntimeError("LLM returned no text output.")

    # If the model returned extra prose, attempt to extract the first JSON object.
    plan = None
    try:
        plan = json.loads(text_out)
    except Exception:
        # naive extraction
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            plan = json.loads(text_out[start:end+1])

    if not isinstance(plan, dict) or "themes" not in plan:
        raise RuntimeError("LLM output did not match expected JSON schema.")
    if not isinstance(plan.get("themes"), list):
        raise RuntimeError("LLM output 'themes' must be a list.")

    return plan


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ensure_db_dir()
    # Ensure schema on startup
    conn = sqlite3.connect(LANG_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    ensure_schema(conn)
    conn.close()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
