#!/usr/bin/env python3
import os
import sqlite3
import json
from flask import Flask, g, jsonify, request, abort
from werkzeug.exceptions import HTTPException
import threading
import time
import traceback
from typing import Optional, List
from datetime import datetime, timezone

from openai import OpenAI
from pydantic import BaseModel

DB_DIR = "/var/www/site/data"
DB_PATH = os.path.join(DB_DIR, "lang.db")

USAGE_DB_PATH = os.path.join(DB_DIR, "llm_usage.db")
USAGE_APP_NAME = "lang"  # schema comment says 'jid'|'crayon', but not enforced; use a stable tag

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

    # Langs (collections of lang_words)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS langs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE,
          lang_word_ids TEXT NOT NULL,  -- JSON array of ints
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_langs_name
          ON langs(name);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_langs_updated
          ON langs(updated_at);
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

    # Temporary writings (LLM outputs)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS temporary_writings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_id INTEGER NOT NULL,
          identifier TEXT NOT NULL,        -- JSON (e.g., {"lang_id": 1})
          prompt_type TEXT NOT NULL,       -- e.g. "create_lang_words"
          text TEXT NOT NULL,              -- JSON output from LLM
          model TEXT,
          modifier TEXT,
          task_id INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_temp_writings_lang_id
          ON temporary_writings(lang_id);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_temp_writings_lang_id_created
          ON temporary_writings(lang_id, created_at);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_temp_writings_prompt_type
          ON temporary_writings(prompt_type);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_temp_writings_created_at
          ON temporary_writings(created_at);
        """
    )

    # LLM task queue
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lang_id INTEGER NOT NULL,
          task_type TEXT NOT NULL,         -- "create_lang_words"
          identifier TEXT NOT NULL,        -- JSON {"lang_id": ...}
          payload TEXT NOT NULL,           -- JSON {modifier, ...}
          status TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|error
          error TEXT,
          result_writing_id INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_llm_tasks_lang_id
          ON llm_tasks(lang_id);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_llm_tasks_lang_status
          ON llm_tasks(lang_id, status);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_llm_tasks_status
          ON llm_tasks(status);
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_llm_tasks_created_at
          ON llm_tasks(created_at);
        """
    )


    # Lightweight migration if an older DB created child_words without 'link'
    # (Your production DB currently does not have it.)
    if not _table_has_column(db, "child_words", "link"):
        db.execute("ALTER TABLE child_words ADD COLUMN link TEXT;")

    db.commit()

def init_usage_db():
    """
    Initialize the separate usage database for LLM metrics.
    """
    dbu = _connect_usage_db()
    try:
        # Events table
        dbu.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              app TEXT NOT NULL,
              model TEXT NOT NULL,
              endpoint TEXT NOT NULL,
              email TEXT,
              request_id TEXT,
              tokens_in INTEGER NOT NULL DEFAULT 0,
              tokens_out INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              duration_ms INTEGER NOT NULL DEFAULT 0,
              cost_usd REAL NOT NULL DEFAULT 0.0,
              meta TEXT  -- JSON
            );
            """
        )
        dbu.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_ts
              ON usage_events(ts);
            """
        )
        dbu.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_model
              ON usage_events(model);
            """
        )
        dbu.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_events_app
              ON usage_events(app);
            """
        )

        # Totals all time
        dbu.execute(
            """
            CREATE TABLE IF NOT EXISTS totals_all_time (
              id INTEGER PRIMARY KEY CHECK (id = 1),
              tokens_in INTEGER NOT NULL DEFAULT 0,
              tokens_out INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              calls INTEGER NOT NULL DEFAULT 0,
              last_ts TEXT NOT NULL
            );
            """
        )

        # Totals by model
        dbu.execute(
            """
            CREATE TABLE IF NOT EXISTS totals_by_model (
              model TEXT PRIMARY KEY,
              tokens_in INTEGER NOT NULL DEFAULT 0,
              tokens_out INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              calls INTEGER NOT NULL DEFAULT 0,
              first_ts TEXT NOT NULL,
              last_ts TEXT NOT NULL
            );
            """
        )

        # Totals daily
        dbu.execute(
            """
            CREATE TABLE IF NOT EXISTS totals_daily (
              day TEXT NOT NULL,
              model TEXT NOT NULL,
              tokens_in INTEGER NOT NULL DEFAULT 0,
              tokens_out INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              calls INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (day, model)
            );
            """
        )

        dbu.commit()
    finally:
        dbu.close()

@app.before_request
def _init_once():
    init_db()

@app.before_first_request
def _boot():
    init_db()
    init_usage_db()
    _start_llm_worker_once()

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


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _connect_usage_db():
    # Separate DB from lang.db
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _extract_token_usage(resp) -> tuple[int, int, int]:
    """
    Best-effort extraction for OpenAI Responses API usage.
    Returns (tokens_in, tokens_out, total_tokens).
    """
    tokens_in = tokens_out = total = 0

    # OpenAI SDK commonly exposes .usage as an object or dict with input/output/total tokens
    usage = getattr(resp, "usage", None)
    if usage:
        # dict-like
        if isinstance(usage, dict):
            tokens_in = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or (tokens_in + tokens_out) or 0)
            return tokens_in, tokens_out, total

        # object-like
        tokens_in = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (tokens_in + tokens_out) or 0)
        return tokens_in, tokens_out, total

    # fallback: sometimes nested
    for attr in ("input_tokens", "prompt_tokens"):
        v = getattr(resp, attr, None)
        if v is not None:
            tokens_in = int(v)
            break
    for attr in ("output_tokens", "completion_tokens"):
        v = getattr(resp, attr, None)
        if v is not None:
            tokens_out = int(v)
            break
    total = tokens_in + tokens_out
    return tokens_in, tokens_out, total

def log_llm_usage_event(
    *,
    model: str,
    endpoint: str,
    request_id: str | None,
    tokens_in: int,
    tokens_out: int,
    total_tokens: int,
    duration_ms: int,
    meta: dict | None = None,
    email: str | None = None,
):
    """
    Writes to llm_usage.db usage_events + updates totals tables.
    """
    ts = _utc_iso_now()
    day = _utc_day()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)

    dbu = _connect_usage_db()
    try:
        # 1) usage_events
        dbu.execute(
            """
            INSERT INTO usage_events
              (ts, app, model, endpoint, email, request_id, tokens_in, tokens_out, total_tokens, duration_ms, cost_usd, meta)
            VALUES
              (?,  ?,   ?,     ?,        ?,     ?,          ?,         ?,          ?,            ?,           0.0,     ?)
            """,
            (ts, USAGE_APP_NAME, model, endpoint, email, request_id or "", int(tokens_in), int(tokens_out), int(total_tokens), int(duration_ms), meta_json),
        )

        # 2) totals_all_time (single row id=1)
        dbu.execute(
            """
            INSERT INTO totals_all_time (id, tokens_in, tokens_out, total_tokens, calls, last_ts)
            VALUES (1, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
              tokens_in = tokens_in + excluded.tokens_in,
              tokens_out = tokens_out + excluded.tokens_out,
              total_tokens = total_tokens + excluded.total_tokens,
              calls = calls + 1,
              last_ts = excluded.last_ts
            """,
            (int(tokens_in), int(tokens_out), int(total_tokens), ts),
        )

        # 3) totals_by_model
        dbu.execute(
            """
            INSERT INTO totals_by_model (model, tokens_in, tokens_out, total_tokens, calls, first_ts, last_ts)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
              tokens_in = tokens_in + excluded.tokens_in,
              tokens_out = tokens_out + excluded.tokens_out,
              total_tokens = total_tokens + excluded.total_tokens,
              calls = calls + 1,
              last_ts = excluded.last_ts
            """,
            (model, int(tokens_in), int(tokens_out), int(total_tokens), ts, ts),
        )

        # 4) totals_daily
        dbu.execute(
            """
            INSERT INTO totals_daily (day, model, tokens_in, tokens_out, total_tokens, calls)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(day, model) DO UPDATE SET
              tokens_in = tokens_in + excluded.tokens_in,
              tokens_out = tokens_out + excluded.tokens_out,
              total_tokens = total_tokens + excluded.total_tokens,
              calls = calls + 1
            """,
            (day, model, int(tokens_in), int(tokens_out), int(total_tokens)),
        )

        dbu.commit()
    finally:
        dbu.close()


class OrbitTheme(BaseModel):
    name: str
    description: str
    orbiting_phrases: List[str]
    orbiting_ideas: Optional[List[str]] = None

class LangWordPlan(BaseModel):
    primary_phrase: str
    optional_modifier: Optional[str] = None
    themes: List[OrbitTheme]


PROMPT_CREATE_LANG_WORDS = """You will be given:
a Primary Phrase (a system, entity, idea, institution, scale, or pattern), and
an Optional Modifier that may add context, constraints, or a preferred lens.
The phrase may range from highly abstract (e.g., physical limits) to large-scale systems (e.g., economies) to concrete institutions (e.g., government platforms).
Your job is to:
Infer what kind of thing the primary phrase represents.
Incorporate the modifier if provided to guide emphasis, scope, or perspective.
Generate 4–8 distinct themes that meaningfully describe how the thing works, is structured, or is experienced.
For each theme, build a set of orbiting words/phrases that cluster naturally around it.
Inputs
Primary Phrase: [REQUIRED]
Optional Modifier: [OPTIONAL — may specify perspective, domain, audience, timeframe, or concern]
Examples of modifiers:
“from a policy perspective”
“focusing on failure modes”
“emphasizing technical constraints”
“for first-time users”
“historical evolution”
“economic and incentive structures”
“ethical and societal implications”
How to Use the Modifier
If a modifier is provided, prioritize themes and orbiting phrases that align with it.
Do not force the modifier into every theme—apply it where it naturally fits.
If no modifier is given, default to a balanced, systems-level analysis.
Output Structure
Use the following format:
Theme N: [Short, precise theme name]
One-sentence description of what this theme captures.
word / phrase
word / phrase
word / phrase
word / phrase
word / phrase
(Optionally include 1–2 “orbiting ideas” if they clarify deeper structure or consequences.)
Guidelines
Do not simply restate the primary phrase.
Keep themes conceptually distinct and non-redundant.
Orbiting phrases may include:
components or structures
processes or dynamics
stakeholders or agents
constraints, limits, or bottlenecks
emergent behaviors or outcomes
Match abstraction level to the input:
scientific → theory, limits, models
economic → scale, flows, incentives, risk
institutional → governance, procedures, users, friction
Use clear, neutral, descriptive language.
Now analyze the following inputs and generate the themes and orbiting phrase sets:
Primary Phrase: {primary_phrase}
Optional Modifier: {optional_modifier}
"""

LLM_MODEL_CREATE_LANG_WORDS = "gpt-5-mini-2025-08-07"

def _run_create_lang_words_llm(primary_phrase: str, optional_modifier: str | None) -> LangWordPlan:
    start_time = time.time()
    client = OpenAI()
    prompt = PROMPT_CREATE_LANG_WORDS.format(
        primary_phrase=primary_phrase,
        optional_modifier=(optional_modifier or "").strip()
    )

    # Use structured parsing via Responses API
    resp = client.responses.parse(
        model=LLM_MODEL_CREATE_LANG_WORDS,
        input=[
            {"role": "system", "content": "Return a structured JSON plan following the provided schema."},
            {"role": "user", "content": prompt},
        ],
        text_format=LangWordPlan,
    )

    # Log usage
    duration_ms = int((time.time() - start_time) * 1000)
    tokens_in, tokens_out, total_tokens = _extract_token_usage(resp)
    log_llm_usage_event(
        model=LLM_MODEL_CREATE_LANG_WORDS,
        endpoint="responses.parse",
        request_id=getattr(resp, "id", None),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        meta={"primary_phrase": primary_phrase, "optional_modifier": optional_modifier},
    )

    return resp.output_parsed


_worker_started = False
_worker_lock = threading.Lock()

def _now_dt_sql():
    return "datetime('now')"

def _start_llm_worker_once():
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True

    t = threading.Thread(target=_llm_worker_loop, name="llm-worker", daemon=True)
    t.start()

def _llm_worker_loop():
    # NOTE: uses new sqlite connections (not Flask g)
    while True:
        try:
            _process_one_task()
        except Exception:
            # Never crash the worker loop
            traceback.print_exc()
        time.sleep(1.0)

def _process_one_task():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Claim one queued task
    task = db.execute(
        """
        SELECT id, lang_id, task_type, identifier, payload
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

    # Mark running
    db.execute(
        "UPDATE llm_tasks SET status='running', updated_at=datetime('now') WHERE id=?",
        (task_id,)
    )
    db.commit()

    try:
        task_type = task["task_type"]
        lang_id = int(task["lang_id"])
        identifier = json.loads(task["identifier"])
        payload = json.loads(task["payload"])

        if task_type != "create_lang_words":
            raise ValueError(f"Unknown task_type: {task_type}")
        modifier = (payload.get("modifier") or "").strip() or None

        # Fetch lang name for Primary Phrase
        lr = db.execute("SELECT id, name FROM langs WHERE id=?", (lang_id,)).fetchone()
        if lr is None:
            raise ValueError(f"Lang not found: {lang_id}")
        primary_phrase = lr["name"]

        plan = _run_create_lang_words_llm(primary_phrase, modifier)

        # Store in temporary_writings
        writing_id = db.execute(
            """
            INSERT INTO temporary_writings (lang_id, identifier, prompt_type, text, model, modifier, task_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                lang_id,
                json.dumps({"lang_id": lang_id}),
                "create_lang_words",
                json.dumps(plan.model_dump(), ensure_ascii=False),
                LLM_MODEL_CREATE_LANG_WORDS,
                modifier or "",
                task_id,
            )
        ).lastrowid

        db.execute(
            """
            UPDATE llm_tasks
            SET status='done', result_writing_id=?, updated_at=datetime('now')
            WHERE id=?
            """,
            (int(writing_id), task_id)
        )
        db.commit()

    except Exception as e:
        db.execute(
            """
            UPDATE llm_tasks
            SET status='error', error=?, updated_at=datetime('now')
            WHERE id=?
            """,
            (str(e), task_id)
        )
        db.commit()
    finally:
        db.close()


# =========================
# Langs (new)
# =========================

def _lang_row_or_404(db, lang_id: int):
    r = db.execute(
        """
        SELECT id, name, lang_word_ids, created_at, updated_at
        FROM langs
        WHERE id = ?
        """,
        (lang_id,),
    ).fetchone()
    if r is None:
        abort(404, description="Lang not found.")
    return r


def _resolve_latest_words_for_ids(db, ids):
    """
    For a list of lang_word_id ints, return [{lang_word_id, word, version_id, version}]
    in the same order, skipping ids that no longer exist or have no versions.
    """
    out = []
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
            (wid,),
        ).fetchone()
        if vrow is None:
            continue
        out.append({
            "lang_word_id": int(wrow["id"]),
            "word": wrow["word"],
            "version_id": int(vrow["version_id"]),
            "version": int(vrow["version"]),
        })
    return out


@app.get("/api/langs")
def list_langs():
    db = get_db()
    rows = db.execute(
        """
        SELECT id, name, lang_word_ids, created_at, updated_at
        FROM langs
        ORDER BY name ASC, id ASC
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
        out.append({
            "id": int(r["id"]),
            "name": r["name"],
            "lang_word_ids": ids,
            "count": len(ids),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })

    return jsonify(langs=out)


@app.get("/api/write")
def write_tree():
    """
    Returns a nested structure:
    langs -> lang_words -> versions -> child_words
    With counts at each level for display in write.html
    """
    db = get_db()

    langs_rows = db.execute(
        """
        SELECT id, name, lang_word_ids, created_at, updated_at
        FROM langs
        ORDER BY name ASC, id ASC
        """
    ).fetchall()

    langs_out = []

    for lr in langs_rows:
        # parse lang_word_ids
        try:
            ids = json.loads(lr["lang_word_ids"])
            if not isinstance(ids, list):
                ids = []
            ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
        except Exception:
            ids = []

        # keep order as stored in lang_word_ids
        words_out = []
        total_child_words = 0

        for wid in ids:
            wrow = db.execute("SELECT id, word FROM lang_words WHERE id = ?", (wid,)).fetchone()
            if wrow is None:
                continue

            vrows = db.execute(
                """
                SELECT id AS version_id, version
                FROM lang_word_versions
                WHERE lang_word_id = ?
                ORDER BY version DESC
                """,
                (wid,)
            ).fetchall()

            versions_out = []
            child_count_for_word = 0

            for vr in vrows:
                crows = db.execute(
                    """
                    SELECT id, word, link, created_at, updated_at
                    FROM child_words
                    WHERE lang_word_version_id = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (vr["version_id"],)
                ).fetchall()

                children = [{
                    "id": int(cr["id"]),
                    "word": cr["word"],
                    "link": cr["link"] or "",
                    "created_at": cr["created_at"],
                    "updated_at": cr["updated_at"],
                } for cr in crows]

                child_count_for_word += len(children)

                versions_out.append({
                    "version_id": int(vr["version_id"]),
                    "version": int(vr["version"]),
                    "child_words": children,
                    "child_word_count": len(children),
                })

            total_child_words += child_count_for_word

            words_out.append({
                "lang_word_id": int(wrow["id"]),
                "word": wrow["word"],
                "version_count": len(versions_out),
                "child_word_count": child_count_for_word,
                "versions": versions_out,
            })

        langs_out.append({
            "id": int(lr["id"]),
            "name": lr["name"],
            "lang_word_count": len(words_out),
            "child_word_count": total_child_words,
            "lang_words": words_out,
            "created_at": lr["created_at"],
            "updated_at": lr["updated_at"],
        })

    return jsonify(langs=langs_out)


@app.post("/api/langs")
def create_lang():
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    ids = _parse_ids(data.get("lang_word_ids"))  # optional; can be []

    if not name:
        abort(400, description="Missing 'name'.")
    if len(name) > 200:
        abort(400, description="Name too long (max 200 chars).")

    # Validate word ids exist
    if ids:
        qmarks = ",".join("?" for _ in ids)
        found = db.execute(f"SELECT id FROM lang_words WHERE id IN ({qmarks})", ids).fetchall()
        found_ids = {int(r["id"]) for r in found}
        missing = [i for i in ids if i not in found_ids]
        if missing:
            abort(400, description=f"Unknown lang_word_id(s): {', '.join(map(str, missing))}")

    try:
        cur = db.execute(
            "INSERT INTO langs (name, lang_word_ids) VALUES (?, ?)",
            (name, json.dumps(ids)),
        )
        db.commit()
    except sqlite3.IntegrityError:
        abort(409, description="Lang name already exists.")

    return jsonify(ok=True, id=cur.lastrowid, name=name, lang_word_ids=ids), 201


@app.get("/api/langs/<int:lang_id>")
def get_lang(lang_id: int):
    db = get_db()
    r = _lang_row_or_404(db, lang_id)

    try:
        ids = json.loads(r["lang_word_ids"])
        if not isinstance(ids, list):
            ids = []
        ids = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
    except Exception:
        ids = []

    words = _resolve_latest_words_for_ids(db, ids)

    return jsonify({
        "id": int(r["id"]),
        "name": r["name"],
        "lang_word_ids": ids,
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "words": words,  # [{lang_word_id, word, version_id, version}]
    })


@app.put("/api/langs/<int:lang_id>")
def update_lang(lang_id: int):
    require_admin_key()
    db = get_db()
    _lang_row_or_404(db, lang_id)

    data = request.get_json(silent=True) or {}
    name = data.get("name", None)
    ids_val = data.get("lang_word_ids", None)

    updates = []
    params = []

    if name is not None:
        name = str(name).strip()
        if not name:
            abort(400, description="Missing 'name'.")
        if len(name) > 200:
            abort(400, description="Name too long (max 200 chars).")
        updates.append("name = ?")
        params.append(name)

    if ids_val is not None:
        ids = _parse_ids(ids_val)

        if ids:
            qmarks = ",".join("?" for _ in ids)
            found = db.execute(f"SELECT id FROM lang_words WHERE id IN ({qmarks})", ids).fetchall()
            found_ids = {int(rr["id"]) for rr in found}
            missing = [i for i in ids if i not in found_ids]
            if missing:
                abort(400, description=f"Unknown lang_word_id(s): {', '.join(map(str, missing))}")

        updates.append("lang_word_ids = ?")
        params.append(json.dumps(ids))

    if not updates:
        abort(400, description="Nothing to update.")

    updates.append("updated_at = datetime('now')")
    sql = f"UPDATE langs SET {', '.join(updates)} WHERE id = ?"
    params.append(lang_id)

    try:
        db.execute(sql, params)
        db.commit()
    except sqlite3.IntegrityError:
        abort(409, description="Lang name already exists.")

    r2 = _lang_row_or_404(db, lang_id)
    try:
        out_ids = json.loads(r2["lang_word_ids"])
        if not isinstance(out_ids, list):
            out_ids = []
        out_ids = [int(x) for x in out_ids if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
    except Exception:
        out_ids = []

    return jsonify(ok=True, id=int(r2["id"]), name=r2["name"], lang_word_ids=out_ids)


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

@app.delete("/api/lang_words/<int:lang_word_id>")
def delete_lang_word(lang_word_id: int):
    require_admin_key()
    db = get_db()

    row = db.execute("SELECT id, word FROM lang_words WHERE id = ?", (lang_word_id,)).fetchone()
    if row is None:
        abort(404, description="Lang word not found.")

    # ON DELETE CASCADE removes lang_word_versions and child_words
    db.execute("DELETE FROM lang_words WHERE id = ?", (lang_word_id,))
    db.commit()
    return jsonify(ok=True, lang_word_id=lang_word_id, word=row["word"])


@app.delete("/api/lang_word_versions/<int:version_id>")
def delete_lang_word_version(version_id: int):
    require_admin_key()
    db = get_db()

    v = db.execute(
        "SELECT id, lang_word_id, version FROM lang_word_versions WHERE id = ?",
        (version_id,),
    ).fetchone()
    if v is None:
        abort(404, description="Version not found.")

    cnt = db.execute(
        "SELECT COUNT(*) AS c FROM lang_word_versions WHERE lang_word_id = ?",
        (v["lang_word_id"],),
    ).fetchone()

    # Prevent orphaning the parent word with zero versions
    if int(cnt["c"]) <= 1:
        abort(400, description="Cannot delete the only version. Delete the lang word instead.")

    # ON DELETE CASCADE removes child_words for this version
    db.execute("DELETE FROM lang_word_versions WHERE id = ?", (version_id,))
    db.commit()
    return jsonify(ok=True, version_id=version_id, lang_word_id=v["lang_word_id"], version=v["version"])



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


@app.post("/api/write/create_lang_words")
def enqueue_create_lang_words():
    require_admin_key()
    db = get_db()

    data = request.get_json(silent=True) or {}
    lang_id = data.get("lang_id", None)
    modifier = (data.get("modifier") or "").strip()

    if lang_id is None:
        abort(400, description="Missing lang_id.")
    try:
        lang_id = int(lang_id)
    except Exception:
        abort(400, description="Invalid lang_id.")

    # Ensure lang exists
    lr = db.execute("SELECT id FROM langs WHERE id=?", (lang_id,)).fetchone()
    if lr is None:
        abort(404, description="Lang not found.")

    # Check for existing queued/running task for this lang
    existing = db.execute(
        """
        SELECT id, status
        FROM llm_tasks
        WHERE lang_id=? AND task_type=? AND status IN ('queued','running')
        ORDER BY id DESC
        LIMIT 1
        """,
        (lang_id, "create_lang_words"),
    ).fetchone()

    if existing is not None:
        return jsonify(ok=True, task_id=int(existing["id"]), status=existing["status"], deduped=True), 202

    identifier = {"lang_id": lang_id}
    payload = {"modifier": modifier}

    cur = db.execute(
        """
        INSERT INTO llm_tasks (lang_id, task_type, identifier, payload, status, updated_at)
        VALUES (?, ?, ?, ?, 'queued', datetime('now'))
        """,
        (lang_id, "create_lang_words", json.dumps(identifier), json.dumps(payload)),
    )
    db.commit()

    return jsonify(ok=True, task_id=cur.lastrowid), 202


@app.get("/api/write/tasks/<int:task_id>")
def get_task(task_id: int):
    require_admin_key()
    db = get_db()
    r = db.execute(
        """
        SELECT id, lang_id, task_type, identifier, payload, status, error, result_writing_id, created_at, updated_at
        FROM llm_tasks
        WHERE id=?
        """,
        (task_id,),
    ).fetchone()
    if r is None:
        abort(404, description="Task not found.")

    return jsonify({
        "id": int(r["id"]),
        "lang_id": int(r["lang_id"]) if r["lang_id"] is not None else None,
        "task_type": r["task_type"],
        "identifier": json.loads(r["identifier"]),
        "payload": json.loads(r["payload"]),
        "status": r["status"],
        "error": r["error"] or "",
        "result_writing_id": r["result_writing_id"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    })


@app.get("/api/temporary_writings")
def list_temporary_writings():
    require_admin_key()
    db = get_db()

    lang_id = request.args.get("lang_id", "").strip()
    if not lang_id:
        abort(400, description="Missing lang_id.")
    try:
        lang_id_int = int(lang_id)
    except Exception:
        abort(400, description="Invalid lang_id.")

    ident = json.dumps({"lang_id": lang_id_int})

    rows = db.execute(
        """
        SELECT id, lang_id, identifier, prompt_type, text, model, modifier, task_id, created_at, updated_at
        FROM temporary_writings
        WHERE lang_id = ? AND prompt_type = 'create_lang_words'
        ORDER BY id DESC
        LIMIT 50
        """,
        (lang_id_int,),
    ).fetchall()

    out = []
    for r in rows:
        try:
            text_json = json.loads(r["text"])
        except Exception:
            text_json = r["text"]
        out.append({
            "id": int(r["id"]),
            "lang_id": int(r["lang_id"]) if r["lang_id"] is not None else None,
            "identifier": json.loads(r["identifier"]),
            "prompt_type": r["prompt_type"],
            "text": text_json,
            "model": r["model"] or "",
            "modifier": r["modifier"] or "",
            "task_id": r["task_id"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })

    return jsonify(writings=out)



if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
