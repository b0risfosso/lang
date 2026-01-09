PRAGMA foreign_keys = ON;

BEGIN;

-- =========================================================
-- Core word graph
-- =========================================================

CREATE TABLE lang_words (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  word TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE lang_word_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lang_word_id INTEGER NOT NULL,
  version INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(lang_word_id, version),
  FOREIGN KEY (lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
);

CREATE INDEX idx_versions_lang_word_id
  ON lang_word_versions(lang_word_id);

-- Child words (orbiting phrases) under a specific version
CREATE TABLE child_words (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lang_word_version_id INTEGER NOT NULL,
  word TEXT NOT NULL,
  link TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (lang_word_version_id) REFERENCES lang_word_versions(id) ON DELETE CASCADE
);

CREATE INDEX idx_child_words_version_id
  ON child_words(lang_word_version_id);

CREATE INDEX idx_child_words_link
  ON child_words(link);

-- Child lang words: edges from a *parent version* -> child lang_word_id
CREATE TABLE lang_word_children (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_lang_word_version_id INTEGER NOT NULL,
  child_lang_word_id INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(parent_lang_word_version_id, child_lang_word_id),
  FOREIGN KEY (parent_lang_word_version_id) REFERENCES lang_word_versions(id) ON DELETE CASCADE,
  FOREIGN KEY (child_lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
);

CREATE INDEX idx_lwc_parent_version
  ON lang_word_children(parent_lang_word_version_id);

CREATE INDEX idx_lwc_child_word
  ON lang_word_children(child_lang_word_id);

-- =========================================================
-- Sentences (your new lang.html will display these)
-- =========================================================

CREATE TABLE lang_sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lang_word_ids TEXT NOT NULL,   -- JSON array of ints (lang_words.id)
  sentence TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_lang_sentences_updated
  ON lang_sentences(updated_at);

-- OPTIONAL: keep child_sentences if you still want this capability
-- (You deleted child_sentence.html, but the backend can still support it.
--  If you truly want it gone, remove this whole block.)
CREATE TABLE child_sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lang_sentence_id INTEGER NOT NULL,
  child_word_ids TEXT NOT NULL,     -- JSON array of ints (child_words.id)
  sentence TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (lang_sentence_id) REFERENCES lang_sentences(id) ON DELETE CASCADE
);

CREATE INDEX idx_child_sentences_lang_sentence_id
  ON child_sentences(lang_sentence_id);

CREATE INDEX idx_child_sentences_updated
  ON child_sentences(updated_at);

-- =========================================================
-- LLM pipeline (now keyed by parent_lang_word_id)
-- =========================================================

CREATE TABLE llm_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_lang_word_id INTEGER NOT NULL,
  task_type TEXT NOT NULL,         -- e.g. "create_lang_words"
  identifier TEXT NOT NULL,        -- JSON {"parent_lang_word_id": ...}
  payload TEXT NOT NULL,           -- JSON {modifier, ...}
  status TEXT NOT NULL DEFAULT 'queued', -- queued|running|done|error
  error TEXT,
  result_writing_id INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (parent_lang_word_id) REFERENCES lang_words(id) ON DELETE CASCADE
);

CREATE INDEX idx_llm_tasks_status
  ON llm_tasks(status);

CREATE INDEX idx_llm_tasks_parent_id
  ON llm_tasks(parent_lang_word_id);

CREATE INDEX idx_llm_tasks_parent_status
  ON llm_tasks(parent_lang_word_id, status);

CREATE INDEX idx_llm_tasks_created_at
  ON llm_tasks(created_at);

CREATE TABLE temporary_writings (
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

CREATE INDEX idx_temp_parent_id
  ON temporary_writings(parent_lang_word_id);

CREATE INDEX idx_temp_parent_created
  ON temporary_writings(parent_lang_word_id, created_at);

CREATE INDEX idx_temp_prompt_type
  ON temporary_writings(prompt_type);

CREATE INDEX idx_temp_created_at
  ON temporary_writings(created_at);

-- =========================================================
-- updated_at triggers (so updated_at changes automatically)
-- =========================================================

CREATE TRIGGER trg_child_words_updated_at
AFTER UPDATE ON child_words
FOR EACH ROW
BEGIN
  UPDATE child_words SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trg_lang_sentences_updated_at
AFTER UPDATE ON lang_sentences
FOR EACH ROW
BEGIN
  UPDATE lang_sentences SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trg_child_sentences_updated_at
AFTER UPDATE ON child_sentences
FOR EACH ROW
BEGIN
  UPDATE child_sentences SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trg_llm_tasks_updated_at
AFTER UPDATE ON llm_tasks
FOR EACH ROW
BEGIN
  UPDATE llm_tasks SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trg_temporary_writings_updated_at
AFTER UPDATE ON temporary_writings
FOR EACH ROW
BEGIN
  UPDATE temporary_writings SET updated_at = datetime('now') WHERE id = NEW.id;
END;

COMMIT;
