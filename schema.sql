PRAGMA foreign_keys = ON;

-- lang stays the same
CREATE TABLE IF NOT EXISTS lang (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TRIGGER IF NOT EXISTS trg_lang_updated_at
AFTER UPDATE ON lang
FOR EACH ROW
BEGIN
  UPDATE lang SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- star_stuff becomes a TREE node:
-- - lang_id: which lang entry this star tree belongs to
-- - parent_star_id: NULL for root stars, otherwise points to another star node
-- - type: label/category for the node
CREATE TABLE IF NOT EXISTS star_stuff (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lang_id INTEGER NOT NULL,
  parent_star_id INTEGER,          -- nullable = root star
  type TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (lang_id) REFERENCES lang(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_star_id) REFERENCES star_stuff(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_star_lang_id ON star_stuff(lang_id);
CREATE INDEX IF NOT EXISTS idx_star_parent_id ON star_stuff(parent_star_id);
CREATE INDEX IF NOT EXISTS idx_star_type ON star_stuff(type);

CREATE TRIGGER IF NOT EXISTS trg_star_stuff_updated_at
AFTER UPDATE ON star_stuff
FOR EACH ROW
BEGIN
  UPDATE star_stuff SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- star_stuff_texts holds the SEQUENCE/COLLECTION of texts per star node
CREATE TABLE IF NOT EXISTS star_stuff_texts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  star_id INTEGER NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (star_id) REFERENCES star_stuff(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sst_star_id ON star_stuff_texts(star_id);
CREATE INDEX IF NOT EXISTS idx_sst_created_at ON star_stuff_texts(created_at);

CREATE TRIGGER IF NOT EXISTS trg_star_stuff_texts_updated_at
AFTER UPDATE ON star_stuff_texts
FOR EACH ROW
BEGIN
  UPDATE star_stuff_texts SET updated_at = datetime('now') WHERE id = NEW.id;
END;
