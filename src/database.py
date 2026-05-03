import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "/data/memory.db")

def get_db():
    """Get a database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn

def init_db():
    conn = get_db()

    # Step 1: Create base tables first
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT,
            messages TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source_session TEXT,
            source_turn TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            supersedes TEXT,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(user_id, active);
        CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_id);
    """)

    # Step 2: Create FTS5 table SEPARATELY (must happen after base tables exist)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED,
            user_id UNINDEXED,
            content,
            tokenize = 'porter unicode61'
        )
    """)
    conn.commit()

    # Step 3: Additive column migrations
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}

    additions = [
        ("subject",          "TEXT DEFAULT 'user'"),
        ("entities",         "TEXT DEFAULT '[]'"),
        ("temporal",         "TEXT DEFAULT 'current'"),
        ("is_correction_of", "TEXT"),
    ]
    for col, decl in additions:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
            print(f"🔧 Migrated: added column '{col}'")

    # Step 4: Indexes for new columns
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_memories_key
            ON memories(user_id, key, active);
        CREATE INDEX IF NOT EXISTS idx_memories_subject
            ON memories(user_id, subject, active);
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized")


def index_memory_for_fts(conn, memory_id: str, user_id: str, value: str, entities: list):
    """Add or replace a memory in the FTS index."""
    content = f"{value} {' '.join(entities)}"
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.execute(
        "INSERT INTO memories_fts (memory_id, user_id, content) VALUES (?, ?, ?)",
        (memory_id, user_id, content),
    )
def index_memory_for_fts(conn, memory_id: str, user_id: str, value: str, entities: list):
    """Add or replace a memory in the FTS index."""
    # Combine value + entities so keyword search can match either
    content = f"{value} {' '.join(entities)}"
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.execute(
        "INSERT INTO memories_fts (memory_id, user_id, content) VALUES (?, ?, ?)",
        (memory_id, user_id, content),
    )