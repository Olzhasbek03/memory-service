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
    """Create tables if they don't exist yet."""
    conn = get_db()
    conn.executescript("""
        -- Stores every conversation turn we receive
        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT,
            messages TEXT NOT NULL,  -- stored as JSON string
            timestamp TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        -- Stores extracted memories (facts, preferences, etc.)
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,        -- 'fact', 'preference', 'opinion', 'event'
            key TEXT NOT NULL,         -- e.g. 'employment', 'pet', 'location'
            value TEXT NOT NULL,       -- e.g. 'works at Notion as PM'
            confidence REAL DEFAULT 1.0,
            source_session TEXT,
            source_turn TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            supersedes TEXT,           -- id of the memory this replaces
            active INTEGER DEFAULT 1   -- 1=current truth, 0=old/superseded
        );

        -- Stores embeddings for semantic search
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,   -- turn_id or memory_id
            source_type TEXT NOT NULL, -- 'turn' or 'memory'
            content TEXT NOT NULL,     -- the text that was embedded
            embedding TEXT NOT NULL,   -- stored as JSON array of floats
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(user_id, active);
        CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_id);
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")