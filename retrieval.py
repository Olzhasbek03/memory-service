"""
Hybrid retrieval over user memories.

Pipeline:
  1. Embedding search  → ranked list A
  2. BM25 (FTS5)       → ranked list B
  3. RRF fusion        → ranked list C
  4. (L5: reranker)    → ranked list D

We retrieve over MEMORIES (not raw turns) for /recall, because memories
are the structured knowledge unit. Raw turns are searchable via /search.
"""
import json
import math
import sqlite3
from typing import List, Dict, Any, Tuple
import numpy as np
from openai import OpenAI
from .database import get_db

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
RRF_K = 60  # standard literature default


# ─────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    text = text[:8000] if text else ""
    if not text.strip():
        return [0.0] * 1536
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────
# Per-retriever search
# ─────────────────────────────────────────────

def embedding_search(user_id: str, query: str, top_n: int = 30) -> List[Dict[str, Any]]:
    """Return memories ranked by cosine similarity to the query embedding."""
    q_emb = get_embedding(query)
    conn = get_db()
    rows = conn.execute(
        """
        SELECT m.*, e.embedding
        FROM memories m
        LEFT JOIN embeddings e
          ON e.source_id = m.id AND e.source_type = 'memory'
        WHERE m.user_id = ? AND m.active = 1
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    scored = []
    for r in rows:
        if not r["embedding"]:
            continue
        emb = json.loads(r["embedding"])
        s = cosine(q_emb, emb)
        scored.append({**dict(r), "embed_score": s})
    scored.sort(key=lambda x: x["embed_score"], reverse=True)
    return scored[:top_n]


def bm25_search(user_id: str, query: str, top_n: int = 30) -> List[Dict[str, Any]]:
    """Return memories ranked by BM25 over their value + entities."""
    conn = get_db()

    # FTS5 needs a sanitized query. Strip operators and quote tokens.
    safe = _sanitize_fts_query(query)
    if not safe:
        conn.close()
        return []

    try:
        rows = conn.execute(
            """
            SELECT m.*, bm25(memories_fts) AS bm25_score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.memory_id
            WHERE memories_fts.user_id = ?
              AND memories_fts MATCH ?
              AND m.active = 1
            ORDER BY bm25_score
            LIMIT ?
            """,
            (user_id, safe, top_n),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    # bm25() returns LOWER is better; invert for consistency.
    return [{**dict(r), "bm25_score": -r["bm25_score"]} for r in rows]


def _sanitize_fts_query(q: str) -> str:
    """Strip FTS operators; OR-join surviving tokens."""
    bad = set("\"'()*:^-")
    cleaned = "".join(c if c not in bad else " " for c in q)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    if not tokens:
        return ""
    # OR semantics: any token match qualifies. BM25 still ranks well.
    return " OR ".join(tokens)


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists into one using Reciprocal Rank Fusion.
    Each item in input lists must have an 'id' field.
    """
    fused: Dict[str, Dict[str, Any]] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            mid = item["id"]
            contribution = 1.0 / (k + rank)
            if mid in fused:
                fused[mid]["rrf_score"] += contribution
            else:
                fused[mid] = {**item, "rrf_score": contribution}
    out = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return out[:top_n]


# ─────────────────────────────────────────────
# Top-level hybrid search
# ─────────────────────────────────────────────

def hybrid_search(user_id: str, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """Run both retrievers, fuse with RRF, return top_n memories."""
    if not user_id:
        return []
    emb_list = embedding_search(user_id, query, top_n=30)
    bm25_list = bm25_search(user_id, query, top_n=30)
    return rrf_fuse([emb_list, bm25_list], top_n=top_n)