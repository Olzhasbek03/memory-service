"""
Hybrid retrieval: embeddings + BM25 (FTS5) fused with Reciprocal Rank Fusion.
"""
import json
import sqlite3
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from .database import get_db
import os

EMBEDDING_MODEL = "text-embedding-3-small"
RRF_K = 60


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()

def get_embedding(text: str) -> List[float]:
    text = text[:8000] if text else ""
    if not text.strip():
        return [0.0] * 1536
    client = get_client()
    if client is None:
        return [0.0] * 1536
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def cosine(a: List[float], b: List[float]) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embedding_search(user_id: str, query: str, top_n: int = 30) -> List[Dict[str, Any]]:
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
        row_dict = dict(r)
        if not row_dict.get("embedding"):
            # No embedding yet — still include with base score
            scored.append({**row_dict, "embed_score": 0.3})
            continue
        emb = json.loads(row_dict["embedding"])
        s = cosine(q_emb, emb)
        scored.append({**row_dict, "embed_score": s})

    scored.sort(key=lambda x: x["embed_score"], reverse=True)
    return scored[:top_n]


def _sanitize_fts_query(q: str) -> str:
    bad = set("\"'()*:^-")
    cleaned = "".join(c if c not in bad else " " for c in q)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(tokens)


def bm25_search(user_id: str, query: str, top_n: int = 30) -> List[Dict[str, Any]]:
    conn = get_db()
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
    except sqlite3.OperationalError as e:
        print(f" BM25 search error: {e}")
        rows = []
    conn.close()
    return [{**dict(r), "bm25_score": -r["bm25_score"]} for r in rows]


def rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
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


def hybrid_search(user_id: str, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    emb_list = embedding_search(user_id, query, top_n=30)
    bm25_list = bm25_search(user_id, query, top_n=30)
    return rrf_fuse([emb_list, bm25_list], top_n=top_n)