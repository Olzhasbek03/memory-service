import uuid
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # read .env file

from .database import init_db, get_db
from .models import (
    TurnRequest, RecallRequest, SearchRequest,
    RecallResponse, SearchResponse
)
from .extraction import extract_memories
from .recall import recall, store_embedding, cosine_similarity
from .retrieval import get_embedding
import numpy as np

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables on startup
    yield

app = FastAPI(title="Memory Service", lifespan=lifespan)

# ─────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ─────────────────────────────────────────────
# POST /turns
# ─────────────────────────────────────────────
@app.post("/turns", status_code=201)
def ingest_turn(req: TurnRequest):
    turn_id = str(uuid.uuid4())
    conn = get_db()

    # Store the raw turn
    conn.execute("""
        INSERT INTO turns (id, session_id, user_id, messages, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        turn_id,
        req.session_id,
        req.user_id,
        json.dumps([m.dict() for m in req.messages]),
        req.timestamp,
        json.dumps(req.metadata or {})
    ))
    conn.commit()
    conn.close()

    # Embed the raw turn text
    turn_text = " ".join([f"{m.role}: {m.content}" for m in req.messages])
    store_embedding(turn_id, "turn", turn_text)

    # Extract memories — returns list of stored memory dicts with their ids
    stored_memories = extract_memories(
        turn_id=turn_id,
        user_id=req.user_id,
        session_id=req.session_id,
        messages=[m.dict() for m in req.messages],
        timestamp=req.timestamp
    )

    # Store embeddings for each extracted memory
    # This is critical — without this, recall finds nothing
    for mem in stored_memories:
        if mem.get("id") and mem.get("value"):
            store_embedding(mem["id"], "memory", mem["value"])
            print(f"📎 Embedded memory: {mem['key']} = {mem['value'][:50]}")

    return {"id": turn_id}

# ─────────────────────────────────────────────
# POST /recall
# ─────────────────────────────────────────────
@app.post("/recall")
def recall_context(req: RecallRequest):
    try:
        context, citations = recall(
            query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            max_tokens=req.max_tokens
        )
        return {"context": context, "citations": citations}
    except Exception as e:
        print(f"Recall error: {e}")
        return {"context": "", "citations": []}

# ─────────────────────────────────────────────
# POST /search
# ─────────────────────────────────────────────
@app.post("/search")
def search(req: SearchRequest):
    conn = get_db()

    # Get all embeddings
    rows = conn.execute("""
        SELECT e.source_id, e.content, e.embedding,
               t.session_id, t.timestamp, t.metadata
        FROM embeddings e
        JOIN turns t ON t.id = e.source_id
        WHERE e.source_type = 'turn'
        AND (? IS NULL OR t.session_id = ?)
        AND (? IS NULL OR t.user_id = ?)
        ORDER BY t.timestamp DESC
    """, (req.session_id, req.session_id,
          req.user_id, req.user_id)).fetchall()
    conn.close()

    if not rows:
        return {"results": []}

    query_emb = get_embedding(req.query)
    scored = []
    for row in rows:
        emb = json.loads(row["embedding"])
        score = cosine_similarity(query_emb, emb)
        scored.append({
            "content": row["content"],
            "score": score,
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"] or "{}")
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:req.limit]}

# ─────────────────────────────────────────────
# GET /users/{user_id}/memories
# ─────────────────────────────────────────────
@app.get("/users/{user_id}/memories")
def get_memories(user_id: str):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM memories
        WHERE user_id = ?
        ORDER BY active DESC, updated_at DESC
    """, (user_id,)).fetchall()
    conn.close()

    memories = [dict(row) for row in rows]
    return {"memories": memories}

# ─────────────────────────────────────────────
# DELETE /sessions/{session_id}
# ─────────────────────────────────────────────
@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    conn = get_db()
    # Get turn ids for this session first
    turns = conn.execute(
        "SELECT id FROM turns WHERE session_id = ?", (session_id,)
    ).fetchall()
    turn_ids = [t["id"] for t in turns]

    conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM memories WHERE source_session = ?", (session_id,))
    for tid in turn_ids:
        conn.execute("DELETE FROM embeddings WHERE source_id = ?", (tid,))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# DELETE /users/{user_id}
# ─────────────────────────────────────────────
@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str):
    conn = get_db()
    conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM turns WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM embeddings WHERE source_id IN (SELECT id FROM turns WHERE user_id = ?)", (user_id,))
    conn.commit()
    conn.close()