import os
import uuid
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from .database import init_db, get_db
from .models import (
    TurnRequest, RecallRequest, SearchRequest,
)
from .extraction import extract_memories
from .recall import recall, store_embedding
from .retrieval import get_embedding, cosine


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Memory Service", lifespan=lifespan)


# ─────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_failed", "detail": str(exc.errors())[:500]},
    )


@app.exception_handler(HTTPException)
async def http_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception):
    print(f"Unexpected error on {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "an unexpected error occurred"},
    )


# ─────────────────────────────────────────────
# Optional bearer auth
# ─────────────────────────────────────────────

EXPECTED_TOKEN = os.environ.get("MEMORY_AUTH_TOKEN")


async def check_auth(authorization: Optional[str] = Header(None)):
    if not EXPECTED_TOKEN:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")
    return True


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
def ingest_turn(req: TurnRequest, _: bool = Depends(check_auth)):
    turn_id = str(uuid.uuid4())
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO turns
               (id, session_id, user_id, messages, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                turn_id,
                req.session_id,
                req.user_id,
                json.dumps([m.dict() for m in req.messages]),
                req.timestamp,
                json.dumps(req.metadata or {})[:10_000],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Embed the turn text
    turn_text = " ".join(f"{m.role}: {m.content}" for m in req.messages)
    if turn_text.strip():
        try:
            store_embedding(turn_id, "turn", turn_text)
        except Exception as e:
            print(f" Turn embedding failed: {e}")

    # Extract memories
    try:
        stored_memories = extract_memories(
            turn_id=turn_id,
            user_id=req.user_id,
            session_id=req.session_id,
            messages=[m.dict() for m in req.messages],
            timestamp=req.timestamp,
        )
    except Exception as e:
        print(f" Extraction failed (turn still saved): {e}")
        stored_memories = []

    # Embed each extracted memory
    for mem in stored_memories:
        try:
            if mem.get("id") and mem.get("value"):
                store_embedding(mem["id"], "memory", mem["value"])
                print(f" Embedded memory: {mem['key']} = {mem['value'][:50]}")
        except Exception as e:
            print(f"  Memory embedding failed: {e}")

    return {"id": turn_id}


# ─────────────────────────────────────────────
# POST /recall
# ─────────────────────────────────────────────

@app.post("/recall")
def recall_context(req: RecallRequest, _: bool = Depends(check_auth)):
    try:
        context, citations = recall(
            query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            max_tokens=req.max_tokens,
        )
        return {"context": context, "citations": citations}
    except Exception as e:
        print(f" Recall error: {e}")
        return {"context": "", "citations": []}


# ─────────────────────────────────────────────
# POST /search
# ─────────────────────────────────────────────

@app.post("/search")
def search(req: SearchRequest, _: bool = Depends(check_auth)):
    """Search returns both raw turns AND structured memories, ranked by similarity."""
    conn = get_db()
    try:
        turn_rows = conn.execute(
            """SELECT e.source_id, e.content, e.embedding,
                      t.session_id, t.timestamp, t.metadata,
                      'turn' AS kind
               FROM embeddings e
               JOIN turns t ON t.id = e.source_id
               WHERE e.source_type = 'turn'
               AND (? IS NULL OR t.session_id = ?)
               AND (? IS NULL OR t.user_id = ?)""",
            (req.session_id, req.session_id, req.user_id, req.user_id),
        ).fetchall()

        if req.user_id:
            mem_rows = conn.execute(
                """SELECT e.source_id, e.content, e.embedding,
                          m.source_session AS session_id,
                          m.updated_at AS timestamp,
                          '{}' AS metadata,
                          'memory' AS kind
                   FROM embeddings e
                   JOIN memories m ON m.id = e.source_id
                   WHERE e.source_type = 'memory'
                   AND m.user_id = ?
                   AND m.active = 1""",
                (req.user_id,),
            ).fetchall()
        else:
            mem_rows = []

        all_rows = list(turn_rows) + list(mem_rows)
    finally:
        conn.close()

    if not all_rows:
        return {"results": []}

    try:
        query_emb = get_embedding(req.query)
    except Exception as e:
        print(f" Search embedding failed: {e}")
        return {"results": []}

    scored = []
    for row in all_rows:
        try:
            emb = json.loads(row["embedding"])
            score = cosine(query_emb, emb)
            metadata = json.loads(row["metadata"] or "{}") if row["metadata"] else {}
            metadata["_kind"] = row["kind"]
            scored.append({
                "content": row["content"],
                "score": score,
                "session_id": row["session_id"] or "",
                "timestamp": row["timestamp"],
                "metadata": metadata,
            })
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[: req.limit]}


# ─────────────────────────────────────────────
# GET /users/{user_id}/memories
# ─────────────────────────────────────────────

@app.get("/users/{user_id}/memories")
def get_memories(user_id: str, _: bool = Depends(check_auth)):
    if len(user_id) > 200:
        raise HTTPException(status_code=400, detail="user_id too long")
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE user_id = ?
               ORDER BY active DESC, updated_at DESC""",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return {"memories": [dict(row) for row in rows]}


# ─────────────────────────────────────────────
# DELETE /sessions/{session_id}
# ─────────────────────────────────────────────

@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, _: bool = Depends(check_auth)):
    if len(session_id) > 200:
        raise HTTPException(status_code=400, detail="session_id too long")
    conn = get_db()
    try:
        turn_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM turns WHERE session_id = ?", (session_id,)).fetchall()]
        mem_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM memories WHERE source_session = ?", (session_id,)).fetchall()]

        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM memories WHERE source_session = ?", (session_id,))

        for tid in turn_ids:
            conn.execute("DELETE FROM embeddings WHERE source_id = ?", (tid,))
        for mid in mem_ids:
            conn.execute("DELETE FROM embeddings WHERE source_id = ?", (mid,))
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))

        conn.commit()
    finally:
        conn.close()

# ─────────────────────────────────────────────
# DELETE /users/{user_id}
# ─────────────────────────────────────────────

@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str, _: bool = Depends(check_auth)):
    if len(user_id) > 200:
        raise HTTPException(status_code=400, detail="user_id too long")
    conn = get_db()
    try:
        mem_rows = conn.execute(
            "SELECT id FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        turn_rows = conn.execute(
            "SELECT id FROM turns WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM turns WHERE user_id = ?", (user_id,))
        for m in mem_rows:
            conn.execute("DELETE FROM embeddings WHERE source_id = ?", (m["id"],))
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (m["id"],))
        for t in turn_rows:
            conn.execute("DELETE FROM embeddings WHERE source_id = ?", (t["id"],))
        conn.commit()
    finally:
        conn.close()