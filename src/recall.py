import json
import numpy as np
from openai import OpenAI
from .database import get_db
import tiktoken

client = OpenAI()
encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def get_embedding(text: str) -> list:
    """Turn text into a list of numbers (embedding)."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]  # safety truncation
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    """How similar are two embeddings? Returns 0.0 to 1.0."""
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def store_embedding(source_id: str, source_type: str, content: str):
    """Generate and store an embedding for a piece of content."""
    from datetime import datetime
    import uuid
    embedding = get_embedding(content)
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO embeddings
        (id, source_id, source_type, content, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (str(uuid.uuid4()), source_id, source_type, content,
          json.dumps(embedding), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def recall(query: str, session_id: str, user_id: str, max_tokens: int):
    """Find relevant memories and build a context string."""
    conn = get_db()
    query_embedding = get_embedding(query)

    # --- Step 1: Get active memories for this user ---
    memories = []
    if user_id:
        rows = conn.execute("""
            SELECT m.*, e.embedding, e.content as embed_content
            FROM memories m
            LEFT JOIN embeddings e ON e.source_id = m.id
            WHERE m.user_id = ? AND m.active = 1
            ORDER BY m.updated_at DESC
        """, (user_id,)).fetchall()

        for row in rows:
            score = 0.5  # default score
            if row["embedding"]:
                emb = json.loads(row["embedding"])
                score = cosine_similarity(query_embedding, emb)
            memories.append({
                "id": row["id"],
                "type": row["type"],
                "key": row["key"],
                "value": row["value"],
                "score": score,
                "updated_at": row["updated_at"][:10],  # just the date
                "source_turn": row["source_turn"]
            })

    # Sort by score descending
    memories.sort(key=lambda x: x["score"], reverse=True)

    # --- Step 2: Get recent turns for session context ---
    recent_turns = conn.execute("""
        SELECT id, messages, timestamp FROM turns
        WHERE session_id = ?
        ORDER BY timestamp DESC LIMIT 5
    """, (session_id,)).fetchall()
    conn.close()

    # --- Step 3: Build context string within token budget ---
    # Priority: stable facts first, then query-relevant memories, then recent turns
    context_parts = []
    citations = []
    tokens_used = 0

    # High-confidence facts first (the "stable user facts" priority)
    fact_lines = []
    for m in memories:
        if m["score"] > 0.3 or m["type"] in ("fact", "preference"):
            line = f"- {m['value']} (updated {m['updated_at']})"
            fact_lines.append((line, m))

    if fact_lines:
        header = "## Known facts about this user\n"
        block = header + "\n".join(l for l, _ in fact_lines[:10])
        cost = count_tokens(block)
        if tokens_used + cost <= max_tokens:
            context_parts.append(block)
            tokens_used += cost
            for _, m in fact_lines[:10]:
                citations.append({
                    "turn_id": m["source_turn"] or "",
                    "score": m["score"],
                    "snippet": m["value"][:100]
                })

    # Recent session context
    if recent_turns:
        turn_lines = []
        for t in recent_turns:
            msgs = json.loads(t["messages"])
            date = t["timestamp"][:10]
            for msg in msgs:
                if msg["role"] == "user":
                    turn_lines.append(f"- [{date}] User said: {msg['content'][:150]}")
                    citations.append({
                        "turn_id": t["id"],
                        "score": 0.5,
                        "snippet": msg["content"][:100]
                    })
                    break  # just first user message per turn

        if turn_lines:
            header = "\n## Recent conversation context\n"
            block = header + "\n".join(turn_lines[:5])
            cost = count_tokens(block)
            if tokens_used + cost <= max_tokens:
                context_parts.append(block)
                tokens_used += cost

    context = "\n".join(context_parts)
    return context, citations