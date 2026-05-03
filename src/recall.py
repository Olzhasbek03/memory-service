"""
/recall pipeline:
  1. Rewrite query for clearer retrieval
  2. Hybrid retrieve memories (embedding + BM25 → RRF)
  3. Multi-hop expansion if query connects two topics
  4. Priority scoring (type, confidence, recency)
  5. LLM rerank top candidates
  6. Assemble context under token budget
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoder.encode(text or ""))


def store_embedding(source_id: str, source_type: str, content: str):
    """Generate and store an embedding. Called from main.py after extraction."""
    from .retrieval import get_embedding
    from .database import get_db
    emb = get_embedding(content)
    conn = get_db()
    conn.execute(
        """INSERT OR REPLACE INTO embeddings
           (id, source_id, source_type, content, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), source_id, source_type, content,
         json.dumps(emb), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def cosine_similarity(a, b):
    from .retrieval import cosine
    return cosine(a, b)


# ─────────────────────────────────────────────
# Priority scoring
# ─────────────────────────────────────────────

PRIORITY_TYPES = {
    "correction": 100,
    "fact": 90,
    "preference": 80,
    "opinion": 60,
    "event": 50,
}


def _priority_score(memory: Dict[str, Any], retrieval_score: float) -> float:
    type_w = PRIORITY_TYPES.get(memory.get("type", "fact"), 50)
    conf = float(memory.get("confidence") or 0.5)
    return (retrieval_score * 100) + (type_w * 0.5) + (conf * 10)


def _format_memory_line(m: Dict[str, Any]) -> str:
    date = (m.get("updated_at") or "")[:10]
    line = f"- {m['value']}"
    if date:
        line += f" (updated {date})"
    if m.get("type") == "correction":
        line = "- [CORRECTION] " + line[2:]
    return line


# ─────────────────────────────────────────────
# Main recall function
# ─────────────────────────────────────────────

def recall(query: str, session_id: str, user_id: str, max_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
    from .retrieval import hybrid_search
    from .database import get_db

    if not user_id:
        return "", []

    # ── Step 1: Rewrite query ──
    try:
        from .query_processing import rewrite_query, multihop_expand
        rw = rewrite_query(query)
        search_query = rw["rewritten"]
        is_multihop = rw["is_multihop"]
        print(f"🔍 '{query}' → '{search_query}' (multihop={is_multihop})")
    except Exception as e:
        print(f"⚠️  Query rewrite failed, using original: {e}")
        search_query = query
        is_multihop = False
        rw = {"is_multihop": False, "asked_dimension": "other"}

    # ── Step 2: First-pass hybrid retrieval ──
    try:
        candidates = hybrid_search(user_id, search_query, top_n=20)
    except Exception as e:
        print(f"⚠️  Hybrid search failed: {e}")
        candidates = []

    # ── Step 3: Multi-hop expansion ──
    if is_multihop and candidates:
        try:
            from .query_processing import multihop_expand
            extras = multihop_expand(
                user_id=user_id,
                rewrite_meta=rw,
                first_pass_results=candidates,
                hybrid_search_fn=hybrid_search,
                top_n=10,
            )
            candidates = candidates + extras
            print(f"🔀 Added {len(extras)} multi-hop candidates")
        except Exception as e:
            print(f"⚠️  Multi-hop failed: {e}")

    # ── Step 4: Priority scoring ──
    for c in candidates:
        c["final_score"] = _priority_score(c, c.get("rrf_score", 0.0))
        if c.get("multihop_confirmed"):
            c["final_score"] *= 1.2
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Step 5: LLM rerank ──
    if len(candidates) > 3:
        try:
            from .reranking import rerank
            candidates = rerank(query, candidates[:20], top_n=15)
        except Exception as e:
            print(f"⚠️  Reranker failed, skipping: {e}")

    # ── Step 6: Get recent session turns ──
    conn = get_db()
    try:
        recent_turns = conn.execute(
            """SELECT id, messages, timestamp FROM turns
               WHERE session_id = ?
               ORDER BY timestamp DESC LIMIT 5""",
            (session_id,),
        ).fetchall()
    except Exception:
        recent_turns = []
    finally:
        conn.close()

    # ── Step 7: Assemble context under token budget ──
    citations = []
    parts = []
    used = 0
    HEADER_BUDGET = 30

    # Section 1: Stable user facts (corrections, facts, preferences)
    facts_section = "## Known facts about this user"
    fact_lines = []
    used_ids = set()

    for m in candidates:
        if m.get("subject") != "user":
            continue
        if m.get("type") in ("fact", "preference", "correction"):
            line = _format_memory_line(m)
            cost = count_tokens(line)
            if used + cost + HEADER_BUDGET > max_tokens:
                break
            fact_lines.append(line)
            used += cost
            used_ids.add(m["id"])
            citations.append({
                "turn_id": m.get("source_turn") or "",
                "score": float(m.get("final_score", 0)),
                "snippet": m["value"][:140],
            })

    if fact_lines:
        block = facts_section + "\n" + "\n".join(fact_lines)
        parts.append(block)
        used += count_tokens(facts_section) + 2

    # Section 2: Other relevant memories (opinions, events)
    other_lines = []
    for m in candidates:
        if m["id"] in used_ids:
            continue
        line = _format_memory_line(m)
        cost = count_tokens(line)
        if used + cost + HEADER_BUDGET > max_tokens:
            break
        other_lines.append(line)
        used += cost
        used_ids.add(m["id"])
        citations.append({
            "turn_id": m.get("source_turn") or "",
            "score": float(m.get("final_score", 0)),
            "snippet": m["value"][:140],
        })

    if other_lines:
        h = "\n## Other relevant memories"
        parts.append(h + "\n" + "\n".join(other_lines))
        used += count_tokens(h) + 2

    # Section 3: Recent conversation context
    turn_lines = []
    for t in recent_turns:
        try:
            msgs = json.loads(t["messages"])
            date = (t["timestamp"] or "")[:10]
            first_user = next((m for m in msgs if m.get("role") == "user"), None)
            if not first_user:
                continue
            snippet = first_user["content"][:160]
            line = f"- [{date}] User: {snippet}"
            cost = count_tokens(line)
            if used + cost + HEADER_BUDGET > max_tokens:
                break
            turn_lines.append(line)
            used += cost
            citations.append({
                "turn_id": t["id"],
                "score": 0.3,
                "snippet": snippet,
            })
        except Exception:
            continue

    if turn_lines:
        h = "\n## Recent conversation context"
        parts.append(h + "\n" + "\n".join(turn_lines))

    return "\n".join(parts), citations