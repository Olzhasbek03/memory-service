"""
Fact evolution: detect when a new memory supersedes an existing one.
Resolved at write time so /recall is always fast.

Three signals in order of cost:
  1. Same canonical key + same subject → supersede (cheap)
  2. Explicit correction type → always supersede matching key
  3. LLM judge for ambiguous cross-key contradictions (expensive, rare)
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from openai import OpenAI
from .database import get_db

client = OpenAI()
MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """Compare two memories about the same user.

Memory A (existing):
  type: {a_type}
  key: {a_key}
  value: {a_value}

Memory B (new):
  type: {b_type}
  key: {b_key}
  value: {b_value}

Classify the relationship:
- SUPERSEDES: B replaces A. A is no longer true.
  Example: A="works at Stripe" / B="works at Notion"
- DUPLICATE: A and B say the same thing. No update needed.
- ADDS: B adds new info without contradicting A.
  Example: A="has a dog" / B="has a cat" — both can be true.
- UNRELATED: A and B are about different things.

Respond with JSON only:
{{"relationship": "SUPERSEDES|DUPLICATE|ADDS|UNRELATED", "confidence": 0.0-1.0}}"""


def _same_key_candidates(user_id: str, key: str, subject: str) -> List[Dict[str, Any]]:
    """Get active memories with the same key and subject."""
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM memories
           WHERE user_id = ? AND key = ? AND subject = ? AND active = 1""",
        (user_id, key, subject),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _llm_judge(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM if two memories contradict each other."""
    try:
        prompt = JUDGE_PROMPT.format(
            a_type=existing["type"], a_key=existing["key"], a_value=existing["value"],
            b_type=new["type"],     b_key=new["key"],     b_value=new["value"],
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  LLM judge failed: {e}")
        return {"relationship": "UNRELATED", "confidence": 0.0}


def resolve_evolution(new_memory: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Decide what to do with a new memory.

    Returns a decision dict:
    {
      "action": "INSERT" | "INSERT_AND_SUPERSEDE" | "SKIP_DUPLICATE",
      "supersedes_id": <id or None>,
      "reason": "<short string>"
    }
    """
    key = new_memory["key"]
    subject = new_memory["subject"]
    is_correction = new_memory.get("is_correction_of") is not None

    # ── Signal 1: Explicit correction — always supersede ──
    if is_correction:
        candidates = _same_key_candidates(user_id, key, subject)
        if candidates:
            most_recent = max(candidates, key=lambda x: x["updated_at"])
            return {
                "action": "INSERT_AND_SUPERSEDE",
                "supersedes_id": most_recent["id"],
                "reason": "explicit user correction",
            }

    # ── Signal 2: Same canonical key + same subject ──
    candidates = _same_key_candidates(user_id, key, subject)

    if candidates:
        most_recent = max(candidates, key=lambda x: x["updated_at"])

        # Exact duplicate — skip
        if most_recent["value"].strip().lower() == new_memory["value"].strip().lower():
            return {
                "action": "SKIP_DUPLICATE",
                "supersedes_id": None,
                "reason": "exact duplicate",
            }

        # Stable facts need LLM confirmation before superseding
        if most_recent.get("temporal") == "stable" and new_memory.get("temporal") == "stable":
            judge = _llm_judge(most_recent, new_memory)
            if judge["relationship"] == "DUPLICATE":
                return {
                    "action": "SKIP_DUPLICATE",
                    "supersedes_id": None,
                    "reason": "LLM: duplicate stable fact",
                }
            if judge["relationship"] == "SUPERSEDES" and judge["confidence"] > 0.8:
                return {
                    "action": "INSERT_AND_SUPERSEDE",
                    "supersedes_id": most_recent["id"],
                    "reason": "LLM: stable fact superseded",
                }
            # Uncertain — keep both
            return {
                "action": "INSERT",
                "supersedes_id": None,
                "reason": "stable fact uncertain, keeping both",
            }

        # Mutable fact (current/transient) — supersede by default
        return {
            "action": "INSERT_AND_SUPERSEDE",
            "supersedes_id": most_recent["id"],
            "reason": f"same key '{key}', mutable fact updated",
        }

    # ── Default: fresh memory, no conflict ──
    return {
        "action": "INSERT",
        "supersedes_id": None,
        "reason": "no conflict found",
    }


def apply_evolution(
    conn,
    new_memory: Dict[str, Any],
    new_memory_id: str,
    decision: Dict[str, Any],
    now: str,
) -> bool:
    """
    Apply the evolution decision to the database.
    Returns True if the new memory should be inserted, False if skipped.
    """
    action = decision["action"]

    if action == "SKIP_DUPLICATE":
        print(f"⏭️  SKIP: {decision['reason']}")
        return False

    if action == "INSERT_AND_SUPERSEDE":
        # Mark old memory as inactive
        conn.execute(
            "UPDATE memories SET active = 0, updated_at = ? WHERE id = ?",
            (now, decision["supersedes_id"]),
        )
        # Tag new memory with what it supersedes
        new_memory["_supersedes"] = decision["supersedes_id"]
        print(f"♻️  SUPERSEDE: {decision['reason']}")
        return True

    print(f"➕ INSERT: {decision['reason']}")
    return True