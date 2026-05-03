import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from .database import get_db

MODEL = "gpt-4o-mini"


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


# Singular keys: only one active value at a time. New value supersedes old.
SINGULAR_KEYS = {
    "employment", "location", "family_spouse",
    "relationship_status",
}

# Additive keys: multiple values can coexist (multiple pets, hobbies, kids).
# Same-entity overlap is required for supersession.
ADDITIVE_KEYS = {
    "pet", "hobby", "language", "skill",
    "food_allergy", "family_child",
}


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
- DUPLICATE: A and B say the same thing.
- ADDS: B adds new info without contradicting A.
- UNRELATED: A and B are about different things.

Respond with JSON only:
{{"relationship": "SUPERSEDES|DUPLICATE|ADDS|UNRELATED", "confidence": 0.0-1.0}}"""


def _same_key_candidates(user_id: str, key: str, subject: str) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM memories
           WHERE user_id = ? AND key = ? AND subject = ? AND active = 1""",
        (user_id, key, subject),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _entities_overlap(stored_entities_json, new_entities) -> bool:
    try:
        stored = set(json.loads(stored_entities_json or "[]"))
    except Exception:
        stored = set()
    return bool(stored.intersection(set(new_entities or [])))


def _llm_judge(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    client = get_client()
    if client is None:
        return {"relationship": "UNRELATED", "confidence": 0.0}
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
        print(f"  LLM judge failed: {e}")
        return {"relationship": "UNRELATED", "confidence": 0.0}


def resolve_evolution(new_memory: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    key = new_memory["key"]
    subject = new_memory["subject"]
    is_correction = new_memory.get("is_correction_of") is not None

    # Explicit corrections always supersede the matching key
    if is_correction:
        candidates = _same_key_candidates(user_id, key, subject)
        if candidates:
            most_recent = max(candidates, key=lambda x: x["updated_at"])
            return {
                "action": "INSERT_AND_SUPERSEDE",
                "supersedes_id": most_recent["id"],
                "reason": "explicit user correction",
            }

    candidates = _same_key_candidates(user_id, key, subject)

    if candidates:
        new_entities = new_memory.get("entities", [])

        # Additive keys: only supersede if entities overlap (same pet, same hobby).
        # Otherwise insert as a new memory alongside the existing one.
        if key in ADDITIVE_KEYS:
            for cand in candidates:
                if _entities_overlap(cand.get("entities"), new_entities):
                    if cand["value"].strip().lower() == new_memory["value"].strip().lower():
                        return {"action": "SKIP_DUPLICATE", "supersedes_id": None,
                                "reason": "duplicate additive fact"}
                    return {"action": "INSERT_AND_SUPERSEDE",
                            "supersedes_id": cand["id"],
                            "reason": f"additive key '{key}' updated for same entity"}
            return {"action": "INSERT", "supersedes_id": None,
                    "reason": f"additive key '{key}', new entity"}

        # Singular keys: same-key collision → supersede the most recent
        most_recent = max(candidates, key=lambda x: x["updated_at"])

        if most_recent["value"].strip().lower() == new_memory["value"].strip().lower():
            return {"action": "SKIP_DUPLICATE", "supersedes_id": None,
                    "reason": "exact duplicate"}

        # Stable facts need LLM-confirmed contradiction
        if most_recent.get("temporal") == "stable" and new_memory.get("temporal") == "stable":
            judge = _llm_judge(most_recent, new_memory)
            if judge["relationship"] == "DUPLICATE":
                return {"action": "SKIP_DUPLICATE", "supersedes_id": None,
                        "reason": "LLM: duplicate stable fact"}
            if judge["relationship"] == "SUPERSEDES" and judge["confidence"] > 0.8:
                return {"action": "INSERT_AND_SUPERSEDE",
                        "supersedes_id": most_recent["id"],
                        "reason": "LLM: stable fact superseded"}
            return {"action": "INSERT", "supersedes_id": None,
                    "reason": "stable fact uncertain, keeping both"}

        # Mutable singular fact — supersede by default
        return {"action": "INSERT_AND_SUPERSEDE",
                "supersedes_id": most_recent["id"],
                "reason": f"same key '{key}', mutable fact updated"}

    return {"action": "INSERT", "supersedes_id": None,
            "reason": "no conflict found"}


def apply_evolution(conn, new_memory: Dict[str, Any], new_memory_id: str,
                    decision: Dict[str, Any], now: str) -> bool:
    action = decision["action"]

    if action == "SKIP_DUPLICATE":
        print(f"⏭  SKIP: {decision['reason']}")
        return False

    if action == "INSERT_AND_SUPERSEDE":
        conn.execute(
            "UPDATE memories SET active = 0, updated_at = ? WHERE id = ?",
            (now, decision["supersedes_id"]),
        )
        new_memory["_supersedes"] = decision["supersedes_id"]
        print(f"  SUPERSEDE: {decision['reason']}")
        return True

    print(f" INSERT: {decision['reason']}")
    return True