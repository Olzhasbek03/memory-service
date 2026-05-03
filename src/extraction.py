import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .database import get_db, index_memory_for_fts
from .evolution import resolve_evolution, apply_evolution

EXTRACTION_MODEL = "gpt-4o-mini"


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


TOPIC_NORMALIZATION = {
    "job": "employment", "employer": "employment", "work": "employment",
    "occupation": "employment", "profession": "employment", "career": "employment",
    "company": "employment", "workplace": "employment",
    "address": "location", "city": "location", "country": "location",
    "residence": "location", "lives_in": "location", "home": "location",
    "hometown": "location",
    "pet": "pet", "pets": "pet", "dog": "pet", "cat": "pet", "animal": "pet",
    "diet": "food_diet", "food": "food_diet", "vegetarian": "food_diet",
    "vegan": "food_diet",
    "allergy": "food_allergy", "allergies": "food_allergy",
    "spouse": "family_spouse", "wife": "family_spouse", "husband": "family_spouse",
    "partner": "family_spouse",
    "child": "family_child", "children": "family_child", "kid": "family_child",
    "kids": "family_child", "son": "family_child", "daughter": "family_child",
}


def normalize_key(key: str) -> str:
    if not key:
        return "unknown"
    k = key.strip().lower().replace(" ", "_").replace("-", "_")
    return TOPIC_NORMALIZATION.get(k, k)


EXTRACTION_SYSTEM_PROMPT = """You extract durable, queryable memories from conversation turns.

OUTPUT FORMAT
Return ONLY a JSON object with one field: "memories" — an array.
If nothing factual is said, return {"memories": []}. Do not invent.

Each memory has these fields:
{
  "type": "fact" | "preference" | "opinion" | "event" | "correction",
  "key": "<short stable topic slug>",
  "value": "<full sentence in third person about the user>",
  "subject": "user" | "other:<name>",
  "entities": ["<entity1>", "<entity2>"],
  "confidence": 0.0 to 1.0,
  "is_correction_of": "<key being corrected, or null>",
  "temporal": "stable" | "current" | "transient"
}

KEY GUIDELINES
- key is a slug like "employment", "location", "pet", "food_diet",
  "food_allergy", "hobby", "family_spouse".
- value is a complete sentence: "User works at Notion as a PM".

TYPE GUIDELINES
- fact: durable truth ("lives in Berlin", "has a dog named Biscuit")
- preference: stable like/dislike ("vegetarian", "prefers concise answers")
- opinion: stance that may evolve ("thinks TypeScript is overengineered")
- event: time-bound happening ("going to a wedding Saturday")
- correction: explicit fix ("actually, I meant Notion not Stripe")

SUBJECT: "user" if about the user, "other:Name" if about someone else.
ENTITIES: lowercase names of people, pets, places, companies: ["biscuit", "berlin"]
CONFIDENCE: 0.95+ explicit, 0.7-0.9 inferred, 0.5-0.7 ambiguous, below 0.5 skip
TEMPORAL: stable (allergies, name), current (job, location), transient (tonight's plans)

DO NOT extract assistant statements. Return ONLY the JSON object."""


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _call_extractor(conversation: str) -> List[Dict[str, Any]]:
    client = get_client()
    if client is None:
        print(" No OPENAI_API_KEY — skipping extraction")
        return []
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": conversation},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        memories = parsed.get("memories", [])
        return memories if isinstance(memories, list) else []
    except Exception as e:
        print(f" Extraction LLM call failed: {e}")
        return []


def _validate_memory(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    required = {"type", "key", "value", "subject", "confidence"}
    if not all(k in m for k in required):
        return None
    if m["type"] not in {"fact", "preference", "opinion", "event", "correction"}:
        return None
    try:
        conf = float(m["confidence"])
    except (TypeError, ValueError):
        return None
    if conf < 0.5:
        return None
    return {
        "type": m["type"],
        "key": normalize_key(m["key"]),
        "value": str(m["value"])[:1000],
        "subject": str(m.get("subject", "user"))[:80],
        "entities": [e.lower() for e in m.get("entities", []) if isinstance(e, str)][:20],
        "confidence": conf,
        "is_correction_of": m.get("is_correction_of"),
        "temporal": m.get("temporal", "current"),
    }


def extract_memories(
    turn_id: str,
    user_id: Optional[str],
    session_id: str,
    messages: List[Dict[str, Any]],
    timestamp: str,
) -> List[Dict[str, Any]]:
    # When user_id is None, scope memories to this session only
    # so per-session memory still works without a user identifier.
    scope_id = user_id if user_id else f"session:{session_id}"

    conversation = _format_conversation(messages)
    if not conversation.strip():
        return []

    raw_memories = _call_extractor(conversation)
    valid = [v for v in (_validate_memory(m) for m in raw_memories) if v]

    if not valid:
        return []

    stored = []
    conn = get_db()
    now = datetime.utcnow().isoformat()

    for mem in valid:
        decision = resolve_evolution(mem, scope_id)
        memory_id = str(uuid.uuid4())

        should_insert = apply_evolution(conn, mem, memory_id, decision, now)
        if not should_insert:
            continue

        supersedes_val = mem.get("_supersedes")

        conn.execute(
            """INSERT INTO memories
               (id, user_id, type, key, value, confidence, source_session,
                source_turn, created_at, updated_at, supersedes, active,
                subject, entities, temporal, is_correction_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (
                memory_id, scope_id, mem["type"], mem["key"], mem["value"],
                mem["confidence"], session_id, turn_id, now, now,
                supersedes_val, mem["subject"], json.dumps(mem["entities"]),
                mem["temporal"], mem["is_correction_of"],
            ),
        )

        index_memory_for_fts(conn, memory_id, scope_id, mem["value"], mem["entities"])
        stored.append({**mem, "id": memory_id})

    conn.commit()
    conn.close()
    print(f" Extracted {len(stored)} memories for scope {scope_id}")
    return stored