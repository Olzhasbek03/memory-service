"""
Extraction pipeline: raw conversation → structured memories.

Design principles:
- Extract about the USER first; mark anything about other people clearly.
- Capture entities (names, places, orgs) for downstream multi-hop linking.
- Distinguish stable facts (lives in X) from events (going to X tonight)
  via the `type` and `confidence` fields.
- Detect corrections explicitly so the recall pipeline can prioritize them.
- Refuse to extract when nothing factual was said. Empty list is correct.
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .database import get_db

client = OpenAI()
EXTRACTION_MODEL = "gpt-4o-mini"

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
  "food_allergy", "hobby", "family_spouse", "preference_communication".
- key must be stable across mentions. Same topic always gets the same key.
- value is a complete sentence: "User works at Notion as a PM".

TYPE GUIDELINES
- fact: durable truth ("lives in Berlin", "has a dog named Biscuit")
- preference: stable like/dislike ("vegetarian", "prefers concise answers")
- opinion: stance that may evolve ("thinks TypeScript is overengineered")
- event: time-bound happening ("going to a wedding Saturday")
- correction: explicit fix ("actually, I meant Notion not Stripe")

SUBJECT
- "user" if the fact is about the user themselves
- "other:Alice" if about someone else they mentioned by name
- Do not store memories with subject "other:..." unless the relationship
  to the user is clear ("my wife Alice works at Stripe" → subject "other:Alice"
  AND a separate fact with subject "user", key "family_spouse",
  value "Spouse is Alice, who works at Stripe")

ENTITIES
- Names of people, pets, places, companies, products, technologies mentioned
- Lowercase, singular form: ["biscuit", "berlin", "notion", "typescript"]
- Used for cross-memory linking. Be liberal but accurate.

CONFIDENCE
- 0.95+: explicit statements ("I live in Berlin")
- 0.7-0.9: implicit / inferred ("walking Biscuit" → has dog Biscuit)
- 0.5-0.7: ambiguous ("might move to Berlin")
- Below 0.5: do not extract

TEMPORAL
- stable: doesn't change week to week (allergies, name, hometown)
- current: true now but may change (job, location, relationship status)
- transient: time-bound (dinner plans, what they did this morning)

GRANULARITY
- Split lists. "I love Python and Go" → two preference memories.
- Combine inseparable details. "Has a dog Biscuit, golden retriever, age 5"
  → ONE memory: value="Has a golden retriever named Biscuit, age 5"

DO NOT
- Do not extract assistant statements as user facts.
- Do not extract from greetings, acknowledgments, or chitchat.
- Do not invent details not stated. If unsure, use lower confidence.
- Do not extract the assistant's questions back to the user as facts.

EXAMPLES

Conversation:
USER: I just moved to Berlin from NYC last month. My dog Biscuit hated the flight.
ASSISTANT: That sounds rough! How is Biscuit settling in?

Output:
{"memories": [
  {"type":"fact","key":"location","value":"User recently moved to Berlin from NYC","subject":"user","entities":["berlin","nyc"],"confidence":0.97,"is_correction_of":null,"temporal":"current"},
  {"type":"fact","key":"pet","value":"User has a dog named Biscuit","subject":"user","entities":["biscuit"],"confidence":0.95,"is_correction_of":null,"temporal":"stable"}
]}

Conversation:
USER: Sorry, I said Stripe earlier — I actually work at Notion.
ASSISTANT: Got it, thanks for clarifying.

Output:
{"memories": [
  {"type":"correction","key":"employment","value":"User works at Notion (not Stripe as previously said)","subject":"user","entities":["notion","stripe"],"confidence":0.99,"is_correction_of":"employment","temporal":"current"}
]}

Conversation:
USER: Hey thanks!
ASSISTANT: No problem!

Output:
{"memories": []}
"""


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _call_extractor(conversation: str) -> List[Dict[str, Any]]:
    """Call the LLM. Returns a list of raw memory dicts. Never raises."""
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
        if not isinstance(memories, list):
            return []
        return memories
    except Exception as e:
        print(f"⚠️  Extraction LLM call failed: {e}")
        return []


def _validate_memory(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Defensive validation. Drop malformed memories instead of crashing."""
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
        "key": str(m["key"])[:80],
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
    """
    Run extraction and persist memories. Returns the list of stored memory dicts
    (with their assigned ids) so the caller can embed them.
    """
    if not user_id:
        return []  # we only store memories scoped to a user

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
        # Subject filter: we only store user-subject memories in the main table.
        # Other-subject memories are stored too but marked, so they don't pollute
        # "stable user facts" priority in /recall.
        memory_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO memories
            (id, user_id, type, key, value, confidence, source_session,
             source_turn, created_at, updated_at, supersedes, active,
             subject, entities, temporal, is_correction_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?, ?, ?, ?)
            """,
            (
                memory_id, user_id, mem["type"], mem["key"], mem["value"],
                mem["confidence"], session_id, turn_id, now, now,
                mem["subject"], json.dumps(mem["entities"]),
                mem["temporal"], mem["is_correction_of"],
            ),
        )
        stored.append({**mem, "id": memory_id})

    conn.commit()
    conn.close()
    print(f"✅ Extracted {len(stored)} memories for user {user_id}")
    return stored