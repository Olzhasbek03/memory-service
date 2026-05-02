import json
import uuid
from datetime import datetime
from openai import OpenAI
from .database import get_db

client = OpenAI()  # reads OPENAI_API_KEY from environment automatically

EXTRACTION_PROMPT = """You are a memory extraction system. Read the conversation and extract structured facts about the user.

For each fact, output a JSON object with these fields:
- type: one of "fact", "preference", "opinion", "event"
- key: a short topic label like "employment", "location", "pet", "food_preference", "hobby"
- value: the full fact as a clear sentence
- confidence: 0.0 to 1.0 (how certain are you this is a stable fact?)

Rules:
- Extract ONLY facts about the USER (not the assistant)
- Include implicit facts ("I walked Biscuit" → user has a dog named Biscuit)
- Include corrections ("actually I meant..." → extract the corrected fact)
- For temporary events (user is going to a movie tonight), use type "event" and low confidence
- For stable facts (user lives in Berlin), use type "fact" and high confidence
- If nothing factual is said, return an empty list

Return ONLY a JSON array, no other text. Example:
[
  {"type": "fact", "key": "location", "value": "Lives in Berlin, moved from NYC", "confidence": 0.95},
  {"type": "preference", "key": "food", "value": "Vegetarian, allergic to shellfish", "confidence": 0.99},
  {"type": "fact", "key": "pet", "value": "Has a dog named Biscuit", "confidence": 0.95}
]"""

def extract_memories(turn_id: str, user_id: str, session_id: str, messages: list, timestamp: str):
    """Extract structured memories from a conversation turn."""
    if not user_id:
        return  # can't store user memories without a user_id

    # Format messages for the LLM
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
    ])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract facts from this conversation:\n\n{conversation_text}"}
            ],
            temperature=0  # deterministic output
        )

        raw = response.choices[0].message.content.strip()
        extracted = json.loads(raw)

    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
        return

    if not extracted:
        return

    conn = get_db()
    now = datetime.utcnow().isoformat()

    for item in extracted:
        memory_type = item.get("type", "fact")
        key = item.get("key", "unknown")
        value = item.get("value", "")
        confidence = item.get("confidence", 1.0)

        if not value:
            continue

        # Check if we already have an active memory with this key for this user
        existing = conn.execute("""
            SELECT id, value FROM memories
            WHERE user_id = ? AND key = ? AND active = 1
        """, (user_id, key)).fetchone()

        if existing:
            # Mark old memory as superseded
            conn.execute("""
                UPDATE memories SET active = 0, updated_at = ?
                WHERE id = ?
            """, (now, existing["id"]))
            supersedes_id = existing["id"]
            print(f"📝 Superseding old memory: '{existing['value']}' → '{value}'")
        else:
            supersedes_id = None

        # Insert the new memory
        memory_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO memories
            (id, user_id, type, key, value, confidence, source_session,
             source_turn, created_at, updated_at, supersedes, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (memory_id, user_id, memory_type, key, value, confidence,
              session_id, turn_id, now, now, supersedes_id))

    conn.commit()
    conn.close()
    print(f"✅ Extracted {len(extracted)} memories for user {user_id}")