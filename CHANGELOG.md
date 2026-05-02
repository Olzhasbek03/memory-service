# CHANGELOG

## v1 — Working skeleton with real extraction

**What's in this version:**
- All 7 required endpoints implemented (health, turns, recall, search,
  user memories, delete session, delete user)
- SQLite database with persistent Docker volume
- OpenAI gpt-4o-mini for memory extraction
- Structured memory schema: type, key, value, confidence, supersedes, active
- Basic fact evolution: new facts supersede old ones for the same key
- Embedding-based recall using text-embedding-3-small
- Smoke test passes: ingested "moved to Berlin from NYC", recall correctly
  returns Berlin, memories endpoint shows structured fact with type=fact,
  key=location, confidence=0.95

**What's still weak:**
- Recall is pure cosine similarity — no keyword search yet
- No query rewriting
- No multi-hop recall
- No tests written yet
- Context assembly priority logic is basic

**Next:** Add hybrid retrieval (BM25 + embeddings) to improve keyword queries