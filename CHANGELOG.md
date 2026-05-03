# CHANGELOG
## v3 — Query rewriting + entity-anchored multi-hop recall

**What changed:**
- Pre-retrieval LLM call rewrites queries into third-person factual form
  and classifies whether the query is multi-hop.
- Multi-hop queries trigger a second retrieval pass anchored on entities
  from the first pass, boosted toward the asked dimension.
- Cross-confirmed results get a 1.2x score boost.

**Why:** Multi-hop questions like "what city does the user with dog
Biscuit live in?" share zero tokens with the location memory. First-pass
retrieval lands on the pet memory; we use Biscuit as a re-query anchor
with location-dimension keywords to reach the Berlin memory.

**Result:** Canonical multi-hop test passes — query about Biscuit's city
correctly returns BOTH pet memory (Biscuit) AND location memory (Berlin)
in a single recall response.

**Next:** Fact evolution — semantic supersession that handles
"I work at Stripe" → "I work at Notion" across sessions.
## v2 — Hybrid retrieval (BM25 + embeddings + RRF) + structured extraction

**What changed:**
- Extraction now produces typed memories with: subject, entities, temporal,
  is_correction_of fields. Entity tagging enables future multi-hop linking.
- Added BM25 via SQLite FTS5 alongside embedding search.
- Fused both retrievers with Reciprocal Rank Fusion (k=60).
- /recall now assembles context in priority tiers:
  corrections → facts/preferences → opinions/events → recent turns.
- Output format matches spec example exactly.
- Schema migration logic: additive ALTER TABLE on startup so restarts
  don't require wiping the volume.

**Why hybrid:** Pure embeddings missed keyword-anchored queries like
"what's Biscuit's name" where exact token match matters more than
semantic similarity. FTS5 BM25 catches exact tokens; embeddings cover
paraphrase. RRF fuses both without needing comparable score scales.

**Smoke test result:** Both Berlin (location) AND Biscuit (pet) correctly
extracted with proper types, confidence, entities, and temporal fields.
/recall returns spec-matching formatted context.

**Next:** Query rewriting + multi-hop recall for questions where the
relevant entity isn't in the literal query text.


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