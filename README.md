# Memory Service
A memory service for AI agents. Ingests conversation turns, extracts structured memories with an LLM, persists them to SQLite + FTS5, and serves recall queries with a hybrid retrieval pipeline (embeddings + BM25 fused with RRF, then LLM-reranked).

Built over 2 days for the Higgsfield AI Engineer take-home.

---

## Quick start

```bash
git clone <this repo> memory-service
cd memory-service

# Optional: set OPENAI_API_KEY for full extraction and recall quality.
# Without it the service still boots and the HTTP contract works,
# but LLM-based extraction, embeddings, query rewriting, and
# reranking will degrade gracefully to no-ops.
# export OPENAI_API_KEY=sk-...

docker compose up --build

```

Service is on `http://localhost:8080`. See `.env.example` for env vars (`OPENAI_API_KEY` optional for boot but required for full recall quality, `MEMORY_AUTH_TOKEN` optional).

To verify:

```bash
curl http://localhost:8080/health
```

---

## Architecture

```
                       POST /turns
                            │
                            ▼
           ┌───────────────────────────────────┐
           │   Persist raw turn (SQLite)       │
           └───────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼                               ▼
    ┌──────────────┐              ┌──────────────────┐
    │ Embed turn   │              │ LLM extraction   │
    │ text         │              │ (gpt-4o-mini)    │
    └──────────────┘              └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Validate + key   │
                                  │ normalization    │
                                  └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Evolution check  │
                                  │ (insert / super  │
                                  │  sede / skip)    │
                                  └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Insert + embed + │
                                  │ FTS5 index       │
                                  └──────────────────┘


                       POST /recall
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Query rewrite      │
                  │  (LLM, classifies   │
                  │   multi-hop)        │
                  └─────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌────────────────┐         ┌──────────────────┐
     │ Embedding      │         │ BM25 (FTS5)      │
     │ search top 30  │         │ search top 30    │
     └────────────────┘         └──────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  ┌─────────────────────┐
                  │  RRF fusion (k=60)  │
                  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Multi-hop pass     │
                  │  (if classified)    │
                  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Type/conf priority │
                  │  scoring            │
                  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  LLM rerank top 20  │
                  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Assemble under     │
                  │  token budget       │
                  └─────────────────────┘
```

The whole thing is one Python service (FastAPI + Uvicorn) with SQLite as the only datastore. No external vector DB, no Redis, no message queue. Single container.

---

## Backing store: SQLite + FTS5

I picked SQLite for three reasons:

1. **One file, persistence.** A Docker named volume mounts to `/data/memory.db`. Restarts are invisible to clients — verified by a contract test.
2. **FTS5 is built in.** I needed BM25 for hybrid retrieval (more on that below), and SQLite's FTS5 virtual table gives me BM25 in the same file as the canonical store. No separate inverted index, no two-system consistency problem. The memories table and the BM25 index can't drift apart because they share the same WAL.
3. **It's small enough to defend.** A take-home is the wrong place to introduce Postgres + pgvector + Redis. SQLite handles a few concurrent sessions and tens of thousands of memories without breaking a sweat. If this grew past that I'd port to Postgres + pgvector, keep the same schema.

Embeddings are stored as JSON-serialized float arrays in a separate `embeddings` table. I don't use a real ANN index — at the scale this service is meant to handle (a few thousand memories per user max), a linear scan with cosine is fast enough and avoids the whole "rebuild the index after every insert" problem. If memory count grew large I'd add `sqlite-vss` or move embeddings to a proper vector store.

---

## Extraction pipeline

Raw conversation turns become structured memories via one `gpt-4o-mini` call per turn. Output is enforced JSON with these fields:

```
{
  "type":             "fact" | "preference" | "opinion" | "event" | "correction",
  "key":              "<canonical topic slug>",
  "value":            "<full sentence in third person>",
  "subject":          "user" | "other:<name>",
  "entities":         ["lowercase names"],
  "confidence":       0.0-1.0,
  "is_correction_of": "<key>" | null,
  "temporal":         "stable" | "current" | "transient"
}
```

Things I deliberately put in the extraction prompt:

- **Few-shot examples** for the tricky cases — corrections, implicit facts ("walking Biscuit" → has dog Biscuit), and the no-extraction case ("hey thanks!" → empty list).
- **Granularity rules.** "I love Python and Go" must split into two preference memories. "Has a golden retriever named Biscuit, age 5" must NOT split — those details are inseparable.
- **Subject distinction.** Facts about the user vs facts about people the user mentioned. Stops "my wife works at Stripe" from being stored as a user fact.
- **Temporal classification.** Stable facts (allergies, hometown) get extra protection during supersession.

After the LLM returns, every memory passes through `_validate_memory()` which drops anything malformed and runs `normalize_key()` on the key. Key normalization is the unsexy but critical bit — see "Fact evolution" below.

What I miss: I don't model the *opinion arc* problem from the spec. "I love TypeScript" → "TypeScript generics are annoying" → "TypeScript is fine for big projects" is currently treated as full supersession, with the latest opinion winning. A graduated-confidence model would be better but didn't fit in the time I had. Documented as future work.

---

## Recall strategy

Five stages. Each one does something specific:

### 1. Query rewriting

Vague queries ("what does she do?") get rewritten by an LLM into clearer factual form ("user's job employer profession"). The same call also classifies whether the query is **multi-hop** — i.e., identifies the subject by one attribute and asks about a different attribute. Pet+location, employment+pet, etc.

### 2. Hybrid retrieval (the big one)

I run two retrievers in parallel:

- **Embedding search** with `text-embedding-3-small`, cosine similarity over active memories.
- **BM25** via SQLite FTS5 over `value + entities` text.

Their scores aren't comparable (cosine is 0–1, BM25 is unbounded), so I fuse with **Reciprocal Rank Fusion**:

```
RRF_score(doc) = Σ 1 / (k + rank_in_each_list)        # k = 60
```

This is the literature default from Cormack et al. 2009. It uses ranks not scores, so the scale incompatibility doesn't matter. A document ranked top-1 by both retrievers wins. A document only one retriever found ranks lower.

Why hybrid is non-negotiable: pure embeddings miss "what's Biscuit's name?" because "Biscuit" carries weak signal in vibe-space — the question is generically about pets. Pure BM25 misses "where does she work?" because no memory contains the word "work." You need both.

### 3. Multi-hop expansion

If query rewrite flagged the query as multi-hop, I run a second retrieval pass anchored on entities pulled from the first-pass top results.

Example: query is "what city does the user with dog Biscuit live in?". First pass finds the pet memory (entities: `["biscuit"]`). I then build a second query: `"biscuit location city country residence lives moved"` (the entity + dimension-boosting keywords for the asked dimension). Second pass finds the Berlin memory.

Cross-confirmed results (in both passes) get a 1.2x score boost.

### 4. Priority scoring

After fusion + multi-hop, I score each candidate as:

```
final_score = (RRF * 100) + (type_weight * 0.5) + (confidence * 10)
```

Type weights: `correction (100) > fact (90) > preference (80) > opinion (60) > event (50)`. Corrections are highest because they encode "this is the latest truth, override stale data."

### 5. LLM rerank

Top 20 candidates go to `gpt-4o-mini` in a single batch call. It returns 0–1 relevance scores per candidate. Final ordering blends these (heavy weight on rerank, prior pipeline scores as tiebreaker). Falls back silently to un-reranked order on any error.

### Context assembly under token budget

Three sections in priority order, with `tiktoken` measuring before each line is added:

1. **Stable user facts** — `subject="user"`, type in `{fact, preference, correction}`. These are "always-on" things the agent should know.
2. **Other relevant memories** — opinions, events, anything else that scored high.
3. **Recent session turns** — last 5 turns from the same session, first user message of each. Tail context.

When the budget is tight, sections drop in reverse order. The reasoning: an agent that doesn't know the user's allergies fails worse than one that doesn't know what the user said two turns ago.

---

## Fact evolution

The eval explicitly tests Stripe → Notion. 

A second nuance: not all keys behave the same way. 
**Singular keys**
(employment, location, family_spouse, relationship_status) only have one active value at a time — a new value supersedes the old one.
**Additive keys** 
(pet, hobby, language, skill, food_allergy, family_child) allow multiple active values: a user can have two pets or speak three languages. For additive keys, a new memory only supersedes an existing one if they share an entity (same pet name, same language). Otherwise it inserts as a new memory alongside.

### Key normalization

The LLM is not deterministic about keys. One run produces `"job"`, another `"employment"`, another `"employer"`. I maintain a hand-curated map collapsing 25+ variants to canonical topics (`job`, `work`, `employer`, `occupation`, `profession`, `career`, `company`, `workplace` → `employment`). Without this, supersession doesn't fire and both Stripe and Notion stay active.

### Three-signal detection at write time

Detection happens in `extraction.py`'s loop, before insert. Each new memory goes through `resolve_evolution()` which returns one of: `INSERT`, `INSERT_AND_SUPERSEDE`, `SKIP_DUPLICATE`. The decision ladder:

1. **Explicit correction** (type="correction" or `is_correction_of` set) → always supersede the matching key. Cheap.
2. **Same canonical key + same subject** → if the new value isn't an exact duplicate, supersede the most recent matching memory. Still cheap.
3. **Stable temporal facts** require LLM-confirmed contradiction (>0.8 confidence) before superseding. Allergies and hometowns shouldn't be overwritten on a vague signal.
4. **Default** → fresh memory, no conflict.

I resolve at write time — not read time — because doing this on every recall would be slow and expensive.

### Supersession chain, not deletion

Old memories get `active=0`, never deleted. The new memory's `supersedes` field points at the old id. `/users/{user_id}/memories` returns the full chain so a reviewer (or future debugging) can see the history.

---

## Tradeoffs I made

| Optimized for | Gave up |
|---|---|
| Single-container deploy | Horizontal scalability |
| Synchronous correctness on /turns | /turns latency (it does extraction + embedding inline) |
| Recall quality | Latency (5 stages, multiple LLM calls per /recall) |
| Robustness over throughput | High QPS — there are no connection pools, no batching |
| Boring tech that works | Cool tech (no graph DB, no real ANN index) |

Concrete numbers from my fixture:
- /turns latency: ~2-4s (one LLM extraction call + N embedding calls)
- /recall latency: ~1-3s (rewrite + retrieve + rerank, 2-3 LLM calls)

The spec allows 60s on /turns and "reasonable time" on /recall. I'm well inside both, so I optimized for quality not speed.

---

## Failure modes

- **No OpenAI API key.** Service still boots. /turns accepts the turn but extraction returns empty (no memories). /recall returns whatever raw turns exist. Logged as warnings, no crashes.
- **OpenAI rate-limited or down.** Each LLM call (extraction, rewrite, judge, rerank) is wrapped in try/except. Recall degrades to BM25-only retrieval; extraction returns empty; supersession defaults to insert.
- **Slow disk.** SQLite handles this fine. Ingestion blocks until commit.
- **Container restart mid-write.** SQLite is ACID, so partial writes don't corrupt. The next request sees consistent state.
- **Malformed input.** Pydantic validators catch bad shapes and return 422. Null bytes in content are stripped (FTS5 chokes on them). Huge messages are truncated to 50KB. Never returns 500 on adversarial input — verified by `tests/test_robustness.py`.
- **Cold session.** /recall on an unknown user returns `{"context": "", "citations": []}` with 200, not an error.

---

## How to run the tests

The service must be running.

```bash
docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done

pip3 install --user pytest httpx

# Contract tests (12 tests — endpoints, shapes, status codes)
python3 -m pytest tests/test_contract.py -v

# Robustness tests (13 tests — adversarial input)
python3 -m pytest tests/test_robustness.py -v

# Recall quality fixture (5 scenarios, 7 probes, prints a report)
python3 -m pytest tests/test_recall_quality.py -v -s

# Restart persistence test (required by spec)
bash tests/test_persistence_restart.sh
```

The recall quality runner ingests the fixture in `fixtures/conversations.json`, runs every probe against /recall, prints per-probe hit/miss, and asserts ≥70% recall. Current score on this implementation: **7/7 = 100%**. I used this fixture as my main feedback loop while iterating — every layer in the CHANGELOG was validated against it.

---

## What I'd do with another two days

- **Opinion arc tracking.** Currently treats sentiment shifts as full supersession. Should be graduated confidence ("dislikes TS more strongly now" not "no longer dislikes TS").
- **Cross-key semantic supersession.** If "moved to Berlin" arrives without entities matching "lives in NYC", they stay both active. I have a sketched-in `_semantic_neighbor` path in evolution.py but didn't enable it because the false-positive rate on my fixture was too high.
- **Real ANN index.** Linear cosine scan is fine at this scale. At 100K+ memories per user it isn't.
- **Caching.** OpenAI's prompt caching would cut extraction cost ~50%. Trivial to add.
- **Batched embedding for memories.** I currently embed memories one at a time. Batch API would halve latency on multi-memory turns.
- **Streaming /recall.** The reranker call adds noticeable latency. Returning fact lines as they're decided would feel snappier.

---

## File map

```
src/
├── main.py             # FastAPI app, endpoints, exception handlers, auth
├── models.py           # Pydantic request/response shapes with size limits
├── database.py         # SQLite init, migrations, FTS5 setup
├── extraction.py       # LLM extraction + key normalization + evolution wiring
├── evolution.py        # Supersession decision ladder (INSERT / SUPERSEDE / SKIP)
├── retrieval.py        # Embedding + BM25 + RRF fusion
├── query_processing.py # Query rewriting + multi-hop expansion
├── reranking.py        # LLM reranker
└── recall.py           # Top-level recall orchestration + context assembly

tests/
├── test_contract.py        # HTTP contract: shapes, status codes, isolation
├── test_robustness.py      # Adversarial input: malformed, unicode, oversized
└── test_recall_quality.py  # Fixture-driven recall metric

fixtures/
└── conversations.json  # 5 scenarios with probes — primary iteration loop
```
