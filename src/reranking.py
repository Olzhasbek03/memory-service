"""
LLM-based reranker for /recall candidates.
Takes top N candidates from hybrid retrieval, asks the LLM to rate
each one's relevance to the query in a single batch call.
Falls back to original order on any error.
"""
import json
from typing import List, Dict, Any
from openai import OpenAI
import os

MODEL = "gpt-4o-mini"


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()

RERANK_PROMPT = """You rerank memory candidates for relevance to a query.

Query: {query}

Candidates (numbered):
{candidates}

For each candidate, output a relevance score from 0.0 to 1.0:
- 1.0  = directly answers the query
- 0.7  = strongly relevant supporting context
- 0.4  = tangentially related
- 0.0  = irrelevant

Return JSON:
{{"scores": [{{"id": <number>, "score": <0.0-1.0>}}, ...]}}

Score every candidate. Return ONLY JSON."""


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    if not candidates:
        return []
if len(candidates) == 1:
        return candidates

    client = get_client()
    if client is None:
        return candidates[:top_n]

    listing = "\n".join(
        f"{i}. [{c.get('type','?')}/{c.get('key','?')}] {c['value'][:200]}"
        for i, c in enumerate(candidates)
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(
                query=query, candidates=listing)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        scores_data = json.loads(resp.choices[0].message.content)
        score_map = {int(s["id"]): float(s["score"]) for s in scores_data["scores"]}

        for i, c in enumerate(candidates):
            c["rerank_score"] = score_map.get(i, 0.0)

        for c in candidates:
            prior = c.get("final_score", 0.0)
            rr = c.get("rerank_score", 0.0)
            c["final_score"] = (rr * 100) + (prior * 0.1)

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_n]

    except Exception as e:
        print(f" Reranker failed, falling back: {e}")
        return candidates[:top_n]