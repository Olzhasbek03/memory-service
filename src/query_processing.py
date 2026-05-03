"""
Query processing pipeline.

Two-step process before retrieval:
1. rewrite_query()    — clarify, expand pronouns, generate keywords.
2. classify_multihop() — does this query need entity-anchored second pass?

Then after first-pass retrieval:
3. multihop_expand()  — use top-1 result's entities to build a second query
                         targeting the *complement* dimensions.
"""
import json
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

REWRITE_PROMPT = """You optimize queries for a personal-memory retrieval system.

Output JSON:
{
  "rewritten": "<clearer factual query in third person>",
  "keywords": ["<term1>", "<term2>", ...],
  "is_multihop": true | false,
  "asked_dimension": "<one of: pet | location | employment | family | preference | opinion | event | hobby | other>",
  "anchor_dimension": "<the dimension used to identify the user, or null>"
}

GUIDELINES
- rewritten: third-person factual phrasing. Expand pronouns. Drop filler.
  "what does she do?" → "user's job employer profession"
  "where do they live?" → "user residence city country"

- keywords: 2-6 specific terms that would appear in a memory matching this query.

- is_multihop: TRUE iff the query identifies the subject by one attribute
  while asking about a different attribute.
  Examples:
    "What city does the user with the dog Biscuit live in?" → TRUE
       (anchor=pet, asked=location)
    "Where does the user live?" → FALSE
       (no anchor, just asked dimension)
    "Does the user who works at Notion have any pets?" → TRUE
       (anchor=employment, asked=pet)

- asked_dimension: what is the question asking ABOUT?
- anchor_dimension: what attribute identifies the subject (null if none)?

Return ONLY the JSON object."""


def rewrite_query(query: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)
        return {
            "rewritten":         out.get("rewritten", query),
            "keywords":          out.get("keywords", []),
            "is_multihop":       bool(out.get("is_multihop", False)),
            "asked_dimension":   out.get("asked_dimension", "other"),
            "anchor_dimension":  out.get("anchor_dimension"),
        }
    except Exception as e:
        print(f"⚠️  rewrite_query failed: {e}")
        return {
            "rewritten": query, "keywords": [],
            "is_multihop": False, "asked_dimension": "other",
            "anchor_dimension": None,
        }


def _entities_from(results: List[Dict[str, Any]]) -> List[str]:
    """Pull entity tags from a list of memory rows."""
    out = set()
    for r in results:
        raw = r.get("entities", "[]")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []
        else:
            parsed = raw or []
        for e in parsed:
            if isinstance(e, str) and e.strip():
                out.add(e.strip().lower())
    return list(out)


# Keywords associated with each dimension — used to bias the second-pass query
# toward the asked dimension.
DIMENSION_BOOSTERS = {
    "pet":         "pet dog cat animal",
    "location":    "location city country residence lives moved",
    "employment":  "job employer company works profession role title",
    "family":      "family spouse partner child sibling parent",
    "preference":  "prefers likes dislikes favorite",
    "opinion":     "thinks believes feels opinion",
    "event":       "event meeting trip plan upcoming",
    "hobby":       "hobby interest activity",
    "other":       "",
}


def multihop_expand(
    user_id: str,
    rewrite_meta: Dict[str, Any],
    first_pass_results: List[Dict[str, Any]],
    hybrid_search_fn,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build an entity-anchored second-pass query and run it.
    
    The second-pass query combines:
      - entities pulled from the first-pass top results (anchor)
      - dimension-boosting keywords for the *asked* dimension
    """
    if not rewrite_meta["is_multihop"]:
        return []

    # Use top 3 first-pass results — likely the anchor matches
    anchor_entities = _entities_from(first_pass_results[:3])
    if not anchor_entities:
        return []

    asked = rewrite_meta.get("asked_dimension", "other")
    boost = DIMENSION_BOOSTERS.get(asked, "")
    second_query = f"{' '.join(anchor_entities)} {boost}".strip()

    print(f"🔀 Multi-hop second pass: anchors={anchor_entities} asked={asked}")
    print(f"🔀 Second query: '{second_query}'")

    second = hybrid_search_fn(user_id, second_query, top_n=top_n)

    # Deduplicate against first pass; tag the new ones as multi-hop additions
    seen = {r["id"] for r in first_pass_results}
    extras = []
    for r in second:
        if r["id"] in seen:
            # Already in first pass — boost its score (cross-confirmed)
            for fp in first_pass_results:
                if fp["id"] == r["id"]:
                    fp["rrf_score"] = fp.get("rrf_score", 0) + r.get("rrf_score", 0) * 0.5
                    fp["multihop_confirmed"] = True
        else:
            r["multihop_added"] = True
            extras.append(r)

    return extras