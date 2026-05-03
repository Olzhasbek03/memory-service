"""
Recall quality fixture runner.

Loads fixtures/conversations.json, ingests all turns,
runs all probes, and computes:
  - Recall@expected: did expected keywords appear in /recall context?
  - Negative test: did "must_not_contain" stay absent?
  - Memory inspection: did /users/{id}/memories show structured data?

Run with: pytest tests/test_recall_quality.py -v -s
"""
import json
import os
import time
from pathlib import Path
import httpx
import pytest

BASE = "http://localhost:8080"
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "conversations.json"


def _load_scenarios():
    with open(FIXTURE_PATH) as f:
        return json.load(f)["scenarios"]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        yield c


@pytest.fixture(scope="module")
def ingested(client):
    """Wipe + ingest all fixture data once for the module."""
    scenarios = _load_scenarios()

    # Cleanup any prior runs
    for sc in scenarios:
        client.delete(f"/users/{sc['user_id']}")

    # Ingest
    for sc in scenarios:
        for turn in sc["turns"]:
            r = client.post("/turns", json={
                "session_id": turn["session_id"],
                "user_id": sc["user_id"],
                "messages": turn["messages"],
                "timestamp": turn["timestamp"],
                "metadata": {}
            })
            assert r.status_code == 201, f"Ingest failed for {sc['name']}"
    yield scenarios

    # Teardown
    for sc in scenarios:
        client.delete(f"/users/{sc['user_id']}")


def test_print_recall_quality_report(client, ingested):
    """
    Run every probe; print a report with hit/miss breakdown.
    Test 'passes' if at least 70% of probes hit expected keywords.
    """
    total_probes = 0
    hits = 0
    failures = []

    print("\n" + "=" * 70)
    print("RECALL QUALITY REPORT")
    print("=" * 70)

    for sc in ingested:
        print(f"\nScenario: {sc['name']}  (user={sc['user_id']})")
        for probe in sc["probes"]:
            total_probes += 1
            r = client.post("/recall", json={
                "query": probe["query"],
                "session_id": probe["session_id"],
                "user_id": sc["user_id"],
                "max_tokens": 512
            })
            ctx = r.json().get("context", "").lower()

            expected = probe.get("expected_keywords", [])
            must_not = probe.get("must_not_contain", [])

            expected_hits = [kw for kw in expected if kw.lower() in ctx]
            expected_misses = [kw for kw in expected if kw.lower() not in ctx]
            forbidden_hits = [kw for kw in must_not if kw.lower() in ctx]

            # An "expected" probe passes if every expected keyword found
            # AND no forbidden keyword found.
            expected_satisfied = (
                len(expected) == 0 or len(expected_misses) == 0
            )
            negative_satisfied = len(forbidden_hits) == 0

            passed = expected_satisfied and negative_satisfied
            if passed:
                hits += 1

            status = "✅ HIT " if passed else "❌ MISS"
            print(f"  {status}  Q: {probe['query']!r}")
            if expected:
                print(f"          expected: {expected}, found: {expected_hits}")
            if must_not and forbidden_hits:
                print(f"          forbidden present: {forbidden_hits}")
            if not passed:
                failures.append({
                    "scenario": sc["name"],
                    "query": probe["query"],
                    "expected_misses": expected_misses,
                    "forbidden_hits": forbidden_hits,
                    "context_snippet": ctx[:200],
                })

    # Memory inspection summary
    print("\n" + "─" * 70)
    print("STRUCTURED MEMORY CHECK")
    print("─" * 70)
    for sc in ingested:
        r = client.get(f"/users/{sc['user_id']}/memories")
        mems = r.json().get("memories", [])
        active = [m for m in mems if m.get("active")]
        inactive = [m for m in mems if not m.get("active")]
        print(f"  {sc['user_id']}: {len(active)} active, {len(inactive)} superseded")

    # Final score
    score = hits / total_probes if total_probes else 0.0
    print("\n" + "=" * 70)
    print(f"RECALL@expected: {hits}/{total_probes} = {score:.2%}")
    print("=" * 70)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f['scenario']}: {f['query']}")
            if f["expected_misses"]:
                print(f"    missing: {f['expected_misses']}")
            if f["forbidden_hits"]:
                print(f"    forbidden present: {f['forbidden_hits']}")

    # Pass threshold: 70%
    assert score >= 0.70, f"Recall quality {score:.2%} below 70% threshold"