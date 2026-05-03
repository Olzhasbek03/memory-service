"""
Contract tests against a running memory service at http://localhost:8080.

Run with: docker compose up -d, then pytest tests/test_contract.py
These tests verify HTTP shapes, status codes, and persistence.
"""
import json
import time
import uuid
import pytest
import httpx

BASE = "http://localhost:8080"
TIMEOUT = 60.0


def _new_user():
    return f"test-user-{uuid.uuid4().hex[:8]}"


def _new_session():
    return f"test-session-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as c:
        yield c


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


# ─────────────────────────────────────────────
# Contract roundtrip
# ─────────────────────────────────────────────

def test_turns_returns_201_with_id(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user,
        "messages": [
            {"role": "user", "content": "Hello, I'm a test user."},
            {"role": "assistant", "content": "Hi!"}
        ],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    assert r.status_code == 201
    body = r.json()
    assert "id" in body
    assert isinstance(body["id"], str)
    # Cleanup
    client.delete(f"/users/{user}")


def test_recall_returns_correct_shape(client):
    user = _new_user()
    sess = _new_session()
    client.post("/turns", json={
        "session_id": sess,
        "user_id": user,
        "messages": [
            {"role": "user", "content": "I live in Paris."},
            {"role": "assistant", "content": "Nice!"}
        ],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    r = client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": sess,
        "user_id": user,
        "max_tokens": 256
    })
    assert r.status_code == 200
    body = r.json()
    assert "context" in body
    assert "citations" in body
    assert isinstance(body["context"], str)
    assert isinstance(body["citations"], list)
    client.delete(f"/users/{user}")


def test_recall_on_cold_session_returns_empty_not_error(client):
    """No data for this user — must return 200 with empty context, not error."""
    r = client.post("/recall", json={
        "query": "anything",
        "session_id": "no-such-session",
        "user_id": "no-such-user",
        "max_tokens": 256
    })
    assert r.status_code == 200
    body = r.json()
    assert body["context"] == "" or isinstance(body["context"], str)
    assert isinstance(body["citations"], list)


def test_search_returns_correct_shape(client):
    user = _new_user()
    sess = _new_session()
    client.post("/turns", json={
        "session_id": sess,
        "user_id": user,
        "messages": [{"role": "user", "content": "Hello world"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    r = client.post("/search", json={
        "query": "hello",
        "session_id": sess,
        "user_id": user,
        "limit": 5
    })
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    client.delete(f"/users/{user}")


def test_users_memories_returns_structured_data(client):
    user = _new_user()
    client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user,
        "messages": [
            {"role": "user", "content": "I have a dog named Rex."},
            {"role": "assistant", "content": "Cute!"}
        ],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    r = client.get(f"/users/{user}/memories")
    assert r.status_code == 200
    body = r.json()
    assert "memories" in body
    if body["memories"]:
        m = body["memories"][0]
        # Memories must be structured, not raw message text
        assert "type" in m
        assert "key" in m
        assert "value" in m
        assert "active" in m
    client.delete(f"/users/{user}")


# ─────────────────────────────────────────────
# Cleanup endpoints
# ─────────────────────────────────────────────

def test_delete_session_returns_204(client):
    user = _new_user()
    sess = _new_session()
    client.post("/turns", json={
        "session_id": sess,
        "user_id": user,
        "messages": [{"role": "user", "content": "test"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    r = client.delete(f"/sessions/{sess}")
    assert r.status_code == 204
    client.delete(f"/users/{user}")


def test_delete_user_returns_204(client):
    user = _new_user()
    client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user,
        "messages": [{"role": "user", "content": "test"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    r = client.delete(f"/users/{user}")
    assert r.status_code == 204
    # Verify cleanup
    r2 = client.get(f"/users/{user}/memories")
    assert r2.json()["memories"] == []


# ─────────────────────────────────────────────
# Concurrent sessions don't bleed between users
# ─────────────────────────────────────────────

def test_concurrent_users_no_bleed(client):
    user_a = _new_user()
    user_b = _new_user()

    client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user_a,
        "messages": [{"role": "user", "content": "I work at OpenAI."}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user_b,
        "messages": [{"role": "user", "content": "I work at Anthropic."}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })

    a_recall = client.post("/recall", json={
        "query": "Where does the user work?",
        "session_id": "any",
        "user_id": user_a,
        "max_tokens": 256
    }).json()
    b_recall = client.post("/recall", json={
        "query": "Where does the user work?",
        "session_id": "any",
        "user_id": user_b,
        "max_tokens": 256
    }).json()

    assert "OpenAI" in a_recall["context"] or "openai" in a_recall["context"].lower()
    assert "Anthropic" in b_recall["context"] or "anthropic" in b_recall["context"].lower()
    # Crucially, neither leaks the other
    assert "Anthropic" not in a_recall["context"]
    assert "OpenAI" not in b_recall["context"]

    client.delete(f"/users/{user_a}")
    client.delete(f"/users/{user_b}")


# ─────────────────────────────────────────────
# Malformed input doesn't crash
# ─────────────────────────────────────────────

def test_malformed_turn_returns_4xx(client):
    r = client.post("/turns", json={"missing": "everything"})
    assert 400 <= r.status_code < 500


def test_unicode_handled(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user,
        "messages": [
            {"role": "user", "content": "I love 日本 and 🍕 and ñoño café"},
            {"role": "assistant", "content": "Cool!"}
        ],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    assert r.status_code == 201
    client.delete(f"/users/{user}")


def test_empty_messages_handled(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": _new_session(),
        "user_id": user,
        "messages": [],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {}
    })
    # Either 201 (accepted, nothing extracted) or 4xx — must not crash
    assert r.status_code < 500
    if r.status_code == 201:
        client.delete(f"/users/{user}")