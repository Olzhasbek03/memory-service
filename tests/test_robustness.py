"""
Robustness tests: malformed input, edge cases, large payloads.
The service must NEVER 500 on these — only 4xx (validation) or 200 (handled).
"""
import uuid
import httpx
import pytest

BASE = "http://localhost:8080"


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        yield c


def _new_user():
    return f"robust-{uuid.uuid4().hex[:8]}"


# ─────────────────────────────────────────────
# Malformed payloads
# ─────────────────────────────────────────────

def test_empty_body(client):
    r = client.post("/turns", content=b"")
    assert 400 <= r.status_code < 500


def test_invalid_json(client):
    r = client.post(
        "/turns",
        content=b"this is not json {{{",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= r.status_code < 500


def test_wrong_types(client):
    r = client.post("/turns", json={
        "session_id": 12345,                # should be string
        "user_id": ["not", "a", "string"],
        "messages": "should be a list",
        "timestamp": True,
    })
    assert 400 <= r.status_code < 500


def test_missing_required_fields(client):
    r = client.post("/turns", json={"messages": []})
    assert 400 <= r.status_code < 500


def test_invalid_role(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": "s1",
        "user_id": user,
        "messages": [{"role": "wizard", "content": "hi"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {},
    })
    assert 400 <= r.status_code < 500


# ─────────────────────────────────────────────
# Adversarial content
# ─────────────────────────────────────────────

def test_null_bytes_in_content(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": "s1",
        "user_id": user,
        "messages": [{"role": "user", "content": "hello\x00world\x00"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {},
    })
    assert r.status_code == 201
    client.delete(f"/users/{user}")


def test_huge_message_truncated(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": "s1",
        "user_id": user,
        "messages": [{"role": "user", "content": "A" * 200_000}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {},
    })
    # Either accepted (truncated) or 4xx — never 500
    assert r.status_code < 500
    if r.status_code == 201:
        client.delete(f"/users/{user}")


def test_emoji_and_rtl(client):
    user = _new_user()
    r = client.post("/turns", json={
        "session_id": "s1",
        "user_id": user,
        "messages": [{"role": "user",
                      "content": "🐶 العربية 中文 русский 🎉🎊"}],
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {},
    })
    assert r.status_code == 201
    client.delete(f"/users/{user}")


# ─────────────────────────────────────────────
# Recall edge cases
# ─────────────────────────────────────────────

def test_recall_empty_query(client):
    r = client.post("/recall", json={
        "query": "",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 256,
    })
    assert 400 <= r.status_code < 500


def test_recall_huge_max_tokens(client):
    r = client.post("/recall", json={
        "query": "test",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 1_000_000,
    })
    assert 400 <= r.status_code < 500


def test_recall_negative_max_tokens(client):
    r = client.post("/recall", json={
        "query": "test",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": -5,
    })
    assert 400 <= r.status_code < 500


# ─────────────────────────────────────────────
# Cleanup robustness
# ─────────────────────────────────────────────

def test_delete_nonexistent_user(client):
    r = client.delete("/users/never-existed-12345")
    assert r.status_code == 204


def test_delete_nonexistent_session(client):
    r = client.delete("/sessions/never-existed-12345")
    assert r.status_code == 204


def test_get_memories_nonexistent_user(client):
    r = client.get("/users/never-existed-67890/memories")
    assert r.status_code == 200
    assert r.json()["memories"] == []