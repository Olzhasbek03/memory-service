#!/usr/bin/env bash
# Restart persistence test — required by spec.
# Writes data, restarts the container, verifies recall still finds it.

set -e

if [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠️  OPENAI_API_KEY is required for this test."
  echo "The service boots without it, but extraction needs a key."
  exit 1
fi

BASE="http://localhost:8080"

echo "=== Setup: clean start ==="
docker compose down -v >/dev/null 2>&1 || true
docker compose up -d --build >/dev/null 2>&1
until curl -sf $BASE/health >/dev/null; do sleep 1; done
echo " service up"

echo "=== Write a memory ==="
curl -s -X POST $BASE/turns -H "Content-Type: application/json" \
  -d '{
    "session_id": "restart-session",
    "user_id": "restart-user",
    "messages": [{"role": "user", "content": "I live in Reykjavik."}],
    "timestamp": "2025-01-01T00:00:00Z",
    "metadata": {}
  }' >/dev/null
echo " memory written"

echo "=== Restart container ==="
docker compose restart memory-service >/dev/null
until curl -sf $BASE/health >/dev/null; do sleep 1; done
echo " service restarted"

echo "=== Recall after restart ==="
RESULT=$(curl -s -X POST $BASE/recall -H "Content-Type: application/json" \
  -d '{
    "query": "Where does the user live?",
    "session_id": "different-session",
    "user_id": "restart-user",
    "max_tokens": 256
  }')

if echo "$RESULT" | grep -qi "reykjavik"; then
  echo " PASS: persistence works — Reykjavik recalled after restart"
  exit 0
else
  echo " FAIL: Reykjavik not found after restart"
  echo "Response: $RESULT"
  exit 1
fi