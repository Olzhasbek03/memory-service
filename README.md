# Memory Service

A memory service for AI agents. See CHANGELOG.md for design iteration history.

## Quick start

```bash
docker compose up --build
```

Service runs on http://localhost:8080.

## Run tests

The service must be running before tests run.

```bash
# 1. Start service
docker compose up -d

# 2. Wait for health
until curl -sf http://localhost:8080/health; do sleep 1; done

# 3. Install test deps locally
pip3 install pytest httpx --break-system-packages

# 4. Run contract tests
python3 -m pytest tests/test_contract.py -v

# 5. Run recall quality fixture
python3 -m pytest tests/test_recall_quality.py -v -s
```

The recall quality test prints a full report and asserts ≥70% of
expected probes hit. Use this to measure regression after each
pipeline change.