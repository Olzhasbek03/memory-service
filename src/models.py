from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

# Size caps prevent OOM and runaway LLM costs on adversarial input.
MAX_MESSAGE_LEN = 50_000        # 50KB per message
MAX_MESSAGES_PER_TURN = 50      # spec allows multi-message turns
MAX_QUERY_LEN = 4_000
MAX_METADATA_BYTES = 10_000


class Message(BaseModel):
    role: str
    content: str = ""
    name: Optional[str] = None

    @validator("role")
    def _role_ok(cls, v):
        if v not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"invalid role: {v}")
        return v

    @validator("content", pre=True)
    def _content_str(cls, v):
        if v is None:
            return ""
        if not isinstance(v, str):
            v = str(v)
        # Strip null bytes — SQLite stores them but they break FTS5
        v = v.replace("\x00", "")
        if len(v) > MAX_MESSAGE_LEN:
            v = v[:MAX_MESSAGE_LEN]
        return v


class TurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    user_id: Optional[str] = Field(None, max_length=200)
    messages: List[Message] = Field(..., max_items=MAX_MESSAGES_PER_TURN)
    timestamp: str = Field(..., max_length=64)
    metadata: Optional[Dict[str, Any]] = {}

    @validator("messages")
    def _messages_nonempty_or_ok(cls, v):
        # Empty messages allowed — extraction will skip
        return v


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    session_id: str = Field(..., min_length=1, max_length=200)
    user_id: Optional[str] = Field(None, max_length=200)
    max_tokens: int = Field(1024, ge=1, le=8192)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    session_id: Optional[str] = Field(None, max_length=200)
    user_id: Optional[str] = Field(None, max_length=200)
    limit: int = Field(10, ge=1, le=100)


# ─── Response shapes ───────────────────────────────────

class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: List[Citation]


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: str
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: List[SearchResult]