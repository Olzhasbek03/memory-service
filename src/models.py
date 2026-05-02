from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- INCOMING data shapes (what people SEND us) ---

class Message(BaseModel):
    role: str  # "user", "assistant", or "tool"
    content: str
    name: Optional[str] = None  # only for tool messages

class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = {}

class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10

# --- OUTGOING data shapes (what we SEND BACK) ---

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