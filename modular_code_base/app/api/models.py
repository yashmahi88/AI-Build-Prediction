from pydantic import BaseModel
from typing import List, Optional, Dict

# Request Models
class AnalyzeRequest(BaseModel):
    pipeline_content: str

class ChatRequest(BaseModel):
    messages: list
    model: str = "qwen2.5:1.5b" ##"mistral:7b-instruct-q4_0" ##"codellama:7b"
    temperature: float = 0.4
    max_tokens: int = 2000

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:1.5b"##"mistral:7b-instruct-q4_0" ##"codellama:7b"
    temperature: float = 0.4
    max_tokens: int = 2000

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_build_result: str  # SUCCESS/FAILURE
    corrected_confidence: Optional[int] = None
    missed_issues: List[str] = []
    false_positives: List[str] = []
    user_comments: Optional[str] = None
    suggested_rules: Optional[List[dict]] = None
    feedback_type: str = "manual"

# Response Models
class AnalyzeResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    prediction_id: str
    choices: list
    usage: dict

class CompletionResponse(BaseModel):
    choices: list
    model: str
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class StatusResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    document_count: int

class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    was_correct: bool

class OllamaTagsResponse(BaseModel):
    models: list

class OllamaVersionResponse(BaseModel):
    version: str
