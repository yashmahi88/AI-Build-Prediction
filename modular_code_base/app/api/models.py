from pydantic import BaseModel  # Pydantic base class for data validation and serialization
from typing import List, Optional, Dict  # Type hints for lists, optional fields, and dictionaries

# Request Models
class AnalyzeRequest(BaseModel):  # Request model for pipeline analysis endpoint
    pipeline_content: str  # Pipeline script content to analyze (required string field)

class ChatRequest(BaseModel):  # Request model for chat completion endpoint (OpenAI-compatible format)
    messages: list  # List of message dicts with role and content (conversation history)
    model: str = "qwen2.5:1.5b" ##"mistral:7b-instruct-q4_0" ##"codellama:7b"  # LLM model name (default qwen2.5:1.5b, alternatives commented)
    temperature: float = 0.4  # Controls response randomness (0.0=deterministic, 1.0=creative)
    max_tokens: int = 2000  # Maximum tokens to generate in response (limits length)

class CompletionRequest(BaseModel):  # Request model for text completion endpoint
    prompt: str  # Text prompt to complete (required string)
    model: str = "qwen2.5:1.5b"##"mistral:7b-instruct-q4_0" ##"codellama:7b"  # Model for completion (default qwen2.5:1.5b)
    temperature: float = 0.4  # Response randomness control (0.4 balanced)
    max_tokens: int = 2000  # Max tokens in completion (2000 default)

class FeedbackRequest(BaseModel):  # Request model for user feedback on predictions
    prediction_id: str  # UUID linking feedback to specific prediction (required)
    actual_build_result: str  # SUCCESS/FAILURE  # Actual build outcome (required, should be "SUCCESS" or "FAILURE")
    corrected_confidence: Optional[int] = None  # User's corrected confidence score 0-100 (optional)
    missed_issues: List[str] = []  # Issues model failed to detect (optional list, defaults empty)
    false_positives: List[str] = []  # Issues incorrectly flagged (optional list, defaults empty)
    user_comments: Optional[str] = None  # Free-form user feedback text (optional)
    suggested_rules: Optional[List[dict]] = None  # User-suggested rules to add (optional list of dicts)
    feedback_type: str = "manual"  # Feedback source type: "manual" or "automatic" (defaults "manual")

# Response Models
class AnalyzeResponse(BaseModel):  # Response model for analysis endpoint (OpenAI-compatible)
    id: str  # Unique response identifier (e.g. "chatcmpl-123")
    object: str  # Response object type (e.g. "chat.completion")
    created: int  # Unix timestamp of response creation
    model: str  # Model name used for analysis
    prediction_id: str  # UUID of stored prediction in database
    choices: list  # List of completion choices with message and finish_reason
    usage: dict  # Token usage stats (prompt_tokens, completion_tokens, total_tokens)

class CompletionResponse(BaseModel):  # Response model for completion endpoint
    choices: list  # List of completion choices with text and finish_reason
    model: str  # Model name used for completion
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Token usage with defaults

class StatusResponse(BaseModel):  # Response model for health/status check endpoint
    status: str  # Service status string (e.g. "healthy", "degraded")
    vectorstore_loaded: bool  # Whether FAISS vectorstore is loaded (True if ready)
    document_count: int  # Number of documents in vectorstore (0 if not loaded)

class FeedbackResponse(BaseModel):  # Response model for feedback submission
    status: str  # Feedback processing status (e.g. "success", "error")
    feedback_id: str  # UUID of stored feedback record
    was_correct: bool  # Whether prediction matched actual result (True if correct)

class OllamaTagsResponse(BaseModel):  # Response model for listing available Ollama models
    models: list  # List of model dicts with names and metadata

class OllamaVersionResponse(BaseModel):  # Response model for Ollama version info
    version: str  # Ollama server version string (e.g. "0.1.0")
