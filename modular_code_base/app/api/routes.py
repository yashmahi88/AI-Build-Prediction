"""FastAPI routes and endpoints for Yocto Build Analyzer"""
import asyncio
import time
import uuid
import hashlib
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header

from app.api.models import (
    AnalyzeRequest, AnalyzeResponse, ChatRequest, CompletionRequest,
    CompletionResponse, FeedbackRequest, FeedbackResponse,
    StatusResponse, OllamaTagsResponse
)
from app.api.dependencies import (
    get_user_id, check_user_request_limit, acquire_user_lock, release_user_lock
)
from app.core.database import get_db_connection, get_db_cursor
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

# ========= SERVICE FACTORY FUNCTIONS (Lazy Loading - Avoids Circular Imports) =========

def get_enhanced_analysis_service():
    """Factory function to create EnhancedAnalysisService"""
    from app.services.enhanced_analysis_service import EnhancedAnalysisService
    return EnhancedAnalysisService()

def get_vectorstore_service():
    """Factory function to create VectorStoreService"""
    from app.services.vectorstore_service import VectorStoreService
    return VectorStoreService()

# ========= MAIN ANALYSIS ENDPOINT =========
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pipeline(
    request: AnalyzeRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Analyze pipeline and predict build success"""
    user_id = get_user_id(http_request, x_user_id)
    prediction_id = str(uuid.uuid4())
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress"
        )
    
    try:
        await acquire_user_lock(user_id)
        
        enhanced_service = get_enhanced_analysis_service()
        result = await enhanced_service.comprehensive_analyze(
            request.pipeline_content,
            user_id
        )
        
        # Store prediction in database
        try:
            script_hash = hashlib.sha256(
                request.pipeline_content.encode()
            ).hexdigest()
            
            with get_db_connection() as conn:
                with get_db_cursor(conn) as cur:
                    cur.execute("""
                        INSERT INTO predictions (
                            user_id, pipeline_name, predicted_result,
                            confidence_score, violated_rules,
                            pipeline_script_hash, detected_stack, id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        user_id, "api_analyze",
                        result.get('prediction', 'UNKNOWN'),
                        result.get('confidence', 50),
                        result.get('violated_rules', 0),
                        script_hash,
                        json.dumps(result.get('stack', [])),
                        prediction_id
                    ))
        except Exception as db_error:
            print(f"DB warning: {db_error}")
        
        return {
            "id": f"analyze-{prediction_id}",
            "object": "analysis.completion",
            "created": int(time.time()),
            "model": "yocto-analyzer",
            "prediction_id": prediction_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get('analysis', '')
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.pipeline_content.split()),
                "completion_tokens": len(result.get('analysis', '').split()),
                "total_tokens": len(request.pipeline_content.split()) + len(result.get('analysis', '').split())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_user_lock(user_id)

# ========= CHAT COMPLETIONS (OpenAI Compatible) =========
@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions with enhanced RAG analysis"""
    user_id = get_user_id(http_request, x_user_id)
    prediction_id = str(uuid.uuid4())
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        await acquire_user_lock(user_id)
        
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = request.messages[-1].get("content", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        # Use enhanced analysis service
        enhanced_service = get_enhanced_analysis_service()
        result = await enhanced_service.comprehensive_analyze(user_message, user_id)
        
        response_text = result.get('analysis', 'No analysis available')
        
        return {
            "id": f"chatcmpl-{prediction_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "prediction_id": prediction_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_user_lock(user_id)

# ========= COMPLETIONS (OpenAI Compatible) =========
@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible completions endpoint"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        await acquire_user_lock(user_id)
        
        if not request.prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        enhanced_service = get_enhanced_analysis_service()
        result = await enhanced_service.comprehensive_analyze(request.prompt, user_id)
        
        return {
            "choices": [{
                "index": 0,
                "text": result.get('analysis', ''),
                "finish_reason": "stop"
            }],
            "model": request.model
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_user_lock(user_id)

# ========= OLLAMA-STYLE GENERATE =========
@router.post("/api/generate")
async def ollama_generate(
    request: dict,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Ollama-compatible generate endpoint"""
    user_id = get_user_id(http_request, x_user_id)
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        await acquire_user_lock(user_id)
        
        prompt = request.get("prompt", "")
        model = request.get("model", "codellama:7b")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        enhanced_service = get_enhanced_analysis_service()
        result = await enhanced_service.comprehensive_analyze(prompt, user_id)
        response_text = result.get('analysis', '')
        
        return {
            "model": model,
            "created_at": datetime.now().isoformat() + "Z",
            "response": response_text,
            "done": True,
            "context": [],
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": len(response_text.split())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_user_lock(user_id)

# ========= FEEDBACK SUBMISSION =========
@router.post("/api/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Submit feedback on a prediction"""
    user_id = x_user_id or "anonymous"
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT predicted_result FROM predictions WHERE id::text = %s
                """, (feedback.prediction_id,))
                
                prediction = cur.fetchone()
                if not prediction:
                    raise HTTPException(status_code=404, detail="Prediction not found")
                
                correct = (
                    (prediction['predicted_result'] == "PASS" and 
                     feedback.actual_build_result == "SUCCESS") or
                    (prediction['predicted_result'] == "FAIL" and 
                     feedback.actual_build_result == "FAILURE")
                )
                
                cur.execute("""
                    INSERT INTO feedback (
                        prediction_id, user_id, actual_build_result,
                        correct_prediction, corrected_confidence,
                        missed_issues, false_positives, user_comments
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    feedback.prediction_id,
                    user_id,
                    feedback.actual_build_result,
                    correct,
                    feedback.corrected_confidence,
                    feedback.missed_issues,
                    feedback.false_positives,
                    feedback.user_comments
                ))
                
                feedback_id = cur.fetchone()['id']
        
        return {
            "status": "success",
            "feedback_id": str(feedback_id),
            "was_correct": correct
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========= REBUILD VECTOR STORE =========
@router.post("/api/rebuild")
async def rebuild_vectorstore(x_user_id: Optional[str] = Header(None)):
    """Force rebuild of vector store"""
    try:
        vectorstore_service = get_vectorstore_service()
        vectorstore_service.load_or_build()
        
        return {
            "status": "success",
            "message": "Vector store rebuilt",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========= STATUS ENDPOINT =========
@router.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    vectorstore_service = get_vectorstore_service()
    return {
        "status": "operational",
        "vectorstore_loaded": vectorstore_service.vectorstore is not None,
        "document_count": 0
    }

# ========= OLLAMA COMPATIBILITY ENDPOINTS =========
@router.get("/api/tags")
async def get_ollama_tags():
    """Ollama-compatible tags endpoint"""
    return {
        "models": [
            {
                "name": "codellama:7b",
                "modified_at": datetime.now().isoformat(),
                "size": 3825819519,
                "digest": "sha256:example"
            },
            {
                "name": "codellama-rag",
                "modified_at": datetime.now().isoformat(),
                "size": 3825819519,
                "digest": "sha256:example-rag"
            }
        ]
    }

@router.post("/api/show")
async def show_model(request: dict = None):
    """Ollama-compatible show endpoint"""
    if request is None:
        request = {}
    
    return {
        "license": "Apache 2.0",
        "modelfile": "FROM ./model",
        "parameters": "temperature 0.4",
        "template": "{{ .System }}{{ .Prompt }}",
        "details": {
            "format": "gguf",
            "family": "llama",
            "parameter_size": "7B"
        }
    }

@router.get("/api/version")
async def get_version():
    """Ollama version endpoint"""
    return {"version": "0.1.0"}

# ========= MINIO WEBHOOK =========
@router.post("/minio/webhook")
async def minio_webhook(request: Request):
    """Handle MinIO bucket notifications"""
    try:
        event_data = await request.json()
        
        if "Records" not in event_data:
            return {"status": "ignored"}
        
        return {
            "status": "success",
            "message": "Webhook processed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========= DEBUG ENDPOINTS =========
@router.get("/debug/status")
async def debug_status():
    """Debug status endpoint"""
    vectorstore_service = get_vectorstore_service()
    return {
        "vectorstore_loaded": vectorstore_service.vectorstore is not None,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "minio_bucket": settings.minio_bucket,
            "ollama_model": settings.ollama_llm_model,
            "vector_store_path": settings.vector_store_path
        }
    }

# ========= ROOT ENDPOINT =========
@router.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Yocto Build Analyzer - RAG-Enhanced Pipeline Analysis",
        "version": "1.0.0",
        "description": "Advanced Yocto build prediction using vectorstore and workspace analysis",
        "endpoints": {
            "analyze": "POST /analyze - Full pipeline analysis",
            "chat": "POST /v1/chat/completions - OpenAI-compatible chat",
            "completions": "POST /v1/completions - OpenAI-compatible completions",
            "generate": "POST /api/generate - Ollama-compatible generate",
            "feedback": "POST /api/feedback/submit - Submit prediction feedback",
            "rebuild": "POST /api/rebuild - Rebuild vector store",
            "status": "GET /api/status - System status",
            "tags": "GET /api/tags - Available models",
            "version": "GET /api/version - API version"
        }
    }
