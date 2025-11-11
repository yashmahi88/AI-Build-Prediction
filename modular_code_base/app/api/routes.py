"""FastAPI routes and endpoints for Yocto Build Analyzer"""
import asyncio
import time
import uuid
import hashlib
import json
import logging
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
logger = logging.getLogger(__name__)


# ========= SERVICE FACTORY FUNCTIONS =========

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
    user_id = await get_user_id(http_request, x_user_id)
    prediction_id = str(uuid.uuid4())
    lock_acquired = False
    
    if await check_user_request_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"User {user_id} already has a request in progress"
        )
    
    try:
        lock_acquired = await acquire_user_lock(user_id)
        
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
                    rules_applied = result.get('all_rules', [])
                    
                    cur.execute("""
                        INSERT INTO predictions (
                            user_id, pipeline_name, predicted_result,
                            confidence_score, violated_rules,
                            pipeline_script_hash, detected_stack, id,
                            rules_applied, satisfied_rules
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        user_id, "api_analyze",
                        result.get('prediction', 'UNKNOWN'),
                        result.get('confidence', 50),
                        result.get('violated_rules', 0),
                        script_hash,
                        json.dumps(result.get('stack', [])),
                        prediction_id,
                        json.dumps(rules_applied),
                        result.get('satisfied_rules', 0)
                    ))
                    conn.commit()
        except Exception as db_error:
            logger.warning(f"DB warning: {db_error}")
        
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
        logger.exception(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            await release_user_lock(user_id)


# ========= CHAT COMPLETIONS (OpenAI Compatible) =========


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions with enhanced RAG analysis"""
    user_id = await get_user_id(http_request, x_user_id)
    prediction_id = str(uuid.uuid4())
    lock_acquired = False
    
    logger.info(f"Generated prediction_id: {prediction_id}")
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        lock_acquired = await acquire_user_lock(user_id)
        
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = request.messages[-1].get("content", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        enhanced_service = get_enhanced_analysis_service()
        result = await enhanced_service.comprehensive_analyze(user_message, user_id)
        
        response_text = result.get('analysis', 'No analysis available')
        
        # ========= SAVE TO DATABASE =========
        try:
            script_hash = hashlib.sha256(user_message.encode()).hexdigest()
            
            predicted_result = result.get('prediction', 'UNKNOWN')
            confidence_score = result.get('confidence', 50)
            violated_rules = result.get('violated_rules', 0)
            
            with get_db_connection() as conn:
                with get_db_cursor(conn) as cur:
                    rules_applied = result.get('all_rules', [])
                    detected_stack = result.get('stack', [])
                    
                    # Ensure detected_stack is a list of strings
                    if not isinstance(detected_stack, list):
                        detected_stack = []
                    
                    cur.execute("""
                        INSERT INTO predictions (
                            user_id, pipeline_name, predicted_result,
                            confidence_score, violated_rules,
                            pipeline_script_hash, detected_stack, rules_applied, id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """, (
                        user_id, "chat_completion",
                        predicted_result,
                        confidence_score,
                        violated_rules,
                        script_hash,
                        detected_stack,              # Python list → PostgreSQL text[]
                        json.dumps(rules_applied),   # JSON string → PostgreSQL jsonb
                        prediction_id
                    ))
                    conn.commit()
                    logger.info(f"✅ Stored prediction {prediction_id} in DB")
        
        except Exception as db_error:
            logger.error(f"❌ DB save failed: {db_error}")
            import traceback
            logger.error(traceback.format_exc())
        # ========= END DATABASE SAVE =========
        
        logger.info(f" Returning prediction_id: {prediction_id}")
        
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
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            await release_user_lock(user_id)



# ========= COMPLETIONS (OpenAI Compatible) =========

@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """OpenAI-compatible completions endpoint"""
    user_id = await get_user_id(http_request, x_user_id)
    lock_acquired = False
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        lock_acquired = await acquire_user_lock(user_id)
        
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
        logger.exception(f"Completions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            await release_user_lock(user_id)


# ========= OLLAMA-STYLE GENERATE =========

@router.post("/api/generate")
async def ollama_generate(
    request: dict,
    http_request: Request,
    x_user_id: Optional[str] = Header(None)
):
    """Ollama-compatible generate endpoint"""
    user_id = await get_user_id(http_request, x_user_id)
    lock_acquired = False
    
    if await check_user_request_limit(user_id):
        raise HTTPException(status_code=429, detail="Request in progress")
    
    try:
        lock_acquired = await acquire_user_lock(user_id)
        
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
        logger.exception(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            await release_user_lock(user_id)


# ========= FEEDBACK SUBMISSION WITH LEARNING =========

@router.post("/api/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Submit feedback on a prediction and learn from it"""
    user_id = x_user_id or "anonymous"
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Get prediction details
                cur.execute("""
                    SELECT predicted_result, rules_applied, confidence_score
                    FROM predictions WHERE id::text = %s
                """, (feedback.prediction_id,))
                
                prediction = cur.fetchone()
                if not prediction:
                    raise HTTPException(status_code=404, detail="Prediction not found")
                
                # ========= CHANGED: Fixed prediction matching =========
                predicted = prediction['predicted_result'].upper().strip()
                actual = feedback.actual_build_result.upper().strip()
                
                # Map HIGH-RISK/HIGH_RISK to FAIL for comparison
                if 'HIGH' in predicted or 'RISK' in predicted:
                    predicted = 'FAIL'
                
                # Normalize PASS/SUCCESS
                if predicted in ['PASS', 'SUCCESS']:
                    predicted = 'PASS'
                if actual in ['PASS', 'SUCCESS']:
                    actual = 'SUCCESS'
                
                # Check if correct
                correct = (
                    (predicted == "PASS" and actual == "SUCCESS") or
                    (predicted == "FAIL" and actual == "FAILURE")
                )
                # ========= END CHANGED =========
                
                # Convert to Python lists
                missed = feedback.missed_issues if feedback.missed_issues else []
                false_pos = feedback.false_positives if feedback.false_positives else []
                
                # Insert feedback
                cur.execute("""
                    INSERT INTO feedback (
                        prediction_id, user_id, actual_build_result,
                        correct_prediction, corrected_confidence,
                        missed_issues, false_positives, user_comments,
                        feedback_type
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    feedback.prediction_id,
                    user_id,
                    feedback.actual_build_result,
                    correct,
                    feedback.corrected_confidence,
                    missed,
                    false_pos,
                    feedback.user_comments,
                    feedback.feedback_type or "manual"
                ))
                
                feedback_id = cur.fetchone()['id']
                
                # Update prediction with feedback
                cur.execute("""
                    UPDATE predictions
                    SET actual_result = %s, feedback_received_at = NOW()
                    WHERE id = %s::uuid
                """, (feedback.actual_build_result, feedback.prediction_id))
                
                conn.commit()
                logger.info(f"✅ Feedback {feedback_id} submitted (correct={correct})")
        
        # Learn from feedback asynchronously
        try:
            await learn_from_feedback_async(
                str(feedback_id),
                prediction['rules_applied'],
                feedback,
                correct
            )
        except Exception as e:
            logger.warning(f"⚠️ Learning error: {e}")
        
        return {
            "status": "success",
            "feedback_id": str(feedback_id),
            "was_correct": correct
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))





async def learn_from_feedback_async(
    feedback_id: str,
    rules_applied: str,
    feedback: FeedbackRequest,
    was_correct: bool
):
    """Learn from feedback - YOUR ORIGINAL LOGIC"""
    try:
        await asyncio.sleep(0.1)
        
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Verify feedback exists
                cur.execute("SELECT id FROM feedback WHERE id = %s::uuid", (feedback_id,))
                if not cur.fetchone():
                    logger.warning(f"Feedback {feedback_id} not found yet")
                    return
                
                # Parse rules
                rules = json.loads(rules_applied) if isinstance(rules_applied, str) else rules_applied
                
                # Update rule performance
                for rule in rules[:20]:
                    rule_text = rule.get('rule_text', '') if isinstance(rule, dict) else str(rule)
                    if not rule_text or len(rule_text) < 10:
                        continue
                    
                    cur.execute("""
                        INSERT INTO rule_performance (
                            rule_text, rule_type, total_applications, correct_predictions
                        ) VALUES (%s, %s, 1, %s)
                        ON CONFLICT (rule_text) DO UPDATE SET
                            total_applications = rule_performance.total_applications + 1,
                            correct_predictions = rule_performance.correct_predictions + EXCLUDED.correct_predictions,
                            last_updated = NOW()
                    """, (
                        rule_text[:500],
                        rule.get('rule_type', 'unknown') if isinstance(rule, dict) else 'unknown',
                        1 if was_correct else 0
                    ))
                
                # Learn from missed issues
                if feedback.missed_issues:
                    for issue in feedback.missed_issues[:10]:
                        if issue and len(issue.strip()) > 5:
                            cur.execute("""
                                INSERT INTO learned_patterns (
                                    pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
                                ) VALUES ('failure_indicator', %s, %s::uuid, 0.15)
                                ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
                                    occurrences = learned_patterns.occurrences + 1,
                                    confidence_boost = LEAST(learned_patterns.confidence_boost + 0.05, 0.3)
                            """, (issue[:500], feedback_id))
                
                # Learn from false positives
                if feedback.false_positives:
                    for fp in feedback.false_positives[:10]:
                        if fp and len(fp.strip()) > 5:
                            cur.execute("""
                                INSERT INTO learned_patterns (
                                    pattern_type, pattern_text, learned_from_feedback_id, confidence_boost
                                ) VALUES ('false_positive', %s, %s::uuid, -0.1)
                                ON CONFLICT (pattern_type, pattern_text) DO UPDATE SET
                                    occurrences = learned_patterns.occurrences + 1,
                                    confidence_boost = GREATEST(learned_patterns.confidence_boost - 0.05, -0.2)
                            """, (fp[:500], feedback_id))
                
                conn.commit()
                logger.info(f"✅ Learned from feedback {feedback_id}")
    
    except Exception as e:
        logger.exception(f"Learning failed: {e}")


# ========= LEARNING ANALYTICS ENDPOINTS =========

@router.get("/api/feedback/stats")
async def get_feedback_stats(x_user_id: Optional[str] = Header(None)):
    """Get feedback statistics - IS SYSTEM LEARNING?"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT p.id) as total_predictions,
                        COUNT(f.id) as feedback_count,
                        SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) as correct,
                        AVG(p.confidence_score) as avg_confidence
                    FROM predictions p
                    LEFT JOIN feedback f ON f.prediction_id = p.id
                    WHERE p.created_at > NOW() - INTERVAL '30 days'
                """)
                
                stats = cur.fetchone()
                total = stats['feedback_count'] or 0
                correct = stats['correct'] or 0
                accuracy = (correct / total * 100) if total > 0 else 0
                
                # Get learned patterns
                cur.execute("""
                    SELECT COUNT(*) as learned_patterns
                    FROM learned_patterns
                    WHERE occurrences >= 2
                """)
                patterns = cur.fetchone()
                
                return {
                    "status": "success",
                    "feedback_received": total,
                    "correct_predictions": correct,
                    "incorrect_predictions": total - correct,
                    "accuracy_percentage": round(accuracy, 2),
                    "average_confidence": round(float(stats['avg_confidence'] or 0), 2),
                    "learned_patterns": patterns['learned_patterns'] or 0,
                    "is_learning": accuracy > 60,
                    "learning_status": "✅ System is learning!" if accuracy > 60 else "⚠️ Needs more feedback"
                }
    
    except Exception as e:
        logger.exception(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/feedback/list")
async def list_feedback(limit: int = 50, x_user_id: Optional[str] = Header(None)):
    """List recent feedback with predictions"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                    SELECT 
                        f.id as feedback_id,
                        f.prediction_id,
                        p.pipeline_name,
                        p.predicted_result,
                        p.confidence_score,
                        f.actual_build_result,
                        f.correct_prediction,
                        f.corrected_confidence,
                        f.missed_issues,
                        f.false_positives,
                        f.user_comments,
                        f.created_at
                    FROM feedback f
                    JOIN predictions p ON f.prediction_id = p.id
                    ORDER BY f.created_at DESC
                    LIMIT %s
                """, (limit,))
                
                feedbacks = []
                for row in cur.fetchall():
                    feedbacks.append({
                        "feedback_id": str(row['feedback_id']),
                        "prediction_id": str(row['prediction_id']),
                        "pipeline": row['pipeline_name'],
                        "ai_predicted": row['predicted_result'],
                        "ai_confidence": row['confidence_score'],
                        "actual_result": row['actual_build_result'],
                        "was_correct": row['correct_prediction'],
                        "corrected_confidence": row['corrected_confidence'],
                        "missed_issues": row['missed_issues'],
                        "false_positives": row['false_positives'],
                        "comments": row['user_comments'],
                        "submitted_at": row['created_at'].isoformat() if row['created_at'] else None
                    })
                
                return {
                    "status": "success",
                    "feedbacks": feedbacks,
                    "count": len(feedbacks)
                }
    
    except Exception as e:
        logger.exception(f"List feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/learning/patterns")
async def get_learned_patterns(x_user_id: Optional[str] = Header(None)):
    """Show what patterns AI has learned from feedback"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Failure indicators learned
                cur.execute("""
                    SELECT 
                        pattern_text,
                        pattern_type,
                        occurrences,
                        confidence_boost,
                        created_at
                    FROM learned_patterns
                    WHERE pattern_type = 'failure_indicator'
                    ORDER BY occurrences DESC
                    LIMIT 20
                """)
                
                failure_patterns = [
                    {
                        "pattern": row['pattern_text'],
                        "learned_from_feedback_count": row['occurrences'],
                        "confidence_boost": float(row['confidence_boost']),
                        "first_learned": row['created_at'].isoformat() if row['created_at'] else None
                    }
                    for row in cur.fetchall()
                ]
                
                # False positive patterns
                cur.execute("""
                    SELECT 
                        pattern_text,
                        occurrences,
                        confidence_boost
                    FROM learned_patterns
                    WHERE pattern_type = 'false_positive'
                    ORDER BY occurrences DESC
                    LIMIT 20
                """)
                
                false_patterns = [
                    {
                        "pattern": row['pattern_text'],
                        "times_corrected": row['occurrences'],
                        "confidence_penalty": float(row['confidence_boost'])
                    }
                    for row in cur.fetchall()
                ]
                
                return {
                    "status": "success",
                    "failure_indicators_learned": failure_patterns,
                    "false_positives_corrected": false_patterns,
                    "total_patterns_learned": len(failure_patterns) + len(false_patterns)
                }
    
    except Exception as e:
        logger.exception(f"Patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/learning/rules")
async def get_rule_performance(x_user_id: Optional[str] = Header(None)):
    """Show how well each rule performs"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # ========= CHANGED: Fixed ROUND() function =========
                cur.execute("""
                    SELECT 
                        rule_text,
                        rule_type,
                        total_applications,
                        correct_predictions,
                        ROUND(CAST(correct_predictions AS numeric) / CAST(total_applications AS numeric) * 100, 2) as accuracy
                    FROM rule_performance
                    WHERE total_applications > 0
                    ORDER BY accuracy DESC
                    LIMIT 30
                """)
                # ========= END CHANGED =========
                
                rules = []
                for row in cur.fetchall():
                    accuracy = float(row['accuracy'])
                    rules.append({
                        "rule": row['rule_text'][:100],
                        "type": row['rule_type'],
                        "times_applied": row['total_applications'],
                        "times_correct": row['correct_predictions'],
                        "accuracy_percentage": accuracy,
                        "reliability": "✅ Reliable" if accuracy > 80 else "⚠️ Needs review"
                    })
                
                return {
                    "status": "success",
                    "rules": rules,
                    "count": len(rules)
                }
    
    except Exception as e:
        logger.exception(f"Rules error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/api/learning/rules")
# async def get_rule_performance(x_user_id: Optional[str] = Header(None)):
#     """Show how well each rule performs"""
#     try:
#         with get_db_connection() as conn:
#             with get_db_cursor(conn) as cur:
#                 cur.execute("""
#                     SELECT 
#                         rule_text,
#                         rule_type,
#                         total_applications,
#                         correct_predictions,
#                         ROUND((correct_predictions::float / total_applications * 100), 2) as accuracy
#                     FROM rule_performance
#                     WHERE total_applications > 0
#                     ORDER BY accuracy DESC
#                     LIMIT 30
#                 """)
                
#                 rules = []
#                 for row in cur.fetchall():
#                     accuracy = float(row['accuracy'])
#                     rules.append({
#                         "rule": row['rule_text'][:100],
#                         "type": row['rule_type'],
#                         "times_applied": row['total_applications'],
#                         "times_correct": row['correct_predictions'],
#                         "accuracy_percentage": accuracy,
#                         "reliability": "✅ Reliable" if accuracy > 80 else "⚠️ Needs review"
#                     })
                
#                 return {
#                     "status": "success",
#                     "rules": rules,
#                     "count": len(rules)
#                 }
    
#     except Exception as e:
#         logger.exception(f"Rules error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))



@router.get("/api/learning/accuracy-trend")
async def get_accuracy_trend(x_user_id: Optional[str] = Header(None)):
    """Show accuracy improvement over time"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # ========= CHANGED: Fixed ROUND() function =========
                cur.execute("""
                    SELECT 
                        DATE(f.created_at) as date,
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) as correct,
                        ROUND(CAST(SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) AS numeric) / CAST(COUNT(*) AS numeric) * 100, 2) as accuracy
                    FROM feedback f
                    WHERE f.created_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(f.created_at)
                    ORDER BY date ASC
                """)
                # ========= END CHANGED =========
                
                trend = [
                    {
                        "date": row['date'].isoformat(),
                        "feedback_count": row['total_feedback'],
                        "correct_predictions": row['correct'],
                        "accuracy_percent": float(row['accuracy'])
                    }
                    for row in cur.fetchall()
                ]
                
                improving = False
                if len(trend) > 1:
                    improving = trend[-1]['accuracy_percent'] > trend[0]['accuracy_percent']
                
                return {
                    "status": "success",
                    "accuracy_trend": trend,
                    "total_days": len(trend),
                    "improving": improving,
                    "trend_status": "Improving!" if improving else " Needs work"
                }
    
    except Exception as e:
        logger.exception(f"Trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/api/learning/accuracy-trend")
# async def get_accuracy_trend(x_user_id: Optional[str] = Header(None)):
#     """Show accuracy improvement over time"""
#     try:
#         with get_db_connection() as conn:
#             with get_db_cursor(conn) as cur:
#                 cur.execute("""
#                     SELECT 
#                         DATE(f.created_at) as date,
#                         COUNT(*) as total_feedback,
#                         SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END) as correct,
#                         ROUND(SUM(CASE WHEN f.correct_prediction THEN 1 ELSE 0 END)::float / COUNT(*) * 100, 2) as accuracy
#                     FROM feedback f
#                     WHERE f.created_at > NOW() - INTERVAL '30 days'
#                     GROUP BY DATE(f.created_at)
#                     ORDER BY date ASC
#                 """)
                
#                 trend = [
#                     {
#                         "date": row['date'].isoformat(),
#                         "feedback_count": row['total_feedback'],
#                         "correct_predictions": row['correct'],
#                         "accuracy_percent": float(row['accuracy'])
#                     }
#                     for row in cur.fetchall()
#                 ]
                
#                 improving = False
#                 if len(trend) > 1:
#                     improving = trend[-1]['accuracy_percent'] > trend[0]['accuracy_percent']
                
#                 return {
#                     "status": "success",
#                     "accuracy_trend": trend,
#                     "total_days": len(trend),
#                     "improving": improving,
#                     "trend_status": " Improving!" if improving else " Needs work"
#                 }
    
#     except Exception as e:
#         logger.exception(f"Trend error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


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
        logger.exception(f"Rebuild error: {e}")
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
        logger.exception(f"Webhook error: {e}")
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
        "description": "Advanced Yocto build prediction with feedback learning",
        "endpoints": {
            "analyze": "POST /analyze - Full pipeline analysis",
            "chat": "POST /v1/chat/completions - OpenAI-compatible chat",
            "completions": "POST /v1/completions - OpenAI-compatible completions",
            "generate": "POST /api/generate - Ollama-compatible generate",
            "feedback": "POST /api/feedback/submit - Submit prediction feedback",
            "feedback_stats": "GET /api/feedback/stats - Learning statistics",
            "feedback_list": "GET /api/feedback/list - List all feedback",
            "patterns": "GET /api/learning/patterns - Learned patterns",
            "rules": "GET /api/learning/rules - Rule performance",
            "accuracy_trend": "GET /api/learning/accuracy-trend - Accuracy over time",
            "rebuild": "POST /api/rebuild - Rebuild vector store",
            "status": "GET /api/status - System status",
            "tags": "GET /api/tags - Available models",
            "version": "GET /api/version - API version"
        }
    }
