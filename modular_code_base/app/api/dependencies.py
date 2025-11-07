import asyncio
from typing import Dict, Optional
from fastapi import Request, Header
from app.core.database import get_db_connection, get_db_cursor

# User-specific request locks to prevent concurrent requests
user_locks: Dict[str, asyncio.Lock] = {}

def get_user_id(request: Request, x_user_id: Optional[str] = None) -> str:
    """Extract user ID from request"""
    if x_user_id:
        return x_user_id
    
    # Try to get from query params
    user_id = request.query_params.get("user_id")
    if user_id:
        return user_id
    
    # Try to get from path
    user_id = request.path_params.get("user_id")
    if user_id:
        return user_id
    
    # Default to client IP
    return request.client.host if request.client else "anonymous"

async def check_user_request_limit(user_id: str, max_concurrent: int = 1) -> bool:
    """Check if user already has concurrent requests"""
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    
    # Try to acquire lock without blocking (non-blocking check)
    if user_locks[user_id].locked():
        return True  # Already has a request in progress
    
    return False

async def acquire_user_lock(user_id: str):
    """Acquire lock for user request"""
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    
    await user_locks[user_id].acquire()

def release_user_lock(user_id: str):
    """Release lock for user request"""
    if user_id in user_locks and user_locks[user_id].locked():
        user_locks[user_id].release()
