"""FastAPI dependency functions and utilities"""
import asyncio
import logging
from typing import Optional
from fastapi import Request


logger = logging.getLogger(__name__)

# Global user locks dictionary (in-memory)
_user_locks = {}
_lock = asyncio.Lock()


async def get_user_id(http_request: Request, x_user_id: Optional[str] = None) -> str:
    """Get user ID from request header or generate one"""
    if x_user_id:
        return x_user_id
    
    # Try to get from IP address
    client_ip = http_request.client.host if http_request.client else "unknown"
    return f"user_{client_ip}"


async def check_user_request_limit(user_id: str, max_concurrent: int = 1) -> bool:
    """Check if user has reached concurrent request limit"""
    try:
        async with _lock:
            current_count = len(_user_locks.get(user_id, []))
            return current_count >= max_concurrent
    except Exception as e:
        logger.error(f"Error checking limit for {user_id}: {e}")
        return False


async def acquire_user_lock(user_id: str) -> bool:
    """Acquire lock for user request"""
    try:
        async with _lock:
            if user_id not in _user_locks:
                _user_locks[user_id] = []
            
            lock_id = id(asyncio.current_task())
            _user_locks[user_id].append(lock_id)
            
            logger.debug(f"✅ Lock acquired for {user_id} (tasks: {len(_user_locks[user_id])})")
            return True
    
    except Exception as e:
        logger.error(f"Error acquiring lock for {user_id}: {e}")
        return False


async def release_user_lock(user_id: str) -> bool:
    """Release lock for user request"""
    try:
        async with _lock:
            if user_id in _user_locks:
                lock_id = id(asyncio.current_task())
                
                if lock_id in _user_locks[user_id]:
                    _user_locks[user_id].remove(lock_id)
                    logger.debug(f"✅ Lock released for {user_id} (remaining: {len(_user_locks[user_id])})")
                
                # Clean up empty entries
                if not _user_locks[user_id]:
                    del _user_locks[user_id]
                
                return True
            
            return False
    
    except Exception as e:
        logger.error(f"Error releasing lock for {user_id}: {e}")
        return False


def get_db_connection():
    """Get database connection (sync)"""
    from app.core.database import get_db_connection as _get_db_connection
    return _get_db_connection()


def get_db_cursor(conn):
    """Get database cursor (sync)"""
    from app.core.database import get_db_cursor as _get_db_cursor
    return _get_db_cursor(conn)
