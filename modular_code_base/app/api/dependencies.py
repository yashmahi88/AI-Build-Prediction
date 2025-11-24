"""FastAPI dependency functions and utilities"""  
import asyncio  # Python's async library for concurrent programming (handling multiple requests at once)
import logging  # Standard Python logging library to track what's happening
from typing import Optional  # Type hint for optional parameters (can be None)
from fastapi import Request  # FastAPI's Request object containing HTTP request info

logger = logging.getLogger(__name__)  # Create logger instance for this module (app.core.dependencies)


# Global user locks dictionary (in-memory)
_user_locks = {}  # Dictionary tracking active requests per user: {"user_123": [task_id1, task_id2], "user_456": [task_id3]}
_lock = asyncio.Lock()  # Master lock to prevent race conditions when modifying _user_locks dictionary (ensures thread-safety)



async def get_user_id(http_request: Request, x_user_id: Optional[str] = None) -> str:  # Extract or generate user identifier
    """Get user ID from request header or generate one"""  
    if x_user_id:  # If X-User-ID header was provided in request
        return x_user_id  # Use that as the user identifier
    
    # Try to get from IP address
    client_ip = http_request.client.host if http_request.client else "unknown"  # Extract client IP from request (fallback to "unknown" if not available)
    return f"user_{client_ip}"  # Generate user ID like "user_192.168.1.100" based on IP address

async def check_user_request_limit(user_id: str, max_concurrent: int = 1) -> bool:  # Check if user has too many concurrent requests running
    """Check if user has reached concurrent request limit"""  
    try:  # Error handling block
        async with _lock:  # Acquire master lock to safely read _user_locks (prevents race conditions)
            current_count = len(_user_locks.get(user_id, []))  # Count how many active requests this user has (default to empty list [] if user not in dict)
            return current_count >= max_concurrent  # Return True if user hit limit (e.g., 1 concurrent request), False if they can make another
    except Exception as e:  # Catch any unexpected errors
        logger.error(f"Error checking limit for {user_id}: {e}")  # Log the error with user ID and exception details
        return False  # On error, allow request


async def acquire_user_lock(user_id: str) -> bool:  # Reserve a "slot" for this user's request (like taking a ticket)
    """Acquire lock for user request"""  
    try:  # Start error handling
        async with _lock:  # Get master lock to safely modify _user_locks dictionary
            if user_id not in _user_locks:  # If this is the first request from this user
                _user_locks[user_id] = []  # Create empty list to track their tasks
            
            lock_id = id(asyncio.current_task())  # Get unique ID of current async task (like a ticket number)
            _user_locks[user_id].append(lock_id)  # Add this task ID to user's active task list
            
            logger.debug(f"✅ Lock acquired for {user_id} (tasks: {len(_user_locks[user_id])})")  # Log successful lock acquisition with active task count
            return True  
    
    except Exception as e:  # Handle any errors during lock acquisition
        logger.error(f"Error acquiring lock for {user_id}: {e}")  # Log error details
        return False  


async def release_user_lock(user_id: str) -> bool:  # Release the "slot" when request finishes (return the ticket)
    """Release lock for user request"""  
    try:  
        async with _lock:  # Acquire master lock to safely modify _user_locks
            if user_id in _user_locks:  # Check if this user has any active locks
                lock_id = id(asyncio.current_task())  # Get current task's unique ID
                
                if lock_id in _user_locks[user_id]:  # If this task ID exists in user's active tasks
                    _user_locks[user_id].remove(lock_id)  # Remove it from the list
                    logger.debug(f"✅ Lock released for {user_id} (remaining: {len(_user_locks[user_id])})")  # Log release with remaining task count
                
                # Clean up empty entries
                if not _user_locks[user_id]:  # If user has no more active tasks (empty list)
                    del _user_locks[user_id]  # Delete user from dictionary entirely to save memory
                
                return True  # Signal successful release
            
            return False  # User wasn't in locks dict (shouldn't happen, but handle gracefully)
    
    except Exception as e:  # Catch any errors during release
        logger.error(f"Error releasing lock for {user_id}: {e}")  # Log error with details
        return False  # Signal failure

def get_db_connection():  # Get database connection object (synchronous function, not async)
    """Get database connection (sync)"""  
    from app.core.database import get_db_connection as _get_db_connection  # Import actual function from database module
    return _get_db_connection()  

def get_db_cursor(conn):  # Get database cursor from existing connection (for executing SQL queries)
    """Get database cursor (sync)"""  
    from app.core.database import get_db_cursor as _get_db_cursor  # Import cursor function from database module (lazy import)
    return _get_db_cursor(conn)  # Call function with connection parameter and return  object
