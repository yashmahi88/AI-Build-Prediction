import os
import time
from typing import Optional
import threading

class LockManager:
    """Manage build locks across workers"""
    
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.global_lock = threading.Lock()
    
    def acquire(self, timeout: int = 300) -> bool:
        """Acquire build lock"""
        with self.global_lock:
            if os.path.exists(self.lock_file):
                lock_age = time.time() - os.path.getmtime(self.lock_file)
                if lock_age > timeout:
                    os.remove(self.lock_file)
                else:
                    return False
            
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))
            return True
    
    def release(self):
        """Release build lock"""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            print(f"Error releasing lock: {e}")
