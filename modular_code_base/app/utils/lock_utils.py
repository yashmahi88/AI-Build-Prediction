import os  # Operating system interface for file operations (create, delete, check existence) and process ID retrieval
import time  # Time library for checking file modification timestamps and calculating lock age
from typing import Optional  # Type hint for optional parameters (not used in this code but imported for future use)
import threading  # Threading library for creating thread-safe locks to prevent race conditions in multi-threaded environments


class LockManager:  # Class that manages distributed locks using filesystem to coordinate build processes across multiple workers
    """Manage build locks across workers"""  # Docstring explaining this class prevents concurrent builds using file-based locking
    
    def __init__(self, lock_file: str):  # Constructor that initializes lock manager with path to lock file
        self.lock_file = lock_file  # Store path to lock file (e.g., "/tmp/yocto_build.lock") that will be created/deleted to manage lock state
        self.global_lock = threading.Lock()  # Create thread-level lock to make acquire/release operations atomic within this Python process (prevents race conditions between threads in same process)
    
    def acquire(self, timeout: int = 300) -> bool:  # Method to attempt acquiring the build lock with configurable timeout (default 300 seconds = 5 minutes)
        """Acquire build lock"""  # Docstring explaining this method tries to acquire lock and returns True if successful
        with self.global_lock:  # Acquire thread lock using context manager (ensures only one thread can execute this block at a time within this process)
            if os.path.exists(self.lock_file):  # Check if lock file already exists (indicates another process holds the lock)
                lock_age = time.time() - os.path.getmtime(self.lock_file)  # Calculate how long the lock has been held (current time minus file modification time in seconds)
                if lock_age > timeout:  # Check if lock has been held longer than timeout period (indicates stale lock from crashed process)
                    os.remove(self.lock_file)  # Remove stale lock file (cleanup dead lock)
                else:  # Lock is fresh and actively held by another process
                    return False  # Return False to indicate lock acquisition failed (caller should retry or abort)
            
            with open(self.lock_file, 'w') as f:  # Create new lock file (or overwrite if we just removed a stale one) using context manager for automatic close
                f.write(str(os.getpid()))  # Write current process ID to lock file (helps identify which process holds the lock for debugging)
            return True  # Return True to indicate lock was successfully acquired
    
    def release(self):  # Method to release the build lock by deleting the lock file
        """Release build lock"""  # Docstring explaining this method removes the lock file to allow other processes to acquire lock
        try:  # Wrap deletion in try-except to handle errors gracefully
            if os.path.exists(self.lock_file):  # Check if lock file exists before trying to delete (avoid error if already deleted)
                os.remove(self.lock_file)  # Delete lock file to release the lock (makes lock available for other processes)
        except Exception as e:  # Catch any errors during deletion (permission denied, file system errors, etc.)
            print(f"Error releasing lock: {e}")  # Print error message to console with exception details (but don't raise exception to avoid breaking cleanup code)
