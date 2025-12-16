"""File system watcher for MinIO data directory"""  # Module docstring describing this file monitors MinIO storage for file changes and triggers vector store refreshes
import os  # Operating system interface for file path operations
import time  # Time-related functions for debouncing and timestamps
import asyncio  # Python's async library for handling concurrent operations
import logging  # Standard Python logging library to track file system events
from typing import Optional  # Type hint for optional parameters that can be None
from watchdog.observers import Observer  # File system monitoring class that watches directories for changes
from watchdog.events import FileSystemEventHandler  # Base class for handling file system events (create, modify, delete)


logger = logging.getLogger(__name__)  # Create logger instance for this module to output file watcher events



class MinIOFileWatcher(FileSystemEventHandler):  # Event handler class that responds to file system changes in MinIO data directory
    """File system event handler for MinIO data directory changes"""  # Docstring explaining this class processes file events and triggers vector store refreshes
    
    def __init__(self, bucket_name: str, debounce_seconds: int = 3):  # Constructor that initializes the watcher with bucket name and debounce delay (default 3 seconds to batch rapid changes)
        super().__init__()  # Call parent FileSystemEventHandler constructor to initialize base functionality
        self.bucket_name = bucket_name  # Store the MinIO bucket name to filter which files to watch
        self.debounce_seconds = debounce_seconds  # Store debounce period to avoid triggering refresh on every single file change (wait for changes to settle)
        self.loop: Optional[asyncio.AbstractEventLoop] = None  # Reference to asyncio event loop (initially None, set later) for scheduling async callbacks
        self.last_change_time = 0  # Timestamp of most recent file change (used for debouncing logic)
        self.refresh_task: Optional[asyncio.Task] = None  # Reference to the currently scheduled refresh task (None if no refresh pending)
        self.on_change_callback = None  # Callback function to invoke when files change (will be set later, typically the vector store refresh function)
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):  # Method to inject the asyncio event loop after initialization
        """Set the event loop for async operations"""  # Docstring explaining this connects the watcher to the main async event loop
        self.loop = loop  # Store the event loop reference so we can schedule async tasks from synchronous file event handlers
    
    def set_callback(self, callback):  # Method to set the function that should be called when files change
        """Set callback function to call on file changes"""  # Docstring explaining this registers the refresh handler (usually vector store refresh)
        self.on_change_callback = callback  # Store the callback function reference (will be invoked after debounce period)
    
    def on_modified(self, event):  # Handler called by watchdog when a file is modified
        """Handle file modification events"""  # Docstring explaining this responds to file edit events
        if not event.is_directory and self._is_user_content_file(event.src_path):  # Only process if it's a file (not directory) and passes our filter for actual user content
            logger.info(f"User file modified: {event.src_path}")  # Log which file was modified for debugging
            self._schedule_refresh()  # Schedule a debounced refresh of the vector store
    
    def on_created(self, event):  # Handler called by watchdog when a new file is created
        """Handle file creation events"""  # Docstring explaining this responds to new file events
        if not event.is_directory and self._is_user_content_file(event.src_path):  # Only process if it's a file (not directory) and is actual user content
            logger.info(f"User file created: {event.src_path}")  # Log the newly created file
            self._schedule_refresh()  # Schedule a debounced refresh
    
    def on_deleted(self, event):  # Handler called by watchdog when a file is deleted
        """Handle file deletion events"""  # Docstring explaining this responds to file removal events
        if not event.is_directory and self._is_user_content_file(event.src_path):  # Only process if it's a file (not directory) and was actual user content
            logger.info(f"User file deleted: {event.src_path}")  # Log the deleted file path
            self._schedule_refresh()  # Schedule a debounced refresh (need to remove embeddings for deleted files)
    
    def _is_user_content_file(self, filepath: str) -> bool:  # Filter function to distinguish real user files from MinIO system/metadata files
        """Check if the file is actual user content, not MinIO system files"""  # Docstring explaining this prevents triggering on MinIO internal files
        try:  # Wrap in try-except to handle any unexpected path parsing errors
            normalized_path = filepath.replace('\\', '/')  # Normalize path separators to forward slashes for consistent pattern matching (handles Windows backslashes)
            
            # ✅ CRITICAL: Skip MinIO system files completely
            skip_patterns = [  # List of patterns that indicate MinIO internal/system files (not user content)
                '.minio.sys',  # MinIO system metadata directory
                'xl.meta',  # Erasure coding metadata files used by MinIO internally
                '.usage-cache',  # Usage statistics cache files
                '.usage.json',  # Usage metrics JSON files
                '.bloomcycle',  # Bloom filter cycle files for optimization
                'tmp/',  # Temporary files directory
                '.tmp',  # Temporary file extension
                'part.minio',  # Multipart upload parts (incomplete uploads)
                '.backup',  # Backup files
                '.bkp',  # Backup file extension
                '.trash',  # Trash/deleted files directory
                'multipart'  # Multipart upload temporary files
            ]
            
            for pattern in skip_patterns:  # Loop through each skip pattern
                if pattern in normalized_path:  # Check if pattern appears anywhere in the file path
                    return False  # If found, this is a system file - ignore it
            
            # ✅ Only consider files in our bucket
            if f"{self.bucket_name}" in normalized_path:  # Check if the file path contains our configured bucket name
                filename = os.path.basename(filepath)  # Extract just the filename from the full path
                
                # Skip hidden files, temp files, and empty names
                if filename.startswith('.') or filename.endswith('.tmp'):  # Ignore hidden files (starting with dot) and temporary files (ending with .tmp)
                    return False  # This is a hidden/temp file - ignore it
                if not filename.strip():  # Check if filename is empty or only whitespace
                    return False  # Invalid/empty filename - ignore it
                
                logger.info(f"✅ User content file detected: {filename}")  # Log that we found a legitimate user file
                return True  # This is a valid user content file - process it
            
            return False  # File is not in our bucket - ignore it
            
        except Exception as e:  # Catch any errors during file path processing
            logger.error(f"Error checking file: {e}")  # Log the error for debugging
            return False  # On error, default to ignoring the file to be safe
    
    def _schedule_refresh(self):  # Method to schedule a debounced vector store refresh (avoids refreshing on every single file change)
        """Schedule a debounced refresh"""  # Docstring explaining this implements debouncing logic to batch rapid changes
        self.last_change_time = time.time()  # Record current timestamp as the last change time for debounce calculation
        
        # Cancel existing refresh task if any
        if self.refresh_task and not self.refresh_task.done():  # If there's already a pending refresh task that hasn't completed
            self.refresh_task.cancel()  # Cancel it (we'll schedule a new one, effectively resetting the debounce timer)
        
        # Schedule new refresh
        if self.loop and self.on_change_callback:  # Only schedule if we have both an event loop and a callback function configured
            self.refresh_task = asyncio.run_coroutine_threadsafe(  # Schedule async coroutine from synchronous context (file event handlers are sync)
                self._debounced_refresh(),  # The async function to run (waits for debounce period then triggers refresh)
                self.loop  # The event loop to schedule it on
            )
    
    async def _debounced_refresh(self):  # Async method that waits for the debounce period then triggers the refresh callback
        """Wait for debounce period and trigger refresh"""  # Docstring explaining this implements the actual debounce wait logic
        try:  # Wrap in try-except to handle cancellation and errors
            await asyncio.sleep(self.debounce_seconds)  # Wait for the configured debounce period (default 3 seconds) before proceeding
            
            # Check if enough time has passed since last change
            if time.time() - self.last_change_time >= self.debounce_seconds:  # Double-check that no new changes occurred during the sleep (in case of race conditions)
                logger.info("Debounce completed, refreshing vector store...")  # Log that debounce period finished and we're triggering refresh
                if self.on_change_callback:  # If callback is configured
                    await self.on_change_callback(force_refresh=True)  # Call the callback (typically vector store refresh) with force_refresh flag
        
        except asyncio.CancelledError:  # This exception is raised when the task is cancelled (happens when new file changes occur)
            logger.debug("Refresh cancelled due to new changes")  # Log that we cancelled this refresh because more changes came in
        except Exception as e:  # Catch any other unexpected errors during refresh
            logger.error(f"Error in debounced refresh: {e}")  # Log the error for debugging



class FileWatcherManager:  # Manager class that handles the lifecycle of the file watcher (setup, start, stop, restart)
    """Manages the file watcher lifecycle"""  # Docstring explaining this class provides a clean interface for watcher management
    
    def __init__(self, watch_path: str, bucket_name: str, debounce_seconds: int = 3):  # Constructor that configures the watcher with path, bucket, and debounce settings
        self.watch_path = os.path.abspath(os.path.expanduser(watch_path))  # Convert path to absolute path and expand ~ to home directory for robustness
        self.bucket_name = bucket_name  # Store the MinIO bucket name to pass to the event handler
        self.debounce_seconds = debounce_seconds  # Store debounce period (default 3 seconds)
        self.observer: Optional[Observer] = None  # Reference to the watchdog Observer instance (None until started)
        self.event_handler: Optional[MinIOFileWatcher] = None  # Reference to our custom event handler (None until created)
    
    def setup(self, loop: asyncio.AbstractEventLoop, callback) -> bool:  # Method to initialize and start the file watcher with event loop and callback
        """Setup and start the file watcher"""  # Docstring explaining this creates the watcher and begins monitoring
        try:  # Wrap in try-except to handle setup errors gracefully
            if not os.path.exists(self.watch_path):  # Check if the directory to watch actually exists
                logger.error(f"MinIO data path does not exist: {self.watch_path}")  # Log error if path is invalid
                return False  # Return False to indicate setup failed
            
            # Create event handler
            self.event_handler = MinIOFileWatcher(  # Create our custom event handler instance
                bucket_name=self.bucket_name,  # Pass bucket name for filtering
                debounce_seconds=self.debounce_seconds  # Pass debounce period
            )
            self.event_handler.set_loop(loop)  # Inject the asyncio event loop into the handler
            self.event_handler.set_callback(callback)  # Set the callback function (typically vector store refresh)
            
            # Create and start observer
            self.observer = Observer()  # Create watchdog Observer instance (the main file monitoring engine)
            self.observer.schedule(  # Register our event handler with the observer
                self.event_handler,  # Our handler that will process events
                self.watch_path,  # The directory path to watch
                recursive=True  # Watch subdirectories recursively (monitor entire MinIO data tree)
            )
            self.observer.start()  # Start the observer thread (begins monitoring in background)
            
            logger.info(f"✅ File watcher started for: {self.watch_path}")  # Log successful startup
            return True  # Return True to indicate setup succeeded
            
        except Exception as e:  # Catch any errors during setup
            logger.error(f"Could not setup file watcher: {e}")  # Log the error details
            return False  # Return False to indicate setup failed
    
    def stop(self):  # Method to gracefully stop the file watcher and clean up resources
        """Stop the file watcher"""  # Docstring explaining this shuts down monitoring
        if self.observer:  # Only proceed if observer exists
            logger.info("Stopping file watcher...")  # Log that we're stopping the watcher
            self.observer.stop()  # Signal the observer thread to stop monitoring
            self.observer.join()  # Wait for the observer thread to fully terminate (blocking call)
            self.observer = None  # Clear the observer reference (garbage collection)
        
        if self.event_handler and self.event_handler.refresh_task:  # If there's a pending refresh task
            if not self.event_handler.refresh_task.done():  # And it hasn't completed yet
                self.event_handler.refresh_task.cancel()  # Cancel it to prevent orphaned refresh attempts
    
    def is_running(self) -> bool:  # Method to check if the watcher is currently active
        """Check if watcher is running"""  # Docstring explaining this returns watcher status
        return self.observer is not None and self.observer.is_alive()  # Return True only if observer exists and its thread is alive
    
    def restart(self, loop: asyncio.AbstractEventLoop, callback) -> bool:  # Method to restart the watcher (useful for configuration changes or error recovery)
        """Restart the file watcher"""  # Docstring explaining this stops then starts the watcher
        self.stop()  # First stop the existing watcher (if running)
        return self.setup(loop, callback)  # Then setup and start a new watcher instance (return success/failure status)
