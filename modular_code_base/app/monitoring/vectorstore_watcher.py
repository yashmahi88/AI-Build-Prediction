"""Vectorstore file watcher service"""  # Module docstring describing this file monitors vectorstore index files and triggers API reloads when they change
import time  # Time library for debouncing logic and timestamps
import logging  # Standard Python logging library to track reload events
import requests  # HTTP library for making API calls to trigger vectorstore reload endpoint
from watchdog.observers import Observer  # File system monitoring class that watches directories for changes
from watchdog.events import FileSystemEventHandler  # Base class for handling file system events (modify, create, delete)


logger = logging.getLogger(__name__)  # Create logger instance for this module to output watcher events


class VectorStoreReloader(FileSystemEventHandler):  # Event handler class that monitors FAISS vectorstore files and triggers API rebuilds on changes
    """Monitors vectorstore files and triggers reload on changes"""  # Docstring explaining this class watches for vectorstore updates and calls reload API
    
    def __init__(self, api_base_url="http://localhost:8000"):  # Constructor that initializes the reloader with API base URL (defaults to localhost:8000)
        self.api_base_url = api_base_url  # Store the API base URL where the rebuild endpoint lives
        self.last_reload = 0  # Timestamp of the last reload trigger (initialized to 0 meaning no reloads yet)
        self.reload_delay = 3  # Debounce period in seconds (wait 3 seconds to batch rapid changes and avoid triggering reload on every file write)
    
    def on_modified(self, event):  # Handler method called by watchdog when a file is modified in the watched directory
        """Triggered when vectorstore files are modified"""  # Docstring explaining this responds to file modification events
        if event.is_directory:  # Check if the event is for a directory change
            return  # Ignore directory events, only care about file changes
        
        # Only react to FAISS index files
        if 'index.faiss' in event.src_path or 'index.pkl' in event.src_path:  # Check if the modified file is a FAISS index file (index.faiss binary file or index.pkl metadata file)
            current_time = time.time()  # Get current timestamp in seconds since epoch
            
            # Debounce: only reload if enough time passed since last reload
            if current_time - self.last_reload > self.reload_delay:  # Check if at least 3 seconds have passed since last reload (debouncing to avoid excessive reloads)
                logger.info(f" Vectorstore file changed: {event.src_path}")  # Log which file changed 
                self._trigger_reload()  # Call internal method to trigger the API reload
                self.last_reload = current_time  # Update last reload timestamp to current time for next debounce check
    
    def _trigger_reload(self):  # Private method that calls the API endpoint to reload the vectorstore in memory
        """Call the API rebuild endpoint"""  # Docstring explaining this makes HTTP POST request to trigger vectorstore reload
        try:  # Wrap in try-except to handle network errors gracefully
            logger.info(" Triggering vectorstore reload via API...")  # Log that we're about to call the API 
            response = requests.post(  # Make HTTP POST request to the rebuild endpoint
                f"{self.api_base_url}/api/rebuild",  # Construct full URL by combining base URL with /api/rebuild path
                timeout=30  # Set 30-second timeout to prevent hanging if API is slow or unresponsive
            )
            
            if response.status_code == 200:  # Check if API returned success status code
                logger.info("✅ Vectorstore reloaded successfully")  # Log success with checkmark emoji
            else:  # If API returned non-200 status code
                logger.error(f"❌ Reload failed: {response.status_code}")  # Log failure with X emoji and status code for debugging
        
        except Exception as e:  # Catch any exceptions during API call (network errors, timeouts, etc.)
            logger.error(f" Failed to trigger reload: {e}")  # Log error with details 



def start_vectorstore_watcher(vectorstore_path: str, api_base_url="http://localhost:8000"):  # Function to initialize and start the vectorstore file watcher
    """Start watching vectorstore directory for changes"""  # Docstring explaining this sets up monitoring of vectorstore directory
    event_handler = VectorStoreReloader(api_base_url)  # Create instance of our custom event handler with API base URL
    observer = Observer()  # Create watchdog Observer instance (the main file monitoring engine)
    observer.schedule(event_handler, vectorstore_path, recursive=False)  # Register our event handler to watch the vectorstore path (recursive=False means only watch the directory itself, not subdirectories)
    observer.start()  # Start the observer thread (begins monitoring in background)
    logger.info(f" Started watching vectorstore at: {vectorstore_path}")  # Log that watcher is running with the monitored path 
    return observer  # Return the observer instance so caller can stop it later if needed



# """MinIO file change detection and tracking"""
# import os
# import pickle
# import hashlib
# import logging
# from typing import Dict, Set, Tuple
# from datetime import datetime

# logger = logging.getLogger(__name__)


# class FileChangeTracker:
#     """Track file changes in MinIO bucket"""
    
#     def __init__(self, metadata_path: str = "./vectorstore_metadata"):
#         self.metadata_path = metadata_path
#         self.file_metadata: Dict[str, dict] = {}
#         self._load_metadata()
    
#     def _load_metadata(self):
#         """Load existing file metadata"""
#         if os.path.exists(self.metadata_path):
#             try:
#                 with open(self.metadata_path, 'rb') as f:
#                     self.file_metadata = pickle.load(f)
#                 logger.info(f"Loaded metadata for {len(self.file_metadata)} files")
#             except Exception as e:
#                 logger.warning(f"Could not load metadata: {e}")
#                 self.file_metadata = {}
    
#     def _save_metadata(self):
#         """Save file metadata to disk"""
#         try:
#             os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
#             with open(self.metadata_path, 'wb') as f:
#                 pickle.dump(self.file_metadata, f)
#         except Exception as e:
#             logger.error(f"Could not save metadata: {e}")
    
#     def detect_changes(self, current_snapshot: Dict[str, dict]) -> Dict[str, Set[str]]:
#         """
#         Detect changes between current and stored snapshots
        
#         Returns:
#             dict with keys: 'added', 'modified', 'deleted', 'unchanged'
#         """
#         current_keys = set(current_snapshot.keys())
#         stored_keys = set(self.file_metadata.keys())
        
#         added = current_keys - stored_keys
#         deleted = stored_keys - current_keys
#         potentially_modified = current_keys & stored_keys
        
#         modified = set()
#         unchanged = set()
        
#         for file_key in potentially_modified:
#             current_info = current_snapshot[file_key]
#             stored_info = self.file_metadata[file_key]
            
#             # Check if file changed (compare etag or last_modified)
#             if (current_info.get('etag') != stored_info.get('etag') or
#                 current_info.get('last_modified') != stored_info.get('last_modified')):
#                 modified.add(file_key)
#             else:
#                 unchanged.add(file_key)
        
#         # Update stored metadata
#         self.file_metadata = current_snapshot
#         self._save_metadata()
        
#         return {
#             'added': added,
#             'modified': modified,
#             'deleted': deleted,
#             'unchanged': unchanged
#         }
    
#     def remove_file_metadata(self, file_key: str):
#         """Remove metadata for a deleted file"""
#         if file_key in self.file_metadata:
#             del self.file_metadata[file_key]
#             self._save_metadata()
    
#     def get_file_info(self, file_key: str) -> dict:
#         """Get stored metadata for a file"""
#         return self.file_metadata.get(file_key, {})
