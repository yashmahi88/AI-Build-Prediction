"""Vectorstore file watcher service"""
import time
import logging
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class VectorStoreReloader(FileSystemEventHandler):
    """Monitors vectorstore files and triggers reload on changes"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.last_reload = 0
        self.reload_delay = 3  # Wait 3 seconds to batch multiple changes
    
    def on_modified(self, event):
        """Triggered when vectorstore files are modified"""
        if event.is_directory:
            return
        
        # Only react to FAISS index files
        if 'index.faiss' in event.src_path or 'index.pkl' in event.src_path:
            current_time = time.time()
            
            # Debounce: only reload if enough time passed since last reload
            if current_time - self.last_reload > self.reload_delay:
                logger.info(f" Vectorstore file changed: {event.src_path}")
                self._trigger_reload()
                self.last_reload = current_time
    
    def _trigger_reload(self):
        """Call the API rebuild endpoint"""
        try:
            logger.info(" Triggering vectorstore reload via API...")
            response = requests.post(
                f"{self.api_base_url}/api/rebuild",
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Vectorstore reloaded successfully")
            else:
                logger.error(f"❌ Reload failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f" Failed to trigger reload: {e}")


def start_vectorstore_watcher(vectorstore_path: str, api_base_url="http://localhost:8000"):
    """Start watching vectorstore directory for changes"""
    event_handler = VectorStoreReloader(api_base_url)
    observer = Observer()
    observer.schedule(event_handler, vectorstore_path, recursive=False)
    observer.start()
    logger.info(f" Started watching vectorstore at: {vectorstore_path}")
    return observer



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
