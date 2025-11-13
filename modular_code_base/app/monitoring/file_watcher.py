"""File system watcher for MinIO data directory"""
import os
import time
import asyncio
import logging
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class MinIOFileWatcher(FileSystemEventHandler):
    """File system event handler for MinIO data directory changes"""
    
    def __init__(self, bucket_name: str, debounce_seconds: int = 3):
        super().__init__()
        self.bucket_name = bucket_name
        self.debounce_seconds = debounce_seconds
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_change_time = 0
        self.refresh_task: Optional[asyncio.Task] = None
        self.on_change_callback = None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async operations"""
        self.loop = loop
    
    def set_callback(self, callback):
        """Set callback function to call on file changes"""
        self.on_change_callback = callback
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and self._is_user_content_file(event.src_path):
            logger.info(f"User file modified: {event.src_path}")
            self._schedule_refresh()
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and self._is_user_content_file(event.src_path):
            logger.info(f"User file created: {event.src_path}")
            self._schedule_refresh()
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory and self._is_user_content_file(event.src_path):
            logger.info(f"User file deleted: {event.src_path}")
            self._schedule_refresh()
    
    def _is_user_content_file(self, filepath: str) -> bool:
        """Check if the file is actual user content, not MinIO system files"""
        try:
            normalized_path = filepath.replace('\\', '/')
            
            # ✅ CRITICAL: Skip MinIO system files completely
            skip_patterns = [
                '.minio.sys',
                'xl.meta',
                '.usage-cache',
                '.usage.json',
                '.bloomcycle',
                'tmp/',
                '.tmp',
                'part.minio',
                '.backup',
                '.bkp',
                '.trash',
                'multipart'
            ]
            
            for pattern in skip_patterns:
                if pattern in normalized_path:
                    return False
            
            # ✅ Only consider files in our bucket
            if f"{self.bucket_name}" in normalized_path:
                filename = os.path.basename(filepath)
                
                # Skip hidden files, temp files, and empty names
                if filename.startswith('.') or filename.endswith('.tmp'):
                    return False
                if not filename.strip():
                    return False
                
                logger.info(f"✅ User content file detected: {filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return False
    
    def _schedule_refresh(self):
        """Schedule a debounced refresh"""
        self.last_change_time = time.time()
        
        # Cancel existing refresh task if any
        if self.refresh_task and not self.refresh_task.done():
            self.refresh_task.cancel()
        
        # Schedule new refresh
        if self.loop and self.on_change_callback:
            self.refresh_task = asyncio.run_coroutine_threadsafe(
                self._debounced_refresh(),
                self.loop
            )
    
    async def _debounced_refresh(self):
        """Wait for debounce period and trigger refresh"""
        try:
            await asyncio.sleep(self.debounce_seconds)
            
            # Check if enough time has passed since last change
            if time.time() - self.last_change_time >= self.debounce_seconds:
                logger.info("Debounce completed, refreshing vector store...")
                if self.on_change_callback:
                    await self.on_change_callback(force_refresh=True)
        
        except asyncio.CancelledError:
            logger.debug("Refresh cancelled due to new changes")
        except Exception as e:
            logger.error(f"Error in debounced refresh: {e}")


class FileWatcherManager:
    """Manages the file watcher lifecycle"""
    
    def __init__(self, watch_path: str, bucket_name: str, debounce_seconds: int = 3):
        self.watch_path = os.path.abspath(os.path.expanduser(watch_path))
        self.bucket_name = bucket_name
        self.debounce_seconds = debounce_seconds
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[MinIOFileWatcher] = None
    
    def setup(self, loop: asyncio.AbstractEventLoop, callback) -> bool:
        """Setup and start the file watcher"""
        try:
            if not os.path.exists(self.watch_path):
                logger.error(f"MinIO data path does not exist: {self.watch_path}")
                return False
            
            # Create event handler
            self.event_handler = MinIOFileWatcher(
                bucket_name=self.bucket_name,
                debounce_seconds=self.debounce_seconds
            )
            self.event_handler.set_loop(loop)
            self.event_handler.set_callback(callback)
            
            # Create and start observer
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                self.watch_path,
                recursive=True
            )
            self.observer.start()
            
            logger.info(f"✅ File watcher started for: {self.watch_path}")
            return True
            
        except Exception as e:
            logger.error(f"Could not setup file watcher: {e}")
            return False
    
    def stop(self):
        """Stop the file watcher"""
        if self.observer:
            logger.info("Stopping file watcher...")
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        if self.event_handler and self.event_handler.refresh_task:
            if not self.event_handler.refresh_task.done():
                self.event_handler.refresh_task.cancel()
    
    def is_running(self) -> bool:
        """Check if watcher is running"""
        return self.observer is not None and self.observer.is_alive()
    
    def restart(self, loop: asyncio.AbstractEventLoop, callback) -> bool:
        """Restart the file watcher"""
        self.stop()
        return self.setup(loop, callback)
