"""FastAPI main application with vectorstore file monitoring"""
import logging
import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers').setLevel(logging.WARNING)

from app.api.routes import router
from app.services.vectorstore_service import VectorStoreService
from app.core.config import get_settings
from app.core.database import init_db_pool, create_tables

settings = get_settings()
vectorstore_service = VectorStoreService()
vectorstore_watcher_observer = None

def signal_handler(signum, frame):
    logger.info("Ctrl+C detected - initiating graceful shutdown...")
    sys.exit(0)

def reload_vectorstore_from_disk():
    """Reload vectorstore when files change on disk"""
    try:
        logger.info("Vectorstore files changed - reloading...")
        vectorstore_service.load()
        logger.info("Vectorstore reloaded successfully")
    except Exception as e:
        logger.error(f"Error reloading vectorstore: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Yocto Build Analyzer...")
    
    try:
        logger.info("Initializing database...")
        init_db_pool()
        create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")
    
    try:
        logger.info("Initializing vectorstore...")
        vectorstore_service.load_or_build()
        logger.info("Vectorstore ready")
    except Exception as e:
        logger.warning(f"Vectorstore warning: {e}")
    
    global vectorstore_watcher_observer
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        import time
        
        class VectorStoreFileHandler(FileSystemEventHandler):
            def __init__(self):
                self.last_reload = 0
                self.debounce_delay = 3
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                if 'index.faiss' in event.src_path or 'index.pkl' in event.src_path:
                    current_time = time.time()
                    if current_time - self.last_reload > self.debounce_delay:
                        logger.info(f"File changed: {event.src_path}")
                        reload_vectorstore_from_disk()
                        self.last_reload = current_time
        
        vectorstore_path = settings.vector_store_path or "./vectorstore"
        event_handler = VectorStoreFileHandler()
        vectorstore_watcher_observer = Observer()
        vectorstore_watcher_observer.schedule(event_handler, vectorstore_path, recursive=False)
        vectorstore_watcher_observer.start()
        logger.info(f"Vectorstore file watcher active on {vectorstore_path}")
        
    except Exception as e:
        logger.warning(f"Could not setup vectorstore watcher: {e}")
        vectorstore_watcher_observer = None
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 80)
    logger.info("Application ready on http://0.0.0.0:8000")
    logger.info("=" * 80)
    
    yield
    
    logger.info("Shutting down...")
    
    if vectorstore_watcher_observer:
        try:
            logger.info("Stopping vectorstore watcher...")
            vectorstore_watcher_observer.stop()
            vectorstore_watcher_observer.join()
            logger.info("Vectorstore watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping watcher: {e}")
    
    try:
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info("All tasks cancelled")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application stopped")

app = FastAPI(
    title="Yocto Build Analyzer",
    description="RAG-based Yocto build prediction system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "yocto-analyzer",
        "version": "1.0.0",
        "vectorstore_loaded": vectorstore_service.is_loaded(),
        "vectorstore_watcher_active": vectorstore_watcher_observer.is_alive() if vectorstore_watcher_observer else False
    }

@app.get("/")
async def root():
    return {
        "message": "Yocto Build Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "features": {
            "rag_analysis": True,
            "feedback_learning": True,
            "vectorstore_auto_reload": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 80)
    logger.info("Starting Yocto Build Analyzer")
    logger.info("=" * 80)
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        timeout_notify=30,
        timeout_graceful_shutdown=30,
    )
    
    server = uvicorn.Server(config)
    
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
