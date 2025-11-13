"""FastAPI main application with graceful shutdown and MinIO monitoring"""
import logging
import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


# Import after logging is configured
from app.api.routes import router
from app.services.vectorstore_service import VectorStoreService
from app.core.config import get_settings
from app.core.database import init_db_pool, create_tables
from app.monitoring.change_detector import FileChangeTracker
from app.monitoring.file_watcher import FileWatcherManager


settings = get_settings()
vectorstore_service = VectorStoreService()
shutdown_event = asyncio.Event()

# Initialize monitoring components
file_tracker = FileChangeTracker(metadata_path=settings.metadata_path)
watcher_manager = None  # Will be initialized in lifespan


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Ctrl+C detected - initiating graceful shutdown...")
    sys.exit(0)


async def refresh_vectorstore_callback(force_refresh: bool = False):
    """Callback for file watcher - rebuilds vectorstore on MinIO changes"""
    try:
        logger.info("MinIO file change detected - refreshing vectorstore...")
        
        # Get current MinIO snapshot
        from app.services.minio_service import MinIOService
        minio_service = MinIOService()
        
        current_snapshot = {}
        objects = minio_service.list_objects()
        
        for obj in objects:
            current_snapshot[obj.object_name] = {
                'etag': obj.etag,
                'last_modified': obj.last_modified,
                'size': obj.size
            }
        
        # Detect changes
        changes = file_tracker.detect_changes(current_snapshot)
        
        if changes['added'] or changes['modified'] or changes['deleted']:
            logger.info(
                f"Changes detected: "
                f"+{len(changes['added'])} "
                f"~{len(changes['modified'])} "
                f"-{len(changes['deleted'])}"
            )
            
            # Rebuild vectorstore
            logger.info("Rebuilding vectorstore...")
            vectorstore_service.load_or_build(force_rebuild=True)
            logger.info("✅ Vectorstore updated")
        else:
            logger.info("No significant changes detected")
            
    except Exception as e:
        logger.error(f"❌ Error refreshing vectorstore: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager - startup and shutdown"""
    
    # ========= STARTUP =========
    logger.info("Starting Yocto Build Analyzer...")
    
    # Initialize database
    try:
        logger.info("Initializing database pool...")
        init_db_pool()
        create_tables()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")
    
    # Initialize vector store
    try:
        logger.info("Initializing vector store...")
        vectorstore_service.load_or_build()
        logger.info("✅ Vector store ready")
    except Exception as e:
        logger.warning(f"Vector store warning: {e}")
    
    # Setup MinIO file watcher
    global watcher_manager
    if settings.watch_enabled:
        try:
            logger.info("Setting up MinIO file watcher...")
            loop = asyncio.get_event_loop()
            
            watcher_manager = FileWatcherManager(
                watch_path=settings.minio_data_path,
                bucket_name=settings.minio_bucket,
                debounce_seconds=settings.debounce_seconds
            )
            
            if watcher_manager.setup(loop=loop, callback=refresh_vectorstore_callback):
                logger.info(f"✅ File watcher active on {settings.minio_data_path}")
            else:
                logger.warning("File watcher setup failed")
                watcher_manager = None
                
        except Exception as e:
            logger.warning(f"Could not setup file watcher: {e}")
            watcher_manager = None
    else:
        logger.info("File watching disabled (watch_enabled=False)")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 80)
    logger.info("✅ Application ready on http://0.0.0.0:8000")
    logger.info("=" * 80)
    
    yield
    
    # ========= SHUTDOWN =========
    logger.info("Shutting down...")
    
    # Stop file watcher
    if watcher_manager:
        try:
            logger.info("Stopping file watcher...")
            watcher_manager.stop()
            logger.info("✅ File watcher stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping watcher: {e}")
    
    # Cancel all running tasks
    try:
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        
        logger.info("✅ All tasks cancelled")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    
    logger.info("Application stopped")


# Create FastAPI app with lifespan (NO DOCS)
app = FastAPI(
    title="Yocto Build Analyzer",
    description="RAG-based Yocto build prediction system with feedback learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,      # Disable docs
    redoc_url=None,     # Disable redoc
    openapi_url=None    # Disable openapi
)


# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(router)


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle any unhandled exceptions"""
    logger.exception(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# Handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    logger.error(f"❌ Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yocto-analyzer",
        "version": "1.0.0",
        "vectorstore_loaded": vectorstore_service.is_loaded(),
        "file_watcher_active": watcher_manager.is_running() if watcher_manager else False
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Yocto Build Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "features": {
            "rag_analysis": True,
            "feedback_learning": True,
            "minio_monitoring": settings.watch_enabled,
            "workspace_integration": True
        }
    }


# ========= MAIN ENTRY POINT =========

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 80)
    logger.info("Starting Yocto Build Analyzer")
    logger.info("=" * 80)
    
    # Create uvicorn config
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
    
    # Run server
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Server error: {e}")
        sys.exit(1)
