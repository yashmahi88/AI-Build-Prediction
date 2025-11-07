"""FastAPI main application with graceful shutdown"""
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import after logging is configured
from app.api.routes import router
from app.services.vectorstore_service import VectorStoreService
from app.core.config import get_settings
from app.core.database import init_db_pool, create_tables


settings = get_settings()
vectorstore_service = VectorStoreService()
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("⏹️ Ctrl+C detected - initiating graceful shutdown...")
    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager - startup and shutdown"""
    
    # ========= STARTUP =========
    logger.info("🚀 Starting Yocto Build Analyzer...")
    
    # Initialize database
    try:
        logger.info("📦 Initializing database pool...")
        init_db_pool()
        create_tables()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Database init warning: {e}")
    
    # Initialize vector store
    try:
        logger.info("📚 Initializing vector store...")
        vectorstore_service.load_or_build()
        logger.info("✅ Vector store ready")
    except Exception as e:
        logger.warning(f"⚠️ Vector store warning: {e}")
    
    logger.info("✅ Application ready on http://0.0.0.0:8000")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # ========= SHUTDOWN =========
    logger.info("🛑 Shutting down...")
    
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        
        logger.info("✅ All tasks cancelled")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("🛑 Application stopped")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Yocto Build Analyzer",
    description="RAG-based Yocto build prediction system with feedback learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes - IMPORTANT: routes already have /api prefix
app.include_router(router)


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle any unhandled exceptions"""
    logger.exception(f"🔴 Unhandled exception: {exc}")
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


# Health check endpoint (outside router)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yocto-analyzer",
        "version": "1.0.0"
    }


# Root endpoint (outside router)
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Yocto Build Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "health": "GET /health",
            "api_docs": "GET /docs",
            "redoc": "GET /redoc",
            "openapi": "GET /openapi.json"
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
        logger.info("⏹️ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
