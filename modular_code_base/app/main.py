"""FastAPI main application with graceful shutdown"""
import logging
import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.vectorstore_service import VectorStoreService
from app.core.config import get_settings
from app.core.database import init_db_pool, create_tables
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()
vectorstore_service = VectorStoreService()

# Track running tasks for graceful shutdown
running_tasks = set()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("⏹️ Ctrl+C detected - initiating graceful shutdown...")
    
    # Cancel all running tasks
    for task in asyncio.all_tasks():
        task.cancel()
    
    print("\n🛑 Stopping all processes...")
    sys.exit(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Yocto Build Analyzer...")
    
    try:
        print("📦 Initializing database pool...")
        init_db_pool()
        create_tables()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database init warning: {e}")
    
    try:
        print("📚 Initializing vector store...")
        vectorstore_service.load_or_build()
        print("✅ Vector store ready")
    except Exception as e:
        print(f"⚠️ Vector store warning: {e}")
    
    print("✅ Application ready!")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    
    # Cancel all tasks
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()

app = FastAPI(
    title="Yocto Build Analyzer",
    description="RAG-based Yocto build prediction system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"🔴 Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"❌ Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.get("/")
async def root():
    return {
        "message": "Yocto Build Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create server with timeout settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        timeout_keep_alive=30,  # Keep-alive timeout
        timeout_notify=30,      # Notify timeout
    )
    
    server = uvicorn.Server(config)
    
    # Run with signal handling
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
        sys.exit(0)
