"""Application configuration with Pydantic Settings"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Application configuration from environment variables"""
    
    # MinIO Configuration
    minio_endpoint: str = "https://localhost:9000"
    minio_access_key: str = "admin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "test1"
    minio_data_path: str = "/var/minio-data"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:1.5b" ##"mistral:7b-instruct-q4_0" ##"codellama:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Paths
    vector_store_path: str = "./vectorstore"
    metadata_path: str = "./vectorstore_metadata"
    workspace_dir: str = "/var/jenkins_home/workspace/Yocto-Build-Pipeline"
    workspace_state_path: str = "./workspace_files_state.pkl"
    build_lock_file: str = "./vector_store_build.lock"
    
    # PostgreSQL Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_feedback"
    db_user: str = "rag_user"
    db_password: str = ""
    
    # Feature Flags
    force_rebuild_on_startup: bool = False
    force_workspace_rebuild: bool = False
    watch_enabled: bool = True
    check_minio_on_every_query: bool = False
    incremental_updates: bool = True
    
    # Debounce Settings
    debounce_seconds: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
