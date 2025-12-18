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
    minio_bucket_name: str = "jenkins-logs"  # ← ADD THIS
    minio_secure: bool = False  # ← ADD THIS
    minio_data_path: str = "/var/minio-data"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:1.5b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_llm_temperature: float = 0.4  # ← ADD THIS
    ollama_llm_timeout: int = 120  # ← ADD THIS
    
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
    db_password: str = "eic@123456"  # ← UPDATE THIS
    
    # Feature Flags
    force_rebuild_on_startup: bool = False
    force_workspace_rebuild: bool = False
    watch_enabled: bool = True
    check_minio_on_every_query: bool = False
    incremental_updates: bool = True
    
    # Yocto Documentation
    yocto_doc_scrape_enabled: bool = True  # ← ADD THIS
    yocto_doc_update_interval_hours: int = 24  # ← ADD THIS
    
    # Rule Evaluation Thresholds
    # Rule Evaluation Thresholds
    rule_satisfaction_keyword_threshold: float = 0.4
    prediction_high_risk_violation_threshold: float = 0.3
    prediction_fail_violation_threshold: float = 0.5
    prediction_pass_confidence_boost: int = 10
    prediction_fail_max: int = 40
    prediction_high_risk_min: int = 41
    prediction_high_risk_max: int = 69
    prediction_pass_min: int = 70

    
    # Debounce Settings
    debounce_seconds: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
