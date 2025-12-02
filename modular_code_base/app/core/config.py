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
    ollama_llm_model: str = "qwen2.5:1.5b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_llm_temperature: float = 0.1  # Lower = more deterministic, Higher = more creative
    ollama_llm_timeout: int = 180  # Timeout in seconds for LLM calls
    
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
    
    # ========= PREDICTION CONFIGURATION =========
    # Confidence-based outcome thresholds (0-100)
    prediction_fail_max: int = 35       # 0-35% = FAIL
    prediction_high_risk_max: int = 70  # 35-70% = HIGH_RISK
    # 70-100% = PASS (implicit)
    
    # Violation ratio thresholds (0.0 to 1.0) - determines base outcome
    prediction_fail_violation_threshold: float = 0.5  # >= 50% violations = likely FAIL
    prediction_high_risk_violation_threshold: float = 0.3  # >= 30% violations = likely HIGH_RISK
    
    # Confidence boost for PASS predictions
    prediction_pass_confidence_boost: int = 10  # Add 10% to PASS confidence
    
    # Rule satisfaction threshold (0.0 to 1.0)
    rule_satisfaction_keyword_threshold: float = 0.6  # 60% of keywords must match
    
    # Prediction behavior
    prediction_unknown_on_no_rules: bool = True  # Return UNKNOWN when no rules found
    
    # ========= END PREDICTION CONFIGURATION =========
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
