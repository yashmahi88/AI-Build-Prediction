"""PostgreSQL database connection management"""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator, Any, Optional
from app.core.config import get_settings

settings = get_settings()

# Global connection pool
connection_pool = None

def init_db_pool():
    """Initialize database connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            port=settings.db_port,
            connect_timeout=5
        )
        print("✅ PostgreSQL connection pool initialized")
        return True
    except psycopg2.Error as e:
        print(f"  PostgreSQL connection failed: {e}")
        return False
    except Exception as e:
        print(f"  Database initialization error: {e}")
        return False

@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """Context manager for database connections"""
    global connection_pool
    
    if connection_pool is None:
        # Try to initialize if not already done
        if not init_db_pool():
            raise Exception("Database connection pool not available")
    
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
        conn.commit()
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Database error: {e}")
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

@contextmanager
def get_db_cursor(conn):
    """Context manager for database cursor with dict results"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
    finally:
        cursor.close()

def create_tables():
    """Create necessary database tables"""
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Predictions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id VARCHAR(255) NOT NULL,
                        pipeline_name VARCHAR(255),
                        predicted_result VARCHAR(50),
                        confidence_score INT,
                        violated_rules INT,
                        pipeline_script_hash VARCHAR(255),
                        detected_stack JSONB,
                        actual_result VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        feedback_received_at TIMESTAMP
                    )
                """)
                
                # Feedback table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        prediction_id UUID REFERENCES predictions(id),
                        user_id VARCHAR(255),
                        actual_build_result VARCHAR(50),
                        correct_prediction BOOLEAN,
                        corrected_confidence INT,
                        missed_issues TEXT[],
                        false_positives TEXT[],
                        user_comments TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Dynamic knowledge table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dynamic_knowledge (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source VARCHAR(255),
                        rule_text TEXT,
                        confidence_score FLOAT,
                        rule_type VARCHAR(100),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                print("✅ Database tables created/verified")
                return True
    except Exception as e:
        print(f"⚠️  Error creating tables: {e}")
        return False

def close_db_pool():
    """Close all database connections"""
    global connection_pool
    try:
        if connection_pool:
            connection_pool.closeall()
            connection_pool = None
            print("✅ Database pool closed")
    except Exception as e:
        print(f"Error closing pool: {e}")
