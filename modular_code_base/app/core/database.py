"""PostgreSQL database connection management"""  # Short description of what this module does: manages PostgreSQL connections for the app
import psycopg2  # Main PostgreSQL adapter library for Python to talk to the database [web:1]
from psycopg2 import pool  # Import connection pooling utilities from psycopg2 to reuse connections efficiently [web:2]
from psycopg2.extras import RealDictCursor  # Cursor class that returns rows as Python dictionaries instead of tuples [web:6]
from contextlib import contextmanager  # Helper to easily create context managers using the 'with' statement
from typing import Generator, Any, Optional  # Type hints for functions: Generator for yields, Any for generic types, Optional for nullable types
from app.core.config import get_settings  # Import function to load application configuration (like DB credentials) from central config


settings = get_settings()  # Load configuration once (e.g., db_host, db_name, db_user, db_password, db_port)


# Global connection pool
connection_pool = None  # Start with no pool; will be initialized later when first needed


def init_db_pool():  # Function to initialize the global database connection pool
    """Initialize database connection pool"""  # Docstring explaining this function sets up the pool
    global connection_pool  # Declare that we will modify the global connection_pool variable
    try:  # Try block to catch database-related errors
        connection_pool = psycopg2.pool.SimpleConnectionPool(  # Create a pool that keeps a limited number of open DB connections [web:2][web:4]
            minconn=1,  # Minimum 1 connection kept ready in the pool
            maxconn=10,  # Maximum 10 connections allowed at the same time
            host=settings.db_host,  # Database server hostname or IP from config
            database=settings.db_name,  # Database name to connect to from config
            user=settings.db_user,  # Database username from config
            password=settings.db_password,  # Database password from config
            port=settings.db_port,  # Database port (usually 5432) from config
            connect_timeout=5  # Fail connection attempt if it takes more than 5 seconds
        )
        print("✅ PostgreSQL connection pool initialized")  # Log success message to console for debugging
        return True  # Indicate that initialization worked
    except psycopg2.Error as e:  # Catch errors specifically from psycopg2 (database-related)
        print(f"  PostgreSQL connection failed: {e}")  # Print detailed error message for troubleshooting
        return False  # Indicate initialization failed
    except Exception as e:  # Catch any other unexpected errors
        print(f"  Database initialization error: {e}")  # Print generic database init error
        return False  # Indicate failure as well


@contextmanager  # Turn this function into a context manager so it can be used with 'with get_db_connection() as conn:'
def get_db_connection() -> Generator[Any, None, None]:  # Provide a database connection from the pool and ensure proper cleanup
    """Context manager for database connections"""  # Docstring explaining this wraps pool connections in a safe 'with' block
    global connection_pool  # Use the global pool variable
    
    if connection_pool is None:  # If the pool has not been created yet
        # Try to initialize if not already done
        if not init_db_pool():  # Attempt to set up the pool; if it fails
            raise Exception("Database connection pool not available")  # Stop execution with a clear error message
    
    conn = None  # Start with no connection assigned
    try:  # Try block to handle DB operations safely
        conn = connection_pool.getconn()  # Borrow a connection from the pool for this request [web:2][web:5]
        yield conn  # Give the connection to the caller (code inside the 'with' block runs here)
        conn.commit()  # If everything inside the 'with' block worked, commit the transaction to the database
    except psycopg2.Error as e:  # Handle database-specific errors
        if conn:  # If a connection was acquired
            conn.rollback()  # Undo any partial changes to keep database consistent
        print(f"Database error: {e}")  # Log the error message for debugging
        raise  # Re-raise the error so the caller knows something went wrong
    except Exception as e:  # Handle any non-psycopg2 errors
        if conn:  # If a connection exists
            conn.rollback()  # Roll back changes just in case
        raise  # Re-raise the exception for the caller to handle
    finally:  # This block always runs, whether there was an error or not
        if conn:  # If a connection was borrowed
            connection_pool.putconn(conn)  # Return the connection back to the pool so it can be reused [web:2][web:5]


@contextmanager  # Make this function usable with 'with get_db_cursor(conn) as cur:'
def get_db_cursor(conn):  # Create a cursor from a given connection
    """Context manager for database cursor with dict results"""  # Docstring: explains it returns dict-like rows
    cursor = conn.cursor(cursor_factory=RealDictCursor)  # Create a cursor that returns each row as a dictionary keyed by column name [web:6][web:9]
    try:  # Try block to safely use the cursor
        yield cursor  # Give the cursor to the caller for executing SQL queries
    finally:  # Always execute this after the 'with' block
        cursor.close()  # Close the cursor to free resources on the server and client


def create_tables():  # Function to create required database tables if they do not exist
    """Create necessary database tables"""  # Docstring describing that this sets up schema for predictions/feedback/knowledge
    try:  # Wrap whole table creation in a try block
        with get_db_connection() as conn:  # Open a managed database connection from the pool
            with get_db_cursor(conn) as cur:  # Open a managed cursor that returns dict rows
                # Predictions table
                cur.execute("""  # Run SQL to create 'predictions' table if it doesn't exist
                    CREATE TABLE IF NOT EXISTS predictions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  # Unique ID for each prediction, auto-generated UUID (requires pgcrypto extension) [web:7][web:10]
                        user_id VARCHAR(255) NOT NULL,  # ID of the user who triggered the prediction
                        pipeline_name VARCHAR(255),  # Name of the CI/CD pipeline the prediction belongs to
                        predicted_result VARCHAR(50),  # Predicted outcome (e.g., "success", "failure")
                        confidence_score INT,  # Confidence score (0–100 or similar scale) for the prediction
                        violated_rules INT,  # Number of rules or checks that were violated
                        pipeline_script_hash VARCHAR(255),  # Hash of pipeline script to uniquely identify version/content
                        detected_stack JSONB,  # JSON field describing detected tech stack or metadata (PostgreSQL JSONB type)
                        actual_result VARCHAR(50),  # Actual final result of the pipeline run for comparison
                        created_at TIMESTAMP DEFAULT NOW(),  # Timestamp when this prediction row was created
                        updated_at TIMESTAMP DEFAULT NOW(),  # Timestamp for last update (should be updated on changes)
                        feedback_received_at TIMESTAMP  # When feedback was received for this prediction (nullable)
                    )
                """)
                
                # Feedback table
                cur.execute("""  # Run SQL to create 'feedback' table if it doesn't exist
                    CREATE TABLE IF NOT EXISTS feedback (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  # Unique ID for each feedback entry, generated as UUID [web:7][web:10]
                        prediction_id UUID REFERENCES predictions(id),  # Link back to a prediction row via foreign key
                        user_id VARCHAR(255),  # ID of user providing the feedback (could match predictions.user_id)
                        actual_build_result VARCHAR(50),  # Real build outcome reported in feedback
                        correct_prediction BOOLEAN,  # Whether the model prediction matched the actual result
                        corrected_confidence INT,  # Adjusted confidence score given by user (if they think model was off)
                        missed_issues TEXT[],  # Array of text values listing issues the model missed
                        false_positives TEXT[],  # Array of text values listing issues that were incorrectly flagged
                        user_comments TEXT,  # Free-form feedback text from user
                        created_at TIMESTAMP DEFAULT NOW()  # When this feedback entry was created
                    )
                """)
                
                # Dynamic knowledge table
                cur.execute("""  # Run SQL to create 'dynamic_knowledge' table if it doesn't exist
                    CREATE TABLE IF NOT EXISTS dynamic_knowledge (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  # Unique ID for each rule/knowledge entry (auto UUID) [web:7][web:10]
                        source VARCHAR(255),  # Where this rule came from (e.g., "user_feedback", "logs", "docs")
                        rule_text TEXT,  # Human-readable description or rule content
                        confidence_score FLOAT,  # Confidence level of this rule being useful or correct
                        rule_type VARCHAR(100),  # Category/type of rule (e.g., "lint", "test", "security")
                        created_at TIMESTAMP DEFAULT NOW()  # Timestamp when this knowledge item was added
                    )
                """)
                
                print("✅ Database tables created/verified")  # Log success that all tables exist or have been created
                return True  # Indicate that table creation/verification succeeded
    except Exception as e:  # Catch any errors during table creation
        print(f"⚠️  Error creating tables: {e}")  # Log detailed error message
        return False  # Indicate failure so caller can react

<<<<<<< Updated upstream
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
=======

def close_db_pool():  # Function to fully close and clean up the connection pool (e.g., on app shutdown)
    """Close all database connections"""  # Docstring describing shutdown behavior for DB pool
    global connection_pool  # Use the global pool reference
    try:  # Try block to handle cleanup safely
        if connection_pool:  # Only act if a pool currently exists
            connection_pool.closeall()  # Close all connections managed by the pool and free resources [web:2][web:5]
            connection_pool = None  # Reset pool reference to None so it can be reinitialized later if needed
            print("✅ Database pool closed")  # Log that pool shutdown completed
    except Exception as e:  # Catch any unexpected issues during closing
        print(f"Error closing pool: {e}")  # Log error so it can be investigated
>>>>>>> Stashed changes
