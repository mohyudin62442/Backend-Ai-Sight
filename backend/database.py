"""
Complete Database Configuration
Fixes all database connection and session management issues
"""

import os
from sqlalchemy import create_engine, MetaData, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool, QueuePool
import structlog

logger = structlog.get_logger()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://aioptimization:aioptimization@localhost:5432/aioptimization')
TEST_DATABASE_URL = os.getenv('TEST_DATABASE_URL', 'postgresql://aioptimization:aioptimization@localhost:5432/aioptimization_test')

# Use test database if in test environment
if os.getenv('ENVIRONMENT') == 'test':
    DATABASE_URL = TEST_DATABASE_URL

def create_database_engine(url: str = None):
    """Create database engine with proper configuration"""
    db_url = url or DATABASE_URL
    
    # Base engine configuration
    engine_kwargs = {
        'echo': os.getenv('DEBUG') == 'true' and os.getenv('ENVIRONMENT') != 'test',
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'connect_args': {}
    }
    
    # Production configuration
    if os.getenv('ENVIRONMENT') == 'production':
        engine_kwargs.update({
            'pool_size': 20,
            'max_overflow': 10,
            'poolclass': QueuePool,
            'pool_timeout': 30
        })
    # Test configuration
    elif os.getenv('ENVIRONMENT') == 'test' or 'test' in db_url:
        engine_kwargs.update({
            'pool_size': 5,
            'max_overflow': 0,
            'poolclass': StaticPool if 'sqlite' in db_url else QueuePool,
            'pool_timeout': 10
        })
        
        # SQLite specific settings
        if 'sqlite' in db_url:
            engine_kwargs['connect_args']['check_same_thread'] = False
    # Development configuration
    else:
        engine_kwargs.update({
            'pool_size': 10,
            'max_overflow': 5,
            'poolclass': QueuePool,
            'pool_timeout': 20
        })
    
    try:
        engine = create_engine(db_url, **engine_kwargs)
        
        # Add connection event listeners
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if 'sqlite' in db_url:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        
        @event.listens_for(engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ensure connection is valid on checkout"""
            if hasattr(dbapi_connection, 'ping'):
                try:
                    dbapi_connection.ping(reconnect=True)
                except Exception:
                    connection_proxy._pool.logger.warning("Connection ping failed, will reconnect")
                    raise
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Log success (but mask password in URL)
        safe_url = db_url.split('@')[-1] if '@' in db_url else db_url
        logger.info(f"Database engine created successfully: {safe_url}")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

# Create the main engine
engine = create_database_engine()

# Create sessionmaker with proper configuration
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Important for testing
)

# Create base class for models
Base = declarative_base()

# Database dependency for FastAPI
def get_database():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Alternative database dependency (used in some tests)
def get_db():
    """Alternative database dependency name"""
    return get_database()

# Initialize database tables
def init_database(engine_override=None):
    """Initialize database tables"""
    target_engine = engine_override or engine
    
    try:
        # Import all models to ensure they're registered
        import db_models
        
        # Create all tables
        Base.metadata.create_all(bind=target_engine)
        
        # Verify tables were created
        with target_engine.connect() as conn:
            if 'postgresql' in str(target_engine.url):
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                ))
                tables = [row[0] for row in result]
            elif 'sqlite' in str(target_engine.url):
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ))
                tables = [row[0] for row in result]
            else:
                tables = ["unknown_db_type"]
        
        logger.info(f"Database tables initialized successfully: {tables}")
        return tables
        
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise

# Health check function
def check_database_health():
    """Check database connectivity"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# Database utilities
class DatabaseUtils:
    """Database utility functions"""
    
    @staticmethod
    def create_test_engine():
        """Create a separate engine for testing"""
        test_url = TEST_DATABASE_URL
        return create_database_engine(test_url)
    
    @staticmethod
    def reset_database(engine_to_reset=None):
        """Reset database (drop and recreate all tables)"""
        target_engine = engine_to_reset or engine
        
        try:
            # Import models
            import db_models
            
            # Drop all tables
            Base.metadata.drop_all(bind=target_engine)
            
            # Recreate all tables
            Base.metadata.create_all(bind=target_engine)
            
            logger.info("Database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False
    
    @staticmethod
    def get_table_count(engine_to_check=None):
        """Get number of tables in database"""
        target_engine = engine_to_check or engine
        
        try:
            with target_engine.connect() as conn:
                if 'postgresql' in str(target_engine.url):
                    result = conn.execute(text(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
                    ))
                elif 'sqlite' in str(target_engine.url):
                    result = conn.execute(text(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    ))
                else:
                    return -1
                
                return result.scalar()
                
        except Exception as e:
            logger.error(f"Failed to get table count: {e}")
            return 0

# Transaction utilities
def with_transaction(func):
    """Decorator to wrap function in database transaction"""
    def wrapper(*args, **kwargs):
        db = SessionLocal()
        try:
            result = func(db, *args, **kwargs)
            db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            db.close()
    return wrapper

# Connection testing
def test_connection():
    """Test database connection and return status"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test_value"))
            test_value = result.scalar()
            
            if test_value == 1:
                return {
                    'status': 'connected',
                    'database_type': str(engine.url).split('://')[0],
                    'pool_size': engine.pool.size(),
                    'checked_out': engine.pool.checkedout(),
                    'checked_in': engine.pool.checkedin()
                }
            else:
                return {'status': 'error', 'message': 'Unexpected test result'}
                
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }

# Initialize on import if not in test mode
if __name__ == "__main__" or (os.getenv('ENVIRONMENT') != 'test' and 'pytest' not in os.getenv('_', '')):
    try:
        # Only auto-initialize in non-test environments
        if os.getenv('AUTO_INIT_DB', 'true').lower() == 'true':
            init_database()
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")

# Export commonly used items
__all__ = [
    'engine',
    'SessionLocal', 
    'Base',
    'get_database',
    'get_db',
    'init_database',
    'check_database_health',
    'DatabaseUtils',
    'with_transaction',
    'test_connection'
]