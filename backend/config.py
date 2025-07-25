"""
Configuration management for AI Optimization Engine
Handles environment variables and application settings
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Keys
    anthropic_api_key: str
    openai_api_key: Optional[str] = None
    
    # Database
    database_url: str = "postgresql://user:pass@localhost/aioptimization"
    database_pool_size: int = 20
    database_max_overflow: int = 40
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_db: int = 0
    
    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    password_min_length: int = 8
    
    # API Settings
    api_rate_limit: int = 100
    api_rate_window: int = 3600  # 1 hour
    api_version: str = "1.0.0"
    
    # Application
    app_name: str = "AI Optimization Engine"
    debug: bool = False
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"
    
    # CORS
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ]
    
    # Tracking
    enable_real_tracking: bool = True
    tracking_api_endpoint: str = "/track-bot"
    tracking_script_version: str = "1.0.0"
    geoip_path: Optional[str] = "./GeoLite2-City.mmdb"
    
    # LLM Settings
    claude_model: str = "claude-3-sonnet-20240229"
    gpt_model: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.3
    request_timeout: int = 30  # seconds
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000  # Maximum number of cached items
    
    # File Upload
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    allowed_log_formats: List[str] = ["nginx", "apache", "cloudflare"]
    temp_upload_path: str = "/tmp/uploads"
    
    # Analysis Settings
    max_concurrent_analyses: int = 10
    analysis_timeout: int = 300  # 5 minutes
    max_queries_per_analysis: int = 50
    max_content_chunks: int = 1000
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30  # seconds
    
    # Email (for notifications)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_tls: bool = True
    notification_from_email: str = "noreply@aioptimization.com"
    
    # Subscription Plans
    plan_limits = {
        "free": {
            "analyses_per_month": 2,
            "brands": 1,
            "api_calls": 100,
            "tracking_events": 1000,
            "log_upload_size": 10 * 1024 * 1024  # 10MB
        },
        "starter": {
            "analyses_per_month": 10,
            "brands": 3,
            "api_calls": 1000,
            "tracking_events": 50000,
            "log_upload_size": 50 * 1024 * 1024  # 50MB
        },
        "growth": {
            "analyses_per_month": 50,
            "brands": 10,
            "api_calls": 5000,
            "tracking_events": 500000,
            "log_upload_size": 100 * 1024 * 1024  # 100MB
        },
        "professional": {
            "analyses_per_month": -1,  # Unlimited
            "brands": -1,  # Unlimited
            "api_calls": 20000,
            "tracking_events": 5000000,
            "log_upload_size": 500 * 1024 * 1024  # 500MB
        }
    }
    
    @validator('anthropic_api_key')
    def validate_anthropic_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError('Invalid Anthropic API key format')
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Initialize settings
settings = Settings()

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': settings.log_level,
            'formatter': 'default' if settings.environment != 'production' else 'json',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/error.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # root logger
            'level': settings.log_level,
            'handlers': ['console', 'file', 'error_file'] if settings.environment == 'production' else ['console']
        },
        'uvicorn': {
            'level': 'INFO',
            'handlers': ['console']
        },
        'sqlalchemy': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
}

# Feature flags
FEATURE_FLAGS = {
    'enable_real_tracking': settings.enable_real_tracking,
    'enable_email_notifications': bool(settings.smtp_host),
    'enable_metrics': settings.enable_metrics,
    'enable_scheduled_analyses': settings.environment == 'production',
    'enable_api_keys': settings.environment != 'development',
    'enable_rate_limiting': settings.environment != 'development'
}

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'default': {
        'rate': settings.api_rate_limit,
        'window': settings.api_rate_window
    },
    'tracking': {
        'rate': 1000,  # Higher limit for tracking endpoints
        'window': 60   # 1 minute
    },
    'analysis': {
        'rate': 10,    # Lower limit for resource-intensive operations
        'window': 3600 # 1 hour
    }
}
