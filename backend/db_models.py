"""
Fixed Database Models - Resolves SQLAlchemy metadata conflict
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON, Text, 
    ForeignKey, CheckConstraint, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Brand(Base):
    """Brand model with proper constraints"""
    __tablename__ = "brands"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    website_url = Column(String(500), nullable=True)
    industry = Column(String(100), nullable=True)
    tracking_enabled = Column(Boolean, default=False)
    tracking_script_installed = Column(Boolean, default=False)
    api_key = Column(String(100), nullable=True, unique=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    analyses = relationship("Analysis", back_populates="brand", cascade="all, delete-orphan")
    metrics_history = relationship("MetricHistory", back_populates="brand", cascade="all, delete-orphan")
    bot_visits = relationship("BotVisit", back_populates="brand", cascade="all, delete-orphan")
    tracking_events = relationship("TrackingEvent", back_populates="brand", cascade="all, delete-orphan")
    user_brands = relationship("UserBrand", back_populates="brand", cascade="all, delete-orphan")

class Analysis(Base):
    """Analysis model with FIXED status constraints"""
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False, index=True)
    status = Column(String(50), nullable=False)
    analysis_type = Column(String(50), default="comprehensive")
    data_source = Column(String(50), default="real")
    
    # Analysis data
    metrics = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Performance metrics
    total_bot_visits_analyzed = Column(Integer, default=0)
    citation_frequency = Column(Float, default=0.0)
    processing_time = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # FIXED: Updated constraint to include all test statuses
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled', 'recovery_test', 'degradation_test')",
            name="check_analysis_status"
        ),
        Index('ix_analyses_brand_created', 'brand_id', 'created_at'),
        Index('ix_analyses_status', 'status'),
    )
    
    # Relationships
    brand = relationship("Brand", back_populates="analyses")
    metrics_history = relationship("MetricHistory", back_populates="analysis", cascade="all, delete-orphan")

class MetricHistory(Base):
    """Metric history tracking"""
    __tablename__ = "metrics_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False, index=True)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=True, index=True)
    
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    platform = Column(String(50), nullable=True)  # anthropic, openai, etc.
    data_source = Column(String(50), default="real")
    
    recorded_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_metrics_brand_name_date', 'brand_id', 'metric_name', 'recorded_at'),
        Index('ix_metrics_analysis', 'analysis_id'),
    )
    
    # Relationships
    brand = relationship("Brand", back_populates="metrics_history")
    analysis = relationship("Analysis", back_populates="metrics_history")

class User(Base):
    """User model with FIXED email validation"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=True)
    full_name = Column(String(200), nullable=True)
    company = Column(String(200), nullable=True)
    
    # User settings
    role = Column(String(50), default="user")
    plan = Column(String(50), default="free")
    plan_expires_at = Column(DateTime, nullable=True)
    
    # Usage tracking
    api_calls_limit = Column(Integer, default=100)
    api_calls_used = Column(Integer, default=0)
    analyses_limit = Column(Integer, default=2)
    analyses_used = Column(Integer, default=0)
    
    # Preferences
    email_notifications = Column(Boolean, default=True)
    weekly_reports = Column(Boolean, default=True)
    timezone = Column(String(50), default="UTC")
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=True)
    
    # FIXED: Better email validation constraint
    __table_args__ = (
        CheckConstraint(
            "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
            name="check_email_format"
        ),
        CheckConstraint(
            "plan IN ('free', 'basic', 'professional', 'enterprise')",
            name="check_plan_type"
        ),
        CheckConstraint(
            "role IN ('user', 'admin', 'analyst')",
            name="check_user_role"
        ),
    )
    
    # Relationships
    user_brands = relationship("UserBrand", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

class UserBrand(Base):
    """User-Brand relationship"""
    __tablename__ = "user_brands"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    role = Column(String(50), default="viewer")  # viewer, editor, admin
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        UniqueConstraint('user_id', 'brand_id', name='unique_user_brand'),
        CheckConstraint(
            "role IN ('viewer', 'editor', 'admin')",
            name="check_user_brand_role"
        ),
    )
    
    # Relationships
    user = relationship("User", back_populates="user_brands")
    brand = relationship("Brand", back_populates="user_brands")

class ApiKey(Base):
    """API key management"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    
    # Permissions
    permissions = Column(JSON, default=list)  # List of allowed operations
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class BotVisit(Base):
    """Bot visit tracking"""
    __tablename__ = "bot_visits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False, index=True)
    
    # Bot information
    bot_name = Column(String(100), nullable=False)
    platform = Column(String(50), nullable=True)  # anthropic, openai, google, etc.
    user_agent = Column(Text, nullable=True)
    
    # Visit details
    timestamp = Column(DateTime, default=func.now())
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    path = Column(String(500), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Float, nullable=True)
    
    # Analysis results
    brand_mentioned = Column(Boolean, default=False)
    content_type = Column(String(50), nullable=True)
    
    __table_args__ = (
        Index('ix_bot_visits_brand_timestamp', 'brand_id', 'timestamp'),
        Index('ix_bot_visits_bot_platform', 'bot_name', 'platform'),
        Index('ix_bot_visits_timestamp', 'timestamp'),
    )
    
    # Relationships
    brand = relationship("Brand", back_populates="bot_visits")

class TrackingEvent(Base):
    """Real-time tracking events - FIXED metadata conflict"""
    __tablename__ = "tracking_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # pageview, engagement, etc.
    session_id = Column(String(100), nullable=True)
    sequence_number = Column(Integer, nullable=True)
    
    # Bot information
    bot_name = Column(String(100), nullable=True)
    platform = Column(String(50), nullable=True)
    
    # Page details
    page_url = Column(String(1000), nullable=True)
    page_title = Column(String(500), nullable=True)
    
    # Engagement metrics
    time_on_page = Column(Integer, nullable=True)  # seconds
    scroll_depth = Column(Integer, nullable=True)  # percentage
    
    # FIXED: Renamed from 'metadata' to 'event_metadata' to avoid SQLAlchemy conflict
    event_metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_tracking_events_brand_timestamp', 'brand_id', 'timestamp'),
        Index('ix_tracking_events_session', 'session_id'),
        Index('ix_tracking_events_type', 'event_type'),
    )
    
    # Relationships
    brand = relationship("Brand", back_populates="tracking_events")

class PerformanceMetrics(Base):
    """System performance metrics"""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric details
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # ms, %, count, etc.
    
    # Context
    operation = Column(String(100), nullable=True)
    endpoint = Column(String(200), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Timing
    recorded_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_perf_metrics_name_date', 'metric_name', 'recorded_at'),
        Index('ix_perf_metrics_operation', 'operation'),
    )

class ScheduledAnalysis(Base):
    """Scheduled analysis jobs"""
    __tablename__ = "scheduled_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Schedule details
    frequency = Column(String(20), nullable=False)  # daily, weekly, monthly
    next_run = Column(DateTime, nullable=False)
    last_run = Column(DateTime, nullable=True)
    
    # Configuration
    analysis_type = Column(String(50), default="comprehensive")
    config = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        CheckConstraint(
            "frequency IN ('daily', 'weekly', 'monthly')",
            name="check_schedule_frequency"
        ),
        Index('ix_scheduled_next_run', 'next_run'),
    )

class TrackingAlert(Base):
    """Tracking alerts and notifications"""
    __tablename__ = "tracking_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=True)
    severity = Column(String(20), default="info")
    
    # Status
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    
    # Timing
    created_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'warning', 'error', 'critical')",
            name="check_alert_severity"
        ),
        Index('ix_alerts_user_unread', 'user_id', 'is_read'),
    )

class DataExport(Base):
    """Data export jobs"""
    __tablename__ = "data_exports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    
    # Export details
    export_type = Column(String(50), nullable=False)  # csv, json, pdf
    data_type = Column(String(50), nullable=False)    # metrics, visits, analyses
    date_range = Column(JSON, nullable=True)
    
    # Status
    status = Column(String(20), default="pending")
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'expired')",
            name="check_export_status"
        ),
        CheckConstraint(
            "export_type IN ('csv', 'json', 'pdf', 'xlsx')",
            name="check_export_type"
        ),
    )

class LlmUsage(Base):
    """LLM API usage tracking"""
    __tablename__ = "llm_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    
    # API details
    provider = Column(String(50), nullable=False)  # anthropic, openai
    model = Column(String(100), nullable=False)
    endpoint = Column(String(100), nullable=True)
    
    # Usage metrics
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    cost_usd = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)
    
    # Request details
    request_type = Column(String(50), nullable=True)  # analysis, query_test, etc.
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    timestamp = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_llm_usage_user_date', 'user_id', 'timestamp'),
        Index('ix_llm_usage_provider_model', 'provider', 'model'),
    )

class QueryTest(Base):
    """Query testing results"""
    __tablename__ = "query_tests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=False)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=True)
    
    # Test results
    llm_provider = Column(String(50), nullable=False)
    response_text = Column(Text, nullable=True)
    brand_mentioned = Column(Boolean, default=False)
    relevance_score = Column(Float, nullable=True)
    
    # Timing
    response_time = Column(Float, nullable=True)
    tested_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_query_tests_brand_date', 'brand_id', 'tested_at'),
        Index('ix_query_tests_provider', 'llm_provider'),
    )