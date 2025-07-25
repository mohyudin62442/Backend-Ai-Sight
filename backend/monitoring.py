"""
Complete Monitoring Implementation for AI Optimization Engine
Implements monitoring as per FRD Section 10.3 & 14
"""

import structlog
import logging
import time
import json
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, Info
import redis
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = structlog.get_logger()

class PrometheusMetrics:
    """Prometheus metrics collection for monitoring"""
    
    def __init__(self):
        # API metrics - FRD requires tracking response times, error rates
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code', 'user_plan']
        )
        
        self.api_response_time = Histogram(
            'api_response_time_seconds',
            'API response time in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 45.0, 60.0, 90.0]  # FRD targets
        )
        
        # LLM API usage tracking - FRD cost monitoring requirement
        self.llm_api_calls = Counter(
            'llm_api_calls_total',
            'Total LLM API calls',
            ['platform', 'model', 'status', 'request_type']
        )
        
        self.llm_api_cost = Counter(
            'llm_api_cost_usd_total',
            'Total LLM API cost in USD',
            ['platform', 'model']
        )
        
        self.llm_tokens_used = Counter(
            'llm_tokens_used_total',
            'Total tokens used',
            ['platform', 'model', 'type']  # input/output
        )
        
        self.llm_response_time = Histogram(
            'llm_response_time_seconds',
            'LLM API response time',
            ['platform', 'model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        )
        
        # Analysis metrics
        self.analyses_total = Counter(
            'analyses_total',
            'Total analyses performed',
            ['status', 'analysis_type', 'data_source']
        )
        
        self.analysis_duration = Histogram(
            'analysis_duration_seconds',
            'Analysis duration in seconds',
            ['analysis_type'],
            buckets=[10, 20, 30, 45, 60, 90, 120, 180]  # FRD: <45s target, 90s max
        )
        
        self.metrics_calculation_time = Histogram(
            'metrics_calculation_seconds',
            'Individual metric calculation time',
            ['metric_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'db_connections_active',
            'Active database connections'
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.db_connection_errors = Counter(
            'db_connection_errors_total',
            'Database connection errors'
        )
        
        # Redis metrics
        self.redis_operations = Counter(
            'redis_operations_total',
            'Redis operations',
            ['operation', 'status']
        )
        
        self.redis_response_time = Histogram(
            'redis_response_time_seconds',
            'Redis operation response time',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # User metrics
        self.active_users = Gauge(
            'active_users_total',
            'Currently active users'
        )
        
        self.subscription_plans = Gauge(
            'subscription_plans_total',
            'Users by subscription plan',
            ['plan']
        )
        
        self.user_analyses_usage = Histogram(
            'user_analyses_usage',
            'Analysis usage by subscription plan',
            ['plan'],
            buckets=[1, 2, 5, 10, 25, 50, 100]
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage'
        )
        
        # Application info
        self.app_info = Info(
            'app_info',
            'Application information'
        )
        
        # Initialize app info
        self.app_info.info({
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        })

class ApplicationMonitor:
    """Comprehensive application monitoring"""
    
    def __init__(self, redis_client, db_session):
        self.metrics = PrometheusMetrics()
        self.redis = redis_client
        self.db = db_session
        self.logger = structlog.get_logger()
        self.start_time = time.time()
    
    def track_api_request(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float,
        user_plan: str = "unknown"
    ):
        """Track API request metrics"""
        self.metrics.api_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code,
            user_plan=user_plan
        ).inc()
        
        self.metrics.api_response_time.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        # Log slow requests (FRD performance requirements)
        target_times = {
            '/health': 0.1,
            '/optimization-metrics': 30.0,
            '/analyze-brand': 45.0,
            '/analyze-queries': 10.0
        }
        
        target_time = target_times.get(endpoint, 2.0)
        
        if duration > target_time:
            self.logger.warning(
                "slow_api_request",
                method=method,
                endpoint=endpoint,
                duration=duration,
                target_time=target_time,
                status_code=status_code,
                user_plan=user_plan
            )
    
    def track_llm_usage(
        self, 
        platform: str, 
        model: str, 
        input_tokens: int,
        output_tokens: int, 
        cost: float, 
        response_time: float,
        status: str,
        request_type: str = "analysis"
    ):
        """Track LLM API usage and costs"""
        self.metrics.llm_api_calls.labels(
            platform=platform,
            model=model,
            status=status,
            request_type=request_type
        ).inc()
        
        if cost > 0:
            self.metrics.llm_api_cost.labels(
                platform=platform,
                model=model
            ).inc(cost)
        
        self.metrics.llm_tokens_used.labels(
            platform=platform,
            model=model,
            type="input"
        ).inc(input_tokens)
        
        self.metrics.llm_tokens_used.labels(
            platform=platform,
            model=model,
            type="output"
        ).inc(output_tokens)
        
        self.metrics.llm_response_time.labels(
            platform=platform,
            model=model
        ).observe(response_time)
        
        # Log expensive calls
        if cost > 1.0:  # $1+ calls
            self.logger.warning(
                "expensive_llm_call",
                platform=platform,
                model=model,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time
            )
        
        # Log slow calls
        if response_time > 10.0:  # 10+ second calls
            self.logger.warning(
                "slow_llm_call",
                platform=platform,
                model=model,
                response_time=response_time,
                total_tokens=input_tokens + output_tokens
            )
    
    def track_analysis(
        self, 
        analysis_type: str, 
        duration: float, 
        status: str,
        brand_name: str, 
        data_source: str,
        metrics_calculated: Optional[Dict[str, float]] = None
    ):
        """Track analysis performance"""
        self.metrics.analyses_total.labels(
            status=status,
            analysis_type=analysis_type,
            data_source=data_source
        ).inc()
        
        self.metrics.analysis_duration.labels(
            analysis_type=analysis_type
        ).observe(duration)
        
        # Log analysis details
        log_data = {
            "analysis_type": analysis_type,
            "duration": duration,
            "status": status,
            "brand_name": brand_name,
            "data_source": data_source
        }
        
        if metrics_calculated:
            log_data.update({
                "overall_score": metrics_calculated.get('overall_score'),
                "attribution_rate": metrics_calculated.get('attribution_rate'),
                "citation_count": metrics_calculated.get('ai_citation_count')
            })
        
        self.logger.info("analysis_completed", **log_data)
        
        # Alert on failures or slow analyses
        if status == "failed":
            self.logger.error(
                "analysis_failed",
                analysis_type=analysis_type,
                brand_name=brand_name,
                duration=duration
            )
        elif duration > 45:  # FRD: should be <45s target
            self.logger.warning(
                "slow_analysis",
                analysis_type=analysis_type,
                duration=duration,
                brand_name=brand_name,
                target_time=45
            )
    
    def track_metric_calculation(self, metric_name: str, duration: float):
        """Track individual metric calculation time"""
        self.metrics.metrics_calculation_time.labels(
            metric_name=metric_name
        ).observe(duration)
    
    def track_db_operation(self, operation: str, table: str, duration: float, success: bool = True):
        """Track database operations"""
        self.metrics.db_query_duration.labels(
            query_type=operation,
            table=table
        ).observe(duration)
        
        if not success:
            self.metrics.db_connection_errors.inc()
        
        # Log slow queries
        if duration > 1.0:  # 1+ second queries
            self.logger.warning(
                "slow_db_query",
                operation=operation,
                table=table,
                duration=duration
            )
    
    def track_redis_operation(self, operation: str, duration: float, success: bool = True):
        """Track Redis operations"""
        status = "success" if success else "error"
        self.metrics.redis_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        self.metrics.redis_response_time.labels(
            operation=operation
        ).observe(duration)
    
    async def update_system_metrics(self):
        """Update system-level metrics periodically"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            self.metrics.system_cpu_usage.set(cpu_percent)
            self.metrics.system_memory_usage.set(memory_percent)
            self.metrics.system_disk_usage.set(disk_percent)
            
            # Database connections
            if self.db:
                try:
                    result = self.db.execute(text("SELECT count(*) FROM pg_stat_activity"))
                    active_connections = result.scalar()
                    self.metrics.db_connections.set(active_connections)
                except Exception as e:
                    self.logger.error("failed_to_get_db_metrics", error=str(e))
                    self.metrics.db_connection_errors.inc()
            
            # Redis info
            if self.redis:
                try:
                    start_time = time.time()
                    redis_info = self.redis.info()
                    redis_duration = time.time() - start_time
                    
                    self.track_redis_operation("info", redis_duration, True)
                    
                    # Redis-specific metrics
                    connected_clients = redis_info.get('connected_clients', 0)
                    used_memory = redis_info.get('used_memory', 0)
                    
                except Exception as e:
                    self.logger.error("failed_to_get_redis_metrics", error=str(e))
                    self.track_redis_operation("info", 0, False)
            
            # Active users (sessions in Redis)
            if self.redis:
                try:
                    session_keys = list(self.redis.scan_iter(match="session:*"))
                    self.metrics.active_users.set(len(session_keys))
                except Exception as e:
                    self.logger.error("failed_to_count_active_users", error=str(e))
            
        except Exception as e:
            self.logger.error("system_metrics_update_failed", error=str(e))

class HealthChecker:
    """Comprehensive health checking as per FRD"""
    
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.logger = structlog.get_logger()
    
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check as per FRD Section 7.1"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {},
            "performance": {},
            "environment": os.getenv('ENVIRONMENT', 'development'),
            "uptime": self._get_uptime()
        }
        
        issues = []
        
        # Database health
        db_health = await self._check_database_health()
        health_status["services"]["database"] = db_health
        if db_health["status"] != "healthy":
            issues.append("Database connection failed")
        
        # Redis health
        redis_health = await self._check_redis_health()
        health_status["services"]["redis"] = redis_health
        if redis_health["status"] != "healthy":
            issues.append("Redis connection failed")
        
        # LLM API health
        llm_health = await self._check_llm_apis()
        health_status["services"].update(llm_health)
        
        # System health
        system_health = self._check_system_health()
        health_status["performance"] = system_health
        
        # Check for performance issues
        if system_health.get("cpu_usage", 0) > 90:
            issues.append("High CPU usage")
        if system_health.get("memory_usage", 0) > 90:
            issues.append("High memory usage")
        if system_health.get("disk_usage", 0) > 90:
            issues.append("High disk usage")
        
        # Overall status
        if issues:
            health_status["status"] = "degraded" if len(issues) <= 2 else "unhealthy"
            health_status["issues"] = issues
        
        # Log health check
        self.logger.info(
            "health_check_completed",
            status=health_status["status"],
            issues_count=len(issues),
            db_healthy=health_status["services"]["database"]["status"] == "healthy",
            redis_healthy=health_status["services"]["redis"]["status"] == "healthy"
        )
        
        return health_status
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            result = self.db.execute(text("SELECT 1"))
            response_time = time.time() - start_time
            
            # Check connection count
            conn_result = self.db.execute(text("SELECT count(*) FROM pg_stat_activity"))
            connection_count = conn_result.scalar()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "active_connections": connection_count,
                "details": "Database accessible"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            start_time = time.time()
            self.redis.ping()
            response_time = time.time() - start_time
            
            # Get Redis info
            redis_info = self.redis.info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "connected_clients": redis_info.get('connected_clients', 0),
                "used_memory_mb": round(redis_info.get('used_memory', 0) / (1024**2), 2),
                "details": "Redis accessible"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Redis connection failed"
            }
    
    async def _check_llm_apis(self) -> Dict[str, Dict[str, Any]]:
        """Check LLM API availability"""
        anthropic_status = {
            "status": "configured" if os.getenv('ANTHROPIC_API_KEY') else "not_configured",
            "details": "API key present" if os.getenv('ANTHROPIC_API_KEY') else "API key missing"
        }
        
        openai_status = {
            "status": "configured" if os.getenv('OPENAI_API_KEY') else "not_configured",
            "details": "API key present" if os.getenv('OPENAI_API_KEY') else "API key missing (optional)"
        }
        
        return {
            "anthropic": anthropic_status,
            "openai": openai_status
        }
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "load_average": list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
            }
            
        except Exception as e:
            self.logger.error("system_health_check_failed", error=str(e))
            return {"error": str(e)}
    
    def _get_uptime(self) -> str:
        """Get application uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            
            return f"{days}d {hours}h {minutes}m"
        except:
            return "unknown"

class MonitoringMiddleware:
    """FastAPI middleware for automatic monitoring"""
    
    def __init__(self, app, monitor: ApplicationMonitor):
        self.monitor = monitor
        app.middleware("http")(self)
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        # Extract user plan from request (you'll need to implement this)
        user_plan = "unknown"  # Default
        
        try:
            # Get user info from request if authenticated
            # This would need to integrate with your auth system
            pass
        except:
            pass
        
        # Process request
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Record metrics
        self.monitor.track_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration,
            user_plan=user_plan
        )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Request-ID"] = str(id(request))
        
        return response