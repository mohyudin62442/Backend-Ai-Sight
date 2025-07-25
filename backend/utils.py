"""
Complete Utility Functions for AI Optimization Engine
Includes authentication, validation, caching, and helper functions as per FRD
"""

import hashlib
import secrets
import string
import re
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import jwt
from passlib.context import CryptContext
import redis
import logging
from functools import wraps
from urllib.parse import urlparse
import requests
import psutil
import os

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthUtils:
    """Authentication and authorization utilities as per FRD Section 11"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_api_key(prefix: str = "aoe") -> str:
        """Generate a secure API key"""
        alphabet = string.ascii_letters + string.digits
        key = ''.join(secrets.choice(alphabet) for _ in range(32))
        return f'{prefix}_{key}'
    
    @staticmethod
    def generate_tracking_key() -> str:
        """Generate a tracking-specific API key"""
        return AuthUtils.generate_api_key(prefix="trk")
    
    @staticmethod
    def create_access_token(
        data: dict, 
        secret_key: str, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({
            "exp": expire, 
            "iat": datetime.utcnow(),
            "type": "access_token"
        })
        encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str, secret_key: str) -> Optional[Dict]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def verify_api_key(api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        import hmac
        return hmac.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            stored_hash
        )

class EnhancedPasswordValidator:
    """Enhanced password validation as per FRD Section 11.1"""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against FRD requirements
        Returns (is_valid, list_of_errors)
        """
        errors = []
        
        # Length requirement
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        # Character requirements
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        common_passwords = [
            'password', '12345678', 'qwerty123', 'password123',
            'admin123', 'letmein', 'welcome123', 'changeme'
        ]
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        # Sequential characters check
        if re.search(r'(012|123|234|345|456|567|678|789|abc|bcd|cde)', password.lower()):
            errors.append("Password should not contain sequential characters")
        
        return len(errors) == 0, errors

class SecurityRateLimiter:
    """Rate limiting implementation as per FRD Section 11.1"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if rate limit exceeded
        Returns (allowed, metadata)
        """
        try:
            now = int(time.time())
            window_start = now - (now % window)
            window_key = f"rate_limit:{key}:{window_start}"
            
            # Use Redis pipeline for atomicity
            pipe = self.redis.pipeline()
            pipe.get(window_key)
            pipe.incr(window_key)
            pipe.expire(window_key, window)
            results = pipe.execute()
            
            current_count = int(results[0]) if results[0] else 0
            new_count = int(results[1])
            
            # Check if limit exceeded
            if new_count > limit:
                # Rollback the increment
                pipe = self.redis.pipeline()
                pipe.decr(window_key)
                pipe.execute()
                
                return False, {
                    'limit': limit,
                    'remaining': 0,
                    'reset': window_start + window,
                    'retry_after': (window_start + window) - now
                }
            
            return True, {
                'limit': limit,
                'remaining': limit - new_count,
                'reset': window_start + window,
                'retry_after': 0
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Allow on error to prevent blocking legitimate requests
            return True, {
                'limit': limit, 
                'remaining': limit, 
                'reset': 0,
                'retry_after': 0
            }
    
    async def check_analysis_limit(self, user_id: str, plan: str = "free") -> bool:
        """Check daily analysis limit based on subscription plan"""
        plan_limits = {
            "free": 2,
            "starter": 10,
            "growth": 50,
            "professional": -1,  # Unlimited
            "enterprise": -1     # Unlimited
        }
        
        daily_limit = plan_limits.get(plan, 2)
        if daily_limit == -1:  # Unlimited
            return True
        
        today = datetime.now().strftime('%Y-%m-%d')
        key = f"analysis_limit:{user_id}:{today}"
        
        try:
            current = await self.redis.get(key)
            current_count = int(current) if current else 0
            
            if current_count >= daily_limit:
                return False
            
            # Increment counter
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)  # 24 hours
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis limit check error: {e}")
            return True  # Allow on error

class CacheUtils:
    """Redis cache utilities with enhanced tracking support"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON for cache key: {key}")
            self.redis_client.delete(key)  # Remove corrupted data
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            self.redis_client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
        return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
        return False
    
    def create_cache_key(self, prefix: str, params: dict) -> str:
        """Create consistent cache key from parameters"""
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        param_str = "_".join([f"{k}:{v}" for k, v in sorted_params])
        key_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{prefix}:{key_hash}"
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter"""
        return self.redis_client.incrby(key, amount)
    
    def get_pattern(self, pattern: str) -> Dict[str, Any]:
        """Get all keys matching pattern"""
        results = {}
        try:
            for key in self.redis_client.scan_iter(match=pattern):
                value = self.get(key)
                if value:
                    results[key] = value
        except Exception as e:
            logger.error(f"Pattern scan error for {pattern}: {e}")
        return results
    
    def set_hash(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash values in cache"""
        try:
            # Convert all values to strings
            string_mapping = {k: json.dumps(v, default=str) for k, v in mapping.items()}
            self.redis_client.hset(key, mapping=string_mapping)
            if ttl:
                self.redis_client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Hash set error for key {key}: {e}")
        return False
    
    def get_hash(self, key: str) -> Dict[str, Any]:
        """Get hash values from cache"""
        try:
            hash_data = self.redis_client.hgetall(key)
            result = {}
            for k, v in hash_data.items():
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v  # Keep as string if not JSON
            return result
        except Exception as e:
            logger.error(f"Hash get error for key {key}: {e}")
        return {}

class ValidationUtils:
    """Input validation utilities as per FRD requirements"""
    
    @staticmethod
    def validate_brand_name(brand_name: str) -> str:
        """Validate brand name format as per FRD"""
        if not brand_name or len(brand_name) < 2 or len(brand_name) > 50:
            raise ValueError("Brand name must be 2-50 characters")
        
        # Only allow safe characters as per FRD
        if not re.match(r"^[a-zA-Z0-9\s&\-'\.]+$", brand_name):
            raise ValueError("Brand name contains invalid characters")
        
        # XSS prevention
        dangerous_patterns = ['<script>', 'javascript:', 'onload=', '<iframe>', 'eval(']
        brand_lower = brand_name.lower()
        for pattern in dangerous_patterns:
            if pattern in brand_lower:
                raise ValueError("Potentially dangerous content detected")
        
        return brand_name.strip()
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL with security checks"""
        if not url:
            return True  # URL is optional
        
        try:
            result = urlparse(url)
            
            # Must have scheme and netloc
            if not result.scheme or not result.netloc:
                return False
            
            # Only allow HTTP/HTTPS
            if result.scheme not in ['http', 'https']:
                return False
            
            # Prevent internal network access
            blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '10.', '192.168.', '172.']
            for blocked in blocked_hosts:
                if blocked in result.netloc.lower():
                    return False
            
            # Length check
            if len(url) > 500:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email or len(email) > 255:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_categories(categories: List[str]) -> bool:
        """Validate product categories as per FRD"""
        if not categories or len(categories) > 10:
            return False
        
        for category in categories:
            if not category or len(category) < 2 or len(category) > 50:
                return False
            # Allow letters, numbers, spaces, and hyphens
            if not re.match(r'^[a-zA-Z0-9\s\-]+$', category):
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path separators and null bytes
        filename = filename.replace('/', '').replace('\\', '').replace('\0', '')
        # Keep only safe characters
        filename = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
        # Limit length
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            return f"{name[:200]}.{ext[:10]}"
        return name[:200]
    
    @staticmethod
    def validate_json_structure(data: Any, schema: Dict) -> Tuple[bool, List[str]]:
        """Validate JSON data against schema"""
        errors = []
        
        def validate_field(value: Any, field_schema: Dict, field_name: str):
            field_type = field_schema.get('type')
            required = field_schema.get('required', False)
            
            if value is None:
                if required:
                    errors.append(f"Field '{field_name}' is required")
                return
            
            if field_type == 'string' and not isinstance(value, str):
                errors.append(f"Field '{field_name}' must be a string")
            elif field_type == 'number' and not isinstance(value, (int, float)):
                errors.append(f"Field '{field_name}' must be a number")
            elif field_type == 'array' and not isinstance(value, list):
                errors.append(f"Field '{field_name}' must be an array")
            elif field_type == 'object' and not isinstance(value, dict):
                errors.append(f"Field '{field_name}' must be an object")
        
        if isinstance(schema, dict):
            for field_name, field_schema in schema.items():
                value = data.get(field_name) if isinstance(data, dict) else None
                validate_field(value, field_schema, field_name)
        
        return len(errors) == 0, errors

class MetricsCalculator:
    """Utility functions for metrics calculations as per FRD"""
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def calculate_weighted_average(values: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average of values"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(values.get(k, 0) * weights.get(k, 0) for k in values)
        return weighted_sum / total_weight
    
    @staticmethod
    def normalize_score(value: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize a score to 0-1 range"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    @staticmethod
    def calculate_trend(data_points: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate trend from time series data"""
        if len(data_points) < 2:
            return {'trend': 'stable', 'change': 0.0, 'confidence': 'low'}
        
        # Sort by date
        data_points.sort(key=lambda x: x[0])
        
        # Simple linear regression
        n = len(data_points)
        x_mean = n / 2
        y_mean = sum(point[1] for point in data_points) / n
        
        numerator = sum((i - x_mean) * (point[1] - y_mean) for i, point in enumerate(data_points))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend
        if slope > 0.1:
            trend = 'increasing'
        elif slope < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate percentage change
        first_value = data_points[0][1]
        last_value = data_points[-1][1]
        change = MetricsCalculator.calculate_percentage_change(first_value, last_value)
        
        # Calculate confidence based on R-squared
        y_pred = [slope * i + (y_mean - slope * x_mean) for i in range(n)]
        ss_res = sum((data_points[i][1] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((data_points[i][1] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence = 'high' if r_squared > 0.7 else 'medium' if r_squared > 0.4 else 'low'
        
        return {
            'trend': trend,
            'change': change,
            'slope': slope,
            'start_value': first_value,
            'end_value': last_value,
            'r_squared': r_squared,
            'confidence': confidence,
            'data_points': len(data_points)
        }
    
    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 7) -> List[float]:
        """Calculate moving average with specified window"""
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(len(values)):
            if i < window - 1:
                # Use available values for initial points
                avg = sum(values[:i+1]) / (i + 1)
            else:
                # Use full window
                avg = sum(values[i-window+1:i+1]) / window
            moving_avg.append(avg)
        
        return moving_avg
    
    @staticmethod
    def detect_anomalies(values: List[float], threshold: float = 2.0) -> List[int]:
        """Detect anomalies using z-score method"""
        if len(values) < 3:
            return []
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_dev)
            if z_score > threshold:
                anomalies.append(i)
        
        return anomalies

class AsyncUtils:
    """Async utility functions"""
    
    @staticmethod
    async def gather_with_concurrency(n: int, *tasks):
        """Run tasks with limited concurrency"""
        semaphore = asyncio.Semaphore(n)
        
        async def sem_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*(sem_task(task) for task in tasks))
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Decorator for retrying async functions with exponential backoff"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay}s"
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"Failed after {max_attempts} attempts: {e}")
                
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_after(seconds: float, coro):
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {seconds} seconds")
            raise

class TrackingUtils:
    """Utilities specific to tracking functionality"""
    
    @staticmethod
    def generate_tracking_snippet(api_endpoint: str, api_key: Optional[str] = None) -> str:
        """Generate tracking script snippet for customers"""
        config = {
            'apiEndpoint': api_endpoint,
            'debug': 'false',
            'version': '1.0.0'
        }
        
        if api_key:
            config['apiKey'] = api_key
        
        return f"""
<!-- AI Optimization Engine Tracking -->
<script>
  window.LLM_TRACKER_CONFIG = {json.dumps(config, indent=2)};
</script>
<script src="{api_endpoint.replace('/track-bot', '')}/tracking/download-script" async></script>
<!-- End AI Optimization Engine Tracking -->
"""
    
    @staticmethod
    def parse_user_agent(user_agent: str) -> Dict[str, Any]:
        """Parse user agent string for bot detection"""
        bot_patterns = {
            'openai': ['GPTBot', 'ChatGPT-User', 'OpenAI', 'OpenAI-GPT'],
            'anthropic': ['Claude-Web', 'anthropic-ai', 'ClaudeBot'],
            'google': ['Google-Extended', 'Bard-Google', 'Gemini', 'GoogleOther'],
            'perplexity': ['PerplexityBot', 'Perplexity-UA'],
            'microsoft': ['BingChat', 'BingPreview', 'msnbot-UDiscovery'],
            'you': ['YouBot'],
            'cohere': ['CohereBot'],
            'commoncrawl': ['CCBot']
        }
        
        user_agent_lower = user_agent.lower()
        
        for platform, patterns in bot_patterns.items():
            for pattern in patterns:
                if pattern.lower() in user_agent_lower:
                    return {
                        'is_bot': True,
                        'platform': platform,
                        'bot_name': pattern,
                        'confidence': 0.9,
                        'user_agent': user_agent
                    }
        
        # Check for suspicious patterns
        suspicious_patterns = ['bot', 'crawler', 'spider', 'scraper', 'ai', 'llm']
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower and 'googlebot' not in user_agent_lower:
                return {
                    'is_bot': True,
                    'platform': 'unknown',
                    'bot_name': 'Unknown-AI-Bot',
                    'confidence': 0.5,
                    'user_agent': user_agent
                }
        
        return {
            'is_bot': False,
            'platform': None,
            'bot_name': None,
            'confidence': 0.0,
            'user_agent': user_agent
        }
    
    @staticmethod
    def estimate_content_value(content: str, semantic_tags: List[str]) -> float:
        """Estimate content value for AI systems"""
        score = 0.0
        
        # Length factor (optimal around 500-1500 words)
        word_count = len(content.split())
        if 500 <= word_count <= 1500:
            score += 0.3
        elif 200 <= word_count <= 2000:
            score += 0.2
        elif word_count > 100:
            score += 0.1
        
        # Semantic richness
        if len(semantic_tags) >= 10:
            score += 0.3
        elif len(semantic_tags) >= 5:
            score += 0.2
        elif len(semantic_tags) >= 3:
            score += 0.1
        
        # Structure indicators
        structure_indicators = ['##', '###', '<h2>', '<h3>', '1.', '•', '-']
        structure_count = sum(1 for indicator in structure_indicators if indicator in content)
        if structure_count >= 5:
            score += 0.2
        elif structure_count >= 3:
            score += 0.1
        
        # Question answering potential
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'which']
        question_count = sum(1 for q in question_indicators if q in content.lower())
        if question_count >= 3:
            score += 0.2
        elif question_count >= 1:
            score += 0.1
        
        return min(1.0, score)

class SystemUtils:
    """System monitoring and health utilities"""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
                'boot_time': psutil.boot_time(),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    @staticmethod
    def check_disk_space(path: str = '/', min_free_gb: int = 1) -> bool:
        """Check if there's enough disk space"""
        try:
            usage = psutil.disk_usage(path)
            free_gb = usage.free / (1024**3)
            return free_gb >= min_free_gb
        except Exception:
            return True  # Assume OK if can't check
    
    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """Get current process information"""
        try:
            process = psutil.Process()
            return {
                'pid': process.pid,
                'memory_mb': process.memory_info().rss / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'create_time': process.create_time(),
                'num_threads': process.num_threads(),
                'status': process.status()
            }
        except Exception as e:
            logger.error(f"Failed to get process info: {e}")
            return {}

class NotificationUtils:
    """Utilities for sending notifications"""
    
    @staticmethod
    async def send_webhook(
        url: str, 
        data: Dict[str, Any], 
        headers: Optional[Dict] = None,
        timeout: int = 10
    ) -> bool:
        """Send webhook notification"""
        try:
            default_headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'AI-Optimization-Engine/1.0'
            }
            if headers:
                default_headers.update(headers)
            
            response = requests.post(
                url,
                json=data,
                headers=default_headers,
                timeout=timeout
            )
            
            return response.status_code in [200, 201, 202, 204]
            
        except Exception as e:
            logger.error(f"Webhook error to {url}: {e}")
            return False
    
    @staticmethod
    def format_metric_change(metric_name: str, old_value: float, new_value: float) -> str:
        """Format metric change for notifications"""
        change = MetricsCalculator.calculate_percentage_change(old_value, new_value)
        direction = "📈" if change > 0 else "📉" if change < 0 else "→"
        
        return f"{metric_name}: {old_value:.2f} → {new_value:.2f} {direction} ({change:+.1f}%)"
    
    @staticmethod
    def create_alert_message(
        alert_type: str, 
        brand_name: str, 
        details: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create formatted alert message"""
        templates = {
            'citation_drop': {
                'title': f'Citation Rate Drop Alert - {brand_name}',
                'message': f"Brand citation rate has dropped to {details.get('current_rate', 0):.1%} "
                          f"(threshold: {details.get('threshold', 0):.1%})"
            },
            'new_platform': {
                'title': f'New AI Platform Detected - {brand_name}',
                'message': f"New AI platform '{details.get('platform', 'unknown')}' detected accessing your content"
            },
            'performance_issue': {
                'title': f'Performance Issue - {brand_name}',
                'message': f"Analysis performance degraded: {details.get('issue', 'unknown issue')}"
            }
        }
        
        template = templates.get(alert_type, {
            'title': f'Alert - {brand_name}',
            'message': f"Alert type: {alert_type}"
        })
        
        return template

# Configuration helper
class ConfigUtils:
    """Configuration management utilities"""
    
    @staticmethod
    def load_env_config() -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'database_url': os.getenv('DATABASE_URL'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'secret_key': os.getenv('SECRET_KEY', 'dev-secret-key'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            'enable_tracking': os.getenv('ENABLE_TRACKING', 'true').lower() == 'true'
        }
    
    @staticmethod
    def validate_required_config(config: Dict[str, Any]) -> List[str]:
        """Validate that required configuration is present"""
        required_keys = ['anthropic_api_key', 'database_url', 'secret_key']
        missing = []
        
        for key in required_keys:
            if not config.get(key):
                missing.append(key)
        
        return missing