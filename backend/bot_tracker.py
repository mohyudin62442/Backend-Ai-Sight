"""
Real-time Bot Tracking Module
Handles client-side tracking data and provides real-time analytics
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import redis
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class BotTrackingEvent:
    """Represents a bot tracking event from client-side script"""
    timestamp: datetime
    bot_name: str
    platform: str
    user_agent: str
    page_url: str
    page_title: str
    referrer: str
    session_id: str
    event_type: str  # 'pageview', 'engagement', 'error'
    event_data: Dict
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class ClientSideBotTracker:
    """
    Handles client-side bot tracking data
    Provides real-time analytics and insights
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
    async def track_bot_visit(self, tracking_data: Dict) -> Dict:
        """
        Process and store bot visit from client-side tracking
        """
        try:
            # Create session ID if not provided
            session_id = tracking_data.get('session_id') or self._generate_session_id(tracking_data)
            
            # Create tracking event
            event = BotTrackingEvent(
                timestamp=datetime.fromisoformat(tracking_data['timestamp']),
                bot_name=tracking_data['bot_name'],
                platform=tracking_data['platform'],
                user_agent=tracking_data['user_agent'],
                page_url=tracking_data['page_url'],
                page_title=tracking_data.get('page_title', ''),
                referrer=tracking_data.get('referrer', ''),
                session_id=session_id,
                event_type='pageview',
                event_data={
                    'page_load_time': tracking_data.get('page_load_time', 0),
                    'content_length': tracking_data.get('content_length', 0),
                    'structured_data': tracking_data.get('structured_data', [])
                }
            )
            
            # Store in Redis
            await self._store_tracking_event(event)
            
            # Update real-time metrics
            await self._update_real_time_metrics(event)
            
            # Check for brand mentions in URL
            brand_mentions = await self._extract_brand_mentions(tracking_data)
            
            return {
                'status': 'tracked',
                'session_id': session_id,
                'brand_mentions': brand_mentions
            }
            
        except Exception as e:
            logger.error(f"Error tracking bot visit: {e}")
            raise
    
    async def track_engagement(self, engagement_data: Dict) -> Dict:
        """
        Track bot engagement metrics (time on page, scroll depth, etc.)
        """
        try:
            session_id = engagement_data.get('session_id')
            if not session_id:
                return {'status': 'error', 'message': 'Session ID required'}
            
            # Store engagement data
            key = f"bot_engagement:{session_id}"
            self.redis_client.hset(key, mapping={
                'time_on_page': engagement_data.get('time_on_page', 0),
                'max_scroll_depth': engagement_data.get('max_scroll_depth', 0),
                'links_clicked': engagement_data.get('links_clicked', 0),
                'timestamp': datetime.now().isoformat()
            })
            self.redis_client.expire(key, 86400)  # 24 hour expiration
            
            # Update engagement metrics
            platform = await self._get_session_platform(session_id)
            if platform:
                engagement_key = f"engagement_metrics:{platform}:{datetime.now().strftime('%Y%m%d')}"
                
                # Increment engagement counters
                self.redis_client.hincrby(engagement_key, 'total_sessions', 1)
                self.redis_client.hincrby(engagement_key, 'total_time', int(engagement_data.get('time_on_page', 0)))
                self.redis_client.hincrby(engagement_key, 'total_scroll', int(engagement_data.get('max_scroll_depth', 0)))
                self.redis_client.expire(engagement_key, 90 * 86400)
            
            return {'status': 'tracked', 'session_id': session_id}
            
        except Exception as e:
            logger.error(f"Error tracking engagement: {e}")
            raise
    
    async def track_custom_event(self, event_data: Dict) -> Dict:
        """
        Track custom events (errors, interactions, etc.)
        """
        try:
            event_type = event_data.get('event_type', 'custom')
            platform = event_data.get('platform', 'unknown')
            
            # Store custom event
            key = f"bot_custom_events:{platform}:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.lpush(key, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'event_data': event_data.get('data', {}),
                'page_url': event_data.get('page_url', '')
            }))
            self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 events
            self.redis_client.expire(key, 30 * 86400)
            
            return {'status': 'tracked', 'event_type': event_type}
            
        except Exception as e:
            logger.error(f"Error tracking custom event: {e}")
            raise
    
    def _generate_session_id(self, tracking_data: Dict) -> str:
        """Generate unique session ID"""
        unique_string = f"{tracking_data['user_agent']}:{tracking_data['timestamp']}:{tracking_data.get('ip', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    async def _store_tracking_event(self, event: BotTrackingEvent):
        """Store tracking event in Redis"""
        # Store in time series
        key = f"bot_tracking:{event.platform}:{event.timestamp.strftime('%Y%m%d')}"
        self.redis_client.zadd(
            key,
            {json.dumps(event.to_dict()): event.timestamp.timestamp()}
        )
        self.redis_client.expire(key, 90 * 86400)
        
        # Update session data
        session_key = f"bot_session:{event.session_id}"
        self.redis_client.hset(session_key, mapping={
            'platform': event.platform,
            'bot_name': event.bot_name,
            'start_time': event.timestamp.isoformat(),
            'last_seen': event.timestamp.isoformat(),
            'page_count': 1
        })
        self.redis_client.expire(session_key, 86400)
        
        # Update daily counters
        counter_key = f"realtime_bot_counter:{event.platform}:{event.timestamp.strftime('%Y%m%d')}"
        self.redis_client.hincrby(counter_key, event.bot_name, 1)
        self.redis_client.expire(counter_key, 90 * 86400)
    
    async def _update_real_time_metrics(self, event: BotTrackingEvent):
        """Update real-time metrics dashboards"""
        # Update platform activity
        activity_key = f"realtime_activity:{event.platform}"
        self.redis_client.lpush(activity_key, json.dumps({
            'timestamp': event.timestamp.isoformat(),
            'bot_name': event.bot_name,
            'page_url': event.page_url,
            'page_title': event.page_title
        }))
        self.redis_client.ltrim(activity_key, 0, 99)  # Keep last 100 activities
        
        # Update hourly stats
        hour_key = f"hourly_stats:{event.platform}:{event.timestamp.strftime('%Y%m%d%H')}"
        self.redis_client.incr(hour_key)
        self.redis_client.expire(hour_key, 7 * 86400)
    
    async def _extract_brand_mentions(self, tracking_data: Dict) -> List[str]:
        """Extract brand mentions from page content and metadata"""
        mentions = []
        
        # Check URL
        page_url = tracking_data.get('page_url', '').lower()
        
        # Check page title
        page_title = tracking_data.get('page_title', '').lower()
        
        # Check meta description
        meta_description = tracking_data.get('meta_description', '').lower()
        
        # Check structured data
        structured_data = tracking_data.get('structured_data', [])
        for data in structured_data:
            if isinstance(data, dict):
                # Look for brand mentions in structured data
                for key, value in data.items():
                    if 'brand' in key.lower() and isinstance(value, str):
                        mentions.append(value)
        
        return list(set(mentions))
    
    async def _get_session_platform(self, session_id: str) -> Optional[str]:
        """Get platform from session data"""
        session_key = f"bot_session:{session_id}"
        platform = self.redis_client.hget(session_key, 'platform')
        return platform.decode() if platform else None
    
    async def get_real_time_dashboard_data(self, platforms: List[str] = None) -> Dict:
        """
        Get real-time dashboard data for monitoring
        """
        if not platforms:
            platforms = ['openai', 'anthropic', 'google', 'perplexity', 'microsoft']
        
        dashboard_data = {
            'current_activity': {},
            'hourly_trends': {},
            'engagement_metrics': {},
            'top_pages': {},
            'active_sessions': 0
        }
        
        now = datetime.now()
        
        for platform in platforms:
            # Get recent activity
            activity_key = f"realtime_activity:{platform}"
            recent_activity = self.redis_client.lrange(activity_key, 0, 19)
            dashboard_data['current_activity'][platform] = [
                json.loads(activity) for activity in recent_activity
            ]
            
            # Get hourly trends for last 24 hours
            hourly_data = []
            for i in range(24):
                hour = now - timedelta(hours=i)
                hour_key = f"hourly_stats:{platform}:{hour.strftime('%Y%m%d%H')}"
                count = self.redis_client.get(hour_key)
                hourly_data.append({
                    'hour': hour.strftime('%H:00'),
                    'count': int(count) if count else 0
                })
            dashboard_data['hourly_trends'][platform] = hourly_data
            
            # Get engagement metrics
            engagement_key = f"engagement_metrics:{platform}:{now.strftime('%Y%m%d')}"
            engagement_data = self.redis_client.hgetall(engagement_key)
            if engagement_data:
                total_sessions = int(engagement_data.get(b'total_sessions', 0))
                total_time = int(engagement_data.get(b'total_time', 0))
                total_scroll = int(engagement_data.get(b'total_scroll', 0))
                
                dashboard_data['engagement_metrics'][platform] = {
                    'avg_time_on_page': (total_time / total_sessions) if total_sessions > 0 else 0,
                    'avg_scroll_depth': (total_scroll / total_sessions) if total_sessions > 0 else 0,
                    'total_sessions': total_sessions
                }
        
        # Count active sessions
        session_pattern = "bot_session:*"
        active_sessions = len(list(self.redis_client.scan_iter(match=session_pattern)))
        dashboard_data['active_sessions'] = active_sessions
        
        return dashboard_data
