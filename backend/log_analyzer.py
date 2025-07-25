"""
Server Log Analyzer Module
Analyzes server logs to track real LLM bot visits and calculate actual citation metrics
"""

import re
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass, asdict
from ua_parser import user_agent_parser
from user_agents import parse as parse_user_agent
import geoip2.database
import redis
import os
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BotVisit:
    """Represents a bot visit from server logs"""
    timestamp: datetime
    bot_name: str
    ip_address: str
    user_agent: str
    path: str
    status_code: int
    response_time: float
    bytes_sent: int
    referer: Optional[str]
    country: Optional[str]
    city: Optional[str]
    platform: str
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class LLMBotPattern:
    """Pattern to identify LLM bots"""
    name: str
    patterns: List[str]
    type: str  # 'crawler', 'api', 'browser'
    platform: str  # 'openai', 'anthropic', 'google', etc.

class ServerLogAnalyzer:
    """
    Analyzes server logs to extract real LLM bot activity data
    Supports multiple log formats and provides real-time analysis
    """
    
    def __init__(self, redis_client: redis.Redis, geoip_path: Optional[str] = None):
        self.redis_client = redis_client
        
        # Define comprehensive LLM bot patterns based on real user agents
        self.llm_bot_patterns = [
            # OpenAI/ChatGPT Bots
            LLMBotPattern(
                name="GPTBot",
                patterns=[
                    "GPTBot/1.0", "GPTBot/1.1", "GPTBot/1.2",
                    "ChatGPT-User", "OpenAI-GPT", "OpenAI-Bot"
                ],
                type="crawler",
                platform="openai"
            ),
            
            # Anthropic/Claude Bots
            LLMBotPattern(
                name="Claude-Web",
                patterns=[
                    "Claude-Web/1.0", "anthropic-ai", "ClaudeBot",
                    "Claude-Web-Crawler", "Anthropic-AI-Bot"
                ],
                type="crawler",
                platform="anthropic"
            ),
            
            # Google Bard/Gemini Bots
            LLMBotPattern(
                name="Google-Extended",
                patterns=[
                    "Google-Extended", "Bard-Google", "Gemini-Google",
                    "GoogleOther", "Google-InspectionTool"
                ],
                type="crawler",
                platform="google"
            ),
            
            # Perplexity Bots
            LLMBotPattern(
                name="PerplexityBot",
                patterns=[
                    "PerplexityBot", "Perplexity-UA", "PerplexityAI"
                ],
                type="crawler",
                platform="perplexity"
            ),
            
            # Bing Chat/Copilot Bots
            LLMBotPattern(
                name="BingChat",
                patterns=[
                    "BingChat/1.0", "BingPreview", "msnbot-UDiscovery",
                    "BingCopilot", "Microsoft-Copilot"
                ],
                type="crawler",
                platform="microsoft"
            ),
            
            # You.com Bot
            LLMBotPattern(
                name="YouBot",
                patterns=["YouBot/1.0", "You.com"],
                type="crawler",
                platform="you"
            ),
            
            # Cohere Bots
            LLMBotPattern(
                name="CohereBot",
                patterns=["CohereBot", "Cohere-AI"],
                type="crawler",
                platform="cohere"
            ),
            
            # Common Crawl (used by many AI companies)
            LLMBotPattern(
                name="CCBot",
                patterns=["CCBot/2.0", "CCBot/3.0", "CCBot/3.1"],
                type="crawler",
                platform="commoncrawl"
            )
        ]
        
        # Initialize GeoIP database
        self.geoip_reader = None
        if geoip_path and os.path.exists(geoip_path):
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_path)
                logger.info("GeoIP database loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load GeoIP database: {e}")
        
        # Compile regex patterns for performance
        self.log_patterns = {
            'nginx': re.compile(
                r'(?P<ip>\d+\.\d+\.\d+\.\d+) - (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] '
                r'"(?P<method>\w+) (?P<path>[^ ]+) (?P<protocol>[^"]+)" '
                r'(?P<status>\d+) (?P<bytes>\d+) "(?P<referer>[^"]*)" '
                r'"(?P<user_agent>[^"]*)" (?P<response_time>[\d.]+)?'
            ),
            'apache': re.compile(
                r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
                r'"(?P<method>\w+) (?P<path>[^ ]+) (?P<protocol>[^"]+)" '
                r'(?P<status>\d+) (?P<bytes>\S+) "(?P<referer>[^"]*)" '
                r'"(?P<user_agent>[^"]*)"'
            ),
            'cloudflare': re.compile(
                r'(?P<timestamp>\S+) (?P<ip>\S+) (?P<method>\w+) (?P<path>\S+) '
                r'(?P<status>\d+) (?P<bytes>\d+) "(?P<user_agent>[^"]*)" '
                r'(?P<response_time>[\d.]+)'
            )
        }
    
    def identify_llm_bot(self, user_agent: str) -> Optional[Tuple[LLMBotPattern, float]]:
        """
        Identify if a user agent belongs to an LLM bot
        Returns the bot pattern and confidence score if matched
        """
        if not user_agent:
            return None
        
        user_agent_lower = user_agent.lower()
        
        for bot_pattern in self.llm_bot_patterns:
            for pattern in bot_pattern.patterns:
                if pattern.lower() in user_agent_lower:
                    # Calculate confidence based on exact match vs partial match
                    confidence = 1.0 if pattern.lower() == user_agent_lower else 0.8
                    return (bot_pattern, confidence)
        
        # Check for suspicious patterns that might be LLM bots
        suspicious_patterns = ['bot', 'crawler', 'spider', 'scraper', 'ai', 'llm']
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower and 'googlebot' not in user_agent_lower:
                # Lower confidence for generic patterns
                return (LLMBotPattern(
                    name="Unknown-AI-Bot",
                    patterns=[user_agent],
                    type="crawler",
                    platform="unknown"
                ), 0.5)
        
        return None
    
    async def parse_log_line(self, line: str, log_format: str = "nginx") -> Optional[Dict]:
        """Parse a single log line based on format"""
        pattern = self.log_patterns.get(log_format)
        if not pattern:
            logger.error(f"Unsupported log format: {log_format}")
            return None
        
        match = pattern.match(line.strip())
        if not match:
            return None
        
        return match.groupdict()
    
    async def analyze_log_file(
        self, 
        log_file_path: str, 
        brand_name: str,
        log_format: str = "nginx",
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict:
        """
        Analyze server log file for LLM bot activity
        Returns comprehensive statistics about AI bot visits
        """
        logger.info(f"Starting log analysis for {log_file_path}")
        
        bot_visits = []
        total_visits = 0
        brand_visits = 0
        llm_bot_visits = defaultdict(lambda: defaultdict(int))
        path_access_frequency = defaultdict(lambda: defaultdict(int))
        hourly_distribution = defaultdict(lambda: defaultdict(int))
        daily_trends = defaultdict(lambda: defaultdict(int))
        bot_confidence_scores = defaultdict(list)
        
        try:
            async with aiofiles.open(log_file_path, 'r') as f:
                async for line in f:
                    total_visits += 1
                    
                    # Parse log line
                    parsed = await self.parse_log_line(line, log_format)
                    if not parsed:
                        continue
                    
                    # Parse timestamp
                    timestamp = self._parse_timestamp(parsed['timestamp'], log_format)
                    
                    # Check date range if specified
                    if date_range:
                        if timestamp < date_range[0] or timestamp > date_range[1]:
                            continue
                    
                    user_agent = parsed.get('user_agent', '')
                    
                    # Check if it's an LLM bot
                    bot_result = self.identify_llm_bot(user_agent)
                    if bot_result:
                        bot_pattern, confidence = bot_result
                        
                        # Get geolocation
                        country, city = await self._get_geolocation(parsed['ip'])
                        
                        # Create bot visit record
                        visit = BotVisit(
                            timestamp=timestamp,
                            bot_name=bot_pattern.name,
                            ip_address=parsed['ip'],
                            user_agent=user_agent,
                            path=parsed['path'],
                            status_code=int(parsed['status']),
                            response_time=float(parsed.get('response_time', 0)),
                            bytes_sent=int(parsed.get('bytes', 0)),
                            referer=parsed.get('referer', ''),
                            country=country,
                            city=city,
                            platform=bot_pattern.platform
                        )
                        
                        bot_visits.append(visit)
                        bot_confidence_scores[bot_pattern.platform].append(confidence)
                        
                        # Update statistics
                        llm_bot_visits[bot_pattern.platform][bot_pattern.name] += 1
                        path_access_frequency[bot_pattern.platform][parsed['path']] += 1
                        hourly_distribution[bot_pattern.platform][timestamp.hour] += 1
                        daily_trends[bot_pattern.platform][timestamp.date().isoformat()] += 1
                        
                        # Check if brand is mentioned in path
                        if brand_name.lower() in parsed['path'].lower():
                            brand_visits += 1
                        
                        # Store in Redis for real-time tracking
                        await self._store_bot_visit(visit, bot_pattern, brand_name)
                    
                    # Progress logging every 10000 lines
                    if total_visits % 10000 == 0:
                        logger.info(f"Processed {total_visits} log entries...")
        
        except Exception as e:
            logger.error(f"Error analyzing log file: {e}")
            raise
        
        # Calculate comprehensive metrics
        analysis_results = {
            'total_requests': total_visits,
            'llm_bot_requests': len(bot_visits),
            'brand_specific_bot_requests': brand_visits,
            'llm_bot_percentage': (len(bot_visits) / total_visits * 100) if total_visits > 0 else 0,
            'brand_citation_rate': (brand_visits / len(bot_visits) * 100) if bot_visits else 0,
            'platform_breakdown': dict(llm_bot_visits),
            'top_accessed_paths': self._get_top_paths(path_access_frequency),
            'hourly_distribution': dict(hourly_distribution),
            'daily_trends': dict(daily_trends),
            'unique_bot_ips': len(set(v.ip_address for v in bot_visits)),
            'bot_visits_detail': [v.to_dict() for v in bot_visits[-1000:]],  # Last 1000 visits
            'crawl_success_rate': self._calculate_crawl_success_rate(bot_visits),
            'geographic_distribution': self._calculate_geographic_distribution(bot_visits),
            'content_interest_map': self._analyze_content_interest(path_access_frequency),
            'average_confidence_scores': {
                platform: sum(scores) / len(scores) if scores else 0
                for platform, scores in bot_confidence_scores.items()
            },
            'response_time_analysis': self._analyze_response_times(bot_visits),
            'error_rate_by_platform': self._calculate_error_rates(bot_visits)
        }
        
        logger.info(f"Log analysis complete. Found {len(bot_visits)} LLM bot visits out of {total_visits} total requests")
        
        return analysis_results
    
    def _parse_timestamp(self, timestamp_str: str, log_format: str) -> datetime:
        """Parse timestamp based on log format"""
        timestamp_formats = {
            'nginx': '%d/%b/%Y:%H:%M:%S',
            'apache': '%d/%b/%Y:%H:%M:%S',
            'cloudflare': '%Y-%m-%dT%H:%M:%S'
        }
        
        try:
            # Remove timezone info if present
            timestamp_str = timestamp_str.split()[0]
            return datetime.strptime(timestamp_str, timestamp_formats.get(log_format, '%d/%b/%Y:%H:%M:%S'))
        except Exception as e:
            logger.warning(f"Failed to parse timestamp: {timestamp_str} - {e}")
            return datetime.now()
    
    async def _get_geolocation(self, ip_address: str) -> Tuple[Optional[str], Optional[str]]:
        """Get country and city from IP address"""
        if not self.geoip_reader:
            return None, None
        
        try:
            response = self.geoip_reader.city(ip_address)
            return response.country.name, response.city.name
        except:
            return None, None
    
    async def _store_bot_visit(self, visit: BotVisit, bot_pattern: LLMBotPattern, brand_name: str):
        """Store bot visit in Redis for real-time tracking"""
        # Store individual visit
        visit_key = f"llm_bot_visit:{bot_pattern.platform}:{visit.timestamp.strftime('%Y%m%d')}"
        visit_data = visit.to_dict()
        
        # Add to sorted set with timestamp as score
        self.redis_client.zadd(
            visit_key,
            {json.dumps(visit_data): visit.timestamp.timestamp()}
        )
        
        # Set expiration to 90 days
        self.redis_client.expire(visit_key, 90 * 24 * 3600)
        
        # Update daily counters
        counter_key = f"llm_bot_counter:{bot_pattern.platform}:{visit.timestamp.strftime('%Y%m%d')}"
        self.redis_client.hincrby(counter_key, bot_pattern.name, 1)
        self.redis_client.expire(counter_key, 90 * 24 * 3600)
        
        # Update brand-specific counters if brand mentioned
        if brand_name.lower() in visit.path.lower():
            brand_key = f"brand_citation:{brand_name.lower()}:{bot_pattern.platform}:{visit.timestamp.strftime('%Y%m%d')}"
            self.redis_client.incr(brand_key)
            self.redis_client.expire(brand_key, 90 * 24 * 3600)
        
        # Store path access patterns
        path_key = f"path_access:{bot_pattern.platform}:{visit.timestamp.strftime('%Y%m%d')}"
        self.redis_client.hincrby(path_key, visit.path, 1)
        self.redis_client.expire(path_key, 90 * 24 * 3600)
    
    def _get_top_paths(self, path_frequency: Dict, limit: int = 50) -> Dict:
        """Get most accessed paths by platform"""
        top_paths = {}
        
        for platform, paths in path_frequency.items():
            sorted_paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
            top_paths[platform] = [
                {'path': path, 'count': count} 
                for path, count in sorted_paths[:limit]
            ]
        
        return top_paths
    
    def _calculate_crawl_success_rate(self, bot_visits: List[BotVisit]) -> Dict:
        """Calculate crawl success rate by status code"""
        platform_status = defaultdict(lambda: defaultdict(int))
        
        for visit in bot_visits:
            platform_status[visit.platform][visit.status_code] += 1
        
        success_rates = {}
        for platform, status_counts in platform_status.items():
            total = sum(status_counts.values())
            successful = sum(count for status, count in status_counts.items() if status < 400)
            
            success_rates[platform] = {
                'total_crawls': total,
                'successful_crawls': successful,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'status_breakdown': dict(status_counts)
            }
        
        return success_rates
    
    def _calculate_geographic_distribution(self, bot_visits: List[BotVisit]) -> Dict:
        """Calculate geographic distribution of bot visits"""
        geo_distribution = defaultdict(lambda: defaultdict(int))
        
        for visit in bot_visits:
            if visit.country:
                geo_distribution[visit.platform][visit.country] += 1
        
        return dict(geo_distribution)
    
    def _analyze_content_interest(self, path_frequency: Dict) -> Dict:
        """Analyze which content types are most accessed by AI bots"""
        content_categories = {
            'product_pages': r'/product[s]?/|/item[s]?/|/p/',
            'category_pages': r'/category/|/categories/|/c/',
            'blog_posts': r'/blog/|/post[s]?/|/article[s]?/',
            'api_endpoints': r'/api/|/v\d+/',
            'static_assets': r'\.(css|js|jpg|jpeg|png|gif|svg|ico|woff|woff2|ttf)$',
            'home_page': r'^/$|^/index\.',
            'about_pages': r'/about|/company|/team',
            'faq_pages': r'/faq|/help|/support',
            'reviews': r'/review[s]?/|/testimonial[s]?/',
            'search': r'/search|/s\?',
            'checkout': r'/cart|/checkout|/order',
            'sitemap': r'sitemap\.xml|/sitemap',
            'robots': r'robots\.txt'
        }
        
        interest_map = defaultdict(lambda: defaultdict(int))
        
        for platform, paths in path_frequency.items():
            categorized_count = 0
            for path, count in paths.items():
                for category, pattern in content_categories.items():
                    if re.search(pattern, path, re.IGNORECASE):
                        interest_map[platform][category] += count
                        categorized_count += count
                        break
                else:
                    interest_map[platform]['other'] += count
            
            # Calculate percentages
            total = sum(interest_map[platform].values())
            if total > 0:
                interest_map[platform] = {
                    category: {
                        'count': count,
                        'percentage': round(count / total * 100, 2)
                    }
                    for category, count in interest_map[platform].items()
                }
        
        return dict(interest_map)
    
    def _analyze_response_times(self, bot_visits: List[BotVisit]) -> Dict:
        """Analyze response times by platform"""
        response_times = defaultdict(list)
        
        for visit in bot_visits:
            if visit.response_time > 0:
                response_times[visit.platform].append(visit.response_time)
        
        analysis = {}
        for platform, times in response_times.items():
            if times:
                analysis[platform] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return analysis
    
    def _calculate_error_rates(self, bot_visits: List[BotVisit]) -> Dict:
        """Calculate error rates by platform"""
        platform_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        for visit in bot_visits:
            platform_errors[visit.platform]['total'] += 1
            if visit.status_code >= 400:
                platform_errors[visit.platform]['errors'] += 1
        
        error_rates = {}
        for platform, counts in platform_errors.items():
            error_rates[platform] = {
                'error_rate': (counts['errors'] / counts['total'] * 100) if counts['total'] > 0 else 0,
                'total_requests': counts['total'],
                'error_count': counts['errors']
            }
        
        return error_rates
    
    async def get_real_time_metrics(self, brand_name: str, days: int = 30) -> Dict:
        """
        Get real-time metrics based on actual bot visits from Redis
        This provides REAL citation data instead of simulations
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics = {
            'real_citation_frequency': 0,
            'real_crawl_frequency': 0,
            'platform_coverage': {},
            'content_accessibility': {},
            'crawl_trends': {},
            'bot_behavior_insights': {},
            'brand_mention_trends': {}
        }
        
        total_bot_visits = 0
        brand_mentions = 0
        platform_visits = defaultdict(int)
        daily_visits = defaultdict(int)
        daily_brand_mentions = defaultdict(int)
        
        # Iterate through each day in the range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            # Get data for each platform
            for platform in ['openai', 'anthropic', 'google', 'perplexity', 'microsoft', 'you', 'cohere']:
                # Get visit counter
                counter_key = f"llm_bot_counter:{platform}:{date_str}"
                daily_counts = self.redis_client.hgetall(counter_key)
                
                for bot, count in daily_counts.items():
                    count = int(count)
                    platform_visits[platform] += count
                    total_bot_visits += count
                    daily_visits[date_str] += count
                
                # Get brand-specific citations
                brand_key = f"brand_citation:{brand_name.lower()}:{platform}:{date_str}"
                brand_count = self.redis_client.get(brand_key)
                if brand_count:
                    brand_mentions += int(brand_count)
                    daily_brand_mentions[date_str] += int(brand_count)
                
                # Get crawl success data
                success_key = f"crawl_success:{platform}:{date_str}"
                success_data = self.redis_client.hgetall(success_key)
                if success_data:
                    total = int(success_data.get(b'total', 0))
                    successful = int(success_data.get(b'successful', 0))
                    if platform not in metrics['content_accessibility']:
                        metrics['content_accessibility'][platform] = []
                    metrics['content_accessibility'][platform].append({
                        'date': date_str,
                        'success_rate': (successful / total * 100) if total > 0 else 0
                    })
            
            current_date += timedelta(days=1)
        
        # Calculate final metrics
        metrics['real_citation_frequency'] = (brand_mentions / total_bot_visits * 100) if total_bot_visits > 0 else 0
        metrics['real_crawl_frequency'] = total_bot_visits / days  # Average daily crawls
        metrics['platform_coverage'] = dict(platform_visits)
        metrics['crawl_trends'] = dict(daily_visits)
        metrics['brand_mention_trends'] = dict(daily_brand_mentions)
        
        # Calculate platform-specific citation rates
        metrics['platform_citation_rates'] = {}
        for platform, visits in platform_visits.items():
            if visits > 0:
                platform_brand_mentions = 0
                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y%m%d')
                    brand_key = f"brand_citation:{brand_name.lower()}:{platform}:{date_str}"
                    count = self.redis_client.get(brand_key)
                    if count:
                        platform_brand_mentions += int(count)
                    current_date += timedelta(days=1)
                
                metrics['platform_citation_rates'][platform] = {
                    'citation_rate': (platform_brand_mentions / visits * 100) if visits > 0 else 0,
                    'total_visits': visits,
                    'brand_mentions': platform_brand_mentions
                }
        
        # Get content access patterns
        metrics['content_patterns'] = await self._get_content_access_patterns(days)
        
        logger.info(f"Real-time metrics calculated: {total_bot_visits} bot visits, {brand_mentions} brand mentions")
        
        return metrics
    
    async def _get_content_access_patterns(self, days: int) -> Dict:
        """Analyze content access patterns from Redis data"""
        patterns = defaultdict(lambda: defaultdict(int))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            for platform in ['openai', 'anthropic', 'google', 'perplexity', 'microsoft']:
                path_key = f"path_access:{platform}:{date_str}"
                path_data = self.redis_client.hgetall(path_key)
                
                for path, count in path_data.items():
                    path_str = path.decode() if isinstance(path, bytes) else path
                    patterns[platform][path_str] += int(count)
            
            current_date += timedelta(days=1)
        
        # Sort and limit results
        top_patterns = {}
        for platform, paths in patterns.items():
            sorted_paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
            top_patterns[platform] = sorted_paths[:20]
        
        return top_patterns
