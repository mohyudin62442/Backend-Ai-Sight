"""
Tracking Manager Module
Coordinates server-side and client-side tracking for comprehensive analytics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from log_analyzer import ServerLogAnalyzer
from bot_tracker import ClientSideBotTracker
import redis

logger = logging.getLogger(__name__)

class TrackingManager:
    """
    Unified tracking manager that combines server log analysis 
    and client-side tracking for complete LLM bot analytics
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", geoip_path: Optional[str] = None):
        self.redis_client = redis.from_url(redis_url)
        self.log_analyzer = ServerLogAnalyzer(self.redis_client, geoip_path)
        self.client_tracker = ClientSideBotTracker(self.redis_client)
        
    async def get_comprehensive_metrics(
        self, 
        brand_name: str,
        days: int = 30,
        include_predictions: bool = True
    ) -> Dict:
        """
        Get comprehensive metrics combining all tracking sources
        """
        # Get server-side metrics
        server_metrics = await self.log_analyzer.get_real_time_metrics(brand_name, days)
        
        # Get client-side metrics
        client_metrics = await self.client_tracker.get_real_time_dashboard_data()
        
        # Combine metrics
        comprehensive_metrics = {
            'tracking_period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            },
            
            # Real citation metrics from server logs
            'citation_metrics': {
                'real_citation_frequency': server_metrics['real_citation_frequency'],
                'platform_citation_rates': server_metrics['platform_citation_rates'],
                'brand_mention_trends': server_metrics['brand_mention_trends'],
                'citation_growth_rate': self._calculate_growth_rate(server_metrics['brand_mention_trends'])
            },
            
            # Crawl frequency metrics
            'crawl_metrics': {
                'real_crawl_frequency': server_metrics['real_crawl_frequency'],
                'platform_coverage': server_metrics['platform_coverage'],
                'crawl_trends': server_metrics['crawl_trends'],
                'peak_crawl_times': self._identify_peak_times(server_metrics['crawl_trends'])
            },
            
            # Content accessibility
            'accessibility_metrics': {
                'content_accessibility': server_metrics['content_accessibility'],
                'content_patterns': server_metrics['content_patterns'],
                'most_accessed_content': self._get_most_accessed_content(server_metrics['content_patterns'])
            },
            
            # Engagement metrics from client-side tracking
            'engagement_metrics': client_metrics['engagement_metrics'],
            
            # Real-time activity
            'real_time_activity': {
                'current_activity': client_metrics['current_activity'],
                'active_sessions': client_metrics['active_sessions'],
                'hourly_trends': client_metrics['hourly_trends']
            },
            
            # Platform insights
            'platform_insights': self._generate_platform_insights(server_metrics, client_metrics),
            
            # Predictions (if enabled)
            'predictions': self._generate_predictions(server_metrics) if include_predictions else None
        }
        
        return comprehensive_metrics
    
    async def get_attribution_analysis(
        self,
        brand_name: str,
        competitor_names: List[str] = None,
        days: int = 30
    ) -> Dict:
        """
        Detailed attribution analysis comparing brand vs competitors
        """
        attribution_data = {
            'brand': brand_name,
            'analysis_period': days,
            'brand_metrics': await self._get_brand_attribution_metrics(brand_name, days),
            'competitor_comparison': {}
        }
        
        # Analyze competitors if provided
        if competitor_names:
            # Filter out the brand itself from competitors to avoid self-comparison
            filtered_competitors = [comp for comp in competitor_names if comp.lower() != brand_name.lower()]
            
            for competitor in filtered_competitors:
                try:
                    attribution_data['competitor_comparison'][competitor] = await self._get_brand_attribution_metrics(competitor, days)
                except Exception as e:
                    logger.warning(f"Failed to get metrics for competitor {competitor}: {e}")
                    # Add placeholder data to avoid breaking the analysis
                    attribution_data['competitor_comparison'][competitor] = {
                        'citation_frequency': 0,
                        'total_bot_visits': 0,
                        'platform_breakdown': {},
                        'trend': {}
                    }
        
        # Calculate relative performance
        if attribution_data['competitor_comparison']:
            attribution_data['relative_performance'] = self._calculate_relative_performance(
                attribution_data['brand_metrics'],
                attribution_data['competitor_comparison']
            )
        else:
            # No competitors to compare against
            attribution_data['relative_performance'] = {
                'citation_index': 100,
                'visit_share': 100,
                'performance_rating': 'no_comparison_available',
                'note': 'No competitor data available for comparison'
            }
        
        return attribution_data
    
    async def generate_tracking_insights(self, brand_name: str) -> Dict:
        """
        Generate actionable insights based on tracking data
        """
        # Get comprehensive metrics
        metrics = await self.get_comprehensive_metrics(brand_name, days=30)
        
        insights = {
            'key_findings': [],
            'opportunities': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Analyze citation frequency
        citation_freq = metrics['citation_metrics']['real_citation_frequency']
        if citation_freq < 10:
            insights['warnings'].append({
                'type': 'low_citation',
                'message': f'Your brand citation frequency is only {citation_freq:.1f}%. This is below the recommended 10% threshold.',
                'impact': 'high'
            })
            insights['recommendations'].append({
                'priority': 'high',
                'action': 'Improve brand visibility in content',
                'details': 'Add more brand mentions in key pages, especially product descriptions and about pages.'
            })
        
        # Analyze platform coverage
        platform_coverage = metrics['crawl_metrics']['platform_coverage']
        uncovered_platforms = [p for p in ['openai', 'anthropic', 'google', 'perplexity'] if p not in platform_coverage or platform_coverage[p] == 0]
        
        if uncovered_platforms:
            insights['opportunities'].append({
                'type': 'platform_expansion',
                'message': f'No activity detected from: {", ".join(uncovered_platforms)}',
                'potential': 'Increase reach by optimizing for these platforms'
            })
        
        # Analyze content patterns
        content_patterns = metrics['accessibility_metrics']['content_patterns']
        for platform, patterns in content_patterns.items():
            if patterns:
                # Check if product pages are being accessed
                product_access = sum(count for path, count in patterns if '/product' in path)
                total_access = sum(count for _, count in patterns)
                
                if total_access > 0 and product_access / total_access < 0.3:
                    insights['opportunities'].append({
                        'type': 'content_optimization',
                        'platform': platform,
                        'message': f'{platform} bots are not frequently accessing product pages',
                        'recommendation': 'Improve internal linking to product pages and ensure they are crawlable'
                    })
        
        # Analyze engagement metrics
        engagement = metrics.get('engagement_metrics', {})
        for platform, eng_data in engagement.items():
            if eng_data.get('avg_time_on_page', 0) < 10:  # Less than 10 seconds
                insights['warnings'].append({
                    'type': 'low_engagement',
                    'platform': platform,
                    'message': f'{platform} bots spend very little time on your pages',
                    'recommendation': 'Improve content quality and loading speed'
                })
        
        # Add key findings
        top_platform = max(platform_coverage.items(), key=lambda x: x[1])[0] if platform_coverage else 'None'
        insights['key_findings'].append(f'Most active LLM platform: {top_platform}')
        insights['key_findings'].append(f'Average daily bot visits: {metrics["crawl_metrics"]["real_crawl_frequency"]:.1f}')
        insights['key_findings'].append(f'Overall citation rate: {citation_freq:.1f}%')
        
        return insights
    
    def _calculate_growth_rate(self, trend_data: Dict) -> float:
        """Calculate growth rate from trend data"""
        if not trend_data or len(trend_data) < 2:
            return 0.0
        
        dates = sorted(trend_data.keys())
        if len(dates) < 2:
            return 0.0
        
        # Compare first week vs last week
        first_week_sum = sum(trend_data[d] for d in dates[:7] if d in trend_data)
        last_week_sum = sum(trend_data[d] for d in dates[-7:] if d in trend_data)
        
        if first_week_sum == 0:
            return 100.0 if last_week_sum > 0 else 0.0
        
        return ((last_week_sum - first_week_sum) / first_week_sum) * 100
    
    def _identify_peak_times(self, crawl_trends: Dict) -> Dict:
        """Identify peak crawl times"""
        hourly_aggregates = {}
        
        for date_str, count in crawl_trends.items():
            # This would need hourly data - simplified for now
            hour = 12  # Placeholder
            if hour not in hourly_aggregates:
                hourly_aggregates[hour] = 0
            hourly_aggregates[hour] += count
        
        if not hourly_aggregates:
            return {}
        
        peak_hour = max(hourly_aggregates.items(), key=lambda x: x[1])
        return {
            'peak_hour': peak_hour[0],
            'peak_hour_visits': peak_hour[1],
            'distribution': hourly_aggregates
        }
    
    def _get_most_accessed_content(self, content_patterns: Dict) -> Dict:
        """Get most accessed content by platform"""
        most_accessed = {}
        
        for platform, patterns in content_patterns.items():
            if patterns:
                # Get top 5 pages
                most_accessed[platform] = patterns[:5]
        
        return most_accessed
    
    def _generate_platform_insights(self, server_metrics: Dict, client_metrics: Dict) -> Dict:
        """Generate platform-specific insights"""
        insights = {}
        
        for platform in server_metrics.get('platform_coverage', {}).keys():
            platform_data = {
                'total_visits': server_metrics['platform_coverage'].get(platform, 0),
                'citation_rate': server_metrics['platform_citation_rates'].get(platform, {}).get('citation_rate', 0),
                'engagement': client_metrics['engagement_metrics'].get(platform, {}),
                'recommendations': []
            }
            
            # Generate platform-specific recommendations
            if platform_data['citation_rate'] < 5:
                platform_data['recommendations'].append(
                    f"Low citation rate for {platform}. Consider creating {platform}-optimized content."
                )
            
            if platform_data['total_visits'] < 10:
                platform_data['recommendations'].append(
                    f"Low crawl frequency from {platform}. Check robots.txt and crawl permissions."
                )
            
            insights[platform] = platform_data
        
        return insights
    
    def _generate_predictions(self, metrics: Dict) -> Dict:
        """Generate predictions based on historical data"""
        # Simple linear projection - in production, use proper ML models
        crawl_trend = metrics.get('crawl_trends', {})
        if len(crawl_trend) < 7:
            return {}
        
        # Calculate average daily growth
        values = list(crawl_trend.values())
        avg_daily_growth = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        
        predictions = {
            'next_7_days': {
                'expected_crawls': int(values[-1] * 7 + avg_daily_growth * 7 * 3.5),
                'expected_citations': int(metrics['real_citation_frequency'] * values[-1] * 7 / 100),
                'confidence': 'medium'
            },
            'next_30_days': {
                'expected_crawls': int(values[-1] * 30 + avg_daily_growth * 30 * 15),
                'expected_citations': int(metrics['real_citation_frequency'] * values[-1] * 30 / 100),
                'confidence': 'low'
            }
        }
        
        return predictions
    
    async def _get_brand_attribution_metrics(self, brand_name: str, days: int) -> Dict:
        """Get attribution metrics for a specific brand"""
        metrics = await self.log_analyzer.get_real_time_metrics(brand_name, days)
        
        return {
            'citation_frequency': metrics['real_citation_frequency'],
            'total_bot_visits': sum(metrics['platform_coverage'].values()),
            'platform_breakdown': metrics['platform_citation_rates'],
            'trend': metrics['brand_mention_trends']
        }
    
    def _calculate_relative_performance(self, brand_metrics: Dict, competitor_metrics: Dict) -> Dict:
        """Calculate relative performance vs competitors"""
        if not competitor_metrics:
            return {
                'citation_index': 100,
                'visit_share': 100,
                'performance_rating': 'no_comparison_available',
                'note': 'No competitor data available for comparison'
            }
        
        # Calculate average competitor citation rate
        competitor_citation_rates = [comp['citation_frequency'] for comp in competitor_metrics.values()]
        avg_competitor_citation = sum(competitor_citation_rates) / len(competitor_citation_rates) if competitor_citation_rates else 0
        
        # Calculate citation index
        citation_index = 100  # Default to 100 if no comparison possible
        if avg_competitor_citation > 0:
            citation_index = (brand_metrics['citation_frequency'] / avg_competitor_citation) * 100
        
        # Calculate visit share - protect against division by zero
        total_competitor_visits = sum(comp['total_bot_visits'] for comp in competitor_metrics.values())
        total_all_visits = brand_metrics['total_bot_visits'] + total_competitor_visits
        
        visit_share = 100  # Default to 100% if no competitor visits
        if total_all_visits > 0:
            visit_share = (brand_metrics['total_bot_visits'] / total_all_visits) * 100
        
        # Determine performance rating
        performance_rating = 'no_comparison_available'
        if avg_competitor_citation > 0:
            if brand_metrics['citation_frequency'] > avg_competitor_citation:
                performance_rating = 'above_average'
            elif brand_metrics['citation_frequency'] < avg_competitor_citation:
                performance_rating = 'below_average'
            else:
                performance_rating = 'average'
        
        return {
            'citation_index': round(citation_index, 2),
            'visit_share': round(visit_share, 2),
            'performance_rating': performance_rating,
            'brand_citation_rate': brand_metrics['citation_frequency'],
            'avg_competitor_citation_rate': round(avg_competitor_citation, 2),
            'total_brand_visits': brand_metrics['total_bot_visits'],
            'total_competitor_visits': total_competitor_visits
        }