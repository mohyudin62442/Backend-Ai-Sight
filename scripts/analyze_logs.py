#!/usr/bin/env python3
"""
Analyze server logs for AI bot activity
Standalone script for log analysis without full engine setup
"""

import os
import sys
import argparse
import json
import gzip
from datetime import datetime, timedelta
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import csv

# Bot patterns for detection
BOT_PATTERNS = {
    'openai': ['GPTBot', 'ChatGPT-User', 'OpenAI-GPT'],
    'anthropic': ['Claude-Web', 'anthropic-ai', 'ClaudeBot'],
    'google': ['Google-Extended', 'Bard-Google', 'Gemini-Google'],
    'perplexity': ['PerplexityBot'],
    'microsoft': ['BingChat', 'BingPreview'],
    'you': ['YouBot'],
    'cohere': ['CohereBot'],
    'commoncrawl': ['CCBot']
}

# Log format patterns
LOG_PATTERNS = {
    'nginx': re.compile(
        r'(?P<ip>\d+\.\d+\.\d+\.\d+) - (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\w+) (?P<path>[^ ]+) (?P<protocol>[^"]+)" '
        r'(?P<status>\d+) (?P<bytes>\d+) "(?P<referer>[^"]*)" '
        r'"(?P<user_agent>[^"]*)"'
    ),
    'apache': re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\w+) (?P<path>[^ ]+) (?P<protocol>[^"]+)" '
        r'(?P<status>\d+) (?P<bytes>\S+) "(?P<referer>[^"]*)" '
        r'"(?P<user_agent>[^"]*)"'
    )
}

def detect_bot(user_agent: str) -> Tuple[bool, str, str]:
    """Detect if user agent is an AI bot"""
    user_agent_lower = user_agent.lower()
    
    for platform, patterns in BOT_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in user_agent_lower:
                return True, platform, pattern
    
    return False, None, None

def parse_log_line(line: str, log_format: str) -> Dict:
    """Parse a single log line"""
    pattern = LOG_PATTERNS.get(log_format)
    if not pattern:
        return None
    
    match = pattern.match(line.strip())
    if match:
        return match.groupdict()
    
    return None

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse log timestamp"""
    # Common log timestamp format: 10/Oct/2023:13:55:36 +0000
    try:
        # Remove timezone for simplicity
        timestamp_str = timestamp_str.split()[0]
        return datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S')
    except:
        return datetime.now()

def analyze_content_type(path: str) -> str:
    """Determine content type from path"""
    path_lower = path.lower()
    
    if re.search(r'/product[s]?/|/item[s]?/', path_lower):
        return 'product'
    elif re.search(r'/category/|/categories/', path_lower):
        return 'category'
    elif re.search(r'/blog/|/post[s]?/', path_lower):
        return 'blog'
    elif re.search(r'/api/', path_lower):
        return 'api'
    elif re.search(r'\.(css|js|jpg|png|gif)', path_lower):
        return 'static'
    elif path == '/' or path.endswith('/index.html'):
        return 'home'
    elif re.search(r'/about|/contact', path_lower):
        return 'info'
    elif re.search(r'/search', path_lower):
        return 'search'
    else:
        return 'other'

def analyze_logs(log_file: str, log_format: str, brand_name: str = None) -> Dict:
    """Analyze log file for bot activity"""
    stats = {
        'total_requests': 0,
        'bot_requests': 0,
        'platforms': defaultdict(int),
        'bots': defaultdict(int),
        'status_codes': defaultdict(int),
        'content_types': defaultdict(int),
        'hourly_distribution': defaultdict(int),
        'daily_distribution': defaultdict(int),
        'top_paths': defaultdict(int),
        'brand_mentions': 0,
        'errors': 0
    }
    
    # Open file (handle gzip if needed)
    if log_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(log_file, mode) as f:
        for line_num, line in enumerate(f, 1):
            stats['total_requests'] += 1
            
            # Parse log line
            parsed = parse_log_line(line, log_format)
            if not parsed:
                stats['errors'] += 1
                continue
            
            # Check if it's a bot
            user_agent = parsed.get('user_agent', '')
            is_bot, platform, bot_name = detect_bot(user_agent)
            
            if is_bot:
                stats['bot_requests'] += 1
                stats['platforms'][platform] += 1
                stats['bots'][bot_name] += 1
                
                # Status code
                status = int(parsed.get('status', 0))
                stats['status_codes'][status] += 1
                
                # Content type
                path = parsed.get('path', '')
                content_type = analyze_content_type(path)
                stats['content_types'][content_type] += 1
                
                # Path tracking
                stats['top_paths'][path] += 1
                
                # Time distribution
                timestamp = parse_timestamp(parsed['timestamp'])
                stats['hourly_distribution'][timestamp.hour] += 1
                stats['daily_distribution'][timestamp.date().isoformat()] += 1
                
                # Brand mentions
                if brand_name and brand_name.lower() in path.lower():
                    stats['brand_mentions'] += 1
            
            # Progress indicator
            if line_num % 10000 == 0:
                print(f"Processed {line_num:,} lines...", end='\r')
    
    print(f"Processed {stats['total_requests']:,} total lines")
    
    # Sort top paths
    stats['top_paths'] = dict(sorted(
        stats['top_paths'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:50])
    
    return stats

def generate_report(stats: Dict, output_format: str = 'text') -> str:
    """Generate analysis report"""
    if output_format == 'json':
        # Convert defaultdicts to regular dicts for JSON serialization
        json_stats = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in stats.items()
        }
        return json.dumps(json_stats, indent=2)
    
    elif output_format == 'csv':
        # Create CSV summary
        lines = []
        lines.append("Metric,Value")
        lines.append(f"Total Requests,{stats['total_requests']}")
        lines.append(f"Bot Requests,{stats['bot_requests']}")
        lines.append(f"Bot Percentage,{stats['bot_requests'] / stats['total_requests'] * 100:.2f}%")
        
        lines.append("\nPlatform,Count")
        for platform, count in sorted(stats['platforms'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{platform},{count}")
        
        lines.append("\nContent Type,Count")
        for ctype, count in sorted(stats['content_types'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{ctype},{count}")
        
        return '\n'.join(lines)
    
    else:  # text format
        report = []
        report.append("=" * 60)
        report.append("AI BOT ACTIVITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Summary
        bot_percentage = (stats['bot_requests'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
        report.append(f"\nSUMMARY:")
        report.append(f"  Total Requests: {stats['total_requests']:,}")
        report.append(f"  Bot Requests: {stats['bot_requests']:,}")
        report.append(f"  Bot Percentage: {bot_percentage:.2f}%")
        
        if stats['brand_mentions'] > 0:
            brand_rate = (stats['brand_mentions'] / stats['bot_requests'] * 100) if stats['bot_requests'] > 0 else 0
            report.append(f"  Brand Mentions: {stats['brand_mentions']:,} ({brand_rate:.2f}% of bot requests)")
        
        # Platform breakdown
        report.append(f"\nBOT PLATFORMS:")
        for platform, count in sorted(stats['platforms'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['bot_requests'] * 100) if stats['bot_requests'] > 0 else 0
            report.append(f"  {platform}: {count:,} ({percentage:.1f}%)")
        
        # Bot breakdown
        report.append(f"\nSPECIFIC BOTS:")
        for bot, count in sorted(stats['bots'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  {bot}: {count:,}")
        
        # Content types
        report.append(f"\nCONTENT ACCESSED:")
        for ctype, count in sorted(stats['content_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['bot_requests'] * 100) if stats['bot_requests'] > 0 else 0
            report.append(f"  {ctype}: {count:,} ({percentage:.1f}%)")
        
        # Status codes
        report.append(f"\nSTATUS CODES:")
        for status, count in sorted(stats['status_codes'].items()):
            percentage = (count / stats['bot_requests'] * 100) if stats['bot_requests'] > 0 else 0
            report.append(f"  {status}: {count:,} ({percentage:.1f}%)")
        
        # Top paths
        report.append(f"\nTOP 20 ACCESSED PATHS:")
        for path, count in list(stats['top_paths'].items())[:20]:
            report.append(f"  {count:4d} - {path}")
        
        # Time patterns
        report.append(f"\nHOURLY DISTRIBUTION:")
        for hour in range(24):
            count = stats['hourly_distribution'].get(hour, 0)
            bar = '█' * int(count / max(stats['hourly_distribution'].values()) * 20) if stats['hourly_distribution'] else ''
            report.append(f"  {hour:02d}:00 {count:4d} {bar}")
        
        report.append("\n" + "=" * 60)
        
        return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='Analyze server logs for AI bot activity')
    parser.add_argument('log_file', help='Path to log file (can be .gz)')
    parser.add_argument('--format', choices=['nginx', 'apache'], default='nginx', help='Log format')
    parser.add_argument('--brand', help='Brand name to track mentions')
    parser.add_argument('--output', choices=['text', 'json', 'csv'], default='text', help='Output format')
    parser.add_argument('--save', help='Save report to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    print(f"Analyzing {args.log_file}...")
    print(f"Log format: {args.format}")
    if args.brand:
        print(f"Tracking brand: {args.brand}")
    
    # Analyze logs
    stats = analyze_logs(args.log_file, args.format, args.brand)
    
    # Generate report
    report = generate_report(stats, args.output)
    
    # Output report
    if args.save:
        with open(args.save, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.save}")
    else:
        print("\n" + report)
    
    # Summary for bot detection
    if stats['bot_requests'] == 0:
        print("\n⚠️  No AI bot activity detected in this log file.")
        print("This could mean:")
        print("  1. AI bots haven't visited yet")
        print("  2. User agents are not being logged")
        print("  3. Bots are being blocked by robots.txt or firewall")
    else:
        print(f"\n✅ Found {stats['bot_requests']:,} AI bot requests across {len(stats['platforms'])} platforms")

if __name__ == '__main__':
    main()
