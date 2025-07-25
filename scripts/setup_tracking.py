#!/usr/bin/env python3
"""
Setup script for AI Optimization Engine tracking
Helps customers configure tracking on their websites
"""

import os
import sys
import json
import argparse
import requests
from urllib.parse import urlparse
import shutil
from pathlib import Path

def print_header():
    """Print script header"""
    print("\n" + "="*50)
    print("AI Optimization Engine - Tracking Setup")
    print("="*50 + "\n")

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def check_api_connection(api_endpoint: str, api_key: str = None) -> bool:
    """Check if API endpoint is reachable"""
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        response = requests.get(f"{api_endpoint}/health", headers=headers, timeout=5)
        return response.status_code == 200
    except:
        return False

def download_tracking_script(api_endpoint: str, output_path: str) -> bool:
    """Download tracking script from API"""
    try:
        script_url = f"{api_endpoint}/tracking/download-script"
        response = requests.get(script_url, timeout=10)
        
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
    except Exception as e:
        print(f"Error downloading script: {e}")
    
    return False

def generate_config(api_endpoint: str, api_key: str = None, debug: bool = False) -> dict:
    """Generate tracking configuration"""
    config = {
        'apiEndpoint': f"{api_endpoint}/track-bot",
        'version': '1.0.0',
        'debug': debug
    }
    
    if api_key:
        config['apiKey'] = api_key
    
    return config

def generate_html_snippet(config: dict, script_path: str = None) -> str:
    """Generate HTML snippet for tracking"""
    config_json = json.dumps(config, indent=2)
    
    if script_path:
        script_src = script_path
    else:
        script_src = f"{config['apiEndpoint'].replace('/track-bot', '')}/tracking/download-script"
    
    return f"""<!-- AI Optimization Engine Tracking -->
<script>
  window.LLM_TRACKER_CONFIG = {config_json};
</script>
<script src="{script_src}" async></script>
<!-- End AI Optimization Engine Tracking -->"""

def setup_wordpress_plugin(config: dict, output_dir: str):
    """Generate WordPress plugin for tracking"""
    plugin_dir = os.path.join(output_dir, 'ai-optimization-tracking')
    os.makedirs(plugin_dir, exist_ok=True)
    
    # Plugin header
    plugin_content = f"""<?php
/**
 * Plugin Name: AI Optimization Engine Tracking
 * Description: Tracks AI/LLM bot visits to provide real citation data
 * Version: 1.0.0
 * Author: AI Optimization Engine
 */

// Prevent direct access
if (!defined('ABSPATH')) {{
    exit;
}}

// Add tracking script to header
function aioe_add_tracking_script() {{
    $config = json_encode({json.dumps(config)});
    ?>
    <!-- AI Optimization Engine Tracking -->
    <script>
      window.LLM_TRACKER_CONFIG = <?php echo $config; ?>;
    </script>
    <script src="{config['apiEndpoint'].replace('/track-bot', '')}/tracking/download-script" async></script>
    <!-- End AI Optimization Engine Tracking -->
    <?php
}}
add_action('wp_head', 'aioe_add_tracking_script');

// Add admin notice
function aioe_admin_notice() {{
    ?>
    <div class="notice notice-success is-dismissible">
        <p>AI Optimization Engine Tracking is active!</p>
    </div>
    <?php
}}
add_action('admin_notices', 'aioe_admin_notice');
"""
    
    with open(os.path.join(plugin_dir, 'ai-optimization-tracking.php'), 'w') as f:
        f.write(plugin_content)
    
    print(f"✅ WordPress plugin created in: {plugin_dir}")

def setup_gtm_container(config: dict, output_dir: str):
    """Generate Google Tag Manager container"""
    container = {
        "exportFormatVersion": 2,
        "exportTime": "2024-01-15 10:00:00",
        "containerVersion": {
            "tag": [{
                "accountId": "YOUR_ACCOUNT_ID",
                "containerId": "YOUR_CONTAINER_ID",
                "tagId": "1",
                "name": "AI Optimization Engine Tracking",
                "type": "html",
                "parameter": [{
                    "type": "TEMPLATE",
                    "key": "html",
                    "value": generate_html_snippet(config)
                }],
                "fingerprint": "1234567890",
                "firingTriggerId": ["2147479553"],  # All Pages trigger
            }]
        }
    }
    
    output_path = os.path.join(output_dir, 'gtm-container.json')
    with open(output_path, 'w') as f:
        json.dump(container, f, indent=2)
    
    print(f"✅ GTM container created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Setup AI Optimization Engine tracking')
    parser.add_argument('--api-endpoint', required=True, help='API endpoint URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--output-dir', default='./tracking-setup', help='Output directory')
    parser.add_argument('--platform', choices=['html', 'wordpress', 'gtm', 'all'], default='all', help='Platform to generate for')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--download-script', action='store_true', help='Download tracking script locally')
    
    args = parser.parse_args()
    
    print_header()
    
    # Validate inputs
    if not validate_url(args.api_endpoint):
        print("❌ Invalid API endpoint URL")
        sys.exit(1)
    
    # Check API connection
    print("Checking API connection...")
    if not check_api_connection(args.api_endpoint, args.api_key):
        print("⚠️  Warning: Could not connect to API endpoint")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("✅ API connection successful")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate configuration
    config = generate_config(args.api_endpoint, args.api_key, args.debug)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'tracker-config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to: {config_path}")
    
    # Download script if requested
    script_path = None
    if args.download_script:
        script_path = os.path.join(args.output_dir, 'llm-tracker.js')
        if download_tracking_script(args.api_endpoint, script_path):
            print(f"✅ Tracking script downloaded to: {script_path}")
        else:
            print("⚠️  Could not download tracking script")
            script_path = None
    
    # Generate platform-specific files
    if args.platform in ['html', 'all']:
        # HTML snippet
        snippet = generate_html_snippet(config, script_path)
        snippet_path = os.path.join(args.output_dir, 'tracking-snippet.html')
        with open(snippet_path, 'w') as f:
            f.write(snippet)
        print(f"✅ HTML snippet saved to: {snippet_path}")
        
        # Example HTML page
        example_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Optimization Tracking Example</title>
    {snippet}
</head>
<body>
    <h1>AI Optimization Tracking Installed</h1>
    <p>The tracking script is now active on this page.</p>
    
    <script>
    // Example of checking if a bot is detected
    window.addEventListener('llmBotDetected', function(event) {{
        console.log('Bot detected:', event.detail);
    }});
    </script>
</body>
</html>"""
        
        example_path = os.path.join(args.output_dir, 'example.html')
        with open(example_path, 'w') as f:
            f.write(example_html)
        print(f"✅ Example HTML page saved to: {example_path}")
    
    if args.platform in ['wordpress', 'all']:
        setup_wordpress_plugin(config, args.output_dir)
    
    if args.platform in ['gtm', 'all']:
        setup_gtm_container(config, args.output_dir)
    
    # Generate README
    readme_content = f"""# AI Optimization Engine Tracking Setup

## Configuration
- API Endpoint: {args.api_endpoint}
- Debug Mode: {args.debug}

## Installation Instructions

### HTML Website
1. Copy the contents of `tracking-snippet.html`
2. Paste into your website's <head> section
3. Save and deploy your changes

### WordPress
1. Upload the `ai-optimization-tracking` folder to `/wp-content/plugins/`
2. Activate the plugin in WordPress admin
3. The tracking will be automatically added to all pages

### Google Tag Manager
1. Import `gtm-container.json` into your GTM container
2. Update the trigger if needed (default: All Pages)
3. Publish the container

## Testing
Open your browser console and look for:
- "LLM Bot detected" messages when AI bots visit
- `window.LLMTracker` object availability

## Support
For help, visit: https://docs.aioptimization.com
"""
    
    readme_path = os.path.join(args.output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✅ Setup complete! Files created in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Install the tracking code on your website")
    print("3. Monitor bot activity in your dashboard")

if __name__ == '__main__':
    main()
