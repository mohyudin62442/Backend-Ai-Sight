# LLM Bot Tracking Setup Guide

## Overview

The LLM Bot Tracking system provides real-time data about how AI systems (ChatGPT, Claude, Gemini, etc.) are accessing and citing your content. This replaces simulated metrics with actual bot behavior data.

## Quick Start

### Step 1: Install the Tracking Script

Add this script to your website's `<head>` section:

```html
<script>
  // Configuration
  window.LLM_TRACKER_CONFIG = {
    apiEndpoint: 'https://api.your-optimization-engine.com/track-bot',
    apiKey: 'YOUR_API_KEY', // Optional if you implement authentication
    debug: false // Set to true to see console logs
  };
</script>
<script src="https://cdn.your-domain.com/llm-tracker.js" async></script>
Or download and self-host:
html
Copy
<script src="/path/to/llm-tracker.js" async></script>
Step 2: Verify Installation
Open your browser's console
Type: window.LLMTracker
You should see the tracker object
To test detection:
javascript
Copy
// Check if current visitor is a bot
console.log(window.LLMTracker.isBot());
console.log(window.LLMTracker.getBotInfo());
