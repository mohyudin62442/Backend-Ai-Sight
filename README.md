# AI Optimization Engine

A comprehensive AI-powered optimization engine that helps e-commerce brands improve their visibility across Large Language Models (LLMs) and Generative AI platforms.

## 🚀 What's New in v2.0

### Real LLM Bot Tracking
- **Server Log Analysis**: Analyze your actual server logs to see which AI bots visit your site
- **Client-Side Tracking**: JavaScript tracking for real-time bot detection
- **Real Citation Metrics**: No more simulations - see actual bot behavior
- **Platform Analytics**: Detailed insights for each AI platform (OpenAI, Anthropic, Google, etc.)

## 🎯 Key Features

### Core Capabilities
- **12 Core Metrics Tracking**: Comprehensive AI visibility metrics with real data
- **Multi-Platform Analysis**: ChatGPT, Claude, Gemini, Perplexity, and more
- **Real-Time Tracking**: Monitor AI bot visits as they happen
- **Server Log Analysis**: Historical analysis of bot activity
- **Semantic Query Analysis**: Understanding user intent and queries
- **Content Gap Identification**: Find missing content opportunities
- **Actionable Recommendations**: Prioritized optimization suggestions
- **Implementation Roadmap**: Step-by-step improvement plan

### Tracking Features
- Detects 15+ different AI/LLM bots
- Geographic analysis of bot traffic
- Content interest mapping
- Crawl success rate monitoring
- Real-time dashboard
- Historical trend analysis

## 📊 Metrics Explained

### Real Metrics (from tracking data)
1. **Chunk Retrieval Frequency**: How often AI bots access your content
2. **Attribution Rate**: Percentage of bot visits to brand-specific pages
3. **AI Citation Count**: Actual count of brand mentions in accessed URLs
4. **AI Model Crawl Success Rate**: Real success rate from HTTP status codes

### Calculated Metrics
5. **Embedding Relevance Score**: Vector similarity between queries and content
6. **Vector Index Presence Rate**: Content indexed in vector databases
7. **Retrieval Confidence Score**: Model certainty when selecting content
8. **RRF Rank Contribution**: Weight in hybrid ranking systems
9. **LLM Answer Coverage**: Number of questions your content answers
10. **Semantic Density Score**: Conceptual richness of content
11. **Zero-Click Surface Presence**: Presence in direct answers
12. **Machine-Validated Authority**: Overall AI recognition

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Redis 6+ (for tracking data)
- PostgreSQL 13+ (optional, for persistent storage)
- Node.js 16+ (for frontend)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-optimization-engine.git
cd ai-optimization-engine

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
