/**
 * LLM Bot Tracking Script v1.0
 * Copyright (c) 2024 AI Optimization Engine
 * 
 * This script tracks LLM/AI bot visits to provide real citation data
 * Include this on your website to enable real-time bot tracking
 */

(function(window, document) {
    'use strict';
    
    // Configuration - Update this with your API endpoint
    const config = {
        apiEndpoint: 'https://your-api-domain.com/api/track-bot',
        apiKey: 'YOUR_API_KEY', // Optional: Add if you implement API key authentication
        version: '1.0.0',
        debug: false // Set to true for console logging
    };
    
    // LLM Bot detection patterns
    const llmBotPatterns = [
        // OpenAI Bots
        { regex: /GPTBot\/\d+\.\d+/i, platform: 'openai', name: 'GPTBot' },
        { regex: /ChatGPT-User/i, platform: 'openai', name: 'ChatGPT-User' },
        { regex: /OpenAI-GPT/i, platform: 'openai', name: 'OpenAI-GPT' },
        
        // Anthropic Bots
        { regex: /Claude-Web\/\d+\.\d+/i, platform: 'anthropic', name: 'Claude-Web' },
        { regex: /anthropic-ai/i, platform: 'anthropic', name: 'Anthropic-AI' },
        { regex: /ClaudeBot/i, platform: 'anthropic', name: 'ClaudeBot' },
        
        // Google Bots
        { regex: /Google-Extended/i, platform: 'google', name: 'Google-Extended' },
        { regex: /Bard-Google/i, platform: 'google', name: 'Bard-Google' },
        { regex: /Gemini-Google/i, platform: 'google', name: 'Gemini-Google' },
        
        // Perplexity
        { regex: /PerplexityBot/i, platform: 'perplexity', name: 'PerplexityBot' },
        
        // Microsoft/Bing
        { regex: /BingChat\/\d+\.\d+/i, platform: 'microsoft', name: 'BingChat' },
        { regex: /BingPreview/i, platform: 'microsoft', name: 'BingPreview' },
        
        // You.com
        { regex: /YouBot/i, platform: 'you', name: 'YouBot' },
        
        // Common Crawl (used by many AI companies)
        { regex: /CCBot\/\d+\.\d+/i, platform: 'commoncrawl', name: 'CCBot' },
        
        // Additional AI Bots
        { regex: /Diffbot/i, platform: 'diffbot', name: 'Diffbot' },
        { regex: /SemrushBot-BA/i, platform: 'semrush', name: 'SemrushBot-BA' }
    ];
    
    // Tracker object
    const LLMTracker = {
        // Session data
        session: {
            id: null,
            startTime: Date.now(),
            pageViews: 0,
            isBot: false,
            botInfo: null
        },
        
        // Initialize tracking
        init: function() {
            try {
                // Detect if current visitor is an LLM bot
                const botInfo = this.detectBot();
                
                if (botInfo) {
                    this.session.isBot = true;
                    this.session.botInfo = botInfo;
                    this.session.id = this.generateSessionId();
                    
                    if (config.debug) {
                        console.log('LLM Bot detected:', botInfo);
                    }
                    
                    // Track page view
                    this.trackPageView();
                    
                    // Set up engagement tracking
                    this.setupEngagementTracking();
                    
                    // Track when page unloads
                    this.setupUnloadTracking();
                    
                    // Expose bot detection to other scripts
                    window.__LLM_BOT_DETECTED__ = botInfo;
                    
                    // Dispatch custom event
                    window.dispatchEvent(new CustomEvent('llmBotDetected', { 
                        detail: botInfo 
                    }));
                }
            } catch (error) {
                if (config.debug) {
                    console.error('LLM Tracker initialization error:', error);
                }
            }
        },
        
        // Detect if user agent is an LLM bot
        detectBot: function() {
            const userAgent = navigator.userAgent;
            
            for (const pattern of llmBotPatterns) {
                if (pattern.regex.test(userAgent)) {
                    return {
                        platform: pattern.platform,
                        name: pattern.name,
                        userAgent: userAgent,
                        detected: true
                    };
                }
            }
            
            return null;
        },
        
        // Generate unique session ID
        generateSessionId: function() {
            return 'llm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },
        
        // Track page view
        trackPageView: function() {
            if (!this.session.isBot) return;
            
            this.session.pageViews++;
            
            const data = {
                event_type: 'pageview',
                session_id: this.session.id,
                timestamp: new Date().toISOString(),
                bot_name: this.session.botInfo.name,
                platform: this.session.botInfo.platform,
                user_agent: this.session.botInfo.userAgent,
                page_url: window.location.href,
                page_title: document.title,
                referrer: document.referrer,
                
                // Page metadata
                meta_description: this.getMetaDescription(),
                meta_keywords: this.getMetaKeywords(),
                
                // Performance metrics
                page_load_time: this.getPageLoadTime(),
                
                // Content metrics
                content_length: document.body ? document.body.innerText.length : 0,
                word_count: this.getWordCount(),
                
                // Structured data
                structured_data: this.getStructuredData(),
                
                // Technical details
                viewport_size: window.innerWidth + 'x' + window.innerHeight,
                screen_resolution: screen.width + 'x' + screen.height,
                
                // Custom tracking
                tracking_version: config.version
            };
            
            // Send tracking data
            this.sendTrackingData(data);
        },
        
        // Set up engagement tracking
        setupEngagementTracking: function() {
            if (!this.session.isBot) return;
            
            let maxScrollDepth = 0;
            let scrollHandler;
            
            // Track scroll depth
            scrollHandler = function() {
                const windowHeight = window.innerHeight;
                const documentHeight = document.documentElement.scrollHeight;
                const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                
                const scrollDepth = Math.round((scrollTop + windowHeight) / documentHeight * 100);
                maxScrollDepth = Math.max(maxScrollDepth, scrollDepth);
            };
            
            // Throttled scroll handler
            let scrollTimeout;
            window.addEventListener('scroll', function() {
                if (scrollTimeout) {
                    clearTimeout(scrollTimeout);
                }
                scrollTimeout = setTimeout(scrollHandler, 100);
            });
            
            // Track time on page and send engagement data periodically
            this.engagementInterval = setInterval(() => {
                this.sendEngagementData({
                    time_on_page: Math.round((Date.now() - this.session.startTime) / 1000),
                    max_scroll_depth: maxScrollDepth
                });
            }, 30000); // Every 30 seconds
        },
        
        // Set up unload tracking
        setupUnloadTracking: function() {
            if (!this.session.isBot) return;
            
            const sendFinalData = () => {
                // Calculate final metrics
                const timeOnPage = Math.round((Date.now() - this.session.startTime) / 1000);
                
                // Send final engagement data
                const data = {
                    event_type: 'session_end',
                    session_id: this.session.id,
                    time_on_page: timeOnPage,
                    page_views: this.session.pageViews,
                    timestamp: new Date().toISOString()
                };
                
                // Use sendBeacon for reliability
                if (navigator.sendBeacon) {
                    navigator.sendBeacon(
                        config.apiEndpoint + '/engagement',
                        JSON.stringify(data)
                    );
                }
            };
            
            // Multiple event listeners for better coverage
            window.addEventListener('beforeunload', sendFinalData);
            window.addEventListener('pagehide', sendFinalData);
            document.addEventListener('visibilitychange', function() {
                if (document.visibilityState === 'hidden') {
                    sendFinalData();
                }
            });
        },
        
        // Send tracking data to server
        sendTrackingData: function(data) {
            if (config.apiKey) {
                data.api_key = config.apiKey;
            }
            
            // Use beacon API if available for better reliability
            if (navigator.sendBeacon) {
                navigator.sendBeacon(config.apiEndpoint, JSON.stringify(data));
            } else {
                // Fallback to fetch
                fetch(config.apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data),
                    keepalive: true
                }).catch(error => {
                    if (config.debug) {
                        console.error('Failed to send tracking data:', error);
                    }
                });
            }
        },
        
        // Send engagement data
        sendEngagementData: function(engagementData) {
            const data = Object.assign({
                event_type: 'engagement',
                session_id: this.session.id,
                platform: this.session.botInfo.platform,
                timestamp: new Date().toISOString()
            }, engagementData);
            
            this.sendTrackingData(data);
        },
        
        // Utility functions
        getMetaDescription: function() {
            const meta = document.querySelector('meta[name="description"]');
            return meta ? meta.content : '';
        },
        
        getMetaKeywords: function() {
            const meta = document.querySelector('meta[name="keywords"]');
            return meta ? meta.content : '';
        },
        
        getPageLoadTime: function() {
            if (window.performance && window.performance.timing) {
                const timing = window.performance.timing;
                return timing.loadEventEnd - timing.navigationStart;
            }
            return 0;
        },
        
        getWordCount: function() {
            const text = document.body ? document.body.innerText : '';
            return text.trim().split(/\s+/).length;
        },
        
        getStructuredData: function() {
            const structuredData = [];
            
            // Find JSON-LD structured data
            const scripts = document.querySelectorAll('script[type="application/ld+json"]');
            scripts.forEach(script => {
                try {
                    const data = JSON.parse(script.textContent);
                    structuredData.push(data);
                } catch (e) {
                    // Invalid JSON
                }
            });
            
            return structuredData;
        },
        
        // Public API for manual tracking
        trackEvent: function(eventName, eventData) {
            if (!this.session.isBot) return;
            
            const data = {
                event_type: 'custom',
                event_name: eventName,
                event_data: eventData,
                session_id: this.session.id,
                platform: this.session.botInfo.platform,
                timestamp: new Date().toISOString()
            };
            
            this.sendTrackingData(data);
        }
    };
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            LLMTracker.init();
        });
    } else {
        LLMTracker.init();
    }
    
    // Expose public API
    window.LLMTracker = {
        detectBot: LLMTracker.detectBot.bind(LLMTracker),
        trackEvent: LLMTracker.trackEvent.bind(LLMTracker),
        isBot: function() { return LLMTracker.session.isBot; },
        getBotInfo: function() { return LLMTracker.session.botInfo; }
    };
    
})(window, document);
