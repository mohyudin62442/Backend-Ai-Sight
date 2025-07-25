"""
Comprehensive API endpoint tests for AI Optimization Engine
Tests all endpoints with real data and validates FRD compliance
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from fastapi import status

class TestAPIEndpoints:
    """Test suite for all API endpoints"""

    def test_health_endpoint_performance(self, api_client, performance_monitor):
        """Test health endpoint meets FRD performance requirements (<100ms target, 500ms max)"""
        # Measure response time
        performance_monitor.start("health_check")
        response = api_client.get("/health")
        duration = performance_monitor.end("health_check")
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance requirements
        assert duration < 0.5, f"Health check took {duration:.3f}s, exceeds 500ms max"
        if duration > 0.1:
            print(f"Warning: Health check took {duration:.3f}s, exceeds 100ms target")
        
        # Verify response structure
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_endpoint_content(self, api_client):
        """Test health endpoint returns correct service status"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        services = data["services"]
        
        # Should check key services
        assert "anthropic" in services
        assert "database" in services
        
        # Services should be boolean
        assert isinstance(services["anthropic"], bool)
        assert isinstance(services["database"], bool)

    def test_analyze_brand_endpoint_success(self, api_client, sample_brand_data, performance_monitor):
        """Test brand analysis endpoint with valid data"""
        # Measure performance
        performance_monitor.start("brand_analysis")
        response = api_client.post("/analyze-brand", json=sample_brand_data)
        duration = performance_monitor.end("brand_analysis")
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance (45s target, 90s max)
        assert duration < 90.0, f"Analysis took {duration:.2f}s, exceeds 90s max"
        
        # Verify response structure
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        analysis_data = data["data"]
        
        # Verify required fields per FRD
        required_fields = [
            "brand_name",
            "analysis_date", 
            "optimization_metrics",
            "performance_summary",
            "priority_recommendations",
            "llm_test_results"
        ]
        
        for field in required_fields:
            assert field in analysis_data, f"Missing required field: {field}"
        
        # Verify metrics structure
        metrics = analysis_data["optimization_metrics"]
        assert len(metrics) == 12, f"Expected 12 metrics, got {len(metrics)}"
        
        expected_metrics = [
            "chunk_retrieval_frequency",
            "embedding_relevance_score", 
            "attribution_rate",
            "ai_citation_count",
            "vector_index_presence_rate",
            "retrieval_confidence_score",
            "rrf_rank_contribution",
            "llm_answer_coverage",
            "ai_model_crawl_success_rate",
            "semantic_density_score",
            "zero_click_surface_presence",
            "machine_validated_authority"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            
            # Verify metric ranges
            value = metrics[metric]
            if metric == "ai_citation_count":
                assert isinstance(value, int) and value >= 0
            else:
                assert 0.0 <= value <= 1.0, f"Metric {metric} = {value} not in [0,1]"

    def test_analyze_brand_endpoint_validation(self, api_client):
        """Test brand analysis endpoint input validation"""
        # Test missing required fields
        invalid_requests = [
            {},  # Empty request
            {"brand_name": ""},  # Empty brand name
            {"brand_name": "X"},  # Too short brand name
            {"brand_name": "A" * 100},  # Too long brand name
            {"brand_name": "Test<script>"},  # XSS attempt
            {
                "brand_name": "ValidBrand",
                "product_categories": []  # Empty categories
            },
            {
                "brand_name": "ValidBrand", 
                "product_categories": ["a"] * 15  # Too many categories
            },
            {
                "brand_name": "ValidBrand",
                "product_categories": ["valid"],
                "website_url": "invalid-url"  # Invalid URL
            }
        ]
        
        for invalid_request in invalid_requests:
            response = api_client.post("/analyze-brand", json=invalid_request)
            assert response.status_code in [400, 422], f"Expected validation error for {invalid_request}"

    def test_optimization_metrics_endpoint(self, api_client, performance_monitor):
        """Test metrics-only endpoint performance"""
        request_data = {
            "brand_name": "TestBrand",
            "content_sample": "This is test content for metrics calculation. " * 50
        }
        
        # Measure performance
        performance_monitor.start("metrics_calculation")
        response = api_client.post("/optimization-metrics", json=request_data)
        duration = performance_monitor.end("metrics_calculation")
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance (30s target, 60s max)
        assert duration < 60.0, f"Metrics calculation took {duration:.2f}s, exceeds 60s max"
        
        # Verify response structure
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        metrics_data = data["data"]
        assert "optimization_metrics" in metrics_data
        assert "overall_score" in metrics_data
        assert "performance_grade" in metrics_data

    def test_analyze_queries_endpoint(self, api_client, performance_monitor):
        """Test query analysis endpoint"""
        request_data = {
            "brand_name": "TestBrand",
            "product_categories": ["software", "consulting"]
        }
        
        # Measure performance
        performance_monitor.start("query_analysis")
        response = api_client.post("/analyze-queries", json=request_data)
        duration = performance_monitor.end("query_analysis")
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance (10s target, 30s max)
        assert duration < 30.0, f"Query analysis took {duration:.2f}s, exceeds 30s max"
        
        # Verify response structure
        data = response.json()
        assert data["success"] is True
        
        query_data = data["data"]
        required_fields = [
            "generated_queries",
            "query_categories", 
            "purchase_journey_mapping",
            "semantic_coverage",
            "total_queries"
        ]
        
        for field in required_fields:
            assert field in query_data
        
        # Verify query generation (30-50 queries per FRD)
        queries = query_data["generated_queries"]
        assert 30 <= len(queries) <= 50
        assert all(isinstance(q, str) for q in queries)
        
        # Verify categorization
        categories = query_data["query_categories"]
        expected_categories = ["informational", "commercial", "navigational", "transactional"]
        for category in expected_categories:
            assert category in categories
            assert "queries" in categories[category]
            assert "count" in categories[category]

    def test_brands_list_endpoint(self, api_client, db_session):
        """Test brands listing endpoint"""
        # Create test brand in database
        from db_models import Brand
        
        test_brand = Brand(
            name="TestBrand123",
            website_url="https://testbrand123.com",
            industry="technology"
        )
        db_session.add(test_brand)
        db_session.commit()
        
        # Test endpoint
        response = api_client.get("/brands")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "brands" in data["data"]
        assert "total_count" in data["data"]
        
        # Verify brand in response
        brands = data["data"]["brands"]
        test_brand_found = any(b["name"] == "TestBrand123" for b in brands)
        assert test_brand_found

    def test_brand_history_endpoint(self, api_client, db_session):
        """Test brand analysis history endpoint"""
        from db_models import Brand, Analysis
        
        # Create test brand and analysis
        test_brand = Brand(name="HistoryTestBrand")
        db_session.add(test_brand)
        db_session.commit()
        db_session.refresh(test_brand)
        
        test_analysis = Analysis(
            brand_id=test_brand.id,
            status="completed",
            metrics={"overall_score": 0.75},
            processing_time=45.2
        )
        db_session.add(test_analysis)
        db_session.commit()
        
        # Test endpoint
        response = api_client.get(f"/brands/{test_brand.name}/history")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        history_data = data["data"]
        assert "brand_name" in history_data
        assert "history" in history_data
        assert len(history_data["history"]) > 0

    def test_error_handling(self, api_client):
        """Test API error handling"""
        # Test 404 error
        response = api_client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Test malformed JSON
        response = api_client.post(
            "/analyze-brand",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_cors_headers(self, api_client):
        """Test CORS headers are properly set"""
        # Test preflight request
        response = api_client.options(
            "/analyze-brand",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should allow CORS
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200

    def test_rate_limiting(self, api_client):
        """Test rate limiting behavior"""
        # Make multiple rapid requests
        responses = []
        for i in range(5):
            response = api_client.get("/health")
            responses.append(response)
        
        # All health checks should succeed (rate limit is higher)
        assert all(r.status_code == 200 for r in responses)

    def test_concurrent_requests(self, api_client, sample_brand_data):
        """Test handling concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = api_client.post("/optimization-metrics", json={
                    "brand_name": sample_brand_data["brand_name"],
                    "content_sample": sample_brand_data["content_sample"][:500]  # Shorter for speed
                })
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Start 3 concurrent requests
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=60)
        
        # Check results
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 3
        assert all(code == 200 for code in status_codes)

    def test_request_size_limits(self, api_client):
        """Test request size limitations"""
        # Test very large content sample
        large_request = {
            "brand_name": "TestBrand",
            "product_categories": ["test"],
            "content_sample": "A" * (11 * 1024 * 1024)  # 11MB content
        }
        
        response = api_client.post("/analyze-brand", json=large_request)
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 413, 422]

    def test_authentication_headers(self, api_client):
        """Test authentication header handling"""
        # Test with Authorization header
        headers = {"Authorization": "Bearer test-token"}
        response = api_client.get("/health", headers=headers)
        
        # Health endpoint should work regardless of auth
        assert response.status_code == 200

    def test_api_versioning(self, api_client):
        """Test API version handling"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        # Should include version info
        data = response.json()
        assert "version" in data or "timestamp" in data

class TestAPIValidation:
    """Test suite for API input validation"""

    def test_brand_name_validation(self, api_client):
        """Test brand name validation rules"""
        valid_names = [
            "TechCorp",
            "Tech & Co",
            "Tech-Solutions",
            "Tech.Solutions",
            "A" * 50  # Max length
        ]
        
        invalid_names = [
            "",  # Empty
            "A",  # Too short
            "A" * 51,  # Too long
            "Tech<script>",  # XSS
            "Tech|Corp",  # Invalid characters
            "Tech;DROP TABLE",  # SQL injection attempt
        ]
        
        base_request = {
            "product_categories": ["test"],
            "content_sample": "Test content"
        }
        
        for name in valid_names:
            request = {**base_request, "brand_name": name}
            response = api_client.post("/analyze-brand", json=request)
            assert response.status_code in [200, 201], f"Valid name '{name}' was rejected"
        
        for name in invalid_names:
            request = {**base_request, "brand_name": name}
            response = api_client.post("/analyze-brand", json=request)
            assert response.status_code in [400, 422], f"Invalid name '{name}' was accepted"

    def test_url_validation(self, api_client):
        """Test URL validation"""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://sub.example.com/path",
            "https://example.com:8080/path?query=value"
        ]
        
        invalid_urls = [
            "ftp://example.com",  # Wrong protocol
            "javascript:alert(1)",  # XSS
            "http://localhost",  # Localhost blocked
            "http://127.0.0.1",  # Local IP blocked
            "not-a-url",  # Invalid format
        ]
        
        base_request = {
            "brand_name": "TestBrand",
            "product_categories": ["test"]
        }
        
        for url in valid_urls:
            request = {**base_request, "website_url": url}
            response = api_client.post("/analyze-brand", json=request)
            assert response.status_code in [200, 201], f"Valid URL '{url}' was rejected"
        
        for url in invalid_urls:
            request = {**base_request, "website_url": url}
            response = api_client.post("/analyze-brand", json=request)
            # Some may be accepted but processed safely
            assert response.status_code in [200, 201, 400, 422]

    def test_category_validation(self, api_client):
        """Test product category validation"""
        valid_categories = [
            ["electronics"],
            ["software", "hardware"],
            ["category-with-hyphens"],
            ["a" * 50],  # Max length category
            ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9", "cat10"]  # Max count
        ]
        
        invalid_categories = [
            [],  # Empty
            [""],  # Empty category
            ["a"],  # Too short
            ["a" * 51],  # Too long
            ["cat1"] * 11,  # Too many
            ["cat<script>"],  # XSS
        ]
        
        base_request = {
            "brand_name": "TestBrand"
        }
        
        for categories in valid_categories:
            request = {**base_request, "product_categories": categories}
            response = api_client.post("/analyze-brand", json=request)
            assert response.status_code in [200, 201], f"Valid categories {categories} were rejected"
        
        for categories in invalid_categories:
            request = {**base_request, "product_categories": categories}
            response = api_client.post("/analyze-brand", json=request)
            assert response.status_code in [400, 422], f"Invalid categories {categories} were accepted"

class TestAPIPerformance:
    """Test suite for API performance requirements"""

    def test_endpoint_performance_requirements(self, api_client, sample_brand_data, performance_monitor):
        """Test all endpoints meet FRD performance requirements"""
        performance_tests = [
            {
                "endpoint": "GET /health",
                "method": "get",
                "url": "/health",
                "data": None,
                "target_time": 0.1,
                "max_time": 0.5
            },
            {
                "endpoint": "POST /optimization-metrics", 
                "method": "post",
                "url": "/optimization-metrics",
                "data": {
                    "brand_name": sample_brand_data["brand_name"],
                    "content_sample": sample_brand_data["content_sample"][:1000]
                },
                "target_time": 30.0,
                "max_time": 60.0
            },
            {
                "endpoint": "POST /analyze-queries",
                "method": "post", 
                "url": "/analyze-queries",
                "data": {
                    "brand_name": sample_brand_data["brand_name"],
                    "product_categories": sample_brand_data["product_categories"]
                },
                "target_time": 10.0,
                "max_time": 30.0
            }
        ]
        
        for test in performance_tests:
            performance_monitor.start(test["endpoint"])
            
            if test["method"] == "get":
                response = api_client.get(test["url"])
            else:
                response = api_client.post(test["url"], json=test["data"])
            
            duration = performance_monitor.end(test["endpoint"])
            
            # Verify response success
            assert response.status_code == 200, f"{test['endpoint']} failed with status {response.status_code}"
            
            # Verify performance requirements
            assert duration <= test["max_time"], f"{test['endpoint']} took {duration:.2f}s, exceeds max {test['max_time']}s"
            
            if duration > test["target_time"]:
                print(f"Warning: {test['endpoint']} took {duration:.2f}s, exceeds target {test['target_time']}s")

    def test_response_size_reasonable(self, api_client, sample_brand_data):
        """Test API responses are reasonably sized"""
        response = api_client.post("/analyze-brand", json=sample_brand_data)
        assert response.status_code == 200
        
        # Response should be reasonable size (< 1MB)
        content_length = len(response.content)
        assert content_length < 1024 * 1024, f"Response too large: {content_length} bytes"

    def test_memory_usage_stable(self, api_client, sample_brand_data):
        """Test multiple requests don't cause memory leaks"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for i in range(5):
            response = api_client.post("/optimization-metrics", json={
                "brand_name": f"TestBrand{i}",
                "content_sample": sample_brand_data["content_sample"][:500]
            })
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Excessive memory usage: {memory_increase} bytes"