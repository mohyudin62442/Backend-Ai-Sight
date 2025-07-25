"""
Performance testing suite for AI Optimization Engine
Tests FRD performance requirements under various load conditions
"""

import pytest
import asyncio
import time
import threading
import concurrent.futures
import psutil
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import statistics

class TestPerformanceRequirements:
    """Test suite for FRD performance requirements"""

    def test_health_check_performance(self, api_client, performance_monitor):
        """Test health check meets FRD requirement: <100ms target, 500ms max"""
        measurements = []
        
        # Test multiple times for consistency
        for i in range(10):
            performance_monitor.start(f"health_check_{i}")
            response = api_client.get("/health")
            duration = performance_monitor.end(f"health_check_{i}")
            
            assert response.status_code == 200
            measurements.append(duration * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        avg_time = statistics.mean(measurements)
        max_time = max(measurements)
        min_time = min(measurements)
        
        print(f"Health check performance: avg={avg_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms")
        
        # FRD Requirements
        assert max_time < 500, f"Health check max time {max_time:.1f}ms exceeds 500ms limit"
        assert avg_time < 100, f"Health check avg time {avg_time:.1f}ms exceeds 100ms target"

    def test_metrics_calculation_performance(self, api_client, sample_brand_data, performance_monitor):
        """Test metrics calculation meets FRD requirement: <30s target, 60s max"""
        request_data = {
            "brand_name": sample_brand_data["brand_name"],
            "content_sample": sample_brand_data["content_sample"]
        }
        
        performance_monitor.start("metrics_calculation")
        response = api_client.post("/optimization-metrics", json=request_data)
        duration = performance_monitor.end("metrics_calculation")
        
        assert response.status_code == 200
        
        # FRD Requirements
        assert duration < 60, f"Metrics calculation took {duration:.2f}s, exceeds 60s max"
        if duration > 30:
            print(f"Warning: Metrics calculation took {duration:.2f}s, exceeds 30s target")

    def test_full_analysis_performance(self, api_client, sample_brand_data, performance_monitor):
        """Test full analysis meets FRD requirement: <45s target, 90s max"""
        performance_monitor.start("full_analysis")
        response = api_client.post("/analyze-brand", json=sample_brand_data)
        duration = performance_monitor.end("full_analysis")
        
        assert response.status_code == 200
        
        # FRD Requirements
        assert duration < 90, f"Full analysis took {duration:.2f}s, exceeds 90s max"
        if duration > 45:
            print(f"Warning: Full analysis took {duration:.2f}s, exceeds 45s target")

    def test_query_analysis_performance(self, api_client, performance_monitor):
        """Test query analysis meets FRD requirement: <10s target, 30s max"""
        request_data = {
            "brand_name": "TestBrand",
            "product_categories": ["software", "consulting"]
        }
        
        performance_monitor.start("query_analysis")
        response = api_client.post("/analyze-queries", json=request_data)
        duration = performance_monitor.end("query_analysis")
        
        assert response.status_code == 200
        
        # FRD Requirements
        assert duration < 30, f"Query analysis took {duration:.2f}s, exceeds 30s max"
        if duration > 10:
            print(f"Warning: Query analysis took {duration:.2f}s, exceeds 10s target")

    def test_dashboard_load_performance(self, api_client, db_session, performance_monitor):
        """Test dashboard loads meet FRD requirement: <2s target, 5s max"""
        from db_models import Brand, Analysis
        
        # Create test data for dashboard
        test_brand = Brand(name="DashboardTestBrand")
        db_session.add(test_brand)
        db_session.commit()
        db_session.refresh(test_brand)
        
        # Test brands list endpoint (simulates dashboard load)
        performance_monitor.start("dashboard_load")
        response = api_client.get("/brands")
        duration = performance_monitor.end("dashboard_load")
        
        assert response.status_code == 200
        
        # FRD Requirements
        assert duration < 5, f"Dashboard load took {duration:.2f}s, exceeds 5s max"
        if duration > 2:
            print(f"Warning: Dashboard load took {duration:.2f}s, exceeds 2s target")

    def test_concurrent_user_performance(self, api_client, sample_brand_data):
        """Test system handles concurrent users per FRD requirement (100 concurrent users)"""
        def make_request(user_id):
            """Make a request as a simulated user"""
            start_time = time.time()
            try:
                # Simulate different types of requests
                if user_id % 3 == 0:
                    response = api_client.get("/health")
                elif user_id % 3 == 1:
                    response = api_client.post("/optimization-metrics", json={
                        "brand_name": f"ConcurrentTestBrand{user_id}",
                        "content_sample": sample_brand_data["content_sample"][:500]
                    })
                else:
                    response = api_client.get("/brands")
                
                duration = time.time() - start_time
                return {
                    "user_id": user_id,
                    "status_code": response.status_code,
                    "duration": duration,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "status_code": 500,
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Test with reduced concurrent users for testing environment
        concurrent_users = 20  # Reduced from 100 for test environment
        
        start_time = time.time()
        
        # Use thread pool to simulate concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_users)]
            results = [future.result(timeout=120) for future in futures]
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        success_rate = len(successful_requests) / len(results) * 100
        avg_response_time = statistics.mean([r["duration"] for r in successful_requests]) if successful_requests else 0
        
        print(f"Concurrent test results:")
        print(f"  Users: {concurrent_users}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Failed requests: {len(failed_requests)}")
        
        # FRD Requirements
        assert success_rate >= 95, f"Success rate {success_rate:.1f}% below 95% requirement"
        assert avg_response_time < 10, f"Average response time {avg_response_time:.2f}s too high"

    def test_daily_analysis_capacity(self, api_client, sample_brand_data):
        """Test system can handle 1,000 analyses per day requirement"""
        # Simulate a burst of analyses (scaled down for testing)
        num_analyses = 10  # Scaled down from realistic daily load
        
        start_time = time.time()
        
        def run_analysis(analysis_id):
            """Run a single analysis"""
            try:
                response = api_client.post("/optimization-metrics", json={
                    "brand_name": f"LoadTestBrand{analysis_id}",
                    "content_sample": sample_brand_data["content_sample"][:300]  # Shorter for speed
                })
                return response.status_code == 200
            except:
                return False
        
        # Run analyses concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_analysis, i) for i in range(num_analyses)]
            results = [future.result(timeout=60) for future in futures]
        
        total_duration = time.time() - start_time
        success_count = sum(results)
        
        # Calculate performance metrics
        analyses_per_second = success_count / total_duration
        projected_daily_capacity = analyses_per_second * 86400  # 24 hours
        
        print(f"Analysis capacity test:")
        print(f"  Completed: {success_count}/{num_analyses}")
        print(f"  Rate: {analyses_per_second:.2f} analyses/second")
        print(f"  Projected daily capacity: {projected_daily_capacity:.0f} analyses")
        
        # Should be able to handle required load
        assert success_count == num_analyses, f"Only {success_count}/{num_analyses} analyses succeeded"
        assert projected_daily_capacity >= 1000, f"Projected capacity {projected_daily_capacity:.0f} below 1000/day requirement"

    def test_uptime_reliability(self, api_client):
        """Test system reliability (99.9% uptime requirement simulation)"""
        # Test system stability over time with repeated requests
        num_requests = 100
        failures = 0
        response_times = []
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = api_client.get("/health")
                duration = time.time() - start_time
                response_times.append(duration)
                
                if response.status_code != 200:
                    failures += 1
            except Exception:
                failures += 1
                response_times.append(10.0)  # Timeout value
            
            # Small delay to simulate realistic usage
            time.sleep(0.1)
        
        uptime_percentage = ((num_requests - failures) / num_requests) * 100
        avg_response_time = statistics.mean(response_times)
        
        print(f"Uptime test results:")
        print(f"  Requests: {num_requests}")
        print(f"  Failures: {failures}")
        print(f"  Uptime: {uptime_percentage:.2f}%")
        print(f"  Average response time: {avg_response_time:.3f}s")
        
        # 99.9% uptime requirement
        assert uptime_percentage >= 99.9, f"Uptime {uptime_percentage:.2f}% below 99.9% requirement"

class TestResourceUtilization:
    """Test system resource usage and optimization"""

    def test_memory_usage_under_load(self, api_client, sample_brand_data):
        """Test memory usage remains reasonable under load"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Run multiple analyses to stress test memory
        for i in range(5):
            response = api_client.post("/optimization-metrics", json={
                "brand_name": f"MemoryTestBrand{i}",
                "content_sample": sample_brand_data["content_sample"]
            })
            assert response.status_code == 200
            
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after analysis {i+1}: {current_memory:.1f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (< 500MB for test load)
        assert memory_increase < 500, f"Excessive memory usage increase: {memory_increase:.1f} MB"

    def test_cpu_usage_efficiency(self, api_client, sample_brand_data):
        """Test CPU usage remains efficient"""
        # Monitor CPU usage during analysis
        cpu_measurements = []
        
        def monitor_cpu():
            """Monitor CPU usage in background"""
            for _ in range(10):
                cpu_measurements.append(psutil.cpu_percent(interval=1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run analysis while monitoring
        response = api_client.post("/analyze-brand", json={
            "brand_name": sample_brand_data["brand_name"],
            "content_sample": sample_brand_data["content_sample"][:1000]  # Shorter for faster test
        })
        assert response.status_code == 200
        
        monitor_thread.join()
        
        if cpu_measurements:
            avg_cpu = statistics.mean(cpu_measurements)
            max_cpu = max(cpu_measurements)
            
            print(f"CPU usage during analysis:")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Peak: {max_cpu:.1f}%")
            
            # CPU usage should be reasonable
            assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}% too high"
            assert max_cpu < 95, f"Peak CPU usage {max_cpu:.1f}% too high"

    def test_database_connection_efficiency(self, db_session):
        """Test database connection usage is efficient"""
        from db_models import Brand, Analysis
        import sqlalchemy
        
        # Get connection pool info
        engine = db_session.bind
        pool = engine.pool
        
        initial_connections = pool.checkedout()
        print(f"Initial DB connections: {initial_connections}")
        
        # Perform database operations
        for i in range(5):
            brand = Brand(name=f"DBTestBrand{i}")
            db_session.add(brand)
            db_session.commit()
            
            # Query brands
            brands = db_session.query(Brand).limit(10).all()
            assert len(brands) > 0
        
        final_connections = pool.checkedout()
        print(f"Final DB connections: {final_connections}")
        
        # Connection usage should be stable
        connection_increase = final_connections - initial_connections
        assert connection_increase <= 2, f"Too many new connections: {connection_increase}"

    def test_cache_performance(self, cache_client):
        """Test Redis cache performance"""
        # Test cache write performance
        write_times = []
        for i in range(100):
            start_time = time.time()
            cache_client.set(f"test_key_{i}", {"data": f"test_value_{i}"})
            write_times.append(time.time() - start_time)
        
        # Test cache read performance
        read_times = []
        for i in range(100):
            start_time = time.time()
            value = cache_client.get(f"test_key_{i}")
            read_times.append(time.time() - start_time)
            assert value is not None
        
        avg_write_time = statistics.mean(write_times) * 1000  # ms
        avg_read_time = statistics.mean(read_times) * 1000    # ms
        
        print(f"Cache performance:")
        print(f"  Average write time: {avg_write_time:.2f}ms")
        print(f"  Average read time: {avg_read_time:.2f}ms")
        
        # Cache operations should be fast
        assert avg_write_time < 10, f"Cache write too slow: {avg_write_time:.2f}ms"
        assert avg_read_time < 5, f"Cache read too slow: {avg_read_time:.2f}ms"

class TestScalabilityLimits:
    """Test system behavior at scale limits"""

    def test_large_content_processing(self, api_client):
        """Test processing very large content samples"""
        # Create progressively larger content
        sizes = [1000, 5000, 10000, 20000]  # words
        
        for size in sizes:
            large_content = ("This is test content for scalability testing. " * size)[:size*10]  # Approximate size
            
            request_data = {
                "brand_name": f"LargeContentTest{size}",
                "content_sample": large_content
            }
            
            start_time = time.time()
            response = api_client.post("/optimization-metrics", json=request_data)
            duration = time.time() - start_time
            
            print(f"Content size ~{size} words: {duration:.2f}s")
            
            # Should handle large content
            assert response.status_code == 200, f"Failed to process {size} word content"
            
            # Time should scale reasonably
            if size <= 10000:
                assert duration < 120, f"Processing {size} words took {duration:.2f}s, too slow"

    def test_many_categories_processing(self, api_client):
        """Test processing maximum number of categories"""
        max_categories = ["category" + str(i) for i in range(10)]  # Max allowed
        
        request_data = {
            "brand_name": "ManyCategoriesTest",
            "product_categories": max_categories
        }
        
        response = api_client.post("/analyze-queries", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        queries = data["data"]["generated_queries"]
        
        # Should generate queries for all categories
        assert len(queries) >= 30, f"Only generated {len(queries)} queries for {len(max_categories)} categories"

    def test_stress_test_rapid_requests(self, api_client):
        """Stress test with rapid successive requests"""
        def rapid_requests():
            """Make rapid requests"""
            results = []
            for i in range(20):
                try:
                    response = api_client.get("/health")
                    results.append(response.status_code == 200)
                except:
                    results.append(False)
                time.sleep(0.05)  # 50ms between requests
            return results
        
        # Run multiple threads making rapid requests
        threads = []
        all_results = []
        
        for _ in range(3):
            thread = threading.Thread(target=lambda: all_results.extend(rapid_requests()))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join(timeout=30)
        
        success_rate = sum(all_results) / len(all_results) * 100 if all_results else 0
        
        print(f"Rapid requests test:")
        print(f"  Total requests: {len(all_results)}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Should handle rapid requests gracefully
        assert success_rate >= 90, f"Success rate {success_rate:.1f}% too low under stress"

class TestPerformanceRegression:
    """Test for performance regressions"""

    def test_baseline_performance_metrics(self, api_client, sample_brand_data, performance_monitor):
        """Establish baseline performance metrics for regression testing"""
        baseline_tests = {
            "health_check": {
                "endpoint": "/health",
                "method": "GET",
                "data": None,
                "expected_max": 0.5
            },
            "optimization_metrics": {
                "endpoint": "/optimization-metrics", 
                "method": "POST",
                "data": {
                    "brand_name": sample_brand_data["brand_name"],
                    "content_sample": sample_brand_data["content_sample"][:500]
                },
                "expected_max": 60
            },
            "query_analysis": {
                "endpoint": "/analyze-queries",
                "method": "POST", 
                "data": {
                    "brand_name": sample_brand_data["brand_name"],
                    "product_categories": sample_brand_data["product_categories"][:2]
                },
                "expected_max": 30
            }
        }
        
        baseline_results = {}
        
        for test_name, test_config in baseline_tests.items():
            performance_monitor.start(test_name)
            
            if test_config["method"] == "GET":
                response = api_client.get(test_config["endpoint"])
            else:
                response = api_client.post(test_config["endpoint"], json=test_config["data"])
            
            duration = performance_monitor.end(test_name)
            
            assert response.status_code == 200
            assert duration <= test_config["expected_max"]
            
            baseline_results[test_name] = duration
            
            print(f"Baseline {test_name}: {duration:.3f}s")
        
        # Store baseline for comparison (in real testing, save to file/database)
        return baseline_results

    def test_performance_consistency(self, api_client):
        """Test performance is consistent across multiple runs"""
        measurements = []
        
        # Run health check multiple times
        for _ in range(20):
            start_time = time.time()
            response = api_client.get("/health")
            duration = time.time() - start_time
            
            assert response.status_code == 200
            measurements.append(duration * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_time = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0
        coefficient_of_variation = (std_dev / mean_time) * 100 if mean_time > 0 else 0
        
        print(f"Performance consistency:")
        print(f"  Mean: {mean_time:.2f}ms")
        print(f"  Std Dev: {std_dev:.2f}ms")
        print(f"  Coefficient of Variation: {coefficient_of_variation:.1f}%")
        
        # Performance should be consistent (CV < 50%)
        assert coefficient_of_variation < 50, f"Performance too inconsistent: CV {coefficient_of_variation:.1f}%"