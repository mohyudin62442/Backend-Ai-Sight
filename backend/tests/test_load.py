import pytest
import asyncio
import time
import threading
import concurrent.futures
import statistics
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import queue

class TestLoadScenarios:
    """Test various load scenarios per FRD requirements"""

    def test_light_load_scenario(self, api_client, load_test_scenarios):
        """Test light load: 5 concurrent users, 10 requests each"""
        scenario = load_test_scenarios['light_load']
        
        def user_session(user_id):
            """Simulate a user session"""
            results = []
            
            for request_num in range(scenario['requests_per_user']):
                start_time = time.time()
                
                try:
                    # Mix of different request types
                    if request_num % 3 == 0:
                        response = api_client.get("/health")
                    elif request_num % 3 == 1:
                        response = api_client.get("/brands")
                    else:
                        response = api_client.post("/optimization-metrics", json={
                            "brand_name": f"LoadTestBrand{user_id}_{request_num}",
                            "content_sample": "Load testing content sample. " * 20
                        })
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                    # Small delay between requests
                    time.sleep(random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Run load test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario['concurrent_users']) as executor:
            futures = [
                executor.submit(user_session, user_id) 
                for user_id in range(scenario['concurrent_users'])
            ]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures, timeout=300):
                all_results.extend(future.result())
        
        total_duration = time.time() - start_time
        
        # Analyze results
        self._analyze_load_test_results(all_results, scenario, total_duration, "Light Load")

    def test_normal_load_scenario(self, api_client, load_test_scenarios):
        """Test normal load: 20 concurrent users, 25 requests each"""
        scenario = load_test_scenarios['normal_load']
        
        def user_session(user_id):
            """Simulate a normal user session with realistic patterns"""
            results = []
            
            # Ramp up - not all users start simultaneously
            time.sleep(random.uniform(0, scenario['ramp_up_time']))
            
            for request_num in range(scenario['requests_per_user']):
                start_time = time.time()
                
                try:
                    # Realistic request distribution
                    rand = random.random()
                    
                    if rand < 0.3:  # 30% health checks
                        response = api_client.get("/health")
                    elif rand < 0.5:  # 20% brand listings
                        response = api_client.get("/brands")
                    elif rand < 0.8:  # 30% metrics calculations
                        response = api_client.post("/optimization-metrics", json={
                            "brand_name": f"NormalLoadBrand{user_id}",
                            "content_sample": "Normal load testing content. " * 30
                        })
                    else:  # 20% query analysis
                        response = api_client.post("/analyze-queries", json={
                            "brand_name": f"NormalLoadBrand{user_id}",
                            "product_categories": ["software", "testing"]
                        })
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200,
                        "request_type": "metrics" if rand >= 0.5 else "simple"
                    })
                    
                    # Realistic think time
                    time.sleep(random.uniform(0.5, 2.0))
                    
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Run load test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario['concurrent_users']) as executor:
            futures = [
                executor.submit(user_session, user_id) 
                for user_id in range(scenario['concurrent_users'])
            ]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures, timeout=600):
                all_results.extend(future.result())
        
        total_duration = time.time() - start_time
        
        # Analyze results
        self._analyze_load_test_results(all_results, scenario, total_duration, "Normal Load")

    def test_stress_load_scenario(self, api_client, load_test_scenarios):
        """Test stress load: 50 concurrent users, 50 requests each"""
        scenario = load_test_scenarios['stress_load']
        
        def aggressive_user_session(user_id):
            """Simulate aggressive user behavior"""
            results = []
            
            # Staggered start
            time.sleep(random.uniform(0, scenario['ramp_up_time']))
            
            for request_num in range(scenario['requests_per_user']):
                start_time = time.time()
                
                try:
                    # More aggressive request pattern
                    if request_num % 4 == 0:
                        response = api_client.get("/health")
                    elif request_num % 4 == 1:
                        response = api_client.post("/optimization-metrics", json={
                            "brand_name": f"StressBrand{user_id}",
                            "content_sample": "Stress test content. " * 10  # Shorter for speed
                        })
                    elif request_num % 4 == 2:
                        response = api_client.post("/analyze-queries", json={
                            "brand_name": f"StressBrand{user_id}",
                            "product_categories": ["stress", "test"]
                        })
                    else:
                        response = api_client.get("/brands")
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                    # Minimal think time for stress testing
                    time.sleep(random.uniform(0.1, 0.3))
                    
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "request_num": request_num,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Run stress test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario['concurrent_users']) as executor:
            futures = [
                executor.submit(aggressive_user_session, user_id) 
                for user_id in range(scenario['concurrent_users'])
            ]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures, timeout=900):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    print(f"User session failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Analyze results
        self._analyze_load_test_results(all_results, scenario, total_duration, "Stress Load")

    def test_burst_load_pattern(self, api_client):
        """Test burst load pattern - sudden spike in traffic"""
        def burst_requests():
            """Make rapid burst of requests"""
            results = []
            
            for i in range(20):  # 20 rapid requests
                start_time = time.time()
                
                try:
                    response = api_client.get("/health")
                    duration = time.time() - start_time
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    results.append({
                        "request_id": i,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Simulate 10 users making burst requests simultaneously
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(burst_requests) for _ in range(10)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=120):
                all_results.extend(future.result())
        
        total_duration = time.time() - start_time
        
        # Analyze burst results
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r["success"])
        success_rate = (successful_requests / total_requests) * 100
        
        avg_response_time = statistics.mean([r["duration"] for r in all_results if r["success"]])
        
        print(f"\nBurst Load Test Results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Total duration: {total_duration:.2f}s")
        
        # Burst load should be handled gracefully
        assert success_rate >= 90, f"Burst load success rate {success_rate:.1f}% too low"
        assert avg_response_time < 1.0, f"Average response time {avg_response_time:.3f}s too high"

    def test_sustained_load_pattern(self, api_client):
        """Test sustained load over longer period"""
        duration_minutes = 2  # 2 minutes sustained load
        requests_per_minute = 30  # 30 requests per minute
        
        def sustained_requester():
            """Make requests at steady rate"""
            results = []
            end_time = time.time() + (duration_minutes * 60)
            request_interval = 60.0 / requests_per_minute  # Seconds between requests
            
            while time.time() < end_time:
                start_time = time.time()
                
                try:
                    response = api_client.get("/health")
                    duration = time.time() - start_time
                    
                    results.append({
                        "timestamp": start_time,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    results.append({
                        "timestamp": start_time,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
                
                # Wait for next request interval
                time.sleep(max(0, request_interval - (time.time() - start_time)))
            
            return results
        
        # Run sustained load
        print(f"\nRunning sustained load test for {duration_minutes} minutes...")
        start_time = time.time()
        
        results = sustained_requester()
        total_duration = time.time() - start_time
        
        # Analyze sustained load
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = (successful_requests / total_requests) * 100
        
        if successful_requests > 0:
            avg_response_time = statistics.mean([r["duration"] for r in results if r["success"]])
            max_response_time = max([r["duration"] for r in results if r["success"]])
        else:
            avg_response_time = 0
            max_response_time = 0
        
        print(f"Sustained Load Test Results:")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Total requests: {total_requests}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        
        # Sustained load requirements
        assert success_rate >= 99, f"Sustained load success rate {success_rate:.1f}% too low"
        assert avg_response_time < 0.5, f"Average response time {avg_response_time:.3f}s too high"

    def test_mixed_workload_scenario(self, api_client, sample_brand_data):
        """Test mixed workload with different request types"""
        def mixed_workload_user(user_id):
            """Simulate realistic mixed workload"""
            results = []
            
            # Each user makes different types of requests
            request_types = [
                {"type": "health", "weight": 0.4},
                {"type": "metrics", "weight": 0.3},
                {"type": "queries", "weight": 0.2},
                {"type": "brands", "weight": 0.1}
            ]
            
            for request_num in range(15):
                # Choose request type based on weights
                rand = random.random()
                cumulative_weight = 0
                
                for req_type in request_types:
                    cumulative_weight += req_type["weight"]
                    if rand <= cumulative_weight:
                        request_type = req_type["type"]
                        break
                
                start_time = time.time()
                
                try:
                    if request_type == "health":
                        response = api_client.get("/health")
                    elif request_type == "metrics":
                        response = api_client.post("/optimization-metrics", json={
                            "brand_name": f"MixedWorkloadBrand{user_id}",
                            "content_sample": sample_brand_data["content_sample"][:500]
                        })
                    elif request_type == "queries":
                        response = api_client.post("/analyze-queries", json={
                            "brand_name": f"MixedWorkloadBrand{user_id}",
                            "product_categories": sample_brand_data["product_categories"][:2]
                        })
                    else:  # brands
                        response = api_client.get("/brands")
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        "user_id": user_id,
                        "request_type": request_type,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                    # Variable think time based on request type
                    if request_type == "health":
                        time.sleep(random.uniform(0.1, 0.5))
                    else:
                        time.sleep(random.uniform(1.0, 3.0))
                    
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "request_type": request_type,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Run mixed workload with 15 concurrent users
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(mixed_workload_user, user_id) 
                for user_id in range(15)
            ]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures, timeout=600):
                all_results.extend(future.result())
        
        total_duration = time.time() - start_time
        
        # Analyze by request type
        by_request_type = {}
        for result in all_results:
            req_type = result["request_type"]
            if req_type not in by_request_type:
                by_request_type[req_type] = []
            by_request_type[req_type].append(result)
        
        print(f"\nMixed Workload Test Results:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total requests: {len(all_results)}")
        
        for req_type, type_results in by_request_type.items():
            successful = sum(1 for r in type_results if r["success"])
            total = len(type_results)
            success_rate = (successful / total) * 100
            
            if successful > 0:
                avg_time = statistics.mean([r["duration"] for r in type_results if r["success"]])
                print(f"  {req_type}: {successful}/{total} ({success_rate:.1f}%), avg {avg_time:.3f}s")
            else:
                print(f"  {req_type}: {successful}/{total} ({success_rate:.1f}%)")
        
        # Overall success rate should be high
        total_successful = sum(1 for r in all_results if r["success"])
        overall_success_rate = (total_successful / len(all_results)) * 100
        assert overall_success_rate >= 95, f"Overall success rate {overall_success_rate:.1f}% too low"

    def _analyze_load_test_results(self, results: List[Dict], scenario: Dict, total_duration: float, test_name: str):
        """Analyze and report load test results"""
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Calculate response time statistics
        successful_durations = [r["duration"] for r in results if r["success"]]
        if successful_durations:
            avg_response_time = statistics.mean(successful_durations)
            median_response_time = statistics.median(successful_durations)
            p95_response_time = statistics.quantiles(successful_durations, n=20)[18]  # 95th percentile
            max_response_time = max(successful_durations)
            min_response_time = min(successful_durations)
        else:
            avg_response_time = median_response_time = p95_response_time = max_response_time = min_response_time = 0
        
        # Calculate throughput
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        successful_requests_per_second = successful_requests / total_duration if total_duration > 0 else 0
        
        # Group by user to analyze user experience
        by_user = {}
        for result in results:
            user_id = result.get("user_id", "unknown")
            if user_id not in by_user:
                by_user[user_id] = []
            by_user[user_id].append(result)
        
        user_success_rates = []
        for user_results in by_user.values():
            user_successful = sum(1 for r in user_results if r["success"])
            user_total = len(user_results)
            user_success_rate = (user_successful / user_total) * 100 if user_total > 0 else 0
            user_success_rates.append(user_success_rate)
        
        # Report results
        print(f"\n{test_name} Results:")
        print(f"  Scenario: {scenario['concurrent_users']} users, {scenario['requests_per_user']} requests each")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {success_rate:.2f}%")
        print(f"  Throughput: {requests_per_second:.2f} req/s")
        print(f"  Successful throughput: {successful_requests_per_second:.2f} req/s")
        print(f"  Response times:")
        print(f"    Average: {avg_response_time:.3f}s")
        print(f"    Median: {median_response_time:.3f}s")
        print(f"    95th percentile: {p95_response_time:.3f}s")
        print(f"    Min: {min_response_time:.3f}s")
        print(f"    Max: {max_response_time:.3f}s")
        
        if user_success_rates:
            print(f"  User experience:")
            print(f"    Average user success rate: {statistics.mean(user_success_rates):.2f}%")
            print(f"    Worst user success rate: {min(user_success_rates):.2f}%")
        
        # Assertions based on load level
        if "Light" in test_name:
            assert success_rate >= 99, f"Light load success rate {success_rate:.2f}% too low"
            assert avg_response_time < 2.0, f"Light load avg response time {avg_response_time:.3f}s too high"
            assert p95_response_time < 5.0, f"Light load P95 response time {p95_response_time:.3f}s too high"
        elif "Normal" in test_name:
            assert success_rate >= 95, f"Normal load success rate {success_rate:.2f}% too low"
            assert avg_response_time < 5.0, f"Normal load avg response time {avg_response_time:.3f}s too high"
            assert p95_response_time < 10.0, f"Normal load P95 response time {p95_response_time:.3f}s too high"
        elif "Stress" in test_name:
            assert success_rate >= 90, f"Stress load success rate {success_rate:.2f}% too low"
            assert avg_response_time < 10.0, f"Stress load avg response time {avg_response_time:.3f}s too high"
            # More lenient requirements for stress testing

class TestResourceLimits:
    """Test system behavior at resource limits"""

    def test_memory_usage_under_load(self, api_client, sample_brand_data):
        """Test memory usage doesn't grow excessively under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        def memory_intensive_requests():
            """Make requests that use memory"""
            for i in range(10):
                response = api_client.post("/optimization-metrics", json={
                    "brand_name": f"MemoryTestBrand{i}",
                    "content_sample": sample_brand_data["content_sample"]
                })
                assert response.status_code == 200
                
                # Check memory after each request
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"  After request {i+1}: {current_memory:.1f} MB")
        
        # Run memory test
        memory_intensive_requests()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory growth should be reasonable
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f} MB, too much"

    def test_cpu_usage_under_load(self, api_client, sample_brand_data):
        """Test CPU usage under load"""
        import psutil
        
        cpu_measurements = []
        
        def cpu_monitor():
            """Monitor CPU usage"""
            for _ in range(15):  # Monitor for 15 seconds
                cpu_measurements.append(psutil.cpu_percent(interval=1))
        
        def cpu_intensive_requests():
            """Make CPU-intensive requests"""
            for i in range(5):
                response = api_client.post("/analyze-queries", json={
                    "brand_name": f"CPUTestBrand{i}",
                    "product_categories": ["cpu", "intensive", "testing", "load"]
                })
                assert response.status_code == 200
        
        # Start CPU monitoring
        import threading
        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()
        
        # Run CPU-intensive requests
        cpu_intensive_requests()
        
        monitor_thread.join()
        
        if cpu_measurements:
            avg_cpu = statistics.mean(cpu_measurements)
            max_cpu = max(cpu_measurements)
            
            print(f"CPU usage during load:")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Peak: {max_cpu:.1f}%")
            
            # CPU usage should be reasonable
            assert avg_cpu < 85, f"Average CPU {avg_cpu:.1f}% too high"
            assert max_cpu < 95, f"Peak CPU {max_cpu:.1f}% too high"

    def test_connection_pool_limits(self, api_client):
        """Test database connection pool behavior under load"""
        def database_heavy_requests():
            """Make requests that use database connections"""
            results = []
            
            for i in range(25):  # More than typical connection pool size
                start_time = time.time()
                
                try:
                    response = api_client.get("/brands")
                    duration = time.time() - start_time
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    results.append({
                        "request_id": i,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Run concurrent database requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(database_heavy_requests) for _ in range(3)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=180):
                all_results.extend(future.result())
        
        # Analyze connection pool behavior
        successful_requests = sum(1 for r in all_results if r["success"])
        total_requests = len(all_results)
        success_rate = (successful_requests / total_requests) * 100
        
        avg_response_time = statistics.mean([r["duration"] for r in all_results if r["success"]])
        
        print(f"Connection pool test:")
        print(f"  Total requests: {total_requests}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average response time: {avg_response_time:.3f}s")
        
        # Should handle connection pool limits gracefully
        assert success_rate >= 95, f"Connection pool success rate {success_rate:.1f}% too low"
        assert avg_response_time < 5.0, f"Connection pool response time {avg_response_time:.3f}s too high"

class TestErrorConditions:
    """Test system behavior under error conditions"""

    def test_malformed_request_load(self, api_client):
        """Test system handles malformed requests under load"""
        def malformed_requests():
            """Send various malformed requests"""
            results = []
            
            malformed_data = [
                {"brand_name": ""},  # Empty brand name
                {"brand_name": "Test", "product_categories": []},  # Empty categories
                {"invalid": "data"},  # Missing required fields
                {"brand_name": "A" * 100},  # Too long brand name
                {"brand_name": "Test", "content_sample": "A" * 1000000},  # Huge content
            ]
            
            for i, data in enumerate(malformed_data * 4):  # Repeat malformed requests
                start_time = time.time()
                
                try:
                    response = api_client.post("/analyze-brand", json=data)
                    duration = time.time() - start_time
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "duration": duration,
                        "handled_gracefully": response.status_code in [400, 422]
                    })
                    
                except Exception as e:
                    results.append({
                        "request_id": i,
                        "status_code": 500,
                        "duration": time.time() - start_time,
                        "handled_gracefully": False,
                        "error": str(e)
                    })
            
            return results
        
        # Send malformed requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(malformed_requests) for _ in range(5)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=120):
                all_results.extend(future.result())
        
        # System should handle malformed requests gracefully
        total_requests = len(all_results)
        gracefully_handled = sum(1 for r in all_results if r["handled_gracefully"])
        graceful_rate = (gracefully_handled / total_requests) * 100
        
        print(f"Malformed request handling:")
        print(f"  Total malformed requests: {total_requests}")
        print(f"  Gracefully handled: {gracefully_handled}")
        print(f"  Graceful handling rate: {graceful_rate:.1f}%")
        
        assert graceful_rate >= 90, f"Graceful handling rate {graceful_rate:.1f}% too low"

    def test_timeout_behavior_under_load(self, api_client):
        """Test timeout behavior under load"""
        def slow_requests():
            """Make requests that might timeout"""
            results = []
            
            # Large content that might cause timeouts
            large_content = "This is very large content for timeout testing. " * 1000
            
            for i in range(10):
                start_time = time.time()
                
                try:
                    response = api_client.post("/optimization-metrics", 
                                            json={
                                                "brand_name": f"TimeoutTestBrand{i}",
                                                "content_sample": large_content
                                            },
                                            timeout=60)  # 60 second timeout
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "duration": duration,
                        "completed": True
                    })
                    
                except Exception as e:
                    duration = time.time() - start_time
                    results.append({
                        "request_id": i,
                        "status_code": 408,  # Timeout
                        "duration": duration,
                        "completed": False,
                        "error": str(e)
                    })
            
            return results
        
        results = slow_requests()
        
        completed_requests = sum(1 for r in results if r["completed"])
        total_requests = len(results)
        completion_rate = (completed_requests / total_requests) * 100
        
        avg_duration = statistics.mean([r["duration"] for r in results])
        
        print(f"Timeout behavior test:")
        print(f"  Total requests: {total_requests}")
        print(f"  Completed requests: {completed_requests}")
        print(f"  Completion rate: {completion_rate:.1f}%")
        print(f"  Average duration: {avg_duration:.2f}s")
        
        # Most requests should complete within reasonable time
        assert completion_rate >= 80, f"Completion rate {completion_rate:.1f}% too low"