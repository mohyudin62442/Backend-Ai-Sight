"""
Complete Database/Redis Integration Tests for AI Optimization Engine
Tests real database and Redis operations without mocks or stubs
Validates cross-component integration and data consistency
"""

import pytest
import asyncio
import time
import json
import threading
from datetime import datetime, timedelta
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sqlalchemy.exc import IntegrityError

from db_models import (
    Brand, Analysis, MetricHistory, User, ApiKey, TrackingEvent, 
    BotVisit, TrackingAlert, ScheduledAnalysis, UserBrand
)
from optimization_engine import OptimizationMetrics
from utils import CacheUtils
from tracking_manager import TrackingManager
from log_analyzer import ServerLogAnalyzer
from bot_tracker import ClientSideBotTracker

class TestDatabaseIntegration:
    """Test real database operations and data persistence"""

    def test_brand_lifecycle_complete(self, db_session):
        """Test complete brand lifecycle with all related data"""
        # Create brand
        brand = Brand(
            name="Complete Integration Test Brand",
            website_url="https://complete-integration.com",
            industry="technology",
            tracking_enabled=True,
            tracking_script_installed=True,
            api_key="test_api_key_12345"
        )
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Create user and associate with brand
        user = User(
            email="integration@test.com",
            password_hash="$2b$12$test_hash",
            full_name="Integration Test User",
            plan="growth",
            is_active=True,
            is_verified=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create brand-user relationship
        user_brand = UserBrand(
            user_id=user.id,
            brand_id=brand.id,
            role="admin"
        )
        db_session.add(user_brand)
        db_session.commit()
        
        # Create analysis with comprehensive data
        analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            analysis_type="comprehensive",
            data_source="real",
            metrics={
                "chunk_retrieval_frequency": 0.78,
                "embedding_relevance_score": 0.84,
                "attribution_rate": 0.45,
                "ai_citation_count": 23,
                "vector_index_presence_rate": 0.95,
                "retrieval_confidence_score": 0.72,
                "rrf_rank_contribution": 0.68,
                "llm_answer_coverage": 0.76,
                "ai_model_crawl_success_rate": 0.89,
                "semantic_density_score": 0.71,
                "zero_click_surface_presence": 0.34,
                "machine_validated_authority": 0.66
            },
            recommendations=[
                {
                    "priority": "high",
                    "category": "AI Visibility",
                    "title": "Improve Attribution Rate",
                    "action_items": ["Create FAQ section", "Add customer testimonials"]
                }
            ],
            processing_time=42.5,
            completed_at=datetime.utcnow()
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        # Create metric history entries
        metrics_data = [
            ("attribution_rate", 0.45, "anthropic"),
            ("attribution_rate", 0.38, "openai"),
            ("embedding_relevance_score", 0.84, None),
            ("semantic_density_score", 0.71, None)
        ]
        
        for metric_name, value, platform in metrics_data:
            metric = MetricHistory(
                brand_id=brand.id,
                analysis_id=analysis.id,
                metric_name=metric_name,
                metric_value=value,
                platform=platform,
                data_source="real"
            )
            db_session.add(metric)
        
        db_session.commit()
        
        # Create bot visits
        bot_visits = [
            BotVisit(
                brand_id=brand.id,
                bot_name="GPTBot",
                platform="openai",
                user_agent="GPTBot/1.0",
                timestamp=datetime.utcnow(),
                ip_address="192.168.1.100",
                path="/products/main-product",
                status_code=200,
                response_time=0.35,
                brand_mentioned=True,
                content_type="product"
            ),
            BotVisit(
                brand_id=brand.id,
                bot_name="Claude-Web",
                platform="anthropic",
                user_agent="Claude-Web/1.0",
                timestamp=datetime.utcnow() - timedelta(hours=2),
                ip_address="192.168.1.101",
                path="/about",
                status_code=200,
                response_time=0.28,
                brand_mentioned=False,
                content_type="info"
            )
        ]
        
        for visit in bot_visits:
            db_session.add(visit)
        
        db_session.commit()
        
        # Create tracking events
        session_id = str(uuid4())
        tracking_events = [
            TrackingEvent(
                brand_id=brand.id,
                event_type="pageview",
                session_id=session_id,
                sequence_number=1,
                bot_name="GPTBot",
                platform="openai",
                page_url="https://complete-integration.com/products",
                page_title="Products - Complete Integration",
                timestamp=datetime.utcnow()
            ),
            TrackingEvent(
                brand_id=brand.id,
                event_type="engagement",
                session_id=session_id,
                sequence_number=2,
                time_on_page=45,
                scroll_depth=80,
                timestamp=datetime.utcnow() + timedelta(seconds=45)
            )
        ]
        
        for event in tracking_events:
            db_session.add(event)
        
        db_session.commit()
        
        # Test complex queries and relationships
        # 1. Brand with all related data
        brand_with_data = db_session.query(Brand).filter(Brand.id == brand.id).first()
        assert len(brand_with_data.analyses) == 1
        assert len(brand_with_data.metrics_history) == 4
        assert len(brand_with_data.bot_visits) == 2
        assert len(brand_with_data.tracking_events) == 2
        
        # 2. Analysis with metrics
        analysis_with_metrics = db_session.query(Analysis).filter(Analysis.id == analysis.id).first()
        assert analysis_with_metrics.metrics["attribution_rate"] == 0.45
        assert len(analysis_with_metrics.recommendations) == 1
        
        # 3. Metric history aggregation
        avg_attribution = db_session.query(func.avg(MetricHistory.metric_value)).filter(
            MetricHistory.brand_id == brand.id,
            MetricHistory.metric_name == "attribution_rate"
        ).scalar()
        assert abs(avg_attribution - 0.415) < 0.001  # (0.45 + 0.38) / 2
        
        # 4. Bot visit analysis
        bot_visit_count = db_session.query(func.count(BotVisit.id)).filter(
            BotVisit.brand_id == brand.id,
            BotVisit.brand_mentioned == True
        ).scalar()
        assert bot_visit_count == 1
        
        # 5. User-brand relationship
        user_brands = db_session.query(UserBrand).filter(UserBrand.user_id == user.id).all()
        assert len(user_brands) == 1
        assert user_brands[0].role == "admin"

    def test_database_constraints_enforcement(self, db_session):
        """Test database constraints are properly enforced"""
        # Test unique constraints
        brand1 = Brand(name="Unique Constraint Test")
        brand2 = Brand(name="Unique Constraint Test")  # Same name
        
        db_session.add(brand1)
        db_session.commit()
        
        db_session.add(brand2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()
        
        # Test foreign key constraints
        orphan_analysis = Analysis(
            brand_id=uuid4(),  # Non-existent brand
            status="pending"
        )
        db_session.add(orphan_analysis)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()
        
        # Test check constraints
        invalid_user = User(
            email="invalid-email",  # Invalid email format
            password_hash="test"
        )
        db_session.add(invalid_user)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_database_transactions_and_rollback(self, db_session):
        """Test database transaction handling and rollback"""
        # Create initial data
        brand = Brand(name="Transaction Test Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Test successful transaction
        analysis1 = Analysis(brand_id=brand.id, status="pending")
        analysis2 = Analysis(brand_id=brand.id, status="processing")
        
        db_session.add(analysis1)
        db_session.add(analysis2)
        db_session.commit()
        
        # Verify both were saved
        count = db_session.query(func.count(Analysis.id)).filter(Analysis.brand_id == brand.id).scalar()
        assert count == 2
        
        # Test transaction rollback
        try:
            analysis3 = Analysis(brand_id=brand.id, status="completed")
            invalid_analysis = Analysis(brand_id=uuid4(), status="failed")  # Invalid foreign key
            
            db_session.add(analysis3)
            db_session.add(invalid_analysis)
            db_session.commit()
        except IntegrityError:
            db_session.rollback()
        
        # Verify rollback - should still be 2 analyses
        count_after_rollback = db_session.query(func.count(Analysis.id)).filter(Analysis.brand_id == brand.id).scalar()
        assert count_after_rollback == 2

    def test_database_performance_with_indexes(self, db_session):
        """Test database performance with proper indexing"""
        # Create test data
        brand = Brand(name="Performance Test Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Create many metric history entries
        start_time = time.time()
        
        for i in range(500):
            metric = MetricHistory(
                brand_id=brand.id,
                metric_name=f"test_metric_{i % 5}",  # 5 different metrics
                metric_value=i * 0.001,
                recorded_at=datetime.utcnow() - timedelta(days=i % 30)  # Last 30 days
            )
            db_session.add(metric)
        
        db_session.commit()
        creation_time = time.time() - start_time
        
        # Test indexed query performance
        start_time = time.time()
        
        # Query using indexed columns (brand_id, metric_name, recorded_at)
        recent_metrics = db_session.query(MetricHistory).filter(
            MetricHistory.brand_id == brand.id,
            MetricHistory.metric_name == "test_metric_0",
            MetricHistory.recorded_at >= datetime.utcnow() - timedelta(days=15)
        ).all()
        
        query_time = time.time() - start_time
        
        print(f"Created 500 metrics in {creation_time:.3f}s")
        print(f"Indexed query took {query_time:.3f}s, returned {len(recent_metrics)} results")
        
        # Performance assertions
        assert creation_time < 2.0, f"Data creation too slow: {creation_time:.3f}s"
        assert query_time < 0.1, f"Indexed query too slow: {query_time:.3f}s"
        assert len(recent_metrics) > 0

    def test_concurrent_database_operations(self, db_session):
        """Test concurrent database operations"""
        brand = Brand(name="Concurrent Test Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        results = []
        errors = []
        
        def create_analyses(thread_id):
            """Create analyses concurrently"""
            try:
                from database import SessionLocal
                local_session = SessionLocal()
                
                for i in range(5):
                    analysis = Analysis(
                        brand_id=brand.id,
                        status="completed",
                        metrics={"test_metric": thread_id + i * 0.1},
                        processing_time=thread_id * 10 + i
                    )
                    local_session.add(analysis)
                
                local_session.commit()
                local_session.close()
                results.append(f"thread_{thread_id}_success")
                
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")
        
        # Run concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_analyses, args=(i,))
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        assert len([r for r in results if "success" in r]) == 3
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Verify data was created correctly
        analysis_count = db_session.query(func.count(Analysis.id)).filter(Analysis.brand_id == brand.id).scalar()
        assert analysis_count == 15  # 3 threads × 5 analyses each

class TestRedisIntegration:
    """Test real Redis operations and caching"""

    def test_redis_basic_operations_real(self, redis_client):
        """Test basic Redis operations with real data"""
        # Test string operations
        test_key = f"test:basic:{int(time.time())}"
        test_value = {"brand": "TestBrand", "score": 0.75, "timestamp": datetime.now().isoformat()}
        
        # Set with expiration
        result = redis_client.setex(test_key, 60, json.dumps(test_value))
        assert result is True
        
        # Get and verify
        stored_value = redis_client.get(test_key)
        assert stored_value is not None
        parsed_value = json.loads(stored_value)
        assert parsed_value["brand"] == "TestBrand"
        assert parsed_value["score"] == 0.75
        
        # Test TTL
        ttl = redis_client.ttl(test_key)
        assert 55 <= ttl <= 60
        
        # Test delete
        redis_client.delete(test_key)
        assert redis_client.get(test_key) is None

    def test_redis_hash_operations_real(self, redis_client):
        """Test Redis hash operations with real brand data"""
        hash_key = f"brand:metrics:{int(time.time())}"
        
        # Set hash fields
        brand_metrics = {
            "attribution_rate": "0.65",
            "citation_count": "42",
            "last_analysis": datetime.now().isoformat(),
            "status": "active"
        }
        
        redis_client.hset(hash_key, mapping=brand_metrics)
        
        # Get individual fields
        attribution_rate = redis_client.hget(hash_key, "attribution_rate")
        assert float(attribution_rate) == 0.65
        
        # Get all fields
        all_data = redis_client.hgetall(hash_key)
        assert len(all_data) == 4
        assert all_data["status"] == "active"
        
        # Increment numeric field
        new_count = redis_client.hincrby(hash_key, "citation_count", 5)
        assert new_count == 47
        
        # Cleanup
        redis_client.delete(hash_key)

    def test_redis_list_operations_tracking(self, redis_client):
        """Test Redis list operations for tracking data"""
        list_key = f"tracking:events:{int(time.time())}"
        
        # Push tracking events
        events = [
            json.dumps({"event": "pageview", "timestamp": time.time(), "bot": "GPTBot"}),
            json.dumps({"event": "engagement", "timestamp": time.time() + 1, "bot": "GPTBot"}),
            json.dumps({"event": "pageview", "timestamp": time.time() + 2, "bot": "Claude-Web"})
        ]
        
        for event in events:
            redis_client.lpush(list_key, event)
        
        # Get list length
        length = redis_client.llen(list_key)
        assert length == 3
        
        # Get recent events
        recent_events = redis_client.lrange(list_key, 0, 1)  # Get 2 most recent
        assert len(recent_events) == 2
        
        # Pop oldest event
        oldest_event = redis_client.rpop(list_key)
        oldest_data = json.loads(oldest_event)
        assert oldest_data["bot"] == "GPTBot"
        
        # Cleanup
        redis_client.delete(list_key)

    def test_redis_sorted_set_operations(self, redis_client):
        """Test Redis sorted sets for time-series data"""
        zset_key = f"bot:visits:{int(time.time())}"
        
        # Add timestamped bot visits
        current_time = time.time()
        visits = [
            (current_time - 3600, json.dumps({"bot": "GPTBot", "path": "/home"})),
            (current_time - 1800, json.dumps({"bot": "Claude-Web", "path": "/products"})),
            (current_time - 900, json.dumps({"bot": "GPTBot", "path": "/about"})),
            (current_time, json.dumps({"bot": "Perplexity", "path": "/contact"}))
        ]
        
        for timestamp, visit_data in visits:
            redis_client.zadd(zset_key, {visit_data: timestamp})
        
        # Get visits from last hour
        one_hour_ago = current_time - 3600
        recent_visits = redis_client.zrangebyscore(zset_key, one_hour_ago, current_time)
        assert len(recent_visits) == 4
        
        # Get most recent visit
        latest_visit = redis_client.zrange(zset_key, -1, -1)
        latest_data = json.loads(latest_visit[0])
        assert latest_data["bot"] == "Perplexity"
        
        # Count visits in time range
        count = redis_client.zcount(zset_key, current_time - 1800, current_time)
        assert count == 3  # Last 30 minutes
        
        # Cleanup
        redis_client.delete(zset_key)

    def test_redis_performance_real_load(self, redis_client):
        """Test Redis performance with realistic load"""
        # Test write performance
        start_time = time.time()
        
        for i in range(1000):
            key = f"perf:test:{i}"
            value = json.dumps({
                "brand_id": f"brand_{i % 10}",
                "metric": "attribution_rate",
                "value": i * 0.001,
                "timestamp": time.time()
            })
            redis_client.setex(key, 300, value)  # 5 minute TTL
        
        write_time = time.time() - start_time
        
        # Test read performance
        start_time = time.time()
        
        for i in range(1000):
            key = f"perf:test:{i}"
            value = redis_client.get(key)
            assert value is not None
            data = json.loads(value)
            assert "brand_id" in data
        
        read_time = time.time() - start_time
        
        print(f"Redis performance: {write_time:.3f}s write, {read_time:.3f}s read for 1000 operations")
        
        # Performance assertions
        assert write_time < 2.0, f"Redis writes too slow: {write_time:.3f}s for 1000 ops"
        assert read_time < 1.0, f"Redis reads too slow: {read_time:.3f}s for 1000 ops"
        
        # Cleanup
        keys_to_delete = [f"perf:test:{i}" for i in range(1000)]
        if keys_to_delete:
            redis_client.delete(*keys_to_delete)

class TestRealTimeDataFlow:
    """Test real-time data flow between components"""

    def test_bot_visit_real_time_processing(self, redis_client, db_session):
        """Test real-time bot visit processing"""
        # Create brand
        brand = Brand(name="Real Time Bot Visit Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate real-time bot visit detection
        bot_visit_data = {
            "brand_id": str(brand.id),
            "bot_name": "GPTBot",
            "platform": "openai",
            "user_agent": "GPTBot/1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "path": "/real-time-test",
            "status_code": 200,
            "response_time": 0.45,
            "brand_mentioned": True
        }
        
        # Store in Redis with timestamp score for time-series
        visit_key = f"bot_visits:{brand.id}:live"
        redis_client.zadd(visit_key, {json.dumps(bot_visit_data): time.time()})
        
        # Increment counters
        counter_key = f"bot_counter:{brand.id}:daily"
        redis_client.hincrby(counter_key, "GPTBot", 1)
        redis_client.hincrby(counter_key, "total_visits", 1)
        redis_client.expire(counter_key, 86400)  # 24 hours
        
        # Process to database (simulating batch job)
        visit_entries = redis_client.zrange(visit_key, 0, -1)
        
        for visit_json in visit_entries:
            visit_data = json.loads(visit_json)
            
            bot_visit = BotVisit(
                brand_id=brand.id,
                bot_name=visit_data["bot_name"],
                platform=visit_data["platform"],
                user_agent=visit_data["user_agent"],
                timestamp=datetime.fromisoformat(visit_data["timestamp"]),
                ip_address=visit_data["ip_address"],
                path=visit_data["path"],
                status_code=visit_data["status_code"],
                response_time=visit_data["response_time"],
                brand_mentioned=visit_data["brand_mentioned"]
            )
            
            db_session.add(bot_visit)
        
        db_session.commit()
        
        # Verify real-time processing
        stored_visit = db_session.query(BotVisit).filter(BotVisit.brand_id == brand.id).first()
        assert stored_visit is not None
        assert stored_visit.bot_name == "GPTBot"
        assert stored_visit.brand_mentioned is True
        
        # Verify counters
        gpt_count = redis_client.hget(counter_key, "GPTBot")
        total_count = redis_client.hget(counter_key, "total_visits")
        assert int(gpt_count) == 1
        assert int(total_count) == 1

    def test_metrics_streaming_updates(self, redis_client, db_session):
        """Test streaming metrics updates"""
        # Create brand and analysis
        brand = Brand(name="Streaming Metrics Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        analysis = Analysis(brand_id=brand.id, status="processing")
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        # Simulate streaming metrics calculation
        metrics_stream_key = f"metrics_stream:{analysis.id}"
        
        # Stream individual metric calculations
        streaming_metrics = [
            {"metric": "chunk_retrieval_frequency", "value": 0.78, "timestamp": time.time()},
            {"metric": "embedding_relevance_score", "value": 0.84, "timestamp": time.time() + 1},
            {"metric": "attribution_rate", "value": 0.56, "timestamp": time.time() + 2},
            {"metric": "ai_citation_count", "value": 34, "timestamp": time.time() + 3}
        ]
        
        for metric_data in streaming_metrics:
            # Add to stream
            redis_client.xadd(metrics_stream_key, metric_data)
            
            # Update live dashboard data
            live_key = f"live_metrics:{brand.id}"
            redis_client.hset(live_key, metric_data["metric"], metric_data["value"])
            redis_client.expire(live_key, 3600)  # 1 hour
        
        # Read stream and process
        stream_data = redis_client.xrange(metrics_stream_key)
        final_metrics = {}
        
        for stream_id, fields in stream_data:
            metric_name = fields[b"metric"].decode()
            metric_value = float(fields[b"value"])
            final_metrics[metric_name] = metric_value
        
        # Update analysis with final metrics
        analysis.metrics = final_metrics
        analysis.status = "completed"
        db_session.commit()
        
        # Store in metrics history
        for metric_name, metric_value in final_metrics.items():
            metric_history = MetricHistory(
                brand_id=brand.id,
                analysis_id=analysis.id,
                metric_name=metric_name,
                metric_value=metric_value,
                data_source="real"
            )
            db_session.add(metric_history)
        
        db_session.commit()
        
        # Verify streaming pipeline
        updated_analysis = db_session.query(Analysis).filter(Analysis.id == analysis.id).first()
        assert updated_analysis.status == "completed"
        assert len(updated_analysis.metrics) == 4
        assert updated_analysis.metrics["attribution_rate"] == 0.56
        
        # Verify live dashboard data
        live_attribution = redis_client.hget(f"live_metrics:{brand.id}", "attribution_rate")
        assert float(live_attribution) == 0.56

class TestCrossComponentIntegration:
    """Test integration between different system components"""

    def test_optimization_engine_database_full_flow(self, optimization_engine, db_session):
        """Test complete flow from engine analysis to database storage"""
        # Create brand in database
        brand = Brand(name="Engine Integration Brand", website_url="https://engine-test.com")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Run real analysis
        async def run_analysis():
            return await optimization_engine.analyze_brand(
                brand_name=brand.name,
                content_sample="Engine integration test content with comprehensive analysis capabilities."
            )
        
        analysis_result = asyncio.run(run_analysis())
        
        # Store analysis results in database
        analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            analysis_type="comprehensive",
            data_source="real",
            metrics=analysis_result["optimization_metrics"],
            recommendations=analysis_result["priority_recommendations"],
            processing_time=analysis_result["analysis_duration"]
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        # Store individual metrics
        for metric_name, metric_value in analysis_result["optimization_metrics"].items():
            if isinstance(metric_value, (int, float)):
                metric_history = MetricHistory(
                    brand_id=brand.id,
                    analysis_id=analysis.id,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    data_source="real"
                )
                db_session.add(metric_history)
        
        db_session.commit()
        
        # Verify complete integration
        stored_analysis = db_session.query(Analysis).filter(Analysis.id == analysis.id).first()
        assert stored_analysis.status == "completed"
        assert "chunk_retrieval_frequency" in stored_analysis.metrics
        assert len(stored_analysis.recommendations) > 0
        
        # Verify metrics history
        metrics_count = db_session.query(func.count(MetricHistory.id)).filter(
            MetricHistory.analysis_id == analysis.id
        ).scalar()
        assert metrics_count >= 10  # Should have most of the 12 metrics

    def test_cache_database_synchronization(self, cache_client, db_session):
        """Test cache and database staying synchronized"""
        # Create brand in database
        brand = Brand(name="Cache Sync Test Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Cache brand data
        brand_cache_key = f"brand:{brand.id}:data"
        brand_data = {
            "id": str(brand.id),
            "name": brand.name,
            "created_at": brand.created_at.isoformat(),
            "cached_at": datetime.now().isoformat()
        }
        
        success = cache_client.set(brand_cache_key, brand_data, ttl=300)
        assert success is True
        
        # Verify cache contains correct data
        cached_data = cache_client.get(brand_cache_key)
        assert cached_data["name"] == brand.name
        assert cached_data["id"] == str(brand.id)
        
        # Update database
        original_name = brand.name
        brand.name = "Updated Cache Sync Brand"
        brand.updated_at = datetime.utcnow()
        db_session.commit()
        
        # Cache should still have old data (demonstrating cache invalidation need)
        cached_data_after_update = cache_client.get(brand_cache_key)
        assert cached_data_after_update["name"] == original_name  # Still old data
        
        # Simulate cache invalidation and refresh
        cache_client.delete(brand_cache_key)
        
        # Get fresh data from database and cache it
        fresh_brand = db_session.query(Brand).filter(Brand.id == brand.id).first()
        fresh_data = {
            "id": str(fresh_brand.id),
            "name": fresh_brand.name,
            "updated_at": fresh_brand.updated_at.isoformat(),
            "cached_at": datetime.now().isoformat()
        }
        
        cache_client.set(brand_cache_key, fresh_data, ttl=300)
        
        # Verify synchronization
        final_cached_data = cache_client.get(brand_cache_key)
        assert final_cached_data["name"] == "Updated Cache Sync Brand"

    def test_tracking_data_pipeline_real(self, redis_client, db_session):
        """Test real tracking data pipeline from Redis to database"""
        # Create brand
        brand = Brand(name="Tracking Pipeline Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate real-time tracking data collection in Redis
        session_id = str(uuid4())
        tracking_events = [
            {
                "brand_id": str(brand.id),
                "session_id": session_id,
                "event_type": "pageview",
                "bot_name": "GPTBot",
                "platform": "openai",
                "page_url": "https://tracking-pipeline.com/products",
                "page_title": "Products - Tracking Pipeline Brand",
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": "GPTBot/1.0"
            },
            {
                "brand_id": str(brand.id),
                "session_id": session_id,
                "event_type": "engagement",
                "time_on_page": 45,
                "scroll_depth": 75,
                "timestamp": (datetime.utcnow() + timedelta(seconds=45)).isoformat()
            }
        ]
        
        # Store in Redis (simulating real-time collection)
        for event in tracking_events:
            redis_client.lpush(f"tracking_queue:{brand.id}", json.dumps(event))
        
        # Process from Redis to database (simulating batch processing)
        processed_events = []
        
        while True:
            event_json = redis_client.rpop(f"tracking_queue:{brand.id}")
            if not event_json:
                break
            
            event_data = json.loads(event_json)
            
            tracking_event = TrackingEvent(
                brand_id=brand.id,
                event_type=event_data["event_type"],
                session_id=event_data["session_id"],
                bot_name=event_data.get("bot_name"),
                platform=event_data.get("platform"),
                page_url=event_data.get("page_url"),
                page_title=event_data.get("page_title"),
                time_on_page=event_data.get("time_on_page"),
                scroll_depth=event_data.get("scroll_depth"),
                timestamp=datetime.fromisoformat(event_data["timestamp"])
            )
            
            db_session.add(tracking_event)
            processed_events.append(tracking_event)
        
        db_session.commit()
        
        # Verify complete pipeline
        stored_events = db_session.query(TrackingEvent).filter(
            TrackingEvent.brand_id == brand.id
        ).order_by(TrackingEvent.timestamp).all()
        
        assert len(stored_events) == 2
        assert stored_events[0].event_type == "pageview"
        assert stored_events[0].bot_name == "GPTBot"
        assert stored_events[1].event_type == "engagement"
        assert stored_events[1].time_on_page == 45

    def test_concurrent_cross_component_operations(self, db_session, redis_client, cache_client):
        """Test concurrent operations across database, Redis, and cache"""
        brand = Brand(name="Concurrent Cross Component Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        results = []
        errors = []
        
        def database_operations():
            """Perform database operations"""
            try:
                from database import SessionLocal
                local_session = SessionLocal()
                
                for i in range(3):
                    analysis = Analysis(
                        brand_id=brand.id,
                        status="completed",
                        metrics={"test_metric": i * 0.1}
                    )
                    local_session.add(analysis)
                
                local_session.commit()
                local_session.close()
                results.append("db_success")
            except Exception as e:
                errors.append(f"db_error: {e}")
        
        def redis_operations():
            """Perform Redis operations"""
            try:
                for i in range(3):
                    key = f"redis_test:{brand.id}:{i}"
                    value = json.dumps({"operation": "redis_test", "index": i})
                    redis_client.setex(key, 60, value)
                results.append("redis_success")
            except Exception as e:
                errors.append(f"redis_error: {e}")
        
        def cache_operations():
            """Perform cache operations"""
            try:
                for i in range(3):
                    key = f"cache_test_{brand.id}_{i}"
                    data = {"operation": "cache_test", "index": i}
                    cache_client.set(key, data, ttl=60)
                results.append("cache_success")
            except Exception as e:
                errors.append(f"cache_error: {e}")
        
        # Run operations concurrently
        threads = [
            threading.Thread(target=database_operations),
            threading.Thread(target=redis_operations),
            threading.Thread(target=cache_operations)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        assert "db_success" in results
        assert "redis_success" in results  
        assert "cache_success" in results
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Verify data was created correctly
        analysis_count = db_session.query(func.count(Analysis.id)).filter(Analysis.brand_id == brand.id).scalar()
        assert analysis_count == 3
        
        # Verify Redis data
        redis_keys = redis_client.keys(f"redis_test:{brand.id}:*")
        assert len(redis_keys) == 3
        
        # Verify cache data
        for i in range(3):
            cached_data = cache_client.get(f"cache_test_{brand.id}_{i}")
            assert cached_data is not None
            assert cached_data["index"] == i

class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""

    def test_database_rollback_on_redis_failure(self, db_session, redis_client):
        """Test database rollback when Redis operations fail"""
        # Create brand
        brand = Brand(name="Error Handling Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate transaction that should rollback
        try:
            # Start database transaction
            analysis = Analysis(
                brand_id=brand.id,
                status="completed",
                metrics={"test_metric": 0.75}
            )
            db_session.add(analysis)
            
            # Simulate Redis operation failure
            # (In real scenario, this might be connection failure)
            invalid_key = "x" * 1000000  # Extremely long key that might fail
            
            try:
                redis_client.set(invalid_key, "test_value")
                # If Redis operation succeeds, commit database
                db_session.commit()
                operation_success = True
            except Exception as redis_error:
                # If Redis fails, rollback database
                db_session.rollback()
                operation_success = False
                print(f"Redis operation failed (expected): {redis_error}")
        
        except Exception as db_error:
            db_session.rollback()
            operation_success = False
            print(f"Database operation failed: {db_error}")
        
        # Verify state consistency
        analysis_count = db_session.query(func.count(Analysis.id)).filter(Analysis.brand_id == brand.id).scalar()
        
        if operation_success:
            assert analysis_count == 1  # Should be committed
        else:
            assert analysis_count == 0  # Should be rolled back

    def test_partial_failure_recovery(self, db_session, redis_client, cache_client):
        """Test recovery from partial system failures"""
        brand = Brand(name="Partial Failure Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate partial failure scenario
        operations_status = {"db": False, "redis": False, "cache": False}
        
        # Database operation
        try:
            analysis = Analysis(brand_id=brand.id, status="completed")
            db_session.add(analysis)
            db_session.commit()
            operations_status["db"] = True
        except Exception as e:
            print(f"Database operation failed: {e}")
        
        # Redis operation (might fail)
        try:
            redis_client.set(f"recovery_test:{brand.id}", "test_data", ex=60)
            operations_status["redis"] = True
        except Exception as e:
            print(f"Redis operation failed: {e}")
        
        # Cache operation (should work if Redis works)
        try:
            cache_client.set(f"cache_recovery_{brand.id}", {"test": "data"}, ttl=60)
            operations_status["cache"] = True
        except Exception as e:
            print(f"Cache operation failed: {e}")
        
        # System should handle partial failures gracefully
        # At minimum, database should work
        assert operations_status["db"] is True
        
        # Verify data consistency
        stored_analysis = db_session.query(Analysis).filter(Analysis.brand_id == brand.id).first()
        assert stored_analysis is not None
        assert stored_analysis.status == "completed"

    def test_data_consistency_validation(self, db_session, redis_client):
        """Test data consistency validation across components"""
        brand = Brand(name="Consistency Validation Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Create data in both systems
        analysis_data = {
            "brand_id": str(brand.id),
            "status": "completed",
            "metrics": {"attribution_rate": 0.67, "citation_count": 28},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in database
        analysis = Analysis(
            brand_id=brand.id,
            status=analysis_data["status"],
            metrics=analysis_data["metrics"]
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        # Store in Redis
        redis_key = f"analysis_cache:{analysis.id}"
        redis_client.setex(redis_key, 3600, json.dumps(analysis_data))
        
        # Validation function
        def validate_consistency():
            """Validate data consistency between database and Redis"""
            # Get from database
            db_analysis = db_session.query(Analysis).filter(Analysis.id == analysis.id).first()
            
            # Get from Redis
            redis_data = redis_client.get(redis_key)
            
            if redis_data:
                redis_analysis = json.loads(redis_data)
                
                # Compare critical fields
                consistency_checks = [
                    db_analysis.status == redis_analysis["status"],
                    db_analysis.metrics["attribution_rate"] == redis_analysis["metrics"]["attribution_rate"],
                    str(db_analysis.brand_id) == redis_analysis["brand_id"]
                ]
                
                return all(consistency_checks)
            
            return False
        
        # Validate initial consistency
        assert validate_consistency() is True
        
        # Simulate inconsistency
        analysis.metrics["attribution_rate"] = 0.85
        db_session.commit()
        
        # Should now be inconsistent
        assert validate_consistency() is False
        
        # Restore consistency
        updated_data = analysis_data.copy()
        updated_data["metrics"]["attribution_rate"] = 0.85
        redis_client.setex(redis_key, 3600, json.dumps(updated_data))
        
        # Should be consistent again
        assert validate_consistency() is True

class TestPerformanceIntegration:
    """Test performance across integrated components"""

    def test_end_to_end_performance(self, optimization_engine, db_session, redis_client):
        """Test end-to-end performance from analysis to storage"""
        # Create brand
        brand = Brand(name="E2E Performance Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Measure complete pipeline performance
        start_time = time.time()
        
        # 1. Run analysis
        async def run_analysis():
            return await optimization_engine.analyze_brand(
                brand_name=brand.name,
                content_sample="End-to-end performance testing content for comprehensive analysis."
            )
        
        analysis_result = asyncio.run(run_analysis())
        analysis_time = time.time() - start_time
        
        # 2. Store in database
        db_start = time.time()
        analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            metrics=analysis_result["optimization_metrics"],
            processing_time=analysis_result["analysis_duration"]
        )
        db_session.add(analysis)
        db_session.commit()
        db_time = time.time() - db_start
        
        # 3. Cache results
        cache_start = time.time()
        cache_key = f"analysis_result:{analysis.id}"
        redis_client.setex(cache_key, 3600, json.dumps(analysis_result["optimization_metrics"], default=str))
        cache_time = time.time() - cache_start
        
        total_time = time.time() - start_time
        
        print(f"E2E Performance:")
        print(f"  Analysis: {analysis_time:.3f}s")
        print(f"  Database: {db_time:.3f}s")
        print(f"  Cache: {cache_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
        
        # Performance assertions
        assert analysis_time < 90.0, f"Analysis too slow: {analysis_time:.3f}s"
        assert db_time < 1.0, f"Database storage too slow: {db_time:.3f}s"
        assert cache_time < 0.1, f"Cache storage too slow: {cache_time:.3f}s"
        assert total_time < 95.0, f"Total pipeline too slow: {total_time:.3f}s"

    def test_concurrent_integration_performance(self, db_session, redis_client, cache_client):
        """Test performance under concurrent integrated operations"""
        brand = Brand(name="Concurrent Integration Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        def integrated_operation(operation_id):
            """Perform integrated operation across all components"""
            start_time = time.time()
            
            try:
                # Database operation
                from database import SessionLocal
                local_session = SessionLocal()
                
                analysis = Analysis(
                    brand_id=brand.id,
                    status="completed",
                    metrics={"test_metric": operation_id * 0.1}
                )
                local_session.add(analysis)
                local_session.commit()
                local_session.refresh(analysis)
                
                # Redis operation
                redis_key = f"concurrent_test:{analysis.id}"
                redis_client.setex(redis_key, 300, json.dumps({"analysis_id": str(analysis.id)}))
                
                # Cache operation
                cache_key = f"cache_concurrent_{analysis.id}"
                cache_client.set(cache_key, {"operation_id": operation_id}, ttl=300)
                
                local_session.close()
                
                return {
                    "operation_id": operation_id,
                    "duration": time.time() - start_time,
                    "success": True
                }
                
            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Run concurrent integrated operations
        start_time = time.time()
        
        with threading.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(integrated_operation, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_ops = [r for r in results if r["success"]]
        failed_ops = [r for r in results if not r["success"]]
        
        avg_operation_time = sum(r["duration"] for r in successful_ops) / len(successful_ops) if successful_ops else 0
        
        print(f"Concurrent Integration Performance:")
        print(f"  Total operations: {len(results)}")
        print(f"  Successful: {len(successful_ops)}")
        print(f"  Failed: {len(failed_ops)}")
        print(f"  Average operation time: {avg_operation_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        
        # Performance assertions
        assert len(successful_ops) >= 8, f"Too many failed operations: {len(failed_ops)}"
        assert avg_operation_time < 2.0, f"Average operation too slow: {avg_operation_time:.3f}s"
        assert total_time < 10.0, f"Total concurrent operations too slow: {total_time:.3f}s"

class TestSystemResilience:
    """Test system resilience and recovery capabilities"""

    def test_redis_failover_handling(self, db_session, redis_client):
        """Test system behavior when Redis becomes unavailable"""
        brand = Brand(name="Redis Failover Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Normal operation
        analysis = Analysis(brand_id=brand.id, status="completed", metrics={"test": 0.5})
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        # Try to cache (should work)
        try:
            cache_key = f"failover_test:{analysis.id}"
            redis_client.setex(cache_key, 60, json.dumps({"status": "cached"}))
            redis_available = True
        except:
            redis_available = False
        
        # System should continue working even if Redis fails
        analysis2 = Analysis(brand_id=brand.id, status="completed", metrics={"test": 0.7})
        db_session.add(analysis2)
        db_session.commit()
        
        # Database operations should succeed regardless of Redis state
        stored_analyses = db_session.query(Analysis).filter(Analysis.brand_id == brand.id).all()
        assert len(stored_analyses) == 2

    def test_database_connection_recovery(self, db_session):
        """Test recovery from database connection issues"""
        # Create initial data
        brand = Brand(name="DB Recovery Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate connection recovery by creating new session
        def test_recovery():
            try:
                from database import SessionLocal
                recovery_session = SessionLocal()
                
                # Should be able to read existing data
                recovered_brand = recovery_session.query(Brand).filter(Brand.id == brand.id).first()
                assert recovered_brand is not None
                assert recovered_brand.name == "DB Recovery Brand"
                
                # Should be able to create new data
                new_analysis = Analysis(brand_id=brand.id, status="recovery_test")
                recovery_session.add(new_analysis)
                recovery_session.commit()
                
                recovery_session.close()
                return True
            except Exception as e:
                print(f"Recovery failed: {e}")
                return False
        
        assert test_recovery() is True

    def test_graceful_degradation(self, db_session, redis_client):
        """Test graceful degradation when components fail"""
        brand = Brand(name="Graceful Degradation Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Component availability status
        components = {
            "database": True,
            "redis": True,
            "cache": True
        }
        
        # Test database operation
        try:
            analysis = Analysis(brand_id=brand.id, status="degradation_test")
            db_session.add(analysis)
            db_session.commit()
        except:
            components["database"] = False
        
        # Test Redis operation
        try:
            redis_client.set("degradation_test", "value", ex=60)
        except:
            components["redis"] = False
        
        # System should work with at least database available
        assert components["database"] is True
        
        # Even if Redis fails, core functionality should work
        analysis2 = Analysis(brand_id=brand.id, status="still_working")
        db_session.add(analysis2)
        db_session.commit()
        
        stored_analyses = db_session.query(Analysis).filter(Analysis.brand_id == brand.id).all()
        assert len(stored_analyses) >= 2

class TestDataMigrationIntegration:
    """Test data migration and versioning scenarios"""

    def test_schema_migration_compatibility(self, db_session):
        """Test backward compatibility during schema migrations"""
        # Create data with current schema
        brand = Brand(
            name="Migration Test Brand",
            website_url="https://migration-test.com",
            industry="technology"
        )
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Create analysis with current metrics format
        current_metrics = {
            "chunk_retrieval_frequency": 0.78,
            "embedding_relevance_score": 0.84,
            "attribution_rate": 0.45
        }
        
        analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            metrics=current_metrics,
            analysis_type="comprehensive"
        )
        db_session.add(analysis)
        db_session.commit()
        
        # Verify data can be read and processed
        stored_analysis = db_session.query(Analysis).filter(Analysis.id == analysis.id).first()
        assert stored_analysis.metrics["attribution_rate"] == 0.45
        assert stored_analysis.analysis_type == "comprehensive"

    def test_data_format_evolution(self, db_session):
        """Test handling of evolving data formats"""
        brand = Brand(name="Data Evolution Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Old format data
        old_metrics = {
            "overall_score": 0.75,
            "legacy_metric": 0.6
        }
        
        old_analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            metrics=old_metrics
        )
        db_session.add(old_analysis)
        
        # New format data
        new_metrics = {
            "chunk_retrieval_frequency": 0.78,
            "embedding_relevance_score": 0.84,
            "attribution_rate": 0.45,
            "ai_citation_count": 23
        }
        
        new_analysis = Analysis(
            brand_id=brand.id,
            status="completed",
            metrics=new_metrics
        )
        db_session.add(new_analysis)
        db_session.commit()
        
        # System should handle both formats
        all_analyses = db_session.query(Analysis).filter(Analysis.brand_id == brand.id).all()
        assert len(all_analyses) == 2
        
        # Should be able to process both old and new formats
        for analysis in all_analyses:
            assert "metrics" in analysis.__dict__
            assert isinstance(analysis.metrics, dict)

class TestMonitoringIntegration:
    """Test monitoring and observability integration"""

    def test_performance_metrics_collection(self, db_session, redis_client):
        """Test collection of performance metrics"""
        brand = Brand(name="Monitoring Test Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Simulate operations and collect metrics
        operation_times = []
        
        for i in range(5):
            start_time = time.time()
            
            # Database operation
            analysis = Analysis(
                brand_id=brand.id,
                status="completed",
                metrics={"test_metric": i * 0.1}
            )
            db_session.add(analysis)
            db_session.commit()
            
            # Redis operation
            redis_client.set(f"monitoring_test_{i}", json.dumps({"value": i}), ex=60)
            
            operation_time = time.time() - start_time
            operation_times.append(operation_time)
        
        # Analyze performance metrics
        avg_time = sum(operation_times) / len(operation_times)
        max_time = max(operation_times)
        min_time = min(operation_times)
        
        print(f"Operation performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        
        # Performance should be reasonable
        assert avg_time < 1.0, f"Average operation time too slow: {avg_time:.3f}s"
        assert max_time < 2.0, f"Slowest operation too slow: {max_time:.3f}s"

    def test_error_tracking_integration(self, db_session):
        """Test error tracking and reporting"""
        brand = Brand(name="Error Tracking Brand")
        db_session.add(brand)
        db_session.commit()
        db_session.refresh(brand)
        
        # Track successful operations
        successful_ops = 0
        failed_ops = 0
        
        # Mix of successful and failed operations
        test_operations = [
            {"valid": True, "data": {"status": "completed"}},
            {"valid": False, "data": {"status": "invalid_status"}},
            {"valid": True, "data": {"status": "pending"}},
            {"valid": False, "data": None},  # Invalid data
            {"valid": True, "data": {"status": "processing"}}
        ]
        
        for op in test_operations:
            try:
                if not op["valid"] or not op["data"]:
                    raise ValueError("Invalid operation data")
                
                analysis = Analysis(
                    brand_id=brand.id,
                    status=op["data"]["status"]
                )
                db_session.add(analysis)
                db_session.commit()
                successful_ops += 1
                
            except Exception as e:
                db_session.rollback()
                failed_ops += 1
                print(f"Operation failed as expected: {e}")
        
        # Verify error tracking
        success_rate = successful_ops / len(test_operations)
        print(f"Success rate: {success_rate:.2%} ({successful_ops}/{len(test_operations)})")
        
        assert successful_ops == 3  # 3 valid operations
        assert failed_ops == 2  # 2 invalid operations
        assert success_rate == 0.6  # 60% success rate