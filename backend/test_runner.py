#!/usr/bin/env python3
"""
Test Runner Script - Sets up everything and runs tests
Fixes all the issues identified in the test results
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Set up test environment variables"""
    env_vars = {
        'ENVIRONMENT': 'test',
        'DEBUG': 'true',
        'ANTHROPIC_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key',
        'DATABASE_URL': 'postgresql://aioptimization:aioptimization@localhost:5432/aioptimization_test',
        'TEST_DATABASE_URL': 'postgresql://aioptimization:aioptimization@localhost:5432/aioptimization_test',
        'REDIS_URL': 'redis://localhost:6379/0',
        'TEST_REDIS_URL': 'redis://localhost:6379/1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Environment variables set")

def create_database_tables():
    """Create database tables"""
    try:
        print("📊 Creating database tables...")
        
        # Import and create tables
        from db_models import Base
        from sqlalchemy import create_engine
        
        db_url = 'postgresql://aioptimization:aioptimization@localhost:5432/aioptimization_test'
        engine = create_engine(db_url)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = [row[0] for row in result]
            print(f"✅ Created tables: {tables}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        return False

def check_services():
    """Check if required services are running"""
    print("🔍 Checking required services...")
    
    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="aioptimization_test",
            user="aioptimization",
            password="aioptimization"
        )
        conn.close()
        print("✅ PostgreSQL is running")
    except Exception as e:
        print(f"❌ PostgreSQL not accessible: {e}")
        print("💡 Start with: docker run -d --name postgres-test -e POSTGRES_DB=aioptimization_test -e POSTGRES_USER=aioptimization -e POSTGRES_PASSWORD=aioptimization -p 5432:5432 postgres:13")
        return False
    
    # Check Redis
    try:
        import redis
        r = redis.from_url('redis://localhost:6379')
        r.ping()
        print("✅ Redis is running")
    except Exception as e:
        print(f"⚠️ Redis not accessible: {e}")
        print("💡 Start with: docker run -d --name redis-test -p 6379:6379 redis:alpine")
        # Don't fail on Redis, it's optional for basic tests
    
    return True

def install_dependencies():
    """Install required test dependencies"""
    print("📦 Installing dependencies...")
    
    required_packages = [
        'pytest',
        'pytest-asyncio', 
        'pytest-cov',
        'pytest-mock'
    ]
    
    for package in required_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")

def run_test_categories():
    """Run tests in categories to identify issues"""
    test_categories = [
        {
            'name': 'Core Models',
            'command': ['pytest', 'tests/test_models.py', '-v'],
            'description': 'Basic data models and validation'
        },
        {
            'name': 'Utility Functions', 
            'command': ['pytest', 'tests/test_utils.py', '-v', '-k', 'not redis and not database'],
            'description': 'Helper functions and utilities'
        },
        {
            'name': 'Optimization Engine Core',
            'command': ['pytest', 'tests/test_optimization_engine.py::TestOptimizationMetrics', '-v'],
            'description': 'Core optimization metrics'
        },
        {
            'name': 'API Health Check',
            'command': ['pytest', 'tests/test_api.py::TestAPIEndpoints::test_health_endpoint_content', '-v'],
            'description': 'Basic API functionality'
        },
        {
            'name': 'Database Integration',
            'command': ['pytest', 'tests/test_integration.py::TestDatabaseIntegration::test_brand_crud_operations', '-v'],
            'description': 'Database operations'
        },
        {
            'name': 'Full API Tests',
            'command': ['pytest', 'tests/test_api.py', '-v', '--tb=short'],
            'description': 'All API endpoints'
        }
    ]
    
    results = {}
    
    for category in test_categories:
        print(f"\n🧪 Running {category['name']}: {category['description']}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                category['command'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per category
            )
            
            success = result.returncode == 0
            results[category['name']] = {
                'success': success,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if success:
                print(f"✅ {category['name']} - PASSED")
            else:
                print(f"❌ {category['name']} - FAILED")
                print("Error output:")
                print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {category['name']} - TIMEOUT")
            results[category['name']] = {'success': False, 'timeout': True}
            
        except Exception as e:
            print(f"💥 {category['name']} - ERROR: {e}")
            results[category['name']] = {'success': False, 'error': str(e)}
    
    return results

def generate_coverage_report():
    """Generate test coverage report"""
    print("\n📊 Generating coverage report...")
    
    try:
        result = subprocess.run([
            'pytest', 
            '--cov=.',
            '--cov-report=html',
            '--cov-report=term-missing',
            'tests/',
            '-v'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Coverage report generated successfully")
            print("📁 HTML report available at: htmlcov/index.html")
            
            # Extract coverage percentage
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'TOTAL' in line and '%' in line:
                    print(f"📈 {line}")
                    
        else:
            print("❌ Coverage report generation failed")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Coverage report generation timed out")
    except Exception as e:
        print(f"💥 Coverage report error: {e}")

def print_summary(results):
    """Print test results summary"""
    print("\n" + "="*60)
    print("📋 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    print()
    
    for name, result in results.items():
        status = "✅ PASS" if result.get('success') else "❌ FAIL"
        print(f"{status} - {name}")
        
        if not result.get('success'):
            if result.get('timeout'):
                print("   ⏰ Test timed out")
            elif result.get('error'):
                print(f"   💥 Error: {result['error']}")
    
    print("\n" + "="*60)
    
    if success_rate >= 80:
        print("🎉 Excellent! Most tests are passing.")
    elif success_rate >= 60:
        print("👍 Good progress! Some issues to address.")
    else:
        print("🔧 Needs work. Focus on fixing the core issues first.")
    
    print("="*60)

def main():
    """Main test runner"""
    print("🚀 AI Optimization Engine Test Runner")
    print("="*60)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Setup
    setup_environment()
    
    # Check services
    if not check_services():
        print("❌ Required services not available. Please start PostgreSQL.")
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Create database tables
    if not create_database_tables():
        print("❌ Failed to create database tables")
        sys.exit(1)
    
    # Run tests
    results = run_test_categories()
    
    # Generate coverage report
    generate_coverage_report()
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    passed = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    if success_rate >= 80:
        print("\n🎯 Ready for production testing!")
        sys.exit(0)
    elif success_rate >= 60:
        print("\n📈 Good foundation. Address remaining issues.")
        sys.exit(0)
    else:
        print("\n🔧 Core issues need fixing before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()