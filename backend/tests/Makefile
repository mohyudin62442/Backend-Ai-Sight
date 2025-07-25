# Makefile for AI Optimization Engine Test Automation
# Comprehensive test automation and CI/CD integration

.PHONY: help test test-unit test-integration test-performance test-load test-all
.PHONY: test-fast test-slow test-coverage test-watch test-debug
.PHONY: setup setup-dev setup-test clean clean-cache clean-logs
.PHONY: lint format check security install deps-dev
.PHONY: services services-start services-stop services-logs
.PHONY: database-setup database-reset cache-setup cache-clear
.PHONY: reports benchmark profile coverage-html

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Directories
TEST_DIR := tests
SRC_DIR := .
LOG_DIR := tests/logs
REPORT_DIR := reports
COVERAGE_DIR := htmlcov

# Test database configuration
TEST_DB_NAME := aioptimization_test
TEST_DB_USER := aioptimization
TEST_DB_PASS := aioptimization
TEST_REDIS_DB := 1

# Default target
help: ## Show this help message
	@echo "AI Optimization Engine - Test Automation"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / { printf "  %-20s %s\n", $1, $2 }' $(MAKEFILE_LIST)

# =============================================================================
# SETUP AND INSTALLATION
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

deps-dev: ## Install development dependencies including test tools
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-timeout pytest-mock pytest-xdist
	$(PIP) install pytest-benchmark pytest-html pytest-json-report pytest-order
	$(PIP) install black flake8 mypy bandit safety pre-commit
	$(PIP) install psutil redis psycopg2-binary

setup: install deps-dev ## Complete setup for development
	@echo "Setting up development environment..."
	mkdir -p $(LOG_DIR) $(REPORT_DIR) $(COVERAGE_DIR)
	@echo "Creating .env file if it doesn't exist..."
	@if [ ! -f .env ]; then \
		echo "ANTHROPIC_API_KEY=your-anthropic-api-key-here" > .env; \
		echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env; \
		echo "TEST_DATABASE_URL=postgresql://$(TEST_DB_USER):$(TEST_DB_PASS)@localhost:5432/$(TEST_DB_NAME)" >> .env; \
		echo "TEST_REDIS_URL=redis://localhost:6379/$(TEST_REDIS_DB)" >> .env; \
		echo "Please update .env file with your actual API keys"; \
	fi

setup-dev: setup services-start database-setup ## Full development setup including services
	@echo "Development environment ready!"

setup-test: ## Setup test environment
	@echo "Setting up test environment..."
	mkdir -p $(LOG_DIR) $(REPORT_DIR) $(COVERAGE_DIR)
	@echo "Checking required environment variables..."
	@if [ -z "$ANTHROPIC_API_KEY" ]; then \
		echo "Warning: ANTHROPIC_API_KEY not set"; \
	fi

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================

services: services-start ## Start all required services

services-start: ## Start PostgreSQL and Redis services
	@echo "Starting services with Docker..."
	$(DOCKER) run -d --name aiopt-postgres \
		-e POSTGRES_USER=$(TEST_DB_USER) \
		-e POSTGRES_PASSWORD=$(TEST_DB_PASS) \
		-e POSTGRES_DB=postgres \
		-p 5432:5432 \
		postgres:15-alpine || echo "PostgreSQL already running"
	
	$(DOCKER) run -d --name aiopt-redis \
		-p 6379:6379 \
		redis:7-alpine || echo "Redis already running"
	
	@echo "Waiting for services to be ready..."
	@sleep 5

services-stop: ## Stop all services
	@echo "Stopping services..."
	$(DOCKER) stop aiopt-postgres aiopt-redis || true
	$(DOCKER) rm aiopt-postgres aiopt-redis || true

services-logs: ## Show service logs
	@echo "PostgreSQL logs:"
	$(DOCKER) logs aiopt-postgres --tail 20
	@echo "Redis logs:"
	$(DOCKER) logs aiopt-redis --tail 20

services-status: ## Check service status
	@echo "Service Status:"
	@$(DOCKER) ps --filter "name=aiopt-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================

database-setup: ## Setup test database
	@echo "Setting up test database..."
	$(DOCKER) exec aiopt-postgres psql -U $(TEST_DB_USER) -c "CREATE DATABASE $(TEST_DB_NAME);" || echo "Database already exists"
	@echo "Running database migrations..."
	$(PYTHON) -c "from database import Base, engine; Base.metadata.create_all(bind=engine)" || echo "Migration completed"

database-reset: ## Reset test database
	@echo "Resetting test database..."
	$(DOCKER) exec aiopt-postgres psql -U $(TEST_DB_USER) -c "DROP DATABASE IF EXISTS $(TEST_DB_NAME);"
	$(DOCKER) exec aiopt-postgres psql -U $(TEST_DB_USER) -c "CREATE DATABASE $(TEST_DB_NAME);"
	$(PYTHON) -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"

database-shell: ## Open database shell
	$(DOCKER) exec -it aiopt-postgres psql -U $(TEST_DB_USER) -d $(TEST_DB_NAME)

# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

cache-setup: ## Setup Redis cache
	@echo "Testing Redis connection..."
	$(DOCKER) exec aiopt-redis redis-cli ping

cache-clear: ## Clear all cache data
	@echo "Clearing Redis cache..."
	$(DOCKER) exec aiopt-redis redis-cli FLUSHALL

cache-shell: ## Open Redis shell
	$(DOCKER) exec -it aiopt-redis redis-cli

# =============================================================================
# TESTING
# =============================================================================

test: test-unit ## Run default test suite (unit tests)

test-all: clean-cache ## Run all tests including load tests
	@echo "Running complete test suite..."
	$(PYTEST) $(TEST_DIR) \
		--maxfail=5 \
		--tb=short \
		--junit-xml=$(REPORT_DIR)/junit.xml \
		--html=$(REPORT_DIR)/report.html \
		--self-contained-html

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTEST) -m "unit or not (integration or performance or load)" \
		--maxfail=10 \
		--tb=short

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	$(PYTEST) -m "integration" \
		--maxfail=5 \
		--tb=short

test-performance: ## Run performance tests
	@echo "Running performance tests..."
	$(PYTEST) -m "performance" \
		--maxfail=3 \
		--tb=short \
		--timeout=600

test-load: ## Run load tests
	@echo "Running load tests..."
	$(PYTEST) -m "load" \
		--maxfail=1 \
		--tb=short \
		--timeout=1800

test-fast: ## Run only fast tests (< 5 seconds)
	@echo "Running fast tests..."
	$(PYTEST) -m "fast or not slow" \
		--maxfail=10

test-slow: ## Run slow tests (> 5 seconds)
	@echo "Running slow tests..."
	$(PYTEST) -m "slow" \
		--maxfail=3 \
		--timeout=300

test-api: ## Run API endpoint tests
	@echo "Running API tests..."
	$(PYTEST) $(TEST_DIR)/test_api.py -v

test-engine: ## Run optimization engine tests
	@echo "Running engine tests..."
	$(PYTEST) $(TEST_DIR)/test_optimization_engine.py -v

test-db: ## Run database tests
	@echo "Running database tests..."
	$(PYTEST) -m "database" -v

test-cache: ## Run cache tests
	@echo "Running cache tests..."
	$(PYTEST) -m "cache" -v

# =============================================================================
# TEST COVERAGE
# =============================================================================

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) \
		--cov=. \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-report=term-missing \
		--cov-report=xml:$(REPORT_DIR)/coverage.xml \
		--cov-fail-under=80

coverage-html: test-coverage ## Generate HTML coverage report
	@echo "Coverage report generated at $(COVERAGE_DIR)/index.html"
	@command -v open >/dev/null 2>&1 && open $(COVERAGE_DIR)/index.html || \
	 command -v xdg-open >/dev/null 2>&1 && xdg-open $(COVERAGE_DIR)/index.html || \
	 echo "Open $(COVERAGE_DIR)/index.html in your browser"

coverage-report: ## Show coverage report in terminal
	$(PYTEST) $(TEST_DIR) --cov=. --cov-report=term-missing --quiet

# =============================================================================
# DEVELOPMENT TOOLS
# =============================================================================

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "Running tests in watch mode..."
	@command -v ptw >/dev/null 2>&1 || $(PIP) install pytest-watch
	ptw $(TEST_DIR) -- --tb=short

test-debug: ## Run tests with debugging enabled
	@echo "Running tests with debugging..."
	$(PYTEST) $(TEST_DIR) -v -s --tb=long --pdb

test-profile: ## Run tests with profiling
	@echo "Running tests with profiling..."
	$(PYTEST) $(TEST_DIR) --profile --profile-svg

benchmark: ## Run benchmark tests
	@echo "Running benchmark tests..."
	$(PYTEST) $(TEST_DIR) -m "benchmark" \
		--benchmark-only \
		--benchmark-sort=mean \
		--benchmark-html=$(REPORT_DIR)/benchmark.html

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linting (flake8)
	@echo "Running linting..."
	flake8 $(SRC_DIR) --exclude=venv,tests --max-line-length=120

format: ## Format code (black)
	@echo "Formatting code..."
	black $(SRC_DIR) --exclude=venv

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	black $(SRC_DIR) --exclude=venv --check

type-check: ## Run type checking (mypy)
	@echo "Running type checking..."
	mypy $(SRC_DIR) --exclude=venv --ignore-missing-imports

security: ## Run security checks (bandit)
	@echo "Running security checks..."
	bandit -r $(SRC_DIR) -x tests,venv

safety-check: ## Check for known security vulnerabilities
	@echo "Checking for known vulnerabilities..."
	safety check

check: lint format-check type-check security ## Run all code quality checks

# =============================================================================
# CONTINUOUS INTEGRATION
# =============================================================================

ci-test: ## Run CI test suite
	@echo "Running CI test suite..."
	$(PYTEST) $(TEST_DIR) \
		--maxfail=1 \
		--tb=short \
		--cov=. \
		--cov-report=xml:$(REPORT_DIR)/coverage.xml \
		--cov-report=term \
		--junit-xml=$(REPORT_DIR)/junit.xml \
		--timeout=300

ci-setup: ## Setup CI environment
	@echo "Setting up CI environment..."
	mkdir -p $(LOG_DIR) $(REPORT_DIR) $(COVERAGE_DIR)
	$(PIP) install -r requirements.txt
	$(MAKE) deps-dev

ci-test-fast: ## Run fast CI tests only
	@echo "Running fast CI tests..."
	$(PYTEST) -m "fast or not slow" \
		--maxfail=5 \
		--tb=line \
		--quiet

# =============================================================================
# REPORTING
# =============================================================================

reports: ## Generate all test reports
	@echo "Generating test reports..."
	$(PYTEST) $(TEST_DIR) \
		--html=$(REPORT_DIR)/report.html \
		--self-contained-html \
		--junit-xml=$(REPORT_DIR)/junit.xml \
		--json-report --json-report-file=$(REPORT_DIR)/report.json

report-summary: ## Show test summary
	@echo "Test Summary:"
	@if [ -f $(REPORT_DIR)/junit.xml ]; then \
		python3 -c "import xml.etree.ElementTree as ET; \
		tree = ET.parse('$(REPORT_DIR)/junit.xml'); \
		root = tree.getroot(); \
		print(f'Tests: {root.get(\"tests\")}'); \
		print(f'Failures: {root.get(\"failures\")}'); \
		print(f'Errors: {root.get(\"errors\")}'); \
		print(f'Time: {root.get(\"time\")}s')"; \
	else \
		echo "No test results found. Run 'make test' first."; \
	fi

# =============================================================================
# CLEANUP
# =============================================================================

clean: clean-cache clean-logs ## Clean all generated files

clean-cache: ## Clean cache and temporary files
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

clean-logs: ## Clean log files
	@echo "Cleaning log files..."
	rm -rf $(LOG_DIR)/*.log
	mkdir -p $(LOG_DIR)

clean-reports: ## Clean test reports
	@echo "Cleaning test reports..."
	rm -rf $(REPORT_DIR)/*
	rm -rf $(COVERAGE_DIR)/*
	mkdir -p $(REPORT_DIR) $(COVERAGE_DIR)

clean-all: clean clean-reports services-stop ## Clean everything and stop services

# =============================================================================
# DOCKER TESTING
# =============================================================================

docker-test: ## Run tests in Docker container
	@echo "Running tests in Docker..."
	$(DOCKER) build -t ai-optimization-test -f Dockerfile.test .
	$(DOCKER) run --rm \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-v $(PWD)/$(REPORT_DIR):/app/$(REPORT_DIR) \
		ai-optimization-test

docker-test-build: ## Build Docker test image
	$(DOCKER) build -t ai-optimization-test -f Dockerfile.test .

# =============================================================================
# LOAD TESTING SCENARIOS
# =============================================================================

load-test-light: ## Run light load test
	@echo "Running light load test..."
	$(PYTEST) -m "light_load" --tb=short -v

load-test-normal: ## Run normal load test
	@echo "Running normal load test..."
	$(PYTEST) -m "normal_load" --tb=short -v

load-test-heavy: ## Run heavy load test
	@echo "Running heavy load test..."
	$(PYTEST) -m "heavy_load" --tb=short -v --timeout=1800

stress-test: ## Run stress tests
	@echo "Running stress tests..."
	$(PYTEST) -m "stress" --tb=short -v --timeout=1800

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

perf-baseline: ## Establish performance baseline
	@echo "Establishing performance baseline..."
	$(PYTEST) $(TEST_DIR)/test_performance.py::TestPerformanceRequirements::test_baseline_performance_metrics \
		-v --tb=short

perf-compare: ## Compare current performance to baseline
	@echo "Comparing performance to baseline..."
	$(PYTEST) $(TEST_DIR)/test_performance.py::TestPerformanceRegression \
		-v --tb=short

monitor: ## Monitor test execution
	@echo "Monitoring test execution..."
	$(PYTEST) $(TEST_DIR) --tb=no --quiet | \
	while read line; do \
		echo "[$(date '+%H:%M:%S')] $line"; \
	done

# =============================================================================
# UTILITIES
# =============================================================================

env-check: ## Check environment setup
	@echo "Checking environment..."
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Pytest: $(shell $(PYTEST) --version)"
	@echo "Docker: $(shell $(DOCKER) --version)"
	@echo ""
	@echo "Environment variables:"
	@echo "ANTHROPIC_API_KEY: $(if $(ANTHROPIC_API_KEY),✓ Set,✗ Not set)"
	@echo "OPENAI_API_KEY: $(if $(OPENAI_API_KEY),✓ Set,✗ Not set)"
	@echo ""
	@echo "Services:"
	@$(DOCKER) ps --filter "name=aiopt-" --format "{{.Names}}: {{.Status}}" || echo "No services running"

test-info: ## Show test information
	@echo "Test Information:"
	@echo "Test directory: $(TEST_DIR)"
	@echo "Log directory: $(LOG_DIR)"
	@echo "Report directory: $(REPORT_DIR)"
	@echo "Coverage directory: $(COVERAGE_DIR)"
	@echo ""
	@echo "Available test markers:"
	$(PYTEST) --markers | grep -E "^@pytest.mark" | head -20

deps-list: ## List all dependencies
	@echo "Production dependencies:"
	@$(PIP) list --format=freeze | grep -E "^(anthropic|openai|fastapi|sqlalchemy|redis|sentence-transformers)"
	@echo ""
	@echo "Development dependencies:"
	@$(PIP) list --format=freeze | grep -E "^(pytest|black|flake8|mypy|bandit)"

# =============================================================================
# SPECIAL TARGETS
# =============================================================================

pre-commit: format lint test-fast ## Run pre-commit checks

pre-push: check test-coverage ## Run pre-push checks

release-test: clean-all setup-test ci-test ## Run release testing

quick-test: ## Quick test for development
	$(PYTEST) -x --tb=short -q

validate: env-check services-status test-fast ## Validate environment and run quick tests

# Make sure intermediate files are not deleted
.SECONDARY: