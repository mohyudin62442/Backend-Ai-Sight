"""
Updated API Models with Pydantic v2 Compatibility
Fixes all 422 validation errors from test results
"""

from pydantic import BaseModel, Field, field_validator, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid

class BrandAnalysisRequest(BaseModel):
    brand_name: str = Field(min_length=2, max_length=50)
    website_url: Optional[str] = None
    product_categories: List[str] = Field(min_length=1, max_length=10)
    content_sample: Optional[str] = Field(default="", max_length=10000)
    competitor_names: Optional[List[str]] = Field(default=[], max_length=5)

    @field_validator('brand_name')
    @classmethod
    def validate_brand_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Brand name must be at least 2 characters')
        return v.strip()

    @field_validator('product_categories')
    @classmethod
    def validate_categories(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one product category is required')
        return [cat.strip() for cat in v if cat.strip()]

    @field_validator('website_url')
    @classmethod
    def validate_url(cls, v):
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            return f'https://{v}'
        return v

class OptimizationMetricsRequest(BaseModel):
    brand_name: str = Field(min_length=2, max_length=50)
    content_sample: Optional[str] = Field(default="", max_length=10000)

    @field_validator('brand_name')
    @classmethod
    def validate_brand_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Brand name must be at least 2 characters')
        return v.strip()

class QueryAnalysisRequest(BaseModel):
    brand_name: str = Field(min_length=2, max_length=50)
    product_categories: Optional[List[str]] = Field(default=[], max_length=10)

    @field_validator('brand_name')
    @classmethod
    def validate_brand_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Brand name must be at least 2 characters')
        return v.strip()

    @field_validator('product_categories')
    @classmethod
    def validate_categories(cls, v):
        if v is None:
            return []
        return [cat.strip() for cat in v if cat.strip()]

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "2.0.0"
    database: str = "connected"
    redis: str = "connected"
    services: Dict[str, bool] = Field(default_factory=dict)

class StandardResponse(BaseModel):
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]

    def model_dump(self, **kwargs):
        """Pydantic v2 compatible method"""
        return {
            "success": self.success,
            "error": self.error
        }

# Core data models for the optimization engine
class OptimizationMetrics:
    """Core metrics class - matches FRD requirements exactly"""
    
    def __init__(self):
        # Initialize all 12 FRD metrics to 0.0
        self.chunk_retrieval_frequency: float = 0.0
        self.embedding_relevance_score: float = 0.0
        self.attribution_rate: float = 0.0
        self.ai_citation_count: int = 0
        self.vector_index_presence_rate: float = 0.0
        self.retrieval_confidence_score: float = 0.0
        self.rrf_rank_contribution: float = 0.0
        self.llm_answer_coverage: float = 0.0
        self.ai_model_crawl_success_rate: float = 0.0
        self.semantic_density_score: float = 0.0
        self.zero_click_surface_presence: float = 0.0
        self.machine_validated_authority: float = 0.0

    def get_overall_score(self) -> float:
        """Calculate overall score from all metrics"""
        numeric_metrics = [
            self.chunk_retrieval_frequency,
            self.embedding_relevance_score,
            self.attribution_rate,
            self.vector_index_presence_rate,
            self.retrieval_confidence_score,
            self.rrf_rank_contribution,
            self.llm_answer_coverage,
            self.ai_model_crawl_success_rate,
            self.semantic_density_score,
            self.zero_click_surface_presence,
            self.machine_validated_authority
        ]
        
        # Normalize citation count (target: 40 citations per 100 queries)
        normalized_citation = min(1.0, self.ai_citation_count / 40.0)
        numeric_metrics.append(normalized_citation)
        
        return sum(numeric_metrics) / len(numeric_metrics)

    def get_performance_grade(self) -> str:
        """Get letter grade based on overall score"""
        score = self.get_overall_score()
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert metrics to dictionary for API responses"""
        return {
            "chunk_retrieval_frequency": self.chunk_retrieval_frequency,
            "embedding_relevance_score": self.embedding_relevance_score,
            "attribution_rate": self.attribution_rate,
            "ai_citation_count": self.ai_citation_count,
            "vector_index_presence_rate": self.vector_index_presence_rate,
            "retrieval_confidence_score": self.retrieval_confidence_score,
            "rrf_rank_contribution": self.rrf_rank_contribution,
            "llm_answer_coverage": self.llm_answer_coverage,
            "ai_model_crawl_success_rate": self.ai_model_crawl_success_rate,
            "semantic_density_score": self.semantic_density_score,
            "zero_click_surface_presence": self.zero_click_surface_presence,
            "machine_validated_authority": self.machine_validated_authority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, int]]) -> 'OptimizationMetrics':
        """Create metrics object from dictionary"""
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics

class ContentChunk:
    """Content chunk for processing"""
    
    def __init__(self, text: str, word_count: int, embedding=None):
        self.text = text
        self.word_count = word_count
        self.embedding = embedding
        self.keywords = None
        self.has_structure = False
        self.confidence_score = 0.0
        self.semantic_tags = None

class BotVisitData(BaseModel):
    """Model for bot visit tracking"""
    platform: str
    bot_name: str
    page_url: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: str
    response_code: int
    page_title: Optional[str] = None
    content_length: Optional[int] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TrackingMetrics(BaseModel):
    """Real-time tracking metrics"""
    total_visits: int
    unique_bots: int
    platform_breakdown: Dict[str, int]
    success_rate: float
    avg_response_time: float
    top_pages: List[Dict[str, Any]]
    
class User(BaseModel):
    """User model for testing"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    full_name: str
    plan: str = "free"
    
    class Config:
        from_attributes = True