"""
Routing Schemas - Production Ready
Pydantic models for prompt routing, analysis, decisions, and feedback.
Comprehensive schema definitions with validation, documentation, and type safety.
"""

from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from pydantic.json_schema import JsonSchemaValue

# ============================================================================
# ENUMS
# ============================================================================

class PromptType(str, Enum):
    """Prompt type classification."""
    CODE = "code"
    QA = "qa"
    CREATIVE = "creative"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Prompt complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class RoutingStrategy(str, Enum):
    """Routing strategy types."""
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    HYBRID = "hybrid"
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"
    EXPLICIT = "explicit"


class SentimentLabel(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# ============================================================================
# PROMPT ANALYSIS SCHEMAS
# ============================================================================

class ComplexityComponents(BaseModel):
    """Complexity score components."""
    
    length: float = Field(
        ..., 
        description="Length-based complexity score (0-1)",
        ge=0,
        le=1
    )
    vocabulary: float = Field(
        ...,
        description="Vocabulary richness score (0-1)",
        ge=0,
        le=1
    )
    structure: float = Field(
        ...,
        description="Sentence structure complexity (0-1)",
        ge=0,
        le=1
    )
    domain: float = Field(
        ...,
        description="Domain complexity score (0-1)",
        ge=0,
        le=1
    )
    reasoning: float = Field(
        ...,
        description="Reasoning requirement score (0-1)",
        ge=0,
        le=1
    )


class ComplexityMetrics(BaseModel):
    """Detailed complexity metrics."""
    
    characters: int = Field(..., description="Total characters", ge=0)
    words: int = Field(..., description="Total words", ge=0)
    unique_words: int = Field(..., description="Unique words", ge=0)
    sentences: int = Field(..., description="Number of sentences", ge=0)
    avg_sentence_length: float = Field(
        ...,
        description="Average words per sentence",
        ge=0
    )


class ComplexityAnalysis(BaseModel):
    """Complete complexity analysis."""
    
    score: float = Field(
        ...,
        description="Overall complexity score (0-1)",
        ge=0,
        le=1
    )
    level: ComplexityLevel = Field(..., description="Complexity level")
    components: ComplexityComponents = Field(..., description="Component scores")
    metrics: ComplexityMetrics = Field(..., description="Detailed metrics")


class Entity(BaseModel):
    """Extracted entity."""
    
    value: str = Field(..., description="Entity value")
    position: int = Field(..., description="Start position in text", ge=0)
    length: int = Field(..., description="Entity length", ge=1)


class EntityExtraction(BaseModel):
    """Entity extraction results."""
    
    code_language: Optional[List[Entity]] = Field(
        None,
        description="Programming languages"
    )
    framework: Optional[List[Entity]] = Field(
        None,
        description="Frameworks and libraries"
    )
    technology: Optional[List[Entity]] = Field(
        None,
        description="Technologies and tools"
    )
    domain: Optional[List[Entity]] = Field(
        None,
        description="Subject domains"
    )
    person: Optional[List[Entity]] = Field(
        None,
        description="People names"
    )
    organization: Optional[List[Entity]] = Field(
        None,
        description="Organizations"
    )
    location: Optional[List[Entity]] = Field(
        None,
        description="Locations"
    )
    date: Optional[List[Entity]] = Field(
        None,
        description="Dates and times"
    )
    number: Optional[List[Entity]] = Field(
        None,
        description="Numerical values"
    )
    
    @property
    def total_count(self) -> int:
        """Get total number of entities."""
        count = 0
        for field in self.model_fields_set:
            entities = getattr(self, field)
            if entities:
                count += len(entities)
        return count


class Keyword(BaseModel):
    """Extracted keyword with relevance."""
    
    keyword: str = Field(..., description="Keyword text", min_length=1)
    count: int = Field(..., description="Occurrence count", ge=1)
    score: float = Field(
        ...,
        description="Relevance score (0-1)",
        ge=0,
        le=1
    )


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    
    score: float = Field(
        ...,
        description="Sentiment score (-1 to 1)",
        ge=-1,
        le=1
    )
    label: SentimentLabel = Field(..., description="Sentiment label")
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0,
        le=1
    )
    positive_words: int = Field(..., description="Positive word count", ge=0)
    negative_words: int = Field(..., description="Negative word count", ge=0)


class LanguageDetection(BaseModel):
    """Language detection results."""
    
    code: str = Field(..., description="Language code (ISO 639-1)", min_length=2, max_length=3)
    name: str = Field(..., description="Language name")
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0,
        le=1
    )
    is_primary: bool = Field(..., description="Is primary language")
    alternatives: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Alternative language detections"
    )


class PromptAnalysisResponse(BaseModel):
    """
    Complete prompt analysis response.
    
    Provides comprehensive analysis of a prompt including:
    - Type classification
    - Complexity assessment
    - Language detection
    - Entity extraction
    - Keyword extraction
    - Sentiment analysis
    - Token estimation
    - Capability requirements
    """
    
    prompt: str = Field(..., description="Original prompt (truncated if long)")
    length: int = Field(..., description="Prompt length in characters", ge=0)
    
    # Type classification
    prompt_type: PromptType = Field(..., description="Primary prompt type")
    prompt_type_secondary: Optional[PromptType] = Field(
        None,
        description="Secondary prompt type"
    )
    prompt_type_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for all types"
    )
    
    # Complexity
    complexity_score: float = Field(
        ...,
        description="Overall complexity score (0-1)",
        ge=0,
        le=1
    )
    complexity_level: ComplexityLevel = Field(..., description="Complexity level")
    complexity_components: ComplexityComponents = Field(
        ...,
        description="Component complexity scores"
    )
    complexity_metrics: ComplexityMetrics = Field(
        ...,
        description="Detailed complexity metrics"
    )
    
    # Language
    language_code: str = Field(..., description="Detected language code")
    language_name: str = Field(..., description="Detected language name")
    language_confidence: float = Field(
        ...,
        description="Language detection confidence",
        ge=0,
        le=1
    )
    language_alternatives: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Alternative language detections"
    )
    
    # Entities and keywords
    entities: EntityExtraction = Field(
        default_factory=EntityExtraction,
        description="Extracted entities"
    )
    entity_count: int = Field(0, description="Total entities found", ge=0)
    keywords: List[Keyword] = Field(
        default_factory=list,
        description="Extracted keywords with relevance"
    )
    top_keywords: List[str] = Field(
        default_factory=list,
        description="Top 5 keywords"
    )
    
    # Sentiment
    sentiment_score: float = Field(
        0,
        description="Sentiment score (-1 to 1)",
        ge=-1,
        le=1
    )
    sentiment_label: SentimentLabel = Field(
        SentimentLabel.NEUTRAL,
        description="Sentiment label"
    )
    sentiment_confidence: float = Field(
        0,
        description="Sentiment confidence",
        ge=0,
        le=1
    )
    
    # Token estimation
    estimated_tokens: int = Field(
        ...,
        description="Estimated token count",
        ge=0
    )
    estimation_method: Literal["chars_per_token", "model_specific"] = Field(
        "chars_per_token",
        description="Token estimation method"
    )
    
    # Capabilities
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Model capabilities required for this prompt"
    )
    
    # Metadata
    timestamp: str = Field(..., description="Analysis timestamp (ISO 8601)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Write a Python function to calculate fibonacci",
                "length": 45,
                "prompt_type": "code",
                "prompt_type_secondary": "reasoning",
                "prompt_type_scores": {
                    "code": 0.85,
                    "reasoning": 0.45,
                    "qa": 0.20
                },
                "complexity_score": 0.65,
                "complexity_level": "complex",
                "language_code": "en",
                "language_name": "English",
                "language_confidence": 0.98,
                "estimated_tokens": 11,
                "required_capabilities": ["code", "reasoning"]
            }
        }
    )


# ============================================================================
# ROUTING DECISION SCHEMAS
# ============================================================================

class RoutingAlternative(BaseModel):
    """Alternative model considered during routing."""
    
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model display name")
    score: float = Field(
        ...,
        description="Routing score (0-1)",
        ge=0,
        le=1
    )
    latency_ms: Optional[int] = Field(
        None,
        description="Expected latency in milliseconds",
        ge=0
    )
    cost_per_token: Optional[float] = Field(
        None,
        description="Cost per token in USD",
        ge=0
    )
    quality_score: Optional[float] = Field(
        None,
        description="Model quality score (0-1)",
        ge=0,
        le=1
    )
    active_connections: Optional[int] = Field(
        None,
        description="Current active connections",
        ge=0
    )


class RoutingDecisionResponse(BaseModel):
    """
    Complete routing decision response.
    
    Provides the routing decision with:
    - Selected model
    - Decision reasoning
    - Confidence score
    - Alternative models
    - Performance metrics
    - Cache status
    """
    
    request_id: str = Field(..., description="Unique request identifier")
    prompt_hash: str = Field(..., description="Hash of the prompt")
    prompt_length: int = Field(..., description="Prompt length in characters", ge=0)
    
    # Prompt analysis
    prompt_type: PromptType = Field(..., description="Detected prompt type")
    complexity_score: float = Field(
        ...,
        description="Prompt complexity score",
        ge=0,
        le=1
    )
    
    # Routing decision
    selected_model: str = Field(..., description="Selected model ID")
    strategy: RoutingStrategy = Field(..., description="Routing strategy used")
    confidence: float = Field(
        ...,
        description="Decision confidence (0-1)",
        ge=0,
        le=1
    )
    reasoning: str = Field(..., description="Human-readable decision explanation")
    
    # Alternatives
    alternatives: List[RoutingAlternative] = Field(
        default_factory=list,
        description="Top alternative models considered"
    )
    
    # Performance
    analysis_time_ms: float = Field(
        ...,
        description="Time spent on prompt analysis (ms)",
        ge=0
    )
    decision_time_ms: float = Field(
        ...,
        description="Time spent on routing decision (ms)",
        ge=0
    )
    total_time_ms: float = Field(
        ...,
        description="Total routing time (ms)",
        ge=0
    )
    
    # Cache
    cache_hit: bool = Field(False, description="Whether decision was cached")
    
    # Metadata
    timestamp: datetime = Field(..., description="Decision timestamp")
    
    @validator('reasoning')
    def validate_reasoning(cls, v: str) -> str:
        """Ensure reasoning is provided and not too long."""
        if not v or not v.strip():
            raise ValueError('Reasoning cannot be empty')
        if len(v) > 500:
            return v[:497] + '...'
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "abc-123-def-456",
                "prompt_hash": "5d41402abc4b2a76b9719d911017c592",
                "prompt_length": 45,
                "prompt_type": "code",
                "complexity_score": 0.65,
                "selected_model": "grok-beta",
                "strategy": "hybrid",
                "confidence": 0.92,
                "reasoning": "Selected grok-beta for code generation with 120ms expected latency",
                "alternatives": [
                    {
                        "model_id": "starcoder-7b",
                        "model_name": "StarCoder 7B",
                        "score": 0.78,
                        "latency_ms": 450,
                        "cost_per_token": 0.000002
                    }
                ],
                "analysis_time_ms": 35.2,
                "decision_time_ms": 12.8,
                "total_time_ms": 48.0,
                "cache_hit": False,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


class BatchRoutingItem(BaseModel):
    """Individual item in batch routing response."""
    
    index: int = Field(..., description="Original index in batch", ge=0)
    prompt: str = Field(..., description="Original prompt")
    decision: Optional[RoutingDecisionResponse] = Field(
        None,
        description="Routing decision (if successful)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message (if failed)"
    )
    success: bool = Field(..., description="Whether routing succeeded")


class BatchRoutingResponse(BaseModel):
    """Batch routing response."""
    
    request_id: str = Field(..., description="Batch request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    decisions: List[BatchRoutingItem] = Field(
        ...,
        description="Routing decisions for each prompt"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
        ge=0
    )


# ============================================================================
# ROUTING REQUEST SCHEMAS
# ============================================================================

class RoutingRequest(BaseModel):
    """
    Routing request schema.
    
    Request to analyze a prompt and make a routing decision.
    Supports various constraints and preferences.
    """
    
    prompt: str = Field(
        ...,
        description="Prompt to analyze and route",
        min_length=1,
        max_length=10000
    )
    
    strategy: Optional[RoutingStrategy] = Field(
        None,
        description="Routing strategy to use (auto-selected if not specified)"
    )
    
    preferred_models: Optional[List[str]] = Field(
        None,
        description="Preferred model IDs to consider",
        max_length=10
    )
    
    excluded_models: Optional[List[str]] = Field(
        None,
        description="Model IDs to exclude from routing",
        max_length=20
    )
    
    required_capabilities: Optional[List[str]] = Field(
        None,
        description="Required model capabilities",
        example=["code", "reasoning"]
    )
    
    max_latency_ms: Optional[int] = Field(
        None,
        description="Maximum acceptable latency in milliseconds",
        ge=50,
        le=10000
    )
    
    max_cost_usd: Optional[float] = Field(
        None,
        description="Maximum acceptable cost per request in USD",
        ge=0,
        le=1
    )
    
    min_quality_score: Optional[float] = Field(
        None,
        description="Minimum acceptable quality score (0-1)",
        ge=0,
        le=1
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for routing"
    )
    
    @validator('prompt')
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace')
        return v.strip()
    
    @validator('preferred_models')
    def validate_preferred_models(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Remove duplicates from preferred models."""
        if v:
            return list(dict.fromkeys(v))  # Preserve order, remove duplicates
        return v
    
    @validator('excluded_models')
    def validate_excluded_models(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Remove duplicates from excluded models."""
        if v:
            return list(set(v))
        return v
    
    @root_validator
    def validate_preferred_not_excluded(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure preferred models are not also excluded."""
        preferred = values.get('preferred_models', [])
        excluded = values.get('excluded_models', [])
        
        if preferred and excluded:
            conflicts = set(preferred) & set(excluded)
            if conflicts:
                raise ValueError(
                    f"Models cannot be both preferred and excluded: {conflicts}"
                )
        
        return values
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Write a Python function to calculate fibonacci",
                "strategy": "hybrid",
                "preferred_models": ["grok-beta", "starcoder-7b"],
                "required_capabilities": ["code"],
                "max_latency_ms": 500,
                "max_cost_usd": 0.001
            }
        }
    )


class BatchRoutingRequest(BaseModel):
    """Batch routing request schema."""
    
    prompts: List[str] = Field(
        ...,
        description="List of prompts to analyze",
        min_items=1,
        max_items=100
    )
    
    strategy: Optional[RoutingStrategy] = Field(
        None,
        description="Routing strategy to use for all prompts"
    )
    
    parallel: bool = Field(
        True,
        description="Process prompts in parallel"
    )
    
    @validator('prompts')
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """Validate and clean prompts."""
        cleaned = []
        for prompt in v:
            stripped = prompt.strip()
            if stripped:
                cleaned.append(stripped)
        
        if not cleaned:
            raise ValueError('At least one non-empty prompt is required')
        
        return cleaned


# ============================================================================
# ROUTING METRICS SCHEMAS
# ============================================================================

class ModelDistributionItem(BaseModel):
    """Model selection distribution item."""
    
    model_id: str = Field(..., description="Model identifier")
    count: int = Field(..., description="Selection count", ge=0)
    percentage: float = Field(
        ...,
        description="Percentage of total selections",
        ge=0,
        le=100
    )


class StrategyDistributionItem(BaseModel):
    """Strategy usage distribution item."""
    
    strategy: RoutingStrategy = Field(..., description="Routing strategy")
    count: int = Field(..., description="Usage count", ge=0)
    percentage: float = Field(
        ...,
        description="Percentage of total usage",
        ge=0,
        le=100
    )


class PromptTypeDistributionItem(BaseModel):
    """Prompt type distribution item."""
    
    prompt_type: PromptType = Field(..., description="Prompt type")
    count: int = Field(..., description="Occurrence count", ge=0)
    percentage: float = Field(
        ...,
        description="Percentage of total prompts",
        ge=0,
        le=100
    )


class StrategyPerformance(BaseModel):
    """Performance metrics per strategy."""
    
    avg_confidence: float = Field(
        ...,
        description="Average confidence score",
        ge=0,
        le=1
    )
    avg_decision_time_ms: float = Field(
        ...,
        description="Average decision time in milliseconds",
        ge=0
    )
    usage_count: int = Field(..., description="Number of times used", ge=0)
    success_rate: float = Field(
        ...,
        description="Success rate percentage",
        ge=0,
        le=100
    )


class HourlyMetric(BaseModel):
    """Hourly metric data point."""
    
    hour: str = Field(..., description="Hour timestamp (ISO 8601)")
    count: int = Field(..., description="Metric count", ge=0)
    avg_value: Optional[float] = Field(
        None,
        description="Average metric value"
    )


class RoutingMetricsResponse(BaseModel):
    """
    Comprehensive routing metrics response.
    
    Provides analytics on routing performance including:
    - Decision volume and distribution
    - Model and strategy popularity
    - Performance metrics
    - Trends over time
    """
    
    timestamp: datetime = Field(..., description="Metrics snapshot timestamp")
    
    # Volume metrics
    total_decisions: int = Field(..., description="Total routing decisions", ge=0)
    average_confidence: float = Field(
        ...,
        description="Average confidence score",
        ge=0,
        le=1
    )
    average_analysis_time_ms: float = Field(
        ...,
        description="Average analysis time in milliseconds",
        ge=0
    )
    average_decision_time_ms: float = Field(
        ...,
        description="Average decision time in milliseconds",
        ge=0
    )
    cache_hit_rate: float = Field(
        ...,
        description="Routing decision cache hit rate percentage",
        ge=0,
        le=100
    )
    
    # Distributions
    model_distribution: List[ModelDistributionItem] = Field(
        ...,
        description="Model selection distribution"
    )
    strategy_distribution: List[StrategyDistributionItem] = Field(
        ...,
        description="Strategy usage distribution"
    )
    prompt_type_distribution: List[PromptTypeDistributionItem] = Field(
        ...,
        description="Prompt type distribution"
    )
    
    # Performance by strategy
    strategy_performance: Dict[str, StrategyPerformance] = Field(
        ...,
        description="Performance metrics per strategy"
    )
    
    # Trends
    hourly_decisions: List[HourlyMetric] = Field(
        ...,
        description="Hourly decision counts"
    )
    hourly_latency: List[HourlyMetric] = Field(
        ...,
        description="Hourly latency trends"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "total_decisions": 15234,
                "average_confidence": 0.87,
                "average_analysis_time_ms": 32.5,
                "average_decision_time_ms": 15.2,
                "cache_hit_rate": 23.4,
                "model_distribution": [
                    {"model_id": "gpt-3.5-turbo", "count": 5234, "percentage": 34.4},
                    {"model_id": "grok-beta", "count": 4123, "percentage": 27.1}
                ]
            }
        }
    )


# ============================================================================
# ROUTING FEEDBACK SCHEMAS
# ============================================================================

class RoutingFeedbackRequest(BaseModel):
    """
    Routing feedback submission.
    
    User feedback on routing decisions to improve future routing.
    """
    
    decision_id: str = Field(
        ...,
        description="ID of the routing decision being rated"
    )
    
    rating: int = Field(
        ...,
        description="Rating from 1 to 5",
        ge=1,
        le=5
    )
    
    appropriate_model: bool = Field(
        ...,
        description="Whether the selected model was appropriate"
    )
    
    better_model: Optional[str] = Field(
        None,
        description="Model that would have been better (if any)"
    )
    
    comments: Optional[str] = Field(
        None,
        description="Additional feedback comments",
        max_length=1000
    )
    
    @validator('rating')
    def validate_rating(cls, v: int) -> int:
        """Ensure rating is within range."""
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


class RoutingFeedbackResponse(BaseModel):
    """Routing feedback submission response."""
    
    status: Literal["success"] = Field(
        "success",
        description="Submission status"
    )
    message: str = Field(
        ...,
        description="Response message"
    )
    request_id: str = Field(
        ...,
        description="Request identifier for tracking"
    )
    timestamp: datetime = Field(
        ...,
        description="Submission timestamp"
    )


# ============================================================================
# ROUTING RULE SCHEMAS
# ============================================================================

class RoutingCondition(BaseModel):
    """Condition for custom routing rule."""
    
    field: str = Field(
        ...,
        description="Field to evaluate",
        pattern="^(prompt_type|complexity|user_tier|user_id|prompt_length|contains_.*)$"
    )
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"] = Field(
        ...,
        description="Comparison operator"
    )
    value: Any = Field(..., description="Value to compare against")


class RoutingRuleAction(BaseModel):
    """Action for custom routing rule."""
    
    model: str = Field(..., description="Model to route to")
    fallback_model: Optional[str] = Field(
        None,
        description="Fallback model if primary is unavailable"
    )


class RoutingRule(BaseModel):
    """Custom routing rule configuration."""
    
    id: Optional[str] = Field(
        None,
        description="Rule ID (auto-generated if not provided)"
    )
    name: str = Field(
        ...,
        description="Rule name",
        min_length=1,
        max_length=100
    )
    description: Optional[str] = Field(
        None,
        description="Rule description",
        max_length=500
    )
    conditions: List[RoutingCondition] = Field(
        ...,
        description="Rule conditions (all must match)",
        min_items=1
    )
    action: RoutingRuleAction = Field(
        ...,
        description="Rule action"
    )
    priority: int = Field(
        0,
        description="Rule priority (higher = more important)",
        ge=0
    )
    enabled: bool = Field(
        True,
        description="Whether rule is enabled"
    )
    created_at: Optional[datetime] = Field(
        None,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate rule name."""
        if not v.strip():
            raise ValueError('Rule name cannot be empty')
        return v.strip()


# ============================================================================
# ROUTING STRATEGY SCHEMAS
# ============================================================================

class RoutingStrategyInfo(BaseModel):
    """Routing strategy information."""
    
    name: RoutingStrategy = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Strategy weights (if applicable)"
    )
    use_cases: List[str] = Field(
        ...,
        description="Recommended use cases"
    )
    limitations: List[str] = Field(
        ...,
        description="Known limitations"
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "PromptType",
    "ComplexityLevel",
    "RoutingStrategy",
    "SentimentLabel",
    
    # Analysis schemas
    "ComplexityComponents",
    "ComplexityMetrics",
    "ComplexityAnalysis",
    "Entity",
    "EntityExtraction",
    "Keyword",
    "SentimentAnalysis",
    "LanguageDetection",
    "PromptAnalysisResponse",
    
    # Routing decision schemas
    "RoutingAlternative",
    "RoutingDecisionResponse",
    "BatchRoutingItem",
    "BatchRoutingResponse",
    
    # Request schemas
    "RoutingRequest",
    "BatchRoutingRequest",
    
    # Metrics schemas
    "ModelDistributionItem",
    "StrategyDistributionItem",
    "PromptTypeDistributionItem",
    "StrategyPerformance",
    "HourlyMetric",
    "RoutingMetricsResponse",
    
    # Feedback schemas
    "RoutingFeedbackRequest",
    "RoutingFeedbackResponse",
    
    # Rule schemas
    "RoutingCondition",
    "RoutingRuleAction",
    "RoutingRule",
    
    # Strategy info
    "RoutingStrategyInfo",
]