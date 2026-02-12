"""
Cost Calculator - Production Ready
Real-time cost tracking and estimation for LLM inference across multiple providers.
Supports token-based pricing, tiered rates, and cost optimization recommendations.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json

from core.logging import get_logger
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# COST CONFIGURATION
# ============================================================================

class PricingTier(str, Enum):
    """Pricing tiers for different user levels."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class ModelProvider(str, Enum):
    """Model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GROK = "grok"
    META = "meta"
    MISTRAL = "mistral"
    TII = "tii"
    BIGCODE = "bigcode"
    MICROSOFT = "microsoft"
    DEEPSEEK = "deepseek"
    ALIBABA = "alibaba"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


# ============================================================================
# PRICING DATABASE
# ============================================================================

class PricingDatabase:
    """
    Centralized pricing database for all supported models.
    
    Features:
    - Token-based pricing (input/output)
    - Request-based fees
    - Tiered discounts
    - Batch pricing
    - Real-time updates
    """
    
    # OpenAI models
    OPENAI_PRICING = {
        "gpt-4": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.03,    # $0.03 per 1K input tokens
            "output_cost_per_1k": 0.06,   # $0.06 per 1K output tokens
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.5,         # 50% discount for batch
            "context_window": 8192,
            "release_date": "2023-03-14"
        },
        "gpt-4-32k": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.06,
            "output_cost_per_1k": 0.12,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.5,
            "context_window": 32768,
            "release_date": "2023-06-13"
        },
        "gpt-4-turbo": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.01,
            "output_cost_per_1k": 0.03,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.5,
            "context_window": 128000,
            "release_date": "2023-11-06"
        },
        "gpt-3.5-turbo": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.0005,
            "output_cost_per_1k": 0.0015,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.25,
            "context_window": 16384,
            "release_date": "2022-03-01"
        },
        "gpt-3.5-turbo-16k": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.004,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.25,
            "context_window": 16384,
            "release_date": "2023-06-13"
        },
        "text-embedding-ada-002": {
            "provider": ModelProvider.OPENAI,
            "input_cost_per_1k": 0.0001,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 8191,
            "release_date": "2022-12-16"
        }
    }
    
    # Anthropic models
    ANTHROPIC_PRICING = {
        "claude-3-opus": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 200000,
            "release_date": "2024-03-04"
        },
        "claude-3-sonnet": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 200000,
            "release_date": "2024-03-04"
        },
        "claude-3-haiku": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.00025,
            "output_cost_per_1k": 0.00125,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 200000,
            "release_date": "2024-03-04"
        },
        "claude-2.1": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.008,
            "output_cost_per_1k": 0.024,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 200000,
            "release_date": "2023-11-21"
        },
        "claude-2.0": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.008,
            "output_cost_per_1k": 0.024,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 100000,
            "release_date": "2023-07-11"
        },
        "claude-instant-1.2": {
            "provider": ModelProvider.ANTHROPIC,
            "input_cost_per_1k": 0.0008,
            "output_cost_per_1k": 0.0024,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 100000,
            "release_date": "2023-08-09"
        }
    }
    
    # Cohere models
    COHERE_PRICING = {
        "command-r-plus": {
            "provider": ModelProvider.COHERE,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 128000,
            "release_date": "2024-03-11"
        },
        "command-r": {
            "provider": ModelProvider.COHERE,
            "input_cost_per_1k": 0.0005,
            "output_cost_per_1k": 0.0015,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 128000,
            "release_date": "2024-03-11"
        },
        "command": {
            "provider": ModelProvider.COHERE,
            "input_cost_per_1k": 0.0005,
            "output_cost_per_1k": 0.0015,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 4096,
            "release_date": "2022-11-08"
        },
        "command-light": {
            "provider": ModelProvider.COHERE,
            "input_cost_per_1k": 0.0003,
            "output_cost_per_1k": 0.0006,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 4096,
            "release_date": "2022-11-08"
        },
        "embed-english-v3.0": {
            "provider": ModelProvider.COHERE,
            "input_cost_per_1k": 0.0001,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 512,
            "release_date": "2023-11-15"
        }
    }
    
    # Grok (xAI) models
    GROK_PRICING = {
        "grok-beta": {
            "provider": ModelProvider.GROK,
            "input_cost_per_1k": 0.00015,
            "output_cost_per_1k": 0.0006,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 8192,
            "release_date": "2023-11-04"
        },
        "grok-1": {
            "provider": ModelProvider.GROK,
            "input_cost_per_1k": 0.0002,
            "output_cost_per_1k": 0.0008,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.2,
            "context_window": 8192,
            "release_date": "2023-11-04"
        }
    }
    
    # Local models (free, but track compute)
    LOCAL_PRICING = {
        "llama-3.1-8b-instant": {
            "provider": ModelProvider.META,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 8192,
            "compute_cost_per_second": 0.00001  # $0.00001 per second of GPU time
        },
        "mistral-7b": {
            "provider": ModelProvider.MISTRAL,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 32768,
            "compute_cost_per_second": 0.000008
        },
        "mistral-7b-instruct": {
            "provider": ModelProvider.MISTRAL,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 32768,
            "compute_cost_per_second": 0.000008
        },
        "falcon-7b-instruct": {
            "provider": ModelProvider.TII,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000008
        },
        "openchat-3.5": {
            "provider": ModelProvider.MISTRAL,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 8192,
            "compute_cost_per_second": 0.000008
        },
        "zephyr-7b": {
            "provider": ModelProvider.HUGGINGFACE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 32768,
            "compute_cost_per_second": 0.000008
        },
        "neuralbeagle-7b": {
            "provider": ModelProvider.META,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 8192,
            "compute_cost_per_second": 0.000008
        },
        "starcoder-7b": {
            "provider": ModelProvider.BIGCODE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 8192,
            "compute_cost_per_second": 0.000008
        },
        "starcoder2-7b": {
            "provider": ModelProvider.BIGCODE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 16384,
            "compute_cost_per_second": 0.000008
        },
        "phi-2": {
            "provider": ModelProvider.MICROSOFT,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000004
        },
        "tinyllama-1.1b": {
            "provider": ModelProvider.META,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000002
        },
        "gpt-j-6b": {
            "provider": ModelProvider.HUGGINGFACE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000006
        },
        "redpajama-7b": {
            "provider": ModelProvider.HUGGINGFACE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000008
        },
        "mpt-7b-chat": {
            "provider": ModelProvider.HUGGINGFACE,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 2048,
            "compute_cost_per_second": 0.000008
        },
        "qwen2-7b": {
            "provider": ModelProvider.ALIBABA,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 32768,
            "compute_cost_per_second": 0.000008
        },
        "deepseek-coder-6.7b": {
            "provider": ModelProvider.DEEPSEEK,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 16384,
            "compute_cost_per_second": 0.000007
        }
    }
    
    # Custom model pricing (for user-deployed models)
    CUSTOM_PRICING = {
        "custom": {
            "provider": ModelProvider.CUSTOM,
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "fixed_cost_per_request": 0.0,
            "batch_discount": 0.0,
            "context_window": 4096
        }
    }
    
    # Merge all pricing databases
    PRICING = {}
    PRICING.update(OPENAI_PRICING)
    PRICING.update(ANTHROPIC_PRICING)
    PRICING.update(COHERE_PRICING)
    PRICING.update(GROK_PRICING)
    PRICING.update(LOCAL_PRICING)
    PRICING.update(CUSTOM_PRICING)
    
    # Tier discounts
    TIER_DISCOUNTS = {
        PricingTier.FREE: 0.0,      # No discount
        PricingTier.PRO: 0.2,       # 20% discount
        PricingTier.ENTERPRISE: 0.3, # 30% discount
        PricingTier.ADMIN: 0.5      # 50% discount (internal use)
    }
    
    # Volume discounts (annual commit)
    VOLUME_DISCOUNTS = {
        10000: 0.05,    # $10k annual commit -> 5% discount
        50000: 0.10,    # $50k annual commit -> 10% discount
        100000: 0.15,   # $100k annual commit -> 15% discount
        250000: 0.20,   # $250k annual commit -> 20% discount
        500000: 0.25,   # $500k annual commit -> 25% discount
        1000000: 0.30   # $1M annual commit -> 30% discount
    }
    
    @classmethod
    def get_pricing(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing information for a model."""
        return cls.PRICING.get(model_id)
    
    @classmethod
    def get_provider(cls, model_id: str) -> Optional[ModelProvider]:
        """Get provider for a model."""
        pricing = cls.get_pricing(model_id)
        return pricing.get("provider") if pricing else None
    
    @classmethod
    def is_external_api(cls, model_id: str) -> bool:
        """Check if model is an external API (paid)."""
        pricing = cls.get_pricing(model_id)
        if not pricing:
            return False
        return pricing.get("input_cost_per_1k", 0) > 0 or pricing.get("output_cost_per_1k", 0) > 0
    
    @classmethod
    def get_tier_discount(cls, tier: PricingTier) -> float:
        """Get discount percentage for a tier."""
        return cls.TIER_DISCOUNTS.get(tier, 0.0)
    
    @classmethod
    def get_volume_discount(cls, annual_commit_usd: float) -> float:
        """Get volume discount based on annual commit."""
        discount = 0.0
        for threshold, disc in sorted(cls.VOLUME_DISCOUNTS.items()):
            if annual_commit_usd >= threshold:
                discount = disc
            else:
                break
        return discount


# ============================================================================
# COST CALCULATOR
# ============================================================================

class CostCalculator:
    """
    Real-time cost calculation for LLM inference.
    
    Features:
    - Token-based pricing (input/output)
    - Tier discounts
    - Volume discounts
    - Batch pricing
    - Compute cost for local models
    - Cost projections
    - Budget tracking
    """
    
    def __init__(self):
        self.pricing_db = PricingDatabase()
        self._budget_alerts: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "cost_calculator_initialized",
            models_configured=len(self.pricing_db.PRICING),
            external_apis=sum(1 for m in self.pricing_db.PRICING.values() if m.get("input_cost_per_1k", 0) > 0)
        )
    
    # ========================================================================
    # COST CALCULATION
    # ========================================================================
    
    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        tier: PricingTier = PricingTier.FREE,
        annual_commit_usd: float = 0.0,
        is_batch: bool = False,
        compute_seconds: Optional[float] = None,
        custom_rate: Optional[float] = None
    ) -> float:
        """
        Calculate cost for a model inference request.
        
        Args:
            model: Model identifier
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_tokens: Total tokens (alternative to prompt+completion)
            tier: User pricing tier
            annual_commit_usd: Annual commitment amount for volume discount
            is_batch: Whether this is a batch request
            compute_seconds: Compute time in seconds (for local models)
            custom_rate: Custom rate per 1K tokens (overrides pricing)
        
        Returns:
            Cost in USD
        """
        # Get pricing for model
        pricing = self.pricing_db.get_pricing(model)
        
        if not pricing:
            logger.warning(f"No pricing found for model: {model}, assuming free")
            return 0.0
        
        # Use total_tokens if provided, otherwise sum
        if total_tokens is not None:
            prompt_tokens = total_tokens // 2
            completion_tokens = total_tokens - prompt_tokens
        
        # Calculate base cost
        input_cost = 0.0
        output_cost = 0.0
        
        if custom_rate is not None:
            # Custom rate per 1K tokens
            rate_per_token = custom_rate / 1000
            input_cost = prompt_tokens * rate_per_token
            output_cost = completion_tokens * rate_per_token
        else:
            # Standard pricing
            input_cost = (prompt_tokens / 1000) * pricing.get("input_cost_per_1k", 0)
            output_cost = (completion_tokens / 1000) * pricing.get("output_cost_per_1k", 0)
        
        # Add fixed cost per request
        fixed_cost = pricing.get("fixed_cost_per_request", 0.0)
        
        # Add compute cost for local models
        compute_cost = 0.0
        if compute_seconds is not None:
            compute_cost = compute_seconds * pricing.get("compute_cost_per_second", 0)
        elif "compute_cost_per_second" in pricing:
            # Estimate compute time based on tokens (rough approximation)
            estimated_seconds = (prompt_tokens + completion_tokens) / 100  # 100 tokens/sec
            compute_cost = estimated_seconds * pricing["compute_cost_per_second"]
        
        # Calculate subtotal
        subtotal = input_cost + output_cost + fixed_cost + compute_cost
        
        # Apply batch discount
        if is_batch:
            batch_discount = pricing.get("batch_discount", 0)
            subtotal *= (1 - batch_discount)
        
        # Apply tier discount
        tier_discount = self.pricing_db.get_tier_discount(tier)
        subtotal *= (1 - tier_discount)
        
        # Apply volume discount
        volume_discount = self.pricing_db.get_volume_discount(annual_commit_usd)
        subtotal *= (1 - volume_discount)
        
        # Round to 6 decimal places (micro-dollar precision)
        cost = Decimal(str(subtotal)).quantize(
            Decimal('0.000001'),
            rounding=ROUND_HALF_UP
        )
        
        return float(cost)
    
    def calculate_batch_cost(
        self,
        model: str,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        num_requests: int,
        tier: PricingTier = PricingTier.FREE,
        annual_commit_usd: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate cost for a batch of requests.
        
        Args:
            model: Model identifier
            total_prompt_tokens: Total input tokens across all requests
            total_completion_tokens: Total output tokens across all requests
            num_requests: Number of requests in batch
            tier: User pricing tier
            annual_commit_usd: Annual commitment amount
        
        Returns:
            Dictionary with cost breakdown
        """
        # Individual pricing (without batch discount)
        individual_cost = self.calculate_cost(
            model=model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            tier=tier,
            annual_commit_usd=annual_commit_usd,
            is_batch=False
        )
        
        # Batch pricing (with batch discount)
        batch_cost = self.calculate_cost(
            model=model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            tier=tier,
            annual_commit_usd=annual_commit_usd,
            is_batch=True
        )
        
        savings = individual_cost - batch_cost
        savings_percentage = (savings / individual_cost * 100) if individual_cost > 0 else 0
        
        return {
            "model": model,
            "num_requests": num_requests,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "individual_pricing": round(individual_cost, 6),
            "batch_pricing": round(batch_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_percentage": round(savings_percentage, 2),
            "cost_per_request": round(batch_cost / num_requests, 8) if num_requests > 0 else 0,
            "cost_per_1k_tokens": round(
                (batch_cost / (total_prompt_tokens + total_completion_tokens)) * 1000
                if (total_prompt_tokens + total_completion_tokens) > 0 else 0,
                6
            )
        }
    
    # ========================================================================
    # COST COMPARISON
    # ========================================================================
    
    def compare_models(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        models: Optional[List[str]] = None,
        tier: PricingTier = PricingTier.FREE,
        annual_commit_usd: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Compare costs across multiple models.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            models: List of model IDs to compare (all if None)
            tier: User pricing tier
            annual_commit_usd: Annual commitment amount
        
        Returns:
            List of cost comparisons sorted by price
        """
        if models is None:
            # Compare all external API models
            models = [
                model_id for model_id, pricing in self.pricing_db.PRICING.items()
                if pricing.get("input_cost_per_1k", 0) > 0 or pricing.get("output_cost_per_1k", 0) > 0
            ]
        
        comparisons = []
        
        for model_id in models:
            pricing = self.pricing_db.get_pricing(model_id)
            if not pricing:
                continue
            
            cost = self.calculate_cost(
                model=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tier=tier,
                annual_commit_usd=annual_commit_usd
            )
            
            comparisons.append({
                "model": model_id,
                "provider": pricing.get("provider", "unknown").value,
                "cost_usd": round(cost, 6),
                "cost_per_1k_tokens": round(
                    (cost / (prompt_tokens + completion_tokens)) * 1000
                    if (prompt_tokens + completion_tokens) > 0 else 0,
                    6
                ),
                "context_window": pricing.get("context_window", 0),
                "input_cost_per_1k": pricing.get("input_cost_per_1k", 0),
                "output_cost_per_1k": pricing.get("output_cost_per_1k", 0)
            })
        
        # Sort by cost
        comparisons.sort(key=lambda x: x["cost_usd"])
        
        return comparisons
    
    def get_cost_effective_models(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        max_latency_ms: Optional[int] = None,
        min_quality_score: Optional[float] = None,
        tier: PricingTier = PricingTier.FREE
    ) -> List[Dict[str, Any]]:
        """
        Get cost-effective model recommendations.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            max_latency_ms: Maximum acceptable latency
            min_quality_score: Minimum acceptable quality
            tier: User pricing tier
        
        Returns:
            List of model recommendations with cost/benefit analysis
        """
        comparisons = self.compare_models(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tier=tier
        )
        
        recommendations = []
        
        for comp in comparisons:
            model_id = comp["model"]
            
            # Skip if exceeds latency limit
            if max_latency_ms:
                # TODO: Get actual latency from model registry
                pass
            
            # Skip if below quality threshold
            if min_quality_score:
                # TODO: Get quality score from model registry
                pass
            
            # Calculate value score (quality/price)
            # TODO: Incorporate actual quality metrics
            value_score = 1.0 / (comp["cost_usd"] + 0.0001)  # Avoid division by zero
            
            recommendations.append({
                **comp,
                "value_score": round(value_score, 2),
                "recommended": comp == comparisons[0]  # Cheapest is default recommendation
            })
        
        return recommendations
    
    # ========================================================================
    # BUDGET TRACKING
    # ========================================================================
    
    async def set_budget(
        self,
        user_id: str,
        monthly_budget_usd: float,
        alert_thresholds: List[float] = [0.5, 0.8, 0.9, 1.0]
    ) -> Dict[str, Any]:
        """
        Set monthly budget for a user.
        
        Args:
            user_id: User identifier
            monthly_budget_usd: Monthly budget in USD
            alert_thresholds: Percentages at which to alert
        
        Returns:
            Budget configuration
        """
        budget_config = {
            "user_id": user_id,
            "monthly_budget_usd": monthly_budget_usd,
            "alert_thresholds": alert_thresholds,
            "current_spend": 0.0,
            "budget_period_start": datetime.utcnow().isoformat(),
            "alerts_sent": []
        }
        
        self._budget_alerts[user_id] = budget_config
        
        logger.info(
            "budget_configured",
            user_id=user_id,
            monthly_budget_usd=monthly_budget_usd,
            alert_thresholds=alert_thresholds
        )
        
        return budget_config
    
    async def track_spend(
        self,
        user_id: str,
        cost_usd: float,
        model: str,
        tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Track spend and check budget alerts.
        
        Args:
            user_id: User identifier
            cost_usd: Cost of request in USD
            model: Model used
            tokens: Tokens used
        
        Returns:
            Alert information if threshold crossed, None otherwise
        """
        if user_id not in self._budget_alerts:
            return None
        
        budget = self._budget_alerts[user_id]
        budget["current_spend"] += cost_usd
        
        # Calculate usage percentage
        usage_percentage = (budget["current_spend"] / budget["monthly_budget_usd"]) * 100
        
        # Check thresholds
        for threshold in budget["alert_thresholds"]:
            threshold_pct = threshold * 100
            
            if usage_percentage >= threshold_pct:
                threshold_key = f"{threshold:.0%}"
                
                # Check if already alerted for this threshold
                if threshold_key not in budget["alerts_sent"]:
                    budget["alerts_sent"].append(threshold_key)
                    
                    alert = {
                        "user_id": user_id,
                        "type": "budget_threshold",
                        "threshold": threshold,
                        "current_spend": round(budget["current_spend"], 2),
                        "monthly_budget": budget["monthly_budget_usd"],
                        "usage_percentage": round(usage_percentage, 2),
                        "timestamp": datetime.utcnow().isoformat(),
                        "last_request": {
                            "model": model,
                            "cost_usd": round(cost_usd, 6),
                            "tokens": tokens
                        }
                    }
                    
                    logger.info(
                        "budget_threshold_reached",
                        user_id=user_id,
                        threshold=threshold,
                        usage_percentage=round(usage_percentage, 2),
                        current_spend=round(budget["current_spend"], 2)
                    )
                    
                    return alert
        
        return None
    
    async def get_budget_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current budget status for a user."""
        if user_id not in self._budget_alerts:
            return None
        
        budget = self._budget_alerts[user_id]
        usage_percentage = (budget["current_spend"] / budget["monthly_budget_usd"]) * 100
        
        return {
            "user_id": user_id,
            "monthly_budget_usd": budget["monthly_budget_usd"],
            "current_spend_usd": round(budget["current_spend"], 2),
            "remaining_budget_usd": round(budget["monthly_budget_usd"] - budget["current_spend"], 2),
            "usage_percentage": round(usage_percentage, 2),
            "period_start": budget["budget_period_start"],
            "alerts_sent": budget["alerts_sent"]
        }
    
    # ========================================================================
    # COST PROJECTIONS
    # ========================================================================
    
    def project_monthly_cost(
        self,
        model: str,
        avg_daily_requests: int,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
        tier: PricingTier = PricingTier.FREE,
        annual_commit_usd: float = 0.0
    ) -> Dict[str, Any]:
        """
        Project monthly cost based on usage patterns.
        
        Args:
            model: Model identifier
            avg_daily_requests: Average requests per day
            avg_prompt_tokens: Average input tokens per request
            avg_completion_tokens: Average output tokens per request
            tier: User pricing tier
            annual_commit_usd: Annual commitment amount
        
        Returns:
            Monthly cost projection
        """
        daily_requests = avg_daily_requests
        monthly_requests = daily_requests * 30
        yearly_requests = daily_requests * 365
        
        daily_prompt_tokens = daily_requests * avg_prompt_tokens
        daily_completion_tokens = daily_requests * avg_completion_tokens
        monthly_prompt_tokens = daily_prompt_tokens * 30
        monthly_completion_tokens = daily_completion_tokens * 30
        
        # Calculate costs
        daily_cost = self.calculate_cost(
            model=model,
            prompt_tokens=daily_prompt_tokens,
            completion_tokens=daily_completion_tokens,
            tier=tier,
            annual_commit_usd=annual_commit_usd
        )
        
        monthly_cost = self.calculate_cost(
            model=model,
            prompt_tokens=monthly_prompt_tokens,
            completion_tokens=monthly_completion_tokens,
            tier=tier,
            annual_commit_usd=annual_commit_usd
        )
        
        yearly_cost = monthly_cost * 12
        
        return {
            "model": model,
            "tier": tier.value,
            "annual_commit_usd": annual_commit_usd,
            "volume_discount": self.pricing_db.get_volume_discount(annual_commit_usd),
            "usage": {
                "daily_requests": daily_requests,
                "monthly_requests": monthly_requests,
                "yearly_requests": yearly_requests,
                "daily_prompt_tokens": daily_prompt_tokens,
                "daily_completion_tokens": daily_completion_tokens,
                "monthly_tokens": monthly_prompt_tokens + monthly_completion_tokens
            },
            "costs": {
                "daily_usd": round(daily_cost, 2),
                "monthly_usd": round(monthly_cost, 2),
                "yearly_usd": round(yearly_cost, 2),
                "cost_per_request_usd": round(daily_cost / daily_requests, 6) if daily_requests > 0 else 0,
                "cost_per_1k_tokens_usd": round(
                    (daily_cost / (daily_prompt_tokens + daily_completion_tokens)) * 1000
                    if (daily_prompt_tokens + daily_completion_tokens) > 0 else 0,
                    6
                )
            }
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_model_pricing(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing information for a model."""
        return self.pricing_db.get_pricing(model_id)
    
    def list_models_by_provider(self, provider: ModelProvider) -> List[str]:
        """List all models from a specific provider."""
        return [
            model_id for model_id, pricing in self.pricing_db.PRICING.items()
            if pricing.get("provider") == provider
        ]
    
    def list_external_models(self) -> List[str]:
        """List all external (paid) models."""
        return [
            model_id for model_id, pricing in self.pricing_db.PRICING.items()
            if pricing.get("input_cost_per_1k", 0) > 0 or pricing.get("output_cost_per_1k", 0) > 0
        ]
    
    def list_free_models(self) -> List[str]:
        """List all free models."""
        return [
            model_id for model_id, pricing in self.pricing_db.PRICING.items()
            if pricing.get("input_cost_per_1k", 0) == 0 
            and pricing.get("output_cost_per_1k", 0) == 0
            and pricing.get("provider") != ModelProvider.CUSTOM
        ]
    
    def format_cost_usd(self, cost: float) -> str:
        """Format cost in USD with appropriate precision."""
        if cost < 0.000001:
            return f"${cost * 1000000000:.2f} n"
        elif cost < 0.001:
            return f"${cost * 1000000:.2f} µ"
        elif cost < 1:
            return f"${cost * 1000:.2f} m"
        else:
            return f"${cost:.4f}"
    
    def get_pricing_summary(self) -> Dict[str, Any]:
        """Get summary of all pricing information."""
        return {
            "total_models": len(self.pricing_db.PRICING),
            "external_models": len(self.list_external_models()),
            "free_models": len(self.list_free_models()),
            "by_provider": {
                provider.value: len(self.list_models_by_provider(provider))
                for provider in ModelProvider
            },
            "tier_discounts": self.pricing_db.TIER_DISCOUNTS,
            "volume_discounts": self.pricing_db.VOLUME_DISCOUNTS
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_cost_calculator = None


def get_cost_calculator() -> CostCalculator:
    """Get singleton cost calculator instance."""
    global _cost_calculator
    if not _cost_calculator:
        _cost_calculator = CostCalculator()
    return _cost_calculator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CostCalculator",
    "PricingDatabase",
    "PricingTier",
    "ModelProvider",
    "get_cost_calculator"
]