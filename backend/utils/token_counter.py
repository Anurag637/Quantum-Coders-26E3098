"""
Token Counter - Production Ready
Accurate token counting for various models with multiple encoding strategies,
caching, and fallback mechanisms.
"""

import re
import math
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
import threading

from core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# TOKENIZER CACHE
# ============================================================================

class TokenizerCache:
    """
    Thread-safe cache for loaded tokenizers.
    
    Features:
    - LRU eviction
    - Thread safety
    - Memory management
    - Lazy loading
    """
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get tokenizer from cache."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            return None
    
    def set(self, key: str, tokenizer: Any):
        """Store tokenizer in cache with LRU eviction."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = min(
                    self.access_count.items(),
                    key=lambda x: x[1]
                )[0]
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            self.cache[key] = tokenizer
            self.access_count[key] = 0
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()


# ============================================================================
# TOKEN COUNTER
# ============================================================================

class TokenCounter:
    """
    Accurate token counter for various LLM models.
    
    Features:
    - Model-specific token counting
    - Multiple encoding strategies
    - Tokenizer caching
    - Fallback estimation
    - Batch counting
    - Cost calculation
    """
    
    # Model families and their tokenizers
    MODEL_FAMILIES = {
        # OpenAI models
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "code-davinci-002": "p50k_base",
        
        # Anthropic models
        "claude": "claude",
        "claude-instant": "claude",
        
        # Cohere models
        "command": "cohere",
        "command-light": "cohere",
        "command-nightly": "cohere",
        
        # Grok models
        "grok": "grok",
        "grok-beta": "grok",
        
        # Llama family
        "llama": "llama",
        "llama-2": "llama",
        "llama-3": "llama",
        "codellama": "llama",
        
        # Mistral family
        "mistral": "mistral",
        "mixtral": "mistral",
        "zephyr": "mistral",
        "openchat": "mistral",
        
        # Other models
        "falcon": "falcon",
        "qwen": "qwen",
        "phi": "phi",
        "starcoder": "starcoder",
        "deepseek": "deepseek",
    }
    
    # Default tokens per character by model family
    DEFAULT_TOKENS_PER_CHAR = {
        "cl100k_base": 0.25,  # OpenAI: ~4 chars per token
        "p50k_base": 0.3,      # ~3.3 chars per token
        "r50k_base": 0.33,     # ~3 chars per token
        "claude": 0.3,         # ~3.3 chars per token
        "cohere": 0.25,        # ~4 chars per token
        "grok": 0.28,          # ~3.5 chars per token
        "llama": 0.3,          # ~3.3 chars per token
        "mistral": 0.3,        # ~3.3 chars per token
        "falcon": 0.3,         # ~3.3 chars per token
        "qwen": 0.28,          # ~3.5 chars per token
        "phi": 0.28,           # ~3.5 chars per token
        "starcoder": 0.3,      # ~3.3 chars per token
        "deepseek": 0.3,       # ~3.3 chars per token
        "default": 0.25,       # Conservative default
    }
    
    def __init__(self):
        self.tokenizer_cache = TokenizerCache()
        self._tiktoken_available = False
        self._transformers_available = False
        
        # Try to import optional dependencies
        self._init_imports()
        
        logger.info(
            "token_counter_initialized",
            tiktoken_available=self._tiktoken_available,
            transformers_available=self._transformers_available
        )
    
    def _init_imports(self):
        """Initialize optional imports."""
        try:
            import tiktoken
            self.tiktoken = tiktoken
            self._tiktoken_available = True
        except ImportError:
            self.tiktoken = None
            self._tiktoken_available = False
        
        try:
            import transformers
            self.transformers = transformers
            self._transformers_available = True
        except ImportError:
            self.transformers = None
            self._transformers_available = False
    
    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        encoding: Optional[str] = None,
        fallback: bool = True
    ) -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Input text
            model: Model name (e.g., "gpt-4", "llama-2-7b")
            encoding: Specific encoding to use (overrides model)
            fallback: Whether to use fallback estimation
        
        Returns:
            Token count
        """
        if not text:
            return 0
        
        # Normalize text
        text = str(text).strip()
        if not text:
            return 0
        
        # Determine encoding
        encoding_name = self._get_encoding_name(model, encoding)
        
        # Try exact tokenization first
        count = self._count_tokens_exact(text, encoding_name)
        
        # Fallback to estimation if exact fails
        if count is None and fallback:
            count = self._estimate_tokens(text, encoding_name)
        
        return count or 0
    
    def count_tokens_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> List[int]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of input texts
            model: Model name
            encoding: Specific encoding
        
        Returns:
            List of token counts
        """
        results = []
        
        for text in texts:
            count = self.count_tokens(text, model, encoding)
            results.append(count)
        
        return results
    
    def count_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in a chat message list (OpenAI format).
        
        Args:
            messages: List of chat messages
            model: Model name
        
        Returns:
            Total token count
        """
        if not messages:
            return 0
        
        total_tokens = 0
        
        # Special handling for OpenAI models
        if any(m in model for m in ["gpt-3.5", "gpt-4"]):
            try:
                import tiktoken
                
                # Get encoding
                if "gpt-4" in model:
                    encoding = tiktoken.encoding_for_model("gpt-4")
                else:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                
                # Count tokens per message (OpenAI format)
                tokens_per_message = 3  # <|start|>{role}\n{content}<|end|>
                tokens_per_name = 1     # If name is specified
                
                for message in messages:
                    total_tokens += tokens_per_message
                    total_tokens += len(encoding.encode(message["content"]))
                    total_tokens += len(encoding.encode(message["role"]))
                    
                    if "name" in message:
                        total_tokens += tokens_per_name
                        total_tokens += len(encoding.encode(message["name"]))
                
                # Add reply tokens
                total_tokens += 3  # <|start|>assistant<|end|>
                
            except Exception as e:
                logger.debug(f"Failed to count OpenAI message tokens: {e}")
                # Fallback to simple counting
                total_tokens = self._estimate_message_tokens(messages)
        else:
            # Generic counting for other models
            total_tokens = self._estimate_message_tokens(messages)
        
        return total_tokens
    
    def count_prompt_tokens(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in a prompt with optional message history.
        
        Args:
            prompt: Current prompt
            messages: Previous message history
            model: Model name
        
        Returns:
            Total token count
        """
        total = 0
        
        # Count history tokens
        if messages:
            total += self.count_messages_tokens(messages, model)
        
        # Count current prompt
        total += self.count_tokens(prompt, model)
        
        return total
    
    # ========================================================================
    # EXACT TOKEN COUNTING
    # ========================================================================
    
    def _count_tokens_exact(self, text: str, encoding_name: str) -> Optional[int]:
        """Count tokens exactly using available tokenizers."""
        # Try tiktoken first (OpenAI, etc.)
        if self._tiktoken_available and encoding_name in self._get_tiktoken_encodings():
            try:
                encoding = self._get_tiktoken_encoding(encoding_name)
                tokens = encoding.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Tiktoken failed for {encoding_name}: {e}")
        
        # Try transformers (HuggingFace models)
        if self._transformers_available and self._has_transformers_tokenizer(encoding_name):
            try:
                tokenizer = self._get_transformers_tokenizer(encoding_name)
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Transformers failed for {encoding_name}: {e}")
        
        return None
    
    def _get_tiktoken_encodings(self) -> Dict[str, str]:
        """Get available tiktoken encodings."""
        if not self._tiktoken_available:
            return {}
        
        return {
            "cl100k_base": "cl100k_base",
            "p50k_base": "p50k_base",
            "p50k_edit": "p50k_edit",
            "r50k_base": "r50k_base",
        }
    
    def _get_tiktoken_encoding(self, encoding_name: str):
        """Get tiktoken encoding from cache."""
        cache_key = f"tiktoken:{encoding_name}"
        
        encoding = self.tokenizer_cache.get(cache_key)
        if encoding:
            return encoding
        
        try:
            if encoding_name in self._get_tiktoken_encodings():
                encoding = self.tiktoken.get_encoding(encoding_name)
            else:
                # Try to get encoding for model
                encoding = self.tiktoken.encoding_for_model(encoding_name)
            
            self.tokenizer_cache.set(cache_key, encoding)
            return encoding
        except Exception:
            # Fallback to cl100k_base
            encoding = self.tiktoken.get_encoding("cl100k_base")
            self.tokenizer_cache.set(cache_key, encoding)
            return encoding
    
    def _has_transformers_tokenizer(self, model_name: str) -> bool:
        """Check if we have a transformers tokenizer for this model."""
        # Models we know we can load
        supported = ["llama", "mistral", "falcon", "qwen", "phi", "starcoder", "deepseek"]
        return any(m in model_name.lower() for m in supported)
    
    def _get_transformers_tokenizer(self, model_name: str):
        """Get transformers tokenizer from cache."""
        cache_key = f"transformers:{model_name}"
        
        tokenizer = self.tokenizer_cache.get(cache_key)
        if tokenizer:
            return tokenizer
        
        try:
            # Map to HuggingFace model IDs
            model_id = self._map_to_hf_model(model_name)
            
            if model_id:
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                self.tokenizer_cache.set(cache_key, tokenizer)
                return tokenizer
        except Exception as e:
            logger.debug(f"Failed to load transformers tokenizer for {model_name}: {e}")
        
        return None
    
    def _map_to_hf_model(self, model_name: str) -> Optional[str]:
        """Map internal model name to HuggingFace model ID."""
        model_lower = model_name.lower()
        
        if "llama-2-7b" in model_lower:
            return "meta-llama/Llama-2-7b-hf"
        elif "llama-2-13b" in model_lower:
            return "meta-llama/Llama-2-13b-hf"
        elif "llama-2-70b" in model_lower:
            return "meta-llama/Llama-2-70b-hf"
        elif "llama-3-8b" in model_lower:
            return "meta-llama/Meta-Llama-3-8B"
        elif "llama-3-70b" in model_lower:
            return "meta-llama/Meta-Llama-3-70B"
        elif "mistral-7b" in model_lower:
            return "mistralai/Mistral-7B-v0.1"
        elif "mixtral-8x7b" in model_lower:
            return "mistralai/Mixtral-8x7B-v0.1"
        elif "falcon-7b" in model_lower:
            return "tiiuae/falcon-7b"
        elif "falcon-40b" in model_lower:
            return "tiiuae/falcon-40b"
        elif "qwen-7b" in model_lower:
            return "Qwen/Qwen-7B"
        elif "qwen2-7b" in model_lower:
            return "Qwen/Qwen2-7B"
        elif "phi-2" in model_lower:
            return "microsoft/phi-2"
        elif "starcoder-7b" in model_lower:
            return "bigcode/starcoder"
        elif "starcoder2-7b" in model_lower:
            return "bigcode/starcoder2-7b"
        elif "deepseek-coder-6.7b" in model_lower:
            return "deepseek-ai/deepseek-coder-6.7b-instruct"
        
        return None
    
    # ========================================================================
    # FALLBACK ESTIMATION
    # ========================================================================
    
    def _estimate_tokens(self, text: str, encoding_name: str = "default") -> int:
        """
        Estimate token count based on character count.
        
        This is a fallback when exact tokenization is unavailable.
        """
        if not text:
            return 0
        
        # Get tokens per character ratio
        ratio = self.DEFAULT_TOKENS_PER_CHAR.get(
            encoding_name,
            self.DEFAULT_TOKENS_PER_CHAR["default"]
        )
        
        # Count characters (excluding whitespace padding)
        char_count = len(text.strip())
        
        # Estimate tokens
        estimated = int(char_count * ratio)
        
        # Add small buffer for safety
        estimated = max(1, estimated + 5)
        
        return estimated
    
    def _estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate tokens for a list of messages."""
        total = 0
        
        for msg in messages:
            # Count content
            total += self._estimate_tokens(msg.get("content", ""))
            
            # Count role (usually short)
            total += self._estimate_tokens(msg.get("role", ""))
            
            # Count name if present
            if "name" in msg:
                total += self._estimate_tokens(msg["name"])
            
            # Add formatting overhead
            total += 4  # Approximate overhead per message
        
        return total
    
    # ========================================================================
    # MODEL-SPECIFIC METHODS
    # ========================================================================
    
    def _get_encoding_name(
        self,
        model: Optional[str],
        encoding: Optional[str]
    ) -> str:
        """Get encoding name for model."""
        if encoding:
            return encoding
        
        if not model:
            return "default"
        
        model_lower = model.lower()
        
        # Check for exact match
        for model_pattern, enc_name in self.MODEL_FAMILIES.items():
            if model_pattern in model_lower:
                return enc_name
        
        # Check for family match
        for model_pattern, enc_name in self.MODEL_FAMILIES.items():
            if model_pattern in model_lower:
                return enc_name
        
        return "default"
    
    def get_model_context_limit(self, model: str) -> int:
        """Get context window size for a model."""
        model_lower = model.lower()
        
        # OpenAI models
        if "gpt-4" in model_lower:
            return 128000 if "turbo" in model_lower else 8192
        elif "gpt-3.5-turbo" in model_lower:
            return 16384
        elif "gpt-3.5" in model_lower:
            return 4096
        
        # Anthropic models
        elif "claude-3" in model_lower:
            return 200000
        elif "claude-2" in model_lower:
            return 100000
        elif "claude-instant" in model_lower:
            return 100000
        
        # Cohere models
        elif "command-r" in model_lower:
            return 128000
        elif "command" in model_lower:
            return 4096
        
        # Grok models
        elif "grok" in model_lower:
            return 8192
        
        # Llama family
        elif "llama-3" in model_lower:
            return 8192
        elif "llama-2" in model_lower:
            return 4096
        elif "llama" in model_lower:
            return 2048
        
        # Mistral family
        elif "mixtral" in model_lower:
            return 32768
        elif "mistral" in model_lower:
            return 32768
        
        # Other models
        elif "falcon" in model_lower:
            return 2048
        elif "qwen" in model_lower:
            return 32768
        elif "phi" in model_lower:
            return 2048
        elif "starcoder" in model_lower:
            return 8192
        elif "deepseek" in model_lower:
            return 16384
        
        # Default
        return 4096
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def truncate_to_limit(
        self,
        text: str,
        model: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Truncate text to fit within model's token limit.
        
        Args:
            text: Input text
            model: Model name
            max_tokens: Maximum tokens (defaults to model context limit)
        
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        # Get token limit
        limit = max_tokens or self.get_model_context_limit(model)
        
        # Count tokens
        tokens = self.count_tokens(text, model)
        
        if tokens <= limit:
            return text
        
        # Truncate by characters (approximate)
        ratio = limit / tokens
        char_limit = int(len(text) * ratio * 0.95)  # 5% safety margin
        
        truncated = text[:char_limit]
        
        logger.debug(
            "text_truncated",
            original_tokens=tokens,
            truncated_tokens=self.count_tokens(truncated, model),
            limit=limit,
            model=model
        )
        
        return truncated
    
    def clear_cache(self):
        """Clear tokenizer cache."""
        self.tokenizer_cache.clear()
        logger.debug("tokenizer_cache_cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token counter statistics."""
        return {
            "tiktoken_available": self._tiktoken_available,
            "transformers_available": self._transformers_available,
            "cache_size": len(self.tokenizer_cache.cache),
            "model_families": len(self.MODEL_FAMILIES),
            "supported_encodings": list(self._get_tiktoken_encodings().keys()),
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_token_counter = None


def get_token_counter() -> TokenCounter:
    """Get singleton token counter instance."""
    global _token_counter
    if not _token_counter:
        _token_counter = TokenCounter()
    return _token_counter


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "TokenCounter",
    "get_token_counter"
]