"""
Semantic Cache - Production Ready
Intelligent caching using sentence embeddings and similarity search
to return cached responses for semantically similar prompts.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer
import faiss
from redis.asyncio import Redis

from core.logging import get_logger
from core.exceptions import CacheError
from config import settings
from cache.redis_cache import get_redis_cache

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# SEMANTIC CACHE CONFIGURATION
# ============================================================================

@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache."""
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    embedding_dimension: int = 384
    
    # Similarity threshold (0-1)
    similarity_threshold: float = 0.85
    
    # Cache size limits
    max_entries: int = 10000
    max_embedding_cache_size: int = 5000
    
    # TTL settings
    default_ttl: int = 3600  # 1 hour
    negative_ttl: int = 300   # 5 minutes for negative cache
    
    # Performance settings
    batch_size: int = 32
    index_build_interval: int = 60  # seconds
    enable_negative_cache: bool = True
    
    # Index settings
    index_type: str = "flat"  # flat, ivf, hnsw
    nprobe: int = 10  # for IVF index
    hnsw_m: int = 16  # for HNSW index
    
    # Monitoring
    track_similarity_scores: bool = True
    enable_stats: bool = True


# ============================================================================
# SEMANTIC CACHE ENTRY
# ============================================================================

@dataclass
class SemanticCacheEntry:
    """Entry in semantic cache with embedding and metadata."""
    
    # Core data
    prompt: str
    response: str
    model: str
    
    # Embedding
    embedding: List[float] = field(default_factory=list)
    embedding_id: Optional[int] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Performance
    similarity_threshold: float = 0.85
    retrieval_latency_ms: Optional[float] = None
    
    # Usage tracking
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def record_access(self):
        """Record a cache access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "similarity_threshold": self.similarity_threshold,
            "tokens_saved": self.tokens_saved,
            "cost_saved_usd": self.cost_saved_usd
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticCacheEntry':
        """Create from dictionary."""
        entry = cls(
            prompt=data["prompt"],
            response=data["response"],
            model=data["model"],
            embedding=data["embedding"],
            similarity_threshold=data.get("similarity_threshold", 0.85),
            tokens_saved=data.get("tokens_saved", 0),
            cost_saved_usd=data.get("cost_saved_usd", 0.0)
        )
        entry.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            entry.expires_at = datetime.fromisoformat(data["expires_at"])
        entry.access_count = data.get("access_count", 0)
        if data.get("last_accessed"):
            entry.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return entry


# ============================================================================
# EMBEDDING MODEL MANAGER
# ============================================================================

class EmbeddingModelManager:
    """
    Manages sentence transformer models for embeddings.
    
    Features:
    - Lazy model loading
    - Multiple model support
    - GPU/CPU fallback
    - Batch encoding
    - Caching
    """
    
    def __init__(self):
        self.models: Dict[str, SentenceTransformer] = {}
        self.default_model = settings.cache.embedding_model
        self.device = self._detect_device()
        
        logger.info(
            "embedding_model_manager_initialized",
            default_model=self.default_model,
            device=self.device
        )
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
        except ImportError:
            pass
        return "cpu"
    
    def get_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """Get or load embedding model."""
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            logger.info(f"Loading embedding model: {model_name}")
            
            # Load model
            model = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if self.device != "cpu":
                model = model.to(self.device)
            
            self.models[model_name] = model
            
            logger.info(f"Embedding model loaded: {model_name}")
        
        return self.models[model_name]
    
    async def encode(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            model_name: Optional model name
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        
        Returns:
            numpy array of embeddings
        """
        model = self.get_model(model_name)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
        )
        
        return embeddings
    
    async def encode_single(
        self,
        text: str,
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode a single text."""
        embeddings = await self.encode([text], model_name, 1, normalize)
        return embeddings[0]


# ============================================================================
# FAISS INDEX MANAGER
# ============================================================================

class FaissIndexManager:
    """
    Manages FAISS index for similarity search.
    
    Features:
    - Multiple index types (Flat, IVF, HNSW)
    - Incremental updates
    - Persistence
    - GPU support (optional)
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nprobe: int = 10,
        hnsw_m: int = 16
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        
        self.index: Optional[faiss.Index] = None
        self.id_to_key: Dict[int, str] = {}
        self.key_to_id: Dict[str, int] = {}
        self.next_id: int = 0
        
        # GPU support
        self.use_gpu = self._check_gpu_support()
        
        self._initialize_index()
        
        logger.info(
            "faiss_index_initialized",
            dimension=dimension,
            index_type=index_type,
            use_gpu=self.use_gpu
        )
    
    def _check_gpu_support(self) -> bool:
        """Check if FAISS GPU is available."""
        try:
            if faiss.get_num_gpus() > 0:
                return True
        except:
            pass
        return False
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        if self.index_type == "flat":
            # Simple flat index - exact search, slower for large datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "ivf":
            # IVF index - approximate, faster for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                100,  # nlist - number of clusters
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.nprobe
            
        elif self.index_type == "hnsw":
            # HNSW index - graph-based, good balance
            self.index = faiss.IndexHNSWFlat(
                self.dimension,
                self.hnsw_m
            )
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
        
        # Move to GPU if available
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,  # GPU device
                self.index
            )
    
    def add(self, key: str, embedding: np.ndarray) -> int:
        """Add embedding to index."""
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for inner product similarity
        faiss.normalize_L2(embedding)
        
        # Get ID for this key
        if key in self.key_to_id:
            entry_id = self.key_to_id[key]
            # Update existing entry
            self.index.remove_ids(np.array([entry_id]))
        else:
            entry_id = self.next_id
            self.next_id += 1
            self.key_to_id[key] = entry_id
            self.id_to_key[entry_id] = key
        
        # Add to index
        self.index.add_with_ids(embedding, np.array([entry_id]))
        
        return entry_id
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Search for similar embeddings."""
        # Ensure query is 2D and normalized
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Map indices to keys
        keys = []
        for idx in indices[0]:
            if idx != -1:
                keys.append(self.id_to_key.get(idx, "unknown"))
            else:
                keys.append(None)
        
        return scores[0], indices[0], keys
    
    def remove(self, key: str) -> bool:
        """Remove embedding from index."""
        if key in self.key_to_id:
            entry_id = self.key_to_id[key]
            self.index.remove_ids(np.array([entry_id]))
            del self.key_to_id[key]
            del self.id_to_key[entry_id]
            return True
        return False
    
    def size(self) -> int:
        """Get number of entries in index."""
        return self.index.ntotal
    
    def reset(self):
        """Reset index."""
        self._initialize_index()
        self.id_to_key.clear()
        self.key_to_id.clear()
        self.next_id = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            "size": self.size(),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu,
            "next_id": self.next_id
        }
        
        if self.index_type == "ivf":
            stats["nprobe"] = self.nprobe
        elif self.index_type == "hnsw":
            stats["hnsw_m"] = self.hnsw_m
            if hasattr(self.index, 'hnsw'):
                stats["ef_construction"] = self.index.hnsw.efConstruction
                stats["ef_search"] = self.index.hnsw.efSearch
        
        return stats


# ============================================================================
# SEMANTIC CACHE
# ============================================================================

class SemanticCache:
    """
    Intelligent semantic cache using embeddings and similarity search.
    
    Features:
    - Semantic similarity matching
    - Multiple embedding models
    - FAISS vector index
    - Redis persistence
    - Negative caching
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        self.config = config or SemanticCacheConfig()
        
        # Embedding model
        self.embedding_manager = EmbeddingModelManager()
        
        # FAISS index
        self.index = FaissIndexManager(
            dimension=self.config.embedding_dimension,
            index_type=self.config.index_type,
            nprobe=self.config.nprobe,
            hnsw_m=self.config.hnsw_m
        )
        
        # Redis cache for persistence
        self.redis_cache = get_redis_cache()
        self.redis_prefix = "llm:semantic:"
        
        # Negative cache (prompts with no good matches)
        self.negative_cache: Dict[str, datetime] = {}
        self.negative_cache_ttl = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "hits": 0,
            "misses": 0,
            "negative_hits": 0,
            "avg_similarity": 0.0,
            "total_latency_ms": 0,
            "cache_size": 0,
            "last_build_time": None
        }
        
        # Background tasks
        self._build_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(
            "semantic_cache_initialized",
            embedding_model=self.config.embedding_model,
            similarity_threshold=self.config.similarity_threshold,
            max_entries=self.config.max_entries,
            index_type=self.config.index_type
        )
    
    # ========================================================================
    # INITIALIZATION & CLEANUP
    # ========================================================================
    
    async def initialize(self):
        """Initialize semantic cache."""
        # Start background tasks
        self._build_task = asyncio.create_task(self._periodic_index_build())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Load existing cache from Redis
        await self._load_from_redis()
        
        logger.info("semantic_cache_ready")
    
    async def close(self):
        """Close semantic cache."""
        if self._build_task:
            self._build_task.cancel()
            try:
                await self._build_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save to Redis before closing
        await self._save_to_redis()
        
        logger.info("semantic_cache_closed")
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    async def get(
        self,
        prompt: str,
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        max_results: int = 1
    ) -> Dict[str, Any]:
        """
        Get cached response for semantically similar prompt.
        
        Args:
            prompt: Input prompt
            model: Optional model filter
            threshold: Similarity threshold (0-1)
            max_results: Maximum number of results to return
        
        Returns:
            Dict with hit status, response, similarity, etc.
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Use configured threshold if not specified
        threshold = threshold or self.config.similarity_threshold
        
        # Check negative cache first
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        if prompt_hash in self.negative_cache:
            if datetime.utcnow() < self.negative_cache[prompt_hash]:
                self.stats["negative_hits"] += 1
                logger.debug("negative_cache_hit", prompt_hash=prompt_hash)
                return {
                    "hit": False,
                    "cache_type": "negative",
                    "similarity": 0.0,
                    "response": None,
                    "model": None,
                    "latency_ms": (time.time() - start_time) * 1000
                }
        
        # Generate embedding for query
        query_embedding = await self.embedding_manager.encode_single(prompt)
        
        # Search in FAISS index
        scores, indices, keys = self.index.search(query_embedding, k=max_results * 2)
        
        # Filter results by threshold and model
        results = []
        for score, idx, key in zip(scores, indices, keys):
            if score < threshold:
                continue
            
            if key and key != "unknown":
                # Get full entry from Redis
                entry_data = await self.redis_cache.get(f"{self.redis_prefix}{key}")
                
                if entry_data:
                    entry = SemanticCacheEntry.from_dict(
                        json.loads(entry_data.decode('utf-8'))
                    )
                    
                    # Check if expired
                    if entry.is_expired():
                        await self.delete(key)
                        continue
                    
                    # Filter by model if specified
                    if model and entry.model != model:
                        continue
                    
                    entry.record_access()
                    results.append({
                        "key": key,
                        "response": entry.response,
                        "model": entry.model,
                        "similarity": float(score),
                        "tokens_saved": entry.tokens_saved,
                        "cost_saved_usd": entry.cost_saved_usd,
                        "entry": entry
                    })
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Update statistics
        latency_ms = (time.time() - start_time) * 1000
        self.stats["total_latency_ms"] += latency_ms
        
        if results:
            self.stats["hits"] += 1
            self.stats["avg_similarity"] = (
                (self.stats["avg_similarity"] * (self.stats["hits"] - 1) + results[0]["similarity"]) /
                self.stats["hits"]
            )
            
            logger.debug(
                "semantic_cache_hit",
                similarity=results[0]["similarity"],
                threshold=threshold,
                latency_ms=round(latency_ms, 2)
            )
            
            return {
                "hit": True,
                "cache_type": "semantic",
                "similarity": results[0]["similarity"],
                "response": results[0]["response"],
                "model": results[0]["model"],
                "key": results[0]["key"],
                "all_results": results[:max_results],
                "latency_ms": round(latency_ms, 2)
            }
        else:
            self.stats["misses"] += 1
            
            # Add to negative cache
            if self.config.enable_negative_cache:
                self.negative_cache[prompt_hash] = datetime.utcnow() + timedelta(
                    seconds=self.config.negative_ttl
                )
            
            logger.debug(
                "semantic_cache_miss",
                threshold=threshold,
                latency_ms=round(latency_ms, 2)
            )
            
            return {
                "hit": False,
                "cache_type": "miss",
                "similarity": 0.0,
                "response": None,
                "model": None,
                "latency_ms": round(latency_ms, 2)
            }
    
    async def set(
        self,
        prompt: str,
        response: str,
        model: str,
        ttl: Optional[int] = None,
        tokens_saved: int = 0,
        cost_saved_usd: float = 0.0
    ) -> bool:
        """
        Cache a response with its embedding.
        
        Args:
            prompt: Original prompt
            response: Generated response
            model: Model that generated response
            ttl: Time to live in seconds
            tokens_saved: Number of tokens saved by caching
            cost_saved_usd: Cost saved in USD
        
        Returns:
            True if cached successfully
        """
        try:
            # Check cache size limit
            if self.index.size() >= self.config.max_entries:
                # TODO: Implement LRU eviction
                logger.warning("Semantic cache at capacity, skipping")
                return False
            
            # Generate embedding
            embedding = await self.embedding_manager.encode_single(prompt)
            embedding_list = embedding.tolist()
            
            # Create cache entry
            entry = SemanticCacheEntry(
                prompt=prompt,
                response=response,
                model=model,
                embedding=embedding_list,
                similarity_threshold=self.config.similarity_threshold,
                tokens_saved=tokens_saved,
                cost_saved_usd=cost_saved_usd
            )
            
            if ttl:
                entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Generate cache key
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            key = f"{prompt_hash}:{model}"
            
            # Add to FAISS index
            embedding_array = np.array(embedding_list, dtype=np.float32)
            entry.embedding_id = self.index.add(key, embedding_array)
            
            # Store in Redis
            await self.redis_cache.set(
                f"{self.redis_prefix}{key}",
                json.dumps(entry.to_dict()),
                ttl=ttl or self.config.default_ttl
            )
            
            # Remove from negative cache if present
            if prompt_hash in self.negative_cache:
                del self.negative_cache[prompt_hash]
            
            # Update statistics
            self.stats["cache_size"] = self.index.size()
            
            logger.debug(
                "semantic_cache_set",
                key=key,
                model=model,
                ttl=ttl,
                tokens_saved=tokens_saved,
                cost_saved_usd=cost_saved_usd
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set semantic cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        try:
            # Remove from FAISS index
            self.index.remove(key)
            
            # Remove from Redis
            await self.redis_cache.delete(f"{self.redis_prefix}{key}")
            
            # Update statistics
            self.stats["cache_size"] = self.index.size()
            
            logger.debug("semantic_cache_deleted", key=key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete semantic cache: {e}")
            return False
    
    async def clear(self) -> int:
        """Clear all entries from cache."""
        try:
            # Reset FAISS index
            self.index.reset()
            
            # Clear Redis
            pattern = f"{self.redis_prefix}*"
            keys = await self.redis_cache.keys(pattern)
            if keys:
                await self.redis_cache.delete(*keys)
            
            # Clear negative cache
            self.negative_cache.clear()
            
            # Reset statistics
            self.stats["cache_size"] = 0
            self.stats["hits"] = 0
            self.stats["misses"] = 0
            self.stats["negative_hits"] = 0
            self.stats["avg_similarity"] = 0.0
            
            logger.info("semantic_cache_cleared", keys_removed=len(keys) if keys else 0)
            return len(keys) if keys else 0
            
        except Exception as e:
            logger.error(f"Failed to clear semantic cache: {e}")
            return 0
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    async def get_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get cached responses for multiple prompts."""
        # Generate embeddings in batch
        embeddings = await self.embedding_manager.encode(
            prompts,
            batch_size=self.config.batch_size
        )
        
        results = []
        for i, (prompt, embedding) in enumerate(zip(prompts, embeddings)):
            # Search for each embedding
            scores, indices, keys = self.index.search(embedding, k=1)
            
            if scores[0] >= (threshold or self.config.similarity_threshold):
                key = keys[0]
                if key and key != "unknown":
                    entry_data = await self.redis_cache.get(f"{self.redis_prefix}{key}")
                    if entry_data:
                        entry = SemanticCacheEntry.from_dict(
                            json.loads(entry_data.decode('utf-8'))
                        )
                        results.append({
                            "prompt": prompt,
                            "index": i,
                            "hit": True,
                            "similarity": float(scores[0]),
                            "response": entry.response,
                            "model": entry.model
                        })
                        continue
            
            results.append({
                "prompt": prompt,
                "index": i,
                "hit": False,
                "similarity": 0.0,
                "response": None,
                "model": None
            })
        
        return results
    
    async def set_batch(
        self,
        items: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> int:
        """Cache multiple responses in batch."""
        successful = 0
        
        for item in items:
            success = await self.set(
                prompt=item["prompt"],
                response=item["response"],
                model=item["model"],
                ttl=ttl,
                tokens_saved=item.get("tokens_saved", 0),
                cost_saved_usd=item.get("cost_saved_usd", 0.0)
            )
            if success:
                successful += 1
        
        return successful
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    async def _save_to_redis(self):
        """Save cache metadata to Redis."""
        try:
            metadata = {
                "stats": self.stats,
                "negative_cache": {
                    k: v.isoformat()
                    for k, v in self.negative_cache.items()
                },
                "index_stats": self.index.get_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_cache.set(
                f"{self.redis_prefix}_metadata",
                json.dumps(metadata)
            )
            
            logger.debug("semantic_cache_metadata_saved")
            
        except Exception as e:
            logger.error(f"Failed to save semantic cache metadata: {e}")
    
    async def _load_from_redis(self):
        """Load cache metadata from Redis."""
        try:
            metadata_data = await self.redis_cache.get(f"{self.redis_prefix}_metadata")
            
            if metadata_data:
                metadata = json.loads(metadata_data.decode('utf-8'))
                
                # Restore statistics (but keep current session counts)
                current_hits = self.stats["hits"]
                current_misses = self.stats["misses"]
                self.stats = metadata.get("stats", self.stats)
                self.stats["hits"] += current_hits
                self.stats["misses"] += current_misses
                
                # Restore negative cache
                negative_cache = metadata.get("negative_cache", {})
                for k, v in negative_cache.items():
                    self.negative_cache[k] = datetime.fromisoformat(v)
                
                logger.info(
                    "semantic_cache_metadata_loaded",
                    stats=self.stats,
                    negative_cache_size=len(self.negative_cache)
                )
            
        except Exception as e:
            logger.error(f"Failed to load semantic cache metadata: {e}")
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
    async def _periodic_index_build(self):
        """Periodically rebuild/optimize FAISS index."""
        while True:
            try:
                await asyncio.sleep(self.config.index_build_interval)
                
                if self.index.index_type == "ivf" and self.index.size() > 100:
                    # Train IVF index
                    logger.info("Training IVF index...")
                    # TODO: Implement IVF training
                
                self.stats["last_build_time"] = datetime.utcnow().isoformat()
                
                # Save metadata
                await self._save_to_redis()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Index build failed: {e}")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clean up negative cache
                now = datetime.utcnow()
                expired_negative = [
                    k for k, v in self.negative_cache.items()
                    if v < now
                ]
                for k in expired_negative:
                    del self.negative_cache[k]
                
                # TODO: Clean up expired Redis entries
                # This would require scanning all keys
                
                if expired_negative:
                    logger.debug(
                        "semantic_cache_cleanup",
                        expired_negative=len(expired_negative)
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """Compute semantic similarity between two prompts."""
        embeddings = await self.embedding_manager.encode([prompt1, prompt2])
        
        # Normalize and compute cosine similarity
        faiss.normalize_L2(embeddings)
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return float(similarity)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )
        
        avg_latency = (
            self.stats["total_latency_ms"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0
            else 0
        )
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "negative_hits": self.stats["negative_hits"],
            "hit_rate": round(hit_rate * 100, 2),
            "avg_similarity": round(self.stats["avg_similarity"], 3),
            "avg_latency_ms": round(avg_latency, 2),
            "cache_size": self.index.size(),
            "max_size": self.config.max_entries,
            "usage_percent": (self.index.size() / self.config.max_entries * 100),
            "negative_cache_size": len(self.negative_cache),
            "index": self.index.get_stats(),
            "config": {
                "embedding_model": self.config.embedding_model,
                "similarity_threshold": self.config.similarity_threshold,
                "default_ttl": self.config.default_ttl,
                "max_entries": self.config.max_entries,
                "index_type": self.config.index_type
            },
            "last_build_time": self.stats["last_build_time"]
        }
    
    async def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_queries": 0,
            "hits": 0,
            "misses": 0,
            "negative_hits": 0,
            "avg_similarity": 0.0,
            "total_latency_ms": 0,
            "cache_size": self.index.size(),
            "last_build_time": None
        }
        logger.info("semantic_cache_stats_reset")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_semantic_cache = None


def get_semantic_cache() -> SemanticCache:
    """Get singleton semantic cache instance."""
    global _semantic_cache
    if not _semantic_cache:
        _semantic_cache = SemanticCache()
    return _semantic_cache


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SemanticCache",
    "SemanticCacheConfig",
    "SemanticCacheEntry",
    "EmbeddingModelManager",
    "FaissIndexManager",
    "get_semantic_cache"
]