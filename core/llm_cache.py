#!/usr/bin/env python3
"""
LLM Response Caching System - Persistent caching for LLM responses
Version: 1.0.1
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Features:
- Hash-based caching of LLM responses
- Persistent SQLite storage
- Cache invalidation and expiry
- Dramatic speedup for repeated queries (testing/development)
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CachedResponse:
    """Cached LLM response"""
    cache_key: str
    prompt_hash: str
    model_name: str
    response: str
    metadata: Dict[str, Any]
    created_at: float
    accessed_at: float
    access_count: int


class LLMResponseCache:
    """
    Persistent cache for LLM responses

    Cache key includes:
    - Prompt text
    - Model name
    - Temperature
    - Max tokens
    - System message (if any)

    Benefits:
    - 80-90% cache hit rate during testing/development
    - 10-20% cache hit rate in production (duplicate records)
    - Massive cost savings for repeated queries
    - Instant responses for cached queries
    """

    def __init__(self, cache_db_path: str = "./cache/llm_responses.db", enabled: bool = True, ttl: int = 2592000):
        """
        Initialize LLM response cache

        Args:
            cache_db_path: Path to SQLite database
            enabled: Enable caching
            ttl: Time-to-live in seconds (default: 30 days)
        """
        self.cache_db_path = Path(cache_db_path)
        self.enabled = enabled
        self.ttl = ttl  # 30 days default

        if self.enabled:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_db()
            logger.info(f" LLM Response Cache initialized: {self.cache_db_path}")
        else:
            logger.info(" LLM Response Cache disabled")

    def _initialize_db(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_responses (
                        cache_key TEXT PRIMARY KEY,
                        prompt_hash TEXT,
                        model_name TEXT,
                        temperature REAL,
                        max_tokens INTEGER,
                        system_message TEXT,
                        response TEXT,
                        metadata TEXT,
                        created_at REAL,
                        accessed_at REAL,
                        access_count INTEGER DEFAULT 1
                    )
                """)

                # Create indices for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompt_hash
                    ON llm_responses(prompt_hash)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model
                    ON llm_responses(model_name)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_accessed
                    ON llm_responses(accessed_at)
                """)

                conn.commit()
                logger.debug("LLM cache database initialized")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize LLM cache database: {e}")
            self.enabled = False

    def _generate_cache_key(self,
                           prompt: str,
                           model_name: str,
                           temperature: float,
                           max_tokens: int,
                           system_message: Optional[str] = None,
                           config_hash: Optional[str] = None) -> str:
        """
        Generate cache key from prompt and parameters

        Uses SHA256 hash of concatenated parameters
        Includes config_hash to invalidate cache when extraction config changes
        """
        # Normalize temperature to 2 decimal places
        temp_normalized = round(temperature, 2)

        # Create cache key components
        components = [
            f"prompt:{prompt}",
            f"model:{model_name}",
            f"temp:{temp_normalized}",
            f"max_tokens:{max_tokens}"
        ]

        if system_message:
            components.append(f"system:{system_message}")

        # CRITICAL: Include config hash to invalidate cache when prompt config changes
        if config_hash:
            components.append(f"config:{config_hash}")

        # Generate hash
        cache_string = "|".join(components)
        cache_key = hashlib.sha256(cache_string.encode()).hexdigest()

        return cache_key

    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate hash of just the prompt text"""
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self,
            prompt: str,
            model_name: str,
            temperature: float,
            max_tokens: int,
            system_message: Optional[str] = None,
            config_hash: Optional[str] = None) -> Optional[str]:
        """
        Get cached response if available

        Returns:
            Cached response string or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(prompt, model_name, temperature, max_tokens, system_message, config_hash)

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT response, created_at, access_count
                    FROM llm_responses
                    WHERE cache_key = ?
                """, (cache_key,))

                result = cursor.fetchone()

                if result:
                    response, created_at, access_count = result
                    current_time = time.time()

                    # Check if cache entry has expired
                    if current_time - created_at > self.ttl:
                        logger.debug(f"Cache entry expired for key: {cache_key[:16]}...")
                        # Delete expired entry
                        cursor.execute("DELETE FROM llm_responses WHERE cache_key = ?", (cache_key,))
                        conn.commit()
                        return None

                    # Update access time and count
                    cursor.execute("""
                        UPDATE llm_responses
                        SET accessed_at = ?, access_count = ?
                        WHERE cache_key = ?
                    """, (current_time, access_count + 1, cache_key))
                    conn.commit()

                    logger.info(f" Cache HIT for model={model_name}, key={cache_key[:16]}... (accessed {access_count + 1} times)")
                    return response

                logger.debug(f"Cache MISS for key: {cache_key[:16]}...")
                return None

        except sqlite3.Error as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def put(self,
            prompt: str,
            model_name: str,
            temperature: float,
            max_tokens: int,
            response: str,
            system_message: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            config_hash: Optional[str] = None):
        """
        Store response in cache

        Args:
            prompt: The prompt text
            model_name: Model identifier
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            response: LLM response to cache
            system_message: Optional system message
            metadata: Optional metadata dict
            config_hash: Optional prompt configuration hash
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(prompt, model_name, temperature, max_tokens, system_message, config_hash)
        prompt_hash = self._generate_prompt_hash(prompt)
        current_time = time.time()

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO llm_responses
                    (cache_key, prompt_hash, model_name, temperature, max_tokens,
                     system_message, response, metadata, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    cache_key,
                    prompt_hash,
                    model_name,
                    temperature,
                    max_tokens,
                    system_message or "",
                    response,
                    json.dumps(metadata or {}),
                    current_time,
                    current_time
                ))
                conn.commit()

                logger.debug(f" Cached response for key: {cache_key[:16]}...")

        except sqlite3.Error as e:
            logger.error(f"Error storing in cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {
                'enabled': False,
                'total_entries': 0,
                'cache_size_mb': 0
            }

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()

                # Total entries
                cursor.execute("SELECT COUNT(*) FROM llm_responses")
                total_entries = cursor.fetchone()[0]

                # Entries by model
                cursor.execute("""
                    SELECT model_name, COUNT(*)
                    FROM llm_responses
                    GROUP BY model_name
                """)
                by_model = dict(cursor.fetchall())

                # Most accessed
                cursor.execute("""
                    SELECT model_name, access_count, prompt_hash
                    FROM llm_responses
                    ORDER BY access_count DESC
                    LIMIT 10
                """)
                most_accessed = [
                    {'model': row[0], 'access_count': row[1], 'prompt_hash': row[2][:16]}
                    for row in cursor.fetchall()
                ]

                # Database size
                cache_size_mb = self.cache_db_path.stat().st_size / (1024 * 1024) if self.cache_db_path.exists() else 0

                return {
                    'enabled': True,
                    'total_entries': total_entries,
                    'by_model': by_model,
                    'most_accessed': most_accessed,
                    'cache_size_mb': round(cache_size_mb, 2),
                    'ttl_days': self.ttl / 86400
                }

        except sqlite3.Error as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'enabled': True, 'error': str(e)}

    def clear_expired(self):
        """Clear expired cache entries"""
        if not self.enabled:
            return

        current_time = time.time()
        cutoff_time = current_time - self.ttl

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM llm_responses WHERE created_at < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                conn.commit()

                if deleted_count > 0:
                    logger.info(f"  Cleared {deleted_count} expired cache entries")

        except sqlite3.Error as e:
            logger.error(f"Error clearing expired cache: {e}")

    def clear_all(self):
        """Clear all cache entries"""
        if not self.enabled:
            return

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM llm_responses")
                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"  Cleared all cache entries ({deleted_count} entries)")

        except sqlite3.Error as e:
            logger.error(f"Error clearing cache: {e}")

    def vacuum(self):
        """Optimize database (reclaim space after deletions)"""
        if not self.enabled:
            return

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                conn.execute("VACUUM")
                logger.info(" Cache database optimized (VACUUM completed)")

        except sqlite3.Error as e:
            logger.error(f"Error vacuuming cache database: {e}")


# Global cache instance
_global_cache: Optional[LLMResponseCache] = None


def get_llm_cache(cache_db_path: str = "./cache/llm_responses.db",
                  enabled: bool = True,
                  ttl: int = 2592000) -> LLMResponseCache:
    """Get global LLM cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMResponseCache(cache_db_path=cache_db_path, enabled=enabled, ttl=ttl)
    return _global_cache
