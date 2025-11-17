#!/usr/bin/env python3
"""
Adaptive Retry System for LLM Generation Failures

Implements progressive input size reduction and intelligent fallback strategies
to recover from LLM generation failures across all pipeline modes.

Features:
- Progressive clinical note truncation
- Conversation history trimming
- Tool results context reduction
- Automatic minimal prompt fallback
- Exponential backoff for transient errors

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from core.logging_config import get_logger

logger = get_logger(__name__)

# Optional: Smart context preservation with embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - smart context preservation disabled")


class RetryStrategy(Enum):
    """Retry strategies for different failure scenarios"""
    TRANSIENT_ERROR = "transient_error"  # Network/API errors - exponential backoff
    CONTEXT_TOO_LARGE = "context_too_large"  # Input too long - progressive reduction
    INVALID_OUTPUT = "invalid_output"  # LLM output malformed - minimal prompt
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"  # All retries exhausted


@dataclass
class RetryContext:
    """Context for adaptive retry operations"""
    attempt: int = 0
    max_attempts: int = 5
    clinical_text: str = ""
    conversation_history: List[Any] = None
    tool_results: List[Any] = None
    original_clinical_text_length: int = 0
    current_clinical_text_length: int = 0
    history_reduction_level: int = 0
    tool_context_reduction_level: int = 0
    switched_to_minimal: bool = False
    last_error: Optional[str] = None

    # Metrics tracking
    extraction_id: str = ""
    provider: str = ""
    model_name: str = ""
    start_time: float = 0.0

    # Config-driven settings
    reduction_ratios: List[float] = None
    history_levels: List[int] = None
    tool_levels: List[int] = None
    switch_to_minimal_at: int = 4
    preserve_beginning_ratio: float = 0.6
    preserve_ending_ratio: float = 0.4

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.tool_results is None:
            self.tool_results = []
        if not self.extraction_id:
            self.extraction_id = str(uuid.uuid4())
        if self.start_time == 0.0:
            self.start_time = time.time()


class AdaptiveRetryManager:
    """
    Manages adaptive retry strategies for LLM generation failures

    Progressive reduction strategy (configurable):
    1. First retry: Reduce clinical text based on config
    2. Second retry: Further reduction + trim conversation history
    3. Third retry: Aggressive reduction + tool context reduction
    4. Fourth retry: Maximum reduction + switch to minimal prompt
    5. Fifth retry: Emergency fallback - bare minimum context
    """

    def __init__(
        self,
        max_retries: int = 5,
        config: Optional[Any] = None,
        metrics_tracker: Optional[Any] = None
    ):
        """
        Initialize adaptive retry manager

        Args:
            max_retries: Maximum number of retry attempts
            config: AdaptiveRetryConfig instance (optional)
            metrics_tracker: RetryMetricsTracker instance (optional)
        """
        self.max_retries = max_retries
        self.config = config
        self.metrics_tracker = metrics_tracker

        # Smart context preservation
        self.smart_model = None
        if config and config.use_smart_context_preservation:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.smart_model = SentenceTransformer(config.smart_preservation_model)
                    logger.info(f"Smart context preservation enabled with {config.smart_preservation_model}")
                except Exception as e:
                    logger.warning(f"Failed to load smart preservation model: {e}")
            else:
                logger.warning("Smart context preservation requested but sentence-transformers not available")

        logger.info(f"AdaptiveRetryManager initialized with max_retries={max_retries}")

    def execute_with_retry(
        self,
        generation_func: Callable,
        retry_context: RetryContext,
        error_handler: Optional[Callable] = None
    ) -> Any:
        """
        Execute LLM generation with adaptive retry

        Args:
            generation_func: Function to call for LLM generation
            retry_context: Retry context with current state
            error_handler: Optional custom error handler

        Returns:
            LLM generation result

        Raises:
            Exception: If all retries are exhausted
        """
        from core.retry_metrics import RetryAttemptMetrics, ExtractionRetryMetrics

        last_exception = None

        # Initialize metrics tracking
        extraction_metrics = None
        if self.metrics_tracker:
            extraction_metrics = ExtractionRetryMetrics(
                extraction_id=retry_context.extraction_id,
                provider=retry_context.provider,
                model_name=retry_context.model_name,
                original_text_length=retry_context.original_clinical_text_length
            )

        for attempt in range(1, self.max_retries + 1):
            retry_context.attempt = attempt
            attempt_start_time = time.time()

            try:
                logger.info(f"=== ADAPTIVE RETRY ATTEMPT {attempt}/{self.max_retries} ===")

                # Apply adaptive strategy before each attempt
                if attempt > 1:
                    self._apply_adaptive_strategy(retry_context, attempt)

                # Execute generation
                result = generation_func()

                # Success!
                logger.info(f"✓ Generation successful on attempt {attempt}")

                # Record successful attempt
                if extraction_metrics:
                    attempt_metrics = RetryAttemptMetrics(
                        attempt_number=attempt,
                        success=True,
                        clinical_text_length=len(retry_context.clinical_text),
                        conversation_history_length=len(retry_context.conversation_history),
                        tool_context_length=len(retry_context.tool_results),
                        switched_to_minimal=retry_context.switched_to_minimal
                    )
                    extraction_metrics.add_attempt(attempt_metrics)
                    extraction_metrics.final_text_length = len(retry_context.clinical_text)
                    extraction_metrics.total_retry_time_seconds = time.time() - retry_context.start_time
                    self.metrics_tracker.record_extraction(extraction_metrics)

                return result

            except Exception as e:
                last_exception = e
                retry_context.last_error = str(e)

                logger.warning(f"✗ Attempt {attempt} failed: {e}")

                # Record failed attempt
                if extraction_metrics:
                    attempt_metrics = RetryAttemptMetrics(
                        attempt_number=attempt,
                        success=False,
                        error_type=type(e).__name__,
                        clinical_text_length=len(retry_context.clinical_text),
                        conversation_history_length=len(retry_context.conversation_history),
                        tool_context_length=len(retry_context.tool_results),
                        switched_to_minimal=retry_context.switched_to_minimal
                    )
                    extraction_metrics.add_attempt(attempt_metrics)

                # Determine retry strategy
                strategy = self._classify_error(e)

                # Custom error handler
                if error_handler:
                    should_continue = error_handler(retry_context, e, strategy)
                    if not should_continue:
                        logger.error("Error handler requested stop - aborting retries")
                        raise

                # Check if we should retry
                if attempt >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exhausted")
                    break

                # Apply exponential backoff for transient errors
                if self.config and self.config.enable_exponential_backoff and strategy == RetryStrategy.TRANSIENT_ERROR:
                    backoff_time = min(
                        self.config.backoff_base_seconds * (2 ** (attempt - 1)),
                        self.config.backoff_max_seconds
                    )
                    logger.info(f"Transient error detected - backing off {backoff_time:.1f}s before retry")
                    time.sleep(backoff_time)
                elif strategy == RetryStrategy.TRANSIENT_ERROR:
                    # Default backoff
                    backoff_time = min(2 ** (attempt - 1), 16)
                    logger.info(f"Transient error detected - backing off {backoff_time}s before retry")
                    time.sleep(backoff_time)
                else:
                    # Small delay for other errors
                    time.sleep(0.5)

        # All retries exhausted - record final failure
        if extraction_metrics:
            extraction_metrics.successful = False
            extraction_metrics.total_retry_time_seconds = time.time() - retry_context.start_time
            if self.metrics_tracker:
                self.metrics_tracker.record_extraction(extraction_metrics)

        logger.error(f"All {self.max_retries} retry attempts failed")
        logger.error(f"Final error: {retry_context.last_error}")
        raise last_exception

    def _classify_error(self, error: Exception) -> RetryStrategy:
        """
        Classify error type to determine retry strategy

        Args:
            error: Exception that occurred

        Returns:
            RetryStrategy enum
        """
        error_str = str(error).lower()

        # Network/API errors - retry with backoff
        transient_indicators = [
            'timeout', 'connection', 'network', 'unavailable',
            'rate limit', '429', '503', '502', '500'
        ]
        if any(indicator in error_str for indicator in transient_indicators):
            return RetryStrategy.TRANSIENT_ERROR

        # Context too large - reduce input
        context_indicators = [
            'context length', 'token limit', 'too long', 'maximum context',
            'context window', 'input too large'
        ]
        if any(indicator in error_str for indicator in context_indicators):
            return RetryStrategy.CONTEXT_TOO_LARGE

        # Invalid output - try minimal prompt
        output_indicators = [
            'parse', 'json', 'invalid', 'malformed', 'schema'
        ]
        if any(indicator in error_str for indicator in output_indicators):
            return RetryStrategy.INVALID_OUTPUT

        # Default to context reduction
        return RetryStrategy.CONTEXT_TOO_LARGE

    def _apply_adaptive_strategy(self, context: RetryContext, attempt: int):
        """
        Apply adaptive strategy based on retry attempt (config-driven)

        Args:
            context: Current retry context
            attempt: Current attempt number (1-indexed)
        """
        logger.info(f"Applying adaptive strategy for attempt {attempt}")

        # Get reduction strategy from context or use defaults
        ratio_index = attempt - 2  # Attempts 2+ use reduction
        if ratio_index >= 0:
            # Get reduction ratio
            if context.reduction_ratios and ratio_index < len(context.reduction_ratios):
                ratio = context.reduction_ratios[ratio_index]
            else:
                # Fallback to default progressive reduction
                default_ratios = [0.8, 0.6, 0.4, 0.2]
                ratio = default_ratios[min(ratio_index, len(default_ratios) - 1)]

            # Truncate clinical text
            context.clinical_text = self._truncate_clinical_text(
                context.clinical_text,
                ratio,
                use_smart=self.smart_model is not None,
                context_obj=context
            )
            logger.info(f"  → Clinical text reduced to {ratio*100:.0f}% ({len(context.clinical_text)} chars)")

            # Get history reduction level
            if context.history_levels and ratio_index < len(context.history_levels):
                context.history_reduction_level = context.history_levels[ratio_index]
            elif ratio_index == 0:
                context.history_reduction_level = 0
            elif ratio_index == 1:
                context.history_reduction_level = 1
            elif ratio_index == 2:
                context.history_reduction_level = 2
            else:
                context.history_reduction_level = 3

            if context.history_reduction_level > 0:
                logger.info(f"  → Conversation history trimming: level {context.history_reduction_level}")

            # Get tool context reduction level
            if context.tool_levels and ratio_index < len(context.tool_levels):
                context.tool_context_reduction_level = context.tool_levels[ratio_index]
            elif ratio_index >= 2:
                context.tool_context_reduction_level = ratio_index - 1

            if context.tool_context_reduction_level > 0:
                logger.info(f"  → Tool context reduction: level {context.tool_context_reduction_level}")

            # Switch to minimal prompt at configured attempt
            if attempt >= context.switch_to_minimal_at and not context.switched_to_minimal:
                context.switched_to_minimal = True
                logger.info(f"  → Switching to MINIMAL PROMPT")

            # Emergency warnings for final attempts
            if attempt >= self.max_retries - 1:
                logger.warning(f"  → EMERGENCY FALLBACK MODE")
                logger.warning(f"  → Maximum context reduction applied")

    def _truncate_clinical_text(
        self,
        text: str,
        ratio: float,
        use_smart: bool = False,
        context_obj: Optional[RetryContext] = None
    ) -> str:
        """
        Intelligently truncate clinical text while preserving important sections

        Strategies:
        1. Simple: Keep beginning (patient info) and end (recent findings)
        2. Smart: Use embeddings to identify most important sentences

        Args:
            text: Original clinical text
            ratio: Ratio to keep (0.0-1.0)
            use_smart: Use smart embedding-based truncation
            context_obj: RetryContext for config access

        Returns:
            Truncated text
        """
        if not text:
            return text

        target_length = int(len(text) * ratio)

        if target_length >= len(text):
            return text

        # Use smart truncation if available and enabled
        if use_smart and self.smart_model:
            try:
                return self._smart_truncate(text, target_length, context_obj)
            except Exception as e:
                logger.warning(f"Smart truncation failed: {e}, falling back to simple")

        # Simple truncation: Keep beginning and end
        begin_ratio = context_obj.preserve_beginning_ratio if context_obj else 0.6
        end_ratio = context_obj.preserve_ending_ratio if context_obj else 0.4

        begin_length = int(target_length * begin_ratio)
        end_length = int(target_length * end_ratio)

        beginning = text[:begin_length]
        ending = text[-end_length:] if end_length > 0 else ""

        truncated = f"{beginning}\n\n[... CLINICAL TEXT TRUNCATED FOR CONTEXT SIZE ...]\n\n{ending}"

        logger.debug(f"Truncated clinical text: {len(text)} → {len(truncated)} chars (ratio={ratio})")

        return truncated

    def _smart_truncate(
        self,
        text: str,
        target_length: int,
        context_obj: Optional[RetryContext] = None
    ) -> str:
        """
        Use embeddings to identify and preserve most important sentences

        Args:
            text: Original text
            target_length: Target length in characters
            context_obj: RetryContext for additional context

        Returns:
            Smartly truncated text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 3:
            # Too few sentences for smart truncation
            return text[:target_length] + "..."

        # Encode all sentences
        embeddings = self.smart_model.encode(sentences)

        # Calculate importance scores (distance from mean)
        mean_embedding = np.mean(embeddings, axis=0)
        importance_scores = []

        for i, emb in enumerate(embeddings):
            # Distance from mean (less distance = more representative/important)
            distance = np.linalg.norm(emb - mean_embedding)
            # Also prioritize first and last sentences
            position_bonus = 0
            if i == 0:
                position_bonus = 0.3  # First sentence bonus
            elif i == len(sentences) - 1:
                position_bonus = 0.2  # Last sentence bonus

            score = (1.0 / (distance + 1.0)) + position_bonus
            importance_scores.append((i, score, sentences[i]))

        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        # Select sentences until we reach target length
        selected_indices = []
        current_length = 0

        for idx, score, sentence in importance_scores:
            sentence_len = len(sentence) + 2  # +2 for ". "
            if current_length + sentence_len <= target_length:
                selected_indices.append(idx)
                current_length += sentence_len
            elif not selected_indices:
                # Always include at least one sentence
                selected_indices.append(idx)
                break

        # Reconstruct text in original order
        selected_indices.sort()
        selected_sentences = [sentences[i] for i in selected_indices]

        # Add markers for omitted sections
        result = []
        last_idx = -1
        for idx in selected_indices:
            if idx > last_idx + 1:
                result.append("[... CONTENT OMITTED ...]")
            result.append(selected_sentences[selected_indices.index(idx)])
            last_idx = idx

        truncated = ". ".join(result) + "."

        logger.debug(f"Smart truncation: kept {len(selected_sentences)}/{len(sentences)} sentences, {len(truncated)} chars")

        return truncated

    def reduce_conversation_history(
        self,
        messages: List[Any],
        reduction_level: int,
        preserve_system: bool = True,
        preserve_tool_results: bool = True
    ) -> List[Any]:
        """
        Reduce conversation history based on reduction level

        Args:
            messages: List of conversation messages
            reduction_level: 0=none, 1=moderate, 2=aggressive, 3=maximum
            preserve_system: Always keep system message
            preserve_tool_results: Always keep tool result messages

        Returns:
            Reduced message list
        """
        if reduction_level == 0 or not messages:
            return messages

        logger.info(f"Reducing conversation history: level {reduction_level}")

        # Separate messages by importance
        system_msgs = []
        tool_msgs = []
        other_msgs = []

        for msg in messages:
            role = getattr(msg, 'role', msg.get('role', ''))

            if role == 'system' and preserve_system:
                system_msgs.append(msg)
            elif role == 'tool' and preserve_tool_results:
                tool_msgs.append(msg)
            else:
                other_msgs.append(msg)

        # Apply reduction to "other" messages
        if reduction_level == 1:
            # Keep last 10 messages
            other_msgs = other_msgs[-10:]
        elif reduction_level == 2:
            # Keep last 5 messages
            other_msgs = other_msgs[-5:]
        elif reduction_level >= 3:
            # Keep last 2 messages only
            other_msgs = other_msgs[-2:]

        # Reconstruct
        reduced = system_msgs + tool_msgs + other_msgs

        logger.info(f"  History reduced: {len(messages)} → {len(reduced)} messages")

        return reduced

    def reduce_tool_context(
        self,
        tool_results: List[Dict[str, Any]],
        reduction_level: int
    ) -> List[Dict[str, Any]]:
        """
        Reduce tool results context based on reduction level

        Args:
            tool_results: List of tool results
            reduction_level: 0=none, 1=moderate, 2=maximum

        Returns:
            Reduced tool results
        """
        if reduction_level == 0 or not tool_results:
            return tool_results

        logger.info(f"Reducing tool context: level {reduction_level}")

        reduced = []

        for result in tool_results:
            result_copy = result.copy()

            if reduction_level == 1:
                # Truncate RAG content
                if result_copy.get('type') == 'rag':
                    results = result_copy.get('results', [])
                    # Keep only top 3 chunks
                    result_copy['results'] = results[:3]

                    # Truncate each chunk to 500 chars
                    for chunk in result_copy['results']:
                        content = chunk.get('content', '') or chunk.get('text', '')
                        if len(content) > 500:
                            chunk['content'] = content[:500] + "..."

            elif reduction_level >= 2:
                # Maximum reduction
                if result_copy.get('type') == 'rag':
                    results = result_copy.get('results', [])
                    # Keep only top 1 chunk
                    result_copy['results'] = results[:1]

                    # Truncate to 200 chars
                    for chunk in result_copy['results']:
                        content = chunk.get('content', '') or chunk.get('text', '')
                        if len(content) > 200:
                            chunk['content'] = content[:200] + "..."

                # Remove extras details
                if result_copy.get('type') == 'extras':
                    results = result_copy.get('results', [])
                    result_copy['results'] = results[:2]  # Keep only 2 extras

            reduced.append(result_copy)

        logger.info(f"  Tool context reduced from {len(tool_results)} to {len(reduced)} results")

        return reduced


def create_retry_context(
    clinical_text: str,
    conversation_history: Optional[List[Any]] = None,
    tool_results: Optional[List[Any]] = None,
    max_attempts: int = 5,
    config: Optional[Any] = None,
    provider: str = "",
    model_name: str = ""
) -> RetryContext:
    """
    Create a retry context for adaptive retry with configuration

    Args:
        clinical_text: Original clinical text
        conversation_history: Current conversation history
        tool_results: Current tool results
        max_attempts: Maximum retry attempts
        config: AdaptiveRetryConfig instance (optional)
        provider: LLM provider name
        model_name: LLM model name

    Returns:
        RetryContext instance
    """
    context = RetryContext(
        attempt=0,
        max_attempts=max_attempts,
        clinical_text=clinical_text,
        conversation_history=conversation_history or [],
        tool_results=tool_results or [],
        original_clinical_text_length=len(clinical_text),
        current_clinical_text_length=len(clinical_text),
        provider=provider,
        model_name=model_name
    )

    # Apply configuration if provided
    if config:
        context.reduction_ratios = config.clinical_text_reduction_ratios
        context.history_levels = config.history_reduction_levels
        context.tool_levels = config.tool_context_reduction_levels
        context.switch_to_minimal_at = config.switch_to_minimal_at
        context.preserve_beginning_ratio = config.preserve_context_beginning_ratio
        context.preserve_ending_ratio = config.preserve_context_ending_ratio

        # Apply provider-specific overrides
        if provider in config.provider_specific_strategies:
            provider_config = config.provider_specific_strategies[provider]

            if 'max_retries' in provider_config:
                context.max_attempts = provider_config['max_retries']

            if provider_config.get('early_minimal_prompt'):
                context.switch_to_minimal_at = 3

            if provider_config.get('aggressive_reduction'):
                # More aggressive reduction for local models
                context.reduction_ratios = [0.7, 0.5, 0.3, 0.15]

    return context
