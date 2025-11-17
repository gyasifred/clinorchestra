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
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from core.logging_config import get_logger

logger = get_logger(__name__)


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

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.tool_results is None:
            self.tool_results = []


class AdaptiveRetryManager:
    """
    Manages adaptive retry strategies for LLM generation failures

    Progressive reduction strategy:
    1. First retry: Reduce clinical text to 80%
    2. Second retry: Reduce to 60%, trim conversation history
    3. Third retry: Reduce to 40%, aggressive history trim, reduce tool context
    4. Fourth retry: Reduce to 20%, switch to minimal prompt
    5. Fifth retry: Emergency fallback - bare minimum context
    """

    def __init__(self, max_retries: int = 5):
        """
        Initialize adaptive retry manager

        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
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
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            retry_context.attempt = attempt

            try:
                logger.info(f"=== ADAPTIVE RETRY ATTEMPT {attempt}/{self.max_retries} ===")

                # Apply adaptive strategy before each attempt
                if attempt > 1:
                    self._apply_adaptive_strategy(retry_context, attempt)

                # Execute generation
                result = generation_func()

                # Success!
                logger.info(f"✓ Generation successful on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e
                retry_context.last_error = str(e)

                logger.warning(f"✗ Attempt {attempt} failed: {e}")

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
                if strategy == RetryStrategy.TRANSIENT_ERROR:
                    backoff_time = min(2 ** (attempt - 1), 16)  # Max 16 seconds
                    logger.info(f"Transient error detected - backing off {backoff_time}s before retry")
                    time.sleep(backoff_time)
                else:
                    # Small delay for other errors
                    time.sleep(0.5)

        # All retries exhausted
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
        Apply adaptive strategy based on retry attempt

        Args:
            context: Current retry context
            attempt: Current attempt number (1-indexed)
        """
        logger.info(f"Applying adaptive strategy for attempt {attempt}")

        # Progressive clinical text reduction
        if attempt == 2:
            # 80% of original text
            context.clinical_text = self._truncate_clinical_text(
                context.clinical_text, 0.8
            )
            logger.info(f"  → Clinical text reduced to 80% ({len(context.clinical_text)} chars)")

        elif attempt == 3:
            # 60% of original text + conversation history trim
            context.clinical_text = self._truncate_clinical_text(
                context.clinical_text, 0.6
            )
            context.history_reduction_level = 1
            logger.info(f"  → Clinical text reduced to 60% ({len(context.clinical_text)} chars)")
            logger.info(f"  → Conversation history trimming: level 1")

        elif attempt == 4:
            # 40% of original text + aggressive history trim + tool context reduction
            context.clinical_text = self._truncate_clinical_text(
                context.clinical_text, 0.4
            )
            context.history_reduction_level = 2
            context.tool_context_reduction_level = 1
            context.switched_to_minimal = True
            logger.info(f"  → Clinical text reduced to 40% ({len(context.clinical_text)} chars)")
            logger.info(f"  → Conversation history trimming: level 2 (aggressive)")
            logger.info(f"  → Tool context reduction: level 1")
            logger.info(f"  → Switching to MINIMAL PROMPT")

        elif attempt == 5:
            # Emergency: 20% of text + maximum reduction
            context.clinical_text = self._truncate_clinical_text(
                context.clinical_text, 0.2
            )
            context.history_reduction_level = 3
            context.tool_context_reduction_level = 2
            logger.warning(f"  → EMERGENCY FALLBACK: Clinical text reduced to 20% ({len(context.clinical_text)} chars)")
            logger.warning(f"  → Maximum conversation history reduction")
            logger.warning(f"  → Maximum tool context reduction")

    def _truncate_clinical_text(self, text: str, ratio: float) -> str:
        """
        Intelligently truncate clinical text while preserving important sections

        Strategy: Keep beginning (patient info) and end (recent findings)

        Args:
            text: Original clinical text
            ratio: Ratio to keep (0.0-1.0)

        Returns:
            Truncated text
        """
        if not text:
            return text

        target_length = int(len(text) * ratio)

        if target_length >= len(text):
            return text

        # Keep 60% from beginning, 40% from end
        begin_length = int(target_length * 0.6)
        end_length = int(target_length * 0.4)

        beginning = text[:begin_length]
        ending = text[-end_length:] if end_length > 0 else ""

        truncated = f"{beginning}\n\n[... CLINICAL TEXT TRUNCATED FOR CONTEXT SIZE ...]\n\n{ending}"

        logger.debug(f"Truncated clinical text: {len(text)} → {len(truncated)} chars (ratio={ratio})")

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
    max_attempts: int = 5
) -> RetryContext:
    """
    Create a retry context for adaptive retry

    Args:
        clinical_text: Original clinical text
        conversation_history: Current conversation history
        tool_results: Current tool results
        max_attempts: Maximum retry attempts

    Returns:
        RetryContext instance
    """
    return RetryContext(
        attempt=0,
        max_attempts=max_attempts,
        clinical_text=clinical_text,
        conversation_history=conversation_history or [],
        tool_results=tool_results or [],
        original_clinical_text_length=len(clinical_text),
        current_clinical_text_length=len(clinical_text)
    )
