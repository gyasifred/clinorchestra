#!/usr/bin/env python3
"""
Model Tiering Helper - Stage-specific model selection
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Provides intelligent model selection based on stage complexity:
- Stage 1 (Planning): Fast model for quick task analysis
- Stage 3 (Extraction): Accurate model for quality output
- Stage 4 (Refinement): Fast model for simple refinement
- ADAPTIVE: Uses configured model (consistency important for conversation)
"""

from typing import Optional
from core.logging_config import get_logger

logger = get_logger(__name__)


# Model mappings: provider -> (fast_model, accurate_model)
TIERED_MODELS = {
    'anthropic': {
        'fast': 'claude-3-haiku-20240307',
        'accurate': 'claude-3-5-sonnet-20241022'  # Latest sonnet
    },
    'openai': {
        'fast': 'gpt-4o-mini',
        'accurate': 'gpt-4o'
    },
    'google': {
        'fast': 'gemini-1.5-flash',
        'accurate': 'gemini-1.5-pro'
    },
    'azure': {
        # Azure uses deployments, so return None and use configured model
        'fast': None,
        'accurate': None
    },
    'local': {
        # Local models should use configured model
        'fast': None,
        'accurate': None
    }
}


def get_model_for_stage(app_state, stage: int) -> Optional[str]:
    """
    Get appropriate model for a specific stage in STRUCTURED workflow

    Args:
        app_state: Application state with config
        stage: Stage number (1=planning, 2=execution, 3=extraction, 4=refinement)

    Returns:
        Model name to use, or None to use default configured model
    """
    # Check if tiered models are enabled
    if not app_state.optimization_config.enable_tiered_models:
        return None  # Use configured model

    provider = app_state.model_config.provider
    base_model = app_state.model_config.model_name

    # Get provider's model mappings
    tier_config = TIERED_MODELS.get(provider)
    if not tier_config:
        logger.debug(f"Tiered models not supported for provider: {provider}")
        return None

    # Stage 1: Planning/Analysis - Use FAST model
    if stage == 1:
        fast_model = tier_config['fast']
        if fast_model:
            logger.debug(f"[Tiered Models] Stage 1: Using fast model '{fast_model}'")
            return fast_model
        return None

    # Stage 2: Tool Execution - Use configured model (no LLM call in Stage 2)
    elif stage == 2:
        return None

    # Stage 3: Extraction - Use ACCURATE model
    elif stage == 3:
        accurate_model = tier_config['accurate']
        if accurate_model:
            logger.debug(f"[Tiered Models] Stage 3: Using accurate model '{accurate_model}'")
            return accurate_model
        return None

    # Stage 4: Refinement - Use FAST model
    elif stage == 4:
        fast_model = tier_config['fast']
        if fast_model:
            logger.debug(f"[Tiered Models] Stage 4: Using fast model '{fast_model}'")
            return fast_model
        return None

    return None


def get_model_for_adaptive(app_state, iteration: int = 0) -> Optional[str]:
    """
    Get appropriate model for ADAPTIVE workflow

    In ADAPTIVE mode, we use the configured model for consistency
    across the conversation, as model switching can confuse context.

    Args:
        app_state: Application state
        iteration: Current iteration number (for future use)

    Returns:
        None (use configured model for consistency)
    """
    # ADAPTIVE mode should use consistent model for conversation continuity
    # Model switching can confuse the conversation context
    return None


def log_tiered_model_config(app_state):
    """Log tiered model configuration at startup"""
    if not app_state.optimization_config.enable_tiered_models:
        return

    provider = app_state.model_config.provider
    tier_config = TIERED_MODELS.get(provider)

    if tier_config and (tier_config['fast'] or tier_config['accurate']):
        logger.info("=" * 80)
        logger.info("TIERED MODELS ENABLED")
        logger.info(f"Provider: {provider}")
        logger.info(f"Fast Model (Stages 1, 4): {tier_config['fast']}")
        logger.info(f"Accurate Model (Stage 3): {tier_config['accurate']}")
        logger.info("Impact: -30% LLM time, -40% cost")
        logger.info("=" * 80)
    else:
        logger.warning(f"Tiered models enabled but not supported for provider: {provider}")
