#!/usr/bin/env python3
"""
Model-Specific Optimization Profiles
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Provides optimized configurations for different LLM models based on:
- Speed vs quality tradeoffs
- Cost considerations
- Token limits
- Provider-specific optimizations
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelProfile:
    """Optimization profile for a specific model"""
    model_name: str
    provider: str
    temperature: float
    max_tokens: int
    optimization_level: str  # 'speed', 'balanced', 'quality'
    expected_quality: str  # 'basic', 'good', 'high', 'highest'
    cost_per_1k_tokens: float
    recommended_for: list
    notes: str = ""


# Comprehensive model profiles
MODEL_PROFILES = {
    # OpenAI Models - Speed Optimized
    'gpt-4o-mini': ModelProfile(
        model_name='gpt-4o-mini',
        provider='openai',
        temperature=0.1,
        max_tokens=2048,
        optimization_level='speed',
        expected_quality='high',
        cost_per_1k_tokens=0.00015,
        recommended_for=['simple_extraction', 'high_volume', 'development'],
        notes='Fastest OpenAI model. Great for simple-medium complexity tasks. 5-10x faster than GPT-4.'
    ),

    'gpt-4o': ModelProfile(
        model_name='gpt-4o',
        provider='openai',
        temperature=0.05,
        max_tokens=4096,
        optimization_level='balanced',
        expected_quality='highest',
        cost_per_1k_tokens=0.0025,
        recommended_for=['complex_extraction', 'production', 'high_accuracy'],
        notes='Latest GPT-4 variant. Excellent balance of speed and quality.'
    ),

    # OpenAI - Quality Optimized
    'gpt-4': ModelProfile(
        model_name='gpt-4',
        provider='openai',
        temperature=0.01,
        max_tokens=4096,
        optimization_level='quality',
        expected_quality='highest',
        cost_per_1k_tokens=0.03,
        recommended_for=['complex_extraction', 'critical_accuracy', 'research'],
        notes='Highest quality OpenAI model. Best for complex clinical tasks.'
    ),

    'gpt-3.5-turbo': ModelProfile(
        model_name='gpt-3.5-turbo',
        provider='openai',
        temperature=0.2,
        max_tokens=2048,
        optimization_level='speed',
        expected_quality='good',
        cost_per_1k_tokens=0.0005,
        recommended_for=['simple_extraction', 'development', 'prototyping'],
        notes='Fast and cheap. Good for simple extractions.'
    ),

    # Anthropic Models
    'claude-3-haiku-20240307': ModelProfile(
        model_name='claude-3-haiku-20240307',
        provider='anthropic',
        temperature=0.0,
        max_tokens=2048,
        optimization_level='speed',
        expected_quality='high',
        cost_per_1k_tokens=0.00025,
        recommended_for=['speed_critical', 'high_volume', 'simple_extraction'],
        notes='Fastest Claude model. Excellent for high-throughput workloads.'
    ),

    'claude-3-5-sonnet-20241022': ModelProfile(
        model_name='claude-3-5-sonnet-20241022',
        provider='anthropic',
        temperature=0.01,
        max_tokens=4096,
        optimization_level='balanced',
        expected_quality='highest',
        cost_per_1k_tokens=0.003,
        recommended_for=['production', 'complex_extraction', 'high_accuracy'],
        notes='Best Claude model. Excellent for complex clinical extraction.'
    ),

    'claude-3-opus-20240229': ModelProfile(
        model_name='claude-3-opus-20240229',
        provider='anthropic',
        temperature=0.01,
        max_tokens=4096,
        optimization_level='quality',
        expected_quality='highest',
        cost_per_1k_tokens=0.015,
        recommended_for=['complex_extraction', 'critical_accuracy', 'research'],
        notes='Highest quality Claude model. Best reasoning capabilities.'
    ),

    # Google Models
    'gemini-pro': ModelProfile(
        model_name='gemini-pro',
        provider='google',
        temperature=0.1,
        max_tokens=2048,
        optimization_level='balanced',
        expected_quality='high',
        cost_per_1k_tokens=0.00035,
        recommended_for=['balanced_workloads', 'production'],
        notes='Good balance of speed and quality from Google.'
    ),

    # Local Models
    'local-llama-3-8b': ModelProfile(
        model_name='unsloth/llama-3-8b-Instruct-bnb-4bit',
        provider='local',
        temperature=0.01,
        max_tokens=1024,
        optimization_level='memory',
        expected_quality='good',
        cost_per_1k_tokens=0.0,
        recommended_for=['privacy_critical', 'offline', 'low_cost'],
        notes='Free local model. Good for privacy-sensitive workloads. Limited by GPU memory.'
    ),

    'local-mistral-7b': ModelProfile(
        model_name='unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
        provider='local',
        temperature=0.01,
        max_tokens=1024,
        optimization_level='memory',
        expected_quality='good',
        cost_per_1k_tokens=0.0,
        recommended_for=['privacy_critical', 'offline', 'low_cost'],
        notes='Free local model. Compact and efficient.'
    ),
}


class ModelSelector:
    """
    Intelligent model selector based on task requirements

    Automatically recommends the best model for a given task
    based on complexity, volume, and constraints.
    """

    def __init__(self):
        self.profiles = MODEL_PROFILES
        logger.info(f"📋 Model Selector initialized with {len(self.profiles)} profiles")

    def get_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get profile for a specific model"""
        return self.profiles.get(model_name)

    def recommend_model(self,
                       task_complexity: str = 'medium',
                       volume: str = 'medium',
                       budget: str = 'medium',
                       privacy_required: bool = False) -> ModelProfile:
        """
        Recommend optimal model based on requirements

        Args:
            task_complexity: 'simple', 'medium', 'complex'
            volume: 'low' (<100 rows), 'medium' (100-1000), 'high' (>1000)
            budget: 'low', 'medium', 'high'
            privacy_required: Whether data must stay local

        Returns:
            Recommended ModelProfile
        """
        logger.info(f"🔍 Finding optimal model...")
        logger.info(f"   Task complexity: {task_complexity}")
        logger.info(f"   Volume: {volume}")
        logger.info(f"   Budget: {budget}")
        logger.info(f"   Privacy required: {privacy_required}")

        # Privacy constraint
        if privacy_required:
            logger.info("🔒 Privacy required - selecting local model")
            return self.profiles['local-llama-3-8b']

        # Simple tasks
        if task_complexity == 'simple':
            if budget == 'low' or volume == 'high':
                logger.info("✅ Recommended: gpt-4o-mini (fast + cheap)")
                return self.profiles['gpt-4o-mini']
            else:
                logger.info("✅ Recommended: claude-3-haiku (fast + quality)")
                return self.profiles['claude-3-haiku-20240307']

        # Medium complexity
        elif task_complexity == 'medium':
            if budget == 'high':
                logger.info("✅ Recommended: claude-3-5-sonnet (best balanced)")
                return self.profiles['claude-3-5-sonnet-20241022']
            else:
                logger.info("✅ Recommended: gpt-4o (good balanced)")
                return self.profiles['gpt-4o']

        # Complex tasks
        else:  # complex
            if budget == 'high':
                logger.info("✅ Recommended: claude-3-opus (highest quality)")
                return self.profiles['claude-3-opus-20240229']
            else:
                logger.info("✅ Recommended: gpt-4 (high quality)")
                return self.profiles['gpt-4']

    def list_models(self, provider: Optional[str] = None) -> list:
        """List available models, optionally filtered by provider"""
        if provider:
            models = [p for p in self.profiles.values() if p.provider == provider]
        else:
            models = list(self.profiles.values())

        return sorted(models, key=lambda x: x.cost_per_1k_tokens)

    def compare_models(self, model_names: list) -> Dict[str, Any]:
        """Compare multiple models side-by-side"""
        comparison = {}

        for name in model_names:
            profile = self.get_profile(name)
            if profile:
                comparison[name] = {
                    'provider': profile.provider,
                    'cost_per_1k': profile.cost_per_1k_tokens,
                    'quality': profile.expected_quality,
                    'optimization': profile.optimization_level,
                    'max_tokens': profile.max_tokens
                }

        return comparison

    def estimate_cost(self, model_name: str, num_tokens: int) -> float:
        """Estimate cost for processing given number of tokens"""
        profile = self.get_profile(model_name)
        if not profile:
            return 0.0

        cost = (num_tokens / 1000) * profile.cost_per_1k_tokens
        return round(cost, 4)


# Convenience function
def get_recommended_model(task_complexity: str = 'medium',
                         volume: str = 'medium',
                         budget: str = 'medium',
                         privacy_required: bool = False) -> ModelProfile:
    """Get recommended model (convenience function)"""
    selector = ModelSelector()
    return selector.recommend_model(task_complexity, volume, budget, privacy_required)


# Global selector instance
_global_selector = None


def get_model_selector() -> ModelSelector:
    """Get global model selector instance"""
    global _global_selector
    if _global_selector is None:
        _global_selector = ModelSelector()
    return _global_selector
