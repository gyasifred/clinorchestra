#!/usr/bin/env python3
"""
Agent Factory - Create appropriate agent based on configuration
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

🎯 BOTH EXECUTION MODES ARE UNIVERSAL & AUTONOMOUS (AGENTIC)!

Provides two execution modes:
- **STRUCTURED Mode** (v1.0.0): For predictable workflows - systematic 4-stage execution
- **ADAPTIVE Mode** (v1.0.0): For evolving tasks - continuous iterative refinement

BOTH modes are universal - they adapt to ANY clinical task via prompts/schema.
BOTH modes use autonomous LLM decision-making (agentic behavior).
BOTH modes use ASYNC for 60-75% performance improvement!

Selection based on app_state.adaptive_mode_enabled
"""

from typing import Union
from core.logging_config import get_logger
from core.model_tier_helper import log_tiered_model_config

logger = get_logger(__name__)


def create_agent(llm_manager, rag_engine, extras_manager, function_registry,
                 regex_preprocessor, app_state) -> Union['ExtractionAgent', 'AgenticAgent']:
    """
    Create appropriate agent based on app_state configuration

    Args:
        llm_manager: LLM manager instance
        rag_engine: RAG engine instance
        extras_manager: Extras manager instance
        function_registry: Function registry instance
        regex_preprocessor: Regex preprocessor instance
        app_state: Application state with configuration

    Returns:
        ExtractionAgent (STRUCTURED) or AdaptiveAgent (ADAPTIVE) based on config
        Both are autonomous (agentic) - adapt to ANY clinical task!
    """

    # Check if adaptive mode is enabled (config still uses 'agentic_config' internally)
    if app_state.agentic_config.enabled:
        logger.info("=" * 80)
        logger.info("CREATING ADAPTIVE MODE AGENT (v1.0.0) - For Evolving Tasks")
        logger.info("Execution: Continuous Iteration with Dynamic Adaptation + Async Tools")
        logger.info("Universal: Adapts to ANY clinical task via prompts/schema")
        logger.info(f"Max Iterations: {app_state.agentic_config.max_iterations}")
        logger.info(f"Max Tool Calls: {app_state.agentic_config.max_tool_calls}")
        logger.info("=" * 80)

        from core.agentic_agent import AgenticAgent

        agent = AgenticAgent(
            llm_manager=llm_manager,
            rag_engine=rag_engine,
            extras_manager=extras_manager,
            function_registry=function_registry,
            regex_preprocessor=regex_preprocessor,
            app_state=app_state
        )

        # Apply configuration
        if hasattr(agent.context, 'max_iterations'):
            agent.context.max_iterations = app_state.agentic_config.max_iterations
        if hasattr(agent.context, 'max_tool_calls'):
            agent.context.max_tool_calls = app_state.agentic_config.max_tool_calls

        return agent

    else:
        logger.info("=" * 80)
        logger.info("CREATING STRUCTURED MODE AGENT (v1.0.0) - For Predictable Workflows")
        logger.info("Execution: 4-Stage Sequential Pipeline with ASYNC Tools")
        logger.info("Universal: Adapts to ANY clinical task via prompts/schema")
        logger.info("=" * 80)

        # Log tiered model configuration if enabled
        log_tiered_model_config(app_state)

        from core.agent_system import ExtractionAgent

        return ExtractionAgent(
            llm_manager=llm_manager,
            rag_engine=rag_engine,
            extras_manager=extras_manager,
            function_registry=function_registry,
            regex_preprocessor=regex_preprocessor,
            app_state=app_state
        )


def get_agent_info(app_state) -> dict:
    """
    Get information about which agent will be used

    Returns:
        dict with agent information
    """
    if app_state.agentic_config.enabled:
        return {
            'version': '1.0.0',
            'name': 'AdaptiveAgent',
            'mode': 'ADAPTIVE Mode - For Evolving Tasks',
            'features': [
                'Native tool calling',
                'PAUSE/RESUME execution',
                'ASYNC parallel tool execution',
                'Multiple tool iterations',
                'Dynamic adaptation',
                'Context-aware chaining',
                'Universal - adapts to ANY task'
            ],
            'config': {
                'max_iterations': app_state.agentic_config.max_iterations,
                'max_tool_calls': app_state.agentic_config.max_tool_calls,
                'iteration_logging': app_state.agentic_config.iteration_logging,
                'tool_call_logging': app_state.agentic_config.tool_call_logging
            }
        }
    else:
        return {
            'version': '1.0.0',
            'name': 'ExtractionAgent',
            'mode': 'STRUCTURED Mode - For Predictable Workflows',
            'features': [
                'Stage 1: Autonomous task analysis',
                'Stage 2: ASYNC parallel tool execution',
                'Stage 3: Extraction',
                'Stage 4: RAG refinement (optional)',
                'Universal - adapts to ANY task',
                '60-75% performance boost'
            ],
            'config': {
                'max_retries': 3
            }
        }
