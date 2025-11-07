#!/usr/bin/env python3
"""
Agent Factory - Create appropriate agent based on configuration
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

🎯 BOTH PATHWAYS ARE GENERAL & AGENTIC!

Provides two execution pathways:
- Classic ExtractionAgent (v1.0.0): 4-stage pipeline, general & agentic with ASYNC
- Agentic Agent (v1.0.0): Continuous loop, general & agentic with ASYNC

Both agents are universal - they adapt to ANY clinical task via prompts/schema.
Both use ASYNC for 60-75% performance improvement!

Selection based on app_state.agentic_config.enabled
"""

from typing import Union
from core.logging_config import get_logger

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
        ExtractionAgent (v1.0.0) or AgenticAgent (v1.0.0) based on config
        Both are general & agentic - adapt to ANY clinical task!
    """

    # Check if agentic mode is enabled
    if app_state.agentic_config.enabled:
        logger.info("=" * 80)
        logger.info("CREATING AGENTIC AGENT (v1.0.0) - General & Agentic")
        logger.info("Mode: Continuous Loop with PAUSE/RESUME + Async Tool Execution")
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
        logger.info("CREATING CLASSIC EXTRACTION AGENT (v1.0.0) - General & Agentic")
        logger.info("Mode: 4-Stage Pipeline with ASYNC Tool Execution")
        logger.info("Universal: Adapts to ANY clinical task via prompts/schema")
        logger.info("=" * 80)

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
            'name': 'AgenticAgent',
            'mode': 'Continuous Agentic Loop with Async',
            'features': [
                'Native tool calling',
                'PAUSE/RESUME execution',
                'ASYNC parallel tool execution',
                'Multiple tool iterations',
                'Dynamic adaptation',
                'Context-aware chaining'
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
            'mode': '4-Stage Pipeline with ASYNC (General & Agentic)',
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
