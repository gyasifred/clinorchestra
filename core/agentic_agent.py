#!/usr/bin/env python3
"""
ADAPTIVE Execution Mode - Iterative Autonomous Loop

Continuous iterative loop for complex clinical data extraction requiring autonomous adaptation.
LLM analyzes text, calls tools, learns from results, and iterates until extraction completes.

Features:
- Autonomous tool selection and execution
- Parallel async tool execution (60-75% faster)
- Stall detection and recovery
- Automatic minimal prompt fallback
- Native function calling API support

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import json
import time
import re
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from core.json_parser import JSONParser
from core.prompt_templates import format_schema_as_instructions
from core.logging_config import get_logger
from core.performance_monitor import get_performance_monitor, TimingContext
from core.tool_dedup_preventer import create_tool_dedup_preventer
from core.adaptive_retry import AdaptiveRetryManager, create_retry_context

logger = get_logger(__name__)
perf_monitor = get_performance_monitor(enabled=True)


class AgenticState(Enum):
    """States in the agentic loop"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    AWAITING_TOOL_RESULTS = "awaiting_tool_results"  # PAUSED state
    CONTINUING = "continuing"  # RESUMED state
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolCall:
    """Represents a single tool call with support for parameter dependencies"""
    id: str
    type: str  # 'rag', 'function', 'extras'
    name: str
    parameters: Dict[str, Any]
    purpose: Optional[str] = None
    depends_on: Optional[List[str]] = None  # IDs of tool calls this depends on


@dataclass
class ToolResult:
    """Represents a tool execution result"""
    tool_call_id: str
    type: str
    success: bool
    result: Any
    message: Optional[str] = None


@dataclass
class ConversationMessage:
    """Message in the conversation history"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool result messages
    name: Optional[str] = None  # For tool result messages


@dataclass
class AgenticContext:
    """Context for agentic execution"""
    clinical_text: str
    label_context: Optional[str]
    state: AgenticState
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 10  # CHANGED: Reduced from 20 to 10
    total_tool_calls: int = 0
    max_tool_calls: int = 100  # Max tool calls budget (increased from 50 for complex cases)
    tool_calls_this_iteration: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    prompt_variables: Dict[str, Any] = field(default_factory=dict)  # NEW: Additional columns for prompt variables

    # Preprocessing tracking
    original_text: Optional[str] = None
    redacted_text: Optional[str] = None
    normalized_text: Optional[str] = None

    # Stall detection - track repeated tool calls
    tool_call_history: List[str] = field(default_factory=list)
    stall_counter: int = 0
    consecutive_no_progress: int = 0

    # JSON failure tracking - for minimal prompt fallback
    consecutive_json_failures: int = 0  # Track failed JSON parsing attempts
    switched_to_minimal: bool = False  # Flag to track if we switched to minimal prompt

    # PERFORMANCE: Conversation history management
    conversation_window_size: int = 20  # Keep only last 20 messages for context
    total_messages_sent: int = 0  # Track total messages for metrics


class AgenticAgent:
    """
    Truly Agentic Extraction System with Continuous Loop

    The LLM autonomously:
    - Analyzes clinical text
    - Decides what tools to call (if any)
    - Learns from tool results
    - Iterates: calls more tools or completes extraction
    - Adapts strategy based on what it discovers
    """

    def __init__(self, llm_manager, rag_engine, extras_manager, function_registry,
                 regex_preprocessor, app_state):
        """Initialize agentic agent"""
        self.llm_manager = llm_manager
        self.rag_engine = rag_engine
        self.extras_manager = extras_manager
        self.function_registry = function_registry
        self.regex_preprocessor = regex_preprocessor
        self.app_state = app_state

        self.context: Optional[AgenticContext] = None
        self.json_parser = JSONParser()

        # Initialize tool deduplication preventer
        self.tool_dedup_preventer = None  # Created per extraction

        # Initialize adaptive retry manager
        self.retry_manager = AdaptiveRetryManager(max_retries=5)

        logger.info(" AdaptiveAgent v1.0.0 initialized - ADAPTIVE Mode (evolving tasks with ASYNC)")
        logger.info(" Enhanced with: Adaptive Retry + Proactive Tool Deduplication")

    def extract(self, clinical_text: str, label_value: Optional[Any] = None,
                prompt_variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main agentic extraction with continuous loop

        Args:
            clinical_text: The main clinical text to analyze
            label_value: Optional label/diagnosis value
            prompt_variables: Optional dict of additional columns to pass as prompt variables

        Flow:
        1. Initialize context with clinical text
        2. Start conversation with LLM
        3. Loop:
           - LLM analyzes/continues
           - If LLM requests tools -> PAUSE -> Execute -> RESUME
           - If LLM outputs JSON -> Extract and complete
        4. Return result
        """
        try:
            # Initialize context
            label_context = self._get_label_context_string(label_value)

            # Preprocess text (skip if batch preprocessing was already done)
            if self.app_state.optimization_config.use_batch_preprocessing:
                # Text is already preprocessed by batch processor - use as-is
                preprocessed_text = clinical_text
            else:
                # Individual preprocessing (row-by-row mode)
                preprocessed_text = self._preprocess_clinical_text(clinical_text)

            # Get max_iterations and max_tool_calls from app_state config
            max_iterations = self.app_state.agentic_config.max_iterations
            max_tool_calls = self.app_state.agentic_config.max_tool_calls

            self.context = AgenticContext(
                clinical_text=preprocessed_text,
                label_context=label_context,
                state=AgenticState.IDLE,
                original_text=clinical_text,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                prompt_variables=prompt_variables or {}  # NEW: Store prompt variables
            )

            # Initialize tool deduplication preventer for this extraction
            self.tool_dedup_preventer = create_tool_dedup_preventer(max_tool_calls=max_tool_calls)

            logger.info("=" * 80)
            logger.info("ADAPTIVE MODE EXTRACTION STARTED (v1.0.0 - Evolving Tasks & Async)")
            logger.info(f"Configuration: Max Iterations={max_iterations}, Max Tool Calls={max_tool_calls}")
            logger.info(f"Enhanced Features: Adaptive Retry + Proactive Tool Deduplication")
            logger.info("=" * 80)

            # Build initial prompt
            initial_prompt = self._build_agentic_initial_prompt()

            # Add system message if supported
            if self._supports_system_messages():
                system_message = self._build_system_message()
                self.context.conversation_history.append(
                    ConversationMessage(role='system', content=system_message)
                )

            # Add user message
            self.context.conversation_history.append(
                ConversationMessage(role='user', content=initial_prompt)
            )

            # Add iteration and tool call budget planning messages to help LLM plan efficiently
            planning_message = f"""You have a maximum of {max_iterations} iterations and {max_tool_calls} total tool calls to complete this extraction task. Plan your tool usage efficiently to complete within these limits. Each iteration allows you to call tools or output final JSON.

BUDGET:
- Maximum Iterations: {max_iterations}
- Maximum Tool Calls: {max_tool_calls}

Be strategic: prioritize essential tools, avoid redundant calls, and output JSON when you have sufficient information.

🚫 CRITICAL: Do NOT call the same tool with the same parameters multiple times. If you've already called a function with specific parameters, the result is available in tool results - do not recalculate it!"""
            self.context.conversation_history.append(
                ConversationMessage(role='user', content=planning_message)
            )
            logger.info(f"ADAPTIVE MODE: Max iterations={max_iterations}, Max tool calls={max_tool_calls}")

            # Start agentic loop
            self.context.state = AgenticState.ANALYZING
            extraction_complete = False

            while not extraction_complete and self.context.iteration < self.context.max_iterations:
                self.context.iteration += 1
                logger.info("=" * 60)
                logger.info(f"ITERATION {self.context.iteration}/{self.context.max_iterations}")
                logger.info(f"State: {self.context.state.value}")
                logger.info(f"Conversation history size: {len(self.context.conversation_history)} messages")
                logger.info(f"Total tool calls so far: {self.context.total_tool_calls}")
                logger.info(f"Tool results collected: {len(self.context.tool_results)}")
                logger.info("=" * 60)

                # Check tool call budget status
                tool_calls_remaining = self.context.max_tool_calls - self.context.total_tool_calls
                tool_call_usage_pct = (self.context.total_tool_calls / self.context.max_tool_calls) * 100

                # Enforce hard limit: stop if tool calls exceeded
                if self.context.total_tool_calls >= self.context.max_tool_calls:
                    logger.error(f"TOOL CALL LIMIT EXCEEDED: {self.context.total_tool_calls}/{self.context.max_tool_calls}")
                    logger.error("Forcing extraction completion with available information")

                    # Try to extract JSON from current conversation
                    extracted = self._extract_json_from_conversation()
                    if extracted:
                        logger.info(" Extracted JSON from conversation - completing extraction")
                        self.context.final_output = extracted
                        self.context.state = AgenticState.COMPLETED
                        extraction_complete = True
                        continue
                    else:
                        # Force completion message
                        self.context.conversation_history.append(
                            ConversationMessage(
                                role='user',
                                content=f"""CRITICAL: Tool call budget EXHAUSTED ({self.context.total_tool_calls}/{self.context.max_tool_calls}). You MUST output final JSON NOW with available information. No more tools will be executed."""
                            )
                        )

                # Warn LLM if approaching tool call limit (>80% used)
                elif tool_call_usage_pct >= 80:
                    logger.warning(f"TOOL CALL BUDGET WARNING: {self.context.total_tool_calls}/{self.context.max_tool_calls} used ({tool_call_usage_pct:.0f}%)")
                    warning_message = f"""⚠️ TOOL CALL BUDGET WARNING: You have used {self.context.total_tool_calls}/{self.context.max_tool_calls} tool calls ({tool_call_usage_pct:.0f}%). Only {tool_calls_remaining} calls remaining. Prioritize essential tools and prepare to output JSON soon.

🚫 AVOID DUPLICATE CALLS: Do not re-request tools you've already used!"""
                    self.context.conversation_history.append(
                        ConversationMessage(role='user', content=warning_message)
                    )

                # Inject duplicate prevention prompt if we have tool history
                if self.tool_dedup_preventer and len(self.tool_dedup_preventer.call_history) > 0:
                    prevention_prompt = self.tool_dedup_preventer.generate_prevention_prompt()
                    if prevention_prompt:
                        self.context.conversation_history.append(
                            ConversationMessage(role='user', content=prevention_prompt)
                        )

                # CRITICAL: Add keyword variation guidance for RAG/Extras to avoid repeating same queries
                keyword_guidance = self._build_keyword_variation_guidance()
                if keyword_guidance:
                    self.context.conversation_history.append(
                        ConversationMessage(role='user', content=keyword_guidance)
                    )

                # Warn LLM if this is the last iteration
                if self.context.iteration == self.context.max_iterations:
                    warning_message = f"""WARNING: This is iteration {self.context.iteration} of {self.context.max_iterations} (FINAL ITERATION). You MUST output complete valid JSON now. Do not call more tools unless absolutely necessary. Focus on completing the extraction with available information."""
                    self.context.conversation_history.append(
                        ConversationMessage(role='user', content=warning_message)
                    )
                    logger.warning("FINAL ITERATION: Sent completion warning to LLM")

                # LLM generates response (may include tool calls or final JSON)
                logger.debug(f" Calling LLM for iteration {self.context.iteration}...")
                with TimingContext('adaptive_llm_call'):
                    response = self._generate_with_tools()
                logger.debug(f" LLM response received")

                if response is None:
                    logger.error("LLM returned no response")
                    break

                # Parse response
                has_tool_calls, has_json_output = self._parse_llm_response(response)

                if has_tool_calls:
                    # PAUSE - Execute tools
                    logger.info(f"LLM requested {len(self.context.tool_calls_this_iteration)} tools")
                    logger.debug(f" Tool calls requested:")
                    for i, tc in enumerate(self.context.tool_calls_this_iteration):
                        logger.debug(f"  {i+1}. {tc.name}({json.dumps(tc.parameters)})")

                    self.context.state = AgenticState.AWAITING_TOOL_RESULTS

                    # PROACTIVE DEDUPLICATION: Filter duplicates using preventer
                    if self.tool_dedup_preventer:
                        # Convert ToolCall objects to dicts for filtering
                        tool_call_dicts = []
                        for tc in self.context.tool_calls_this_iteration:
                            tool_call_dicts.append({
                                'type': tc.type,
                                'name': tc.name,
                                'parameters': tc.parameters
                            })

                        # Filter duplicates
                        unique_dicts, num_duplicates = self.tool_dedup_preventer.filter_duplicates(tool_call_dicts)

                        if num_duplicates > 0:
                            logger.warning(f"⚠️ PREVENTED {num_duplicates} DUPLICATE TOOL CALLS")
                            logger.info(f"   Original: {len(tool_call_dicts)} calls → Unique: {len(unique_dicts)} calls")

                            # Rebuild tool_calls_this_iteration with only unique calls
                            unique_tool_calls = []
                            for unique_dict in unique_dicts:
                                # Find matching ToolCall object
                                for tc in self.context.tool_calls_this_iteration:
                                    tc_dict = {'type': tc.type, 'name': tc.name, 'parameters': tc.parameters}
                                    if self.tool_dedup_preventer.create_call_signature(tc_dict) == \
                                       self.tool_dedup_preventer.create_call_signature(unique_dict):
                                        unique_tool_calls.append(tc)
                                        break

                            self.context.tool_calls_this_iteration = unique_tool_calls

                        # Log budget status
                        logger.info(f"📊 {self.tool_dedup_preventer.get_budget_status()}")

                    # Track tool calls for stall detection
                    tool_signature = self._get_tool_calls_signature(self.context.tool_calls_this_iteration)
                    self.context.tool_call_history.append(tool_signature)
                    logger.debug(f" Tool signature: {tool_signature}")

                    # Check for repeated tool calls (stall detection)
                    if len(self.context.tool_call_history) >= 3:
                        last_three = self.context.tool_call_history[-3:]
                        if last_three[0] == last_three[1] == last_three[2]:
                            self.context.stall_counter += 1
                            logger.warning(f" STALL DETECTED: Same tools called 3 times in a row (stall count: {self.context.stall_counter})")

                    with TimingContext('adaptive_tool_execution'):
                        tool_results = self._execute_tools(self.context.tool_calls_this_iteration)

                    # RESUME - Add tool results to conversation
                    logger.info(f"Tools executed, resuming with {len(tool_results)} results")
                    self.context.state = AgenticState.CONTINUING
                    self.context.consecutive_no_progress = 0  # Reset no-progress counter

                    # Add tool result messages
                    for result in tool_results:
                        self.context.conversation_history.append(
                            ConversationMessage(
                                role='tool',
                                tool_call_id=result.tool_call_id,
                                name=result.type,
                                content=self._format_tool_result_for_llm(result)
                            )
                        )

                    # Clear tool calls for next iteration
                    self.context.tool_calls_this_iteration = []

                    # If stalled too many times, force completion aggressively
                    if self.context.stall_counter >= 2:
                        logger.warning(" FORCING COMPLETION: Agent stalled - same tools called repeatedly")
                        logger.warning(" Next response MUST be valid JSON - no tool calls allowed")

                        # Try to extract JSON from current responses first
                        extracted = self._extract_json_from_conversation()
                        if extracted:
                            logger.info(" Extracted JSON during force - completing immediately")
                            self.context.final_output = extracted
                            self.context.state = AgenticState.COMPLETED
                            extraction_complete = True
                        else:
                            # Send very strong forcing message
                            self.context.conversation_history.append(
                                ConversationMessage(
                                    role='user',
                                    content="""STOP CALLING TOOLS. You have all the information you need.

OUTPUT ONLY VALID JSON matching the schema. NO TOOL CALLS. NO EXPLANATIONS. ONLY JSON.

Format:
{
  "field1": "value1",
  "field2": "value2"
}

If you call tools again, the task will FAIL. Output JSON NOW."""
                                )
                            )
                            # Set flag to refuse tool calls on next iteration
                            self.context.stall_counter = 0  # Reset but next iteration will check for JSON only

                    # Loop continues - LLM can request more tools or finish

                elif has_json_output:
                    # Extraction complete
                    logger.info(" LLM provided final JSON output - extraction complete")
                    self.context.state = AgenticState.COMPLETED
                    extraction_complete = True

                else:
                    # LLM is thinking/analyzing without calling tools or outputting JSON
                    self.context.consecutive_no_progress += 1
                    logger.info(f"LLM thinking/analyzing without progress (count: {self.context.consecutive_no_progress})")

                    remaining_iters = self.context.max_iterations - self.context.iteration

                    # CRITICAL: Check for repeated JSON failures and switch to minimal prompt
                    if (self.context.consecutive_json_failures >= self.app_state.processing_config.max_retries
                        and not self.context.switched_to_minimal
                        and self.app_state.prompt_config.minimal_prompt
                        and self.app_state.prompt_config.use_minimal):
                        logger.warning(f" SWITCHING TO MINIMAL PROMPT: {self.context.consecutive_json_failures} consecutive JSON failures")
                        self.context.switched_to_minimal = True

                        # Rebuild prompt with minimal version
                        # Force get_prompt_for_processing to use minimal by passing high retry count
                        minimal_prompt = self.app_state.get_prompt_for_processing(retry_count=999)

                        # Inject message telling LLM to use simpler approach
                        self.context.conversation_history.append(
                            ConversationMessage(
                                role='user',
                                content=f"""Previous JSON outputs failed validation. Using SIMPLIFIED task definition:

{minimal_prompt}

CLINICAL TEXT:
{self.context.clinical_text}

LABEL CONTEXT:
{self.context.label_context or "No label provided"}

OUTPUT VALID JSON matching the schema. Be more concise and focus on directly extractable information."""
                            )
                        )
                        logger.info(" Minimal prompt injected - continuing with simpler task definition")

                    # If too many iterations with no progress OR near max iterations, force completion
                    elif self.context.consecutive_no_progress >= 2 or remaining_iters <= 1:
                        logger.warning(f" FORCING COMPLETION: No progress detected (consecutive={self.context.consecutive_no_progress}, remaining={remaining_iters})")

                        # Try to extract any existing JSON
                        extracted = self._extract_json_from_conversation()
                        if extracted:
                            logger.info(" Extracted JSON during no-progress force - completing")
                            self.context.final_output = extracted
                            self.context.state = AgenticState.COMPLETED
                            extraction_complete = True
                        else:
                            self.context.conversation_history.append(
                                ConversationMessage(
                                    role='user',
                                    content="""You must complete the task NOW.

OUTPUT FINAL JSON using the schema provided.
NO MORE TOOL CALLS.
NO EXPLANATIONS.
ONLY JSON.

Example format:
{
  "field_name": "extracted_value"
}"""
                                )
                            )
                            self.context.consecutive_no_progress = 0
                    else:
                        # Add a prompt to encourage either tool use or completion
                        self.context.conversation_history.append(
                            ConversationMessage(
                                role='user',
                                content=f"Continue: Either (1) call tools to gather more information, OR (2) output the final JSON if you have sufficient information. {remaining_iters} iterations remaining."
                            )
                        )

            if self.context.iteration >= self.context.max_iterations:
                logger.warning(f" Max iterations ({self.context.max_iterations}) reached")

                # Try to extract any JSON from the last few responses
                logger.info(" Attempting to extract JSON from conversation history...")
                extracted_json = self._extract_json_from_conversation()

                if extracted_json:
                    logger.info(" Successfully extracted JSON from conversation")
                    self.context.final_output = extracted_json
                    self.context.state = AgenticState.COMPLETED
                else:
                    logger.warning(" No valid JSON found in conversation - marking as failed")
                    self.context.state = AgenticState.FAILED
                    self.context.error = "Max iterations reached without valid JSON output"

            # Final state logging for clear indication of success/failure
            logger.info("=" * 80)
            if self.context.state == AgenticState.COMPLETED:
                if self.context.final_output:
                    logger.info(" EXTRACTION SUCCESS - Agent completed with valid JSON output")
                    logger.info(f"   Iterations: {self.context.iteration}/{self.context.max_iterations}")
                    logger.info(f"   Tool calls: {self.context.total_tool_calls}")
                    logger.info(f"   Final JSON fields: {list(self.context.final_output.keys()) if isinstance(self.context.final_output, dict) else 'N/A'}")
                else:
                    logger.warning(" EXTRACTION INCOMPLETE - Agent completed but no JSON output generated")
                    logger.warning(f"   Iterations: {self.context.iteration}/{self.context.max_iterations}")
            elif self.context.state == AgenticState.FAILED:
                logger.error(" EXTRACTION FAILED - Agent did not complete successfully")
                logger.error(f"   Reason: {self.context.error}")
                logger.error(f"   Iterations: {self.context.iteration}/{self.context.max_iterations}")
            else:
                logger.info(f" EXTRACTION ENDED - Agent state: {self.context.state.value}")
            logger.info("=" * 80)

            return self._build_extraction_result()

        except Exception as e:
            logger.error(f"Agentic extraction failed: {e}", exc_info=True)
            if self.context:
                self.context.state = AgenticState.FAILED
                self.context.error = str(e)
            return self._build_failure_result(str(e))

    def _generate_with_tools(self) -> Optional[Dict[str, Any]]:
        """
        Generate LLM response with tool calling support

        Returns dict with:
        - content: Text content
        - tool_calls: List of tool calls (if any)
        - finish_reason: 'stop', 'tool_calls', etc.
        """
        try:
            # Check if provider supports native tool calling
            if self._supports_native_tool_calling():
                return self._generate_with_native_tool_calling()
            else:
                # Fallback to text-based tool calling
                return self._generate_with_text_based_tools()

        except Exception as e:
            logger.error(f"Generation with tools failed: {e}", exc_info=True)
            return None

    def _supports_native_tool_calling(self) -> bool:
        """Check if provider supports native function calling"""
        return self.llm_manager.provider in ['openai', 'anthropic']

    def _supports_system_messages(self) -> bool:
        """Check if provider supports system messages"""
        return self.llm_manager.provider in ['openai', 'anthropic', 'google']

    def _generate_with_native_tool_calling(self) -> Dict[str, Any]:
        """Generate using native tool calling API"""
        # Use enhanced llm_manager method
        return self.llm_manager.generate_with_tool_calling(
            messages=self._convert_history_to_api_format(),
            tools=self._get_tool_schema(),
            max_tokens=self.app_state.model_config.max_tokens
        )

    def _generate_with_text_based_tools(self) -> Dict[str, Any]:
        """
        Fallback: Text-based tool calling for providers that don't support native tools

        ENHANCED: Includes tool descriptions in prompt so LLM knows what tools exist
        """
        # Build comprehensive prompt with tool descriptions
        messages = self._convert_history_to_text_with_tools()

        # Generate
        response_text = self.llm_manager.generate(
            messages,
            max_tokens=self.app_state.model_config.max_tokens
        )

        # Parse for tool calls or JSON output
        return self._parse_text_response_for_tools(response_text)

    def _get_tool_schema(self, max_functions: int = 100) -> List[Dict[str, Any]]:
        """
        Get tool schema in OpenAI/Anthropic function calling format

        Args:
            max_functions: Maximum number of custom functions to include (default: 100)
                          Some LLM providers have limits (e.g., 128 tools max)
        """
        tools = []

        # RAG tool (only include if initialized with documents)
        if self.rag_engine and self.rag_engine.initialized:
            tools.append({
                'type': 'function',
                'function': {
                    'name': 'query_rag',
                    'description': 'Query retrieval system for guidelines, standards, criteria, and reference information from authoritative sources. Can be called MULTIPLE times with different queries to refine information.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'Focused query with 4-8 specific keywords targeting guidelines, diagnostic criteria, standards, or reference information'
                            },
                            'purpose': {
                                'type': 'string',
                                'description': 'Why you need this information (helps with debugging)'
                            }
                        },
                        'required': ['query']
                    }
                }
            })

        # Extras tool (only include if there are enabled extras)
        if self.extras_manager and len([e for e in self.extras_manager.extras if e.get('enabled', True)]) > 0:
            tools.append({
                'type': 'function',
                'function': {
                    'name': 'query_extras',
                    'description': 'Query supplementary hints/tips/patterns that help understand the task. Use specific keywords to match relevant hints and guidance.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'keywords': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': '3-5 specific keywords related to your task (e.g., domain-specific terms, concepts, assessment types, field names from schema)'
                            }
                        },
                        'required': ['keywords']
                    }
                }
            })

        # Function tool (use consistent instance)
        # INTELLIGENT LIMITING: Only include up to max_functions to avoid exceeding model limits
        if self.function_registry:
            # Get all available functions (ONLY ENABLED ONES from cache)
            functions = self.function_registry.get_all_functions_info()

            # Calculate remaining space for functions (accounting for RAG + Extras)
            remaining_slots = max_functions - len(tools)

            if len(functions) > remaining_slots:
                logger.warning(f"[LIMIT] {len(functions)} functions available, but limiting to {remaining_slots} to avoid exceeding model tool limit")
                # Prioritize functions that might be more relevant
                # For now, just take the first N functions (could be improved with relevance scoring)
                functions = functions[:remaining_slots]

            for func in functions:
                # Build proper JSON Schema for parameters
                # CRITICAL: Remove 'required' field from individual properties (it should only be at parameters level)
                properties = {}
                required_params = []

                for param_name, param_spec in func.get('parameters', {}).items():
                    # Extract only JSON Schema fields, excluding our custom 'required' field
                    prop = {}
                    for key, value in param_spec.items():
                        if key == 'required':
                            # Track which parameters are required (for parameters-level 'required' array)
                            if value:
                                required_params.append(param_name)
                        else:
                            # Include all other fields (type, description, etc.)
                            prop[key] = value
                    properties[param_name] = prop

                # Build tool schema in OpenAI format
                tool_schema = {
                    'type': 'function',
                    'function': {
                        'name': f"call_{func['name']}",
                        'description': f"{func.get('description', 'Calculation or transformation function')}. Can be called multiple times with different inputs.",
                        'parameters': {
                            'type': 'object',
                            'properties': properties
                        }
                    }
                }

                # Only add 'required' array if there are required parameters
                if required_params:
                    tool_schema['function']['parameters']['required'] = required_params

                tools.append(tool_schema)

        logger.info(f"[TOOLS] Built tool schema with {len(tools)} tools (RAG: {1 if self.rag_engine and self.rag_engine.initialized else 0}, Extras: {1 if self.extras_manager and len([e for e in self.extras_manager.extras if e.get('enabled', True)]) > 0 else 0}, Functions: {len([f for f in functions]) if 'functions' in locals() else 0})")
        return tools

    def _extract_json_from_conversation(self) -> Optional[Dict[str, Any]]:
        """
        Try to extract valid JSON from conversation history (last 3 assistant messages)

        This is a fallback when max iterations is reached
        """
        # Check last 3 assistant messages for any JSON
        for msg in reversed(self.context.conversation_history[-6:]):  # Check last 6 messages
            if msg.role == 'assistant' and msg.content:
                parsed = self._try_parse_json(msg.content)
                if parsed:
                    logger.info(f" Found valid JSON in assistant message: {str(parsed)[:100]}...")
                    return parsed

        # No valid JSON found
        return None

    def _detect_duplicate_function_calls(self, current_calls: List[ToolCall]) -> List[Dict[str, str]]:
        """
        Detect if any function calls are duplicates of previous calls

        Returns list of duplicates with function name and parameters
        """
        duplicates = []

        # Get all previous function calls from tool_results
        previous_calls = {}
        for result in self.context.tool_results:
            if result.type == 'function' and result.success:
                # Find the original tool call
                for msg in self.context.conversation_history:
                    if msg.role == 'assistant' and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.id == result.tool_call_id:
                                # Create signature: function_name + sorted parameters
                                func_name = tc.name.replace('call_', '')
                                param_sig = json.dumps(tc.parameters, sort_keys=True)
                                call_sig = f"{func_name}:{param_sig}"
                                previous_calls[call_sig] = {
                                    'function': func_name,
                                    'params': tc.parameters,
                                    'result': result.result
                                }

        # Check current calls against previous
        for tc in current_calls:
            if tc.name.startswith('call_'):
                func_name = tc.name.replace('call_', '')
                param_sig = json.dumps(tc.parameters, sort_keys=True)
                call_sig = f"{func_name}:{param_sig}"

                if call_sig in previous_calls:
                    duplicates.append({
                        'function': func_name,
                        'params': str(tc.parameters),
                        'previous_result': previous_calls[call_sig]['result']
                    })

        return duplicates

    def _get_tool_calls_signature(self, tool_calls: List[ToolCall]) -> str:
        """
        Create a signature for tool calls to detect repeated patterns

        Returns a string representing the tools and their key parameters
        """
        signatures = []
        for tc in tool_calls:
            # Create signature: tool_name + sorted parameter keys
            param_keys = sorted(tc.parameters.keys()) if tc.parameters else []
            sig = f"{tc.name}({','.join(param_keys)})"
            signatures.append(sig)

        return ";".join(sorted(signatures))

    def _build_keyword_variation_guidance(self) -> Optional[str]:
        """
        Build keyword variation guidance for RAG/Extras to avoid repeating same queries

        Analyzes all previous tool results to extract used keywords and instructs
        LLM to use DIFFERENT keywords in subsequent calls to get diverse information.

        Returns:
            String with keyword guidance, or None if no previous tool usage detected
        """
        if not self.context.tool_results:
            return None

        # Extract previously used keywords from tool results
        previous_rag_keywords = []
        previous_extras_keywords = []
        previous_function_calls = []

        for result in self.context.tool_results:
            if result.type == 'rag' and result.success:
                # Extract query string
                query = getattr(result.result, 'query', '') if hasattr(result.result, 'query') else str(result.result)
                if query:
                    # Split query into keywords (simple approach)
                    words = query.lower().split()
                    previous_rag_keywords.extend([w for w in words if len(w) > 3])

            elif result.type == 'extras' and result.success:
                # Extras uses keywords list
                if hasattr(result.result, 'keywords'):
                    keywords = result.result.keywords
                elif isinstance(result.result, dict) and 'keywords' in result.result:
                    keywords = result.result['keywords']
                else:
                    keywords = []
                previous_extras_keywords.extend([k.lower() for k in keywords])

            elif result.type == 'function':
                # Track function names
                func_name = result.tool_call_id.split('_')[0] if '_' in result.tool_call_id else 'unknown'
                previous_function_calls.append(func_name)

        # Remove duplicates and filter
        previous_rag_keywords = list(set([k for k in previous_rag_keywords if len(k) > 3]))
        previous_extras_keywords = list(set(previous_extras_keywords))
        previous_function_calls = list(set(previous_function_calls))

        # Only build guidance if we have previous tool usage
        if not (previous_rag_keywords or previous_extras_keywords or previous_function_calls):
            return None

        # Build guidance message
        guidance_lines = []
        guidance_lines.append("🔄 KEYWORD VARIATION GUIDANCE - Avoid Repeating Queries:")
        guidance_lines.append("━" * 60)

        if previous_rag_keywords:
            guidance_lines.append("")
            guidance_lines.append("📚 RAG Keywords Already Used (DO NOT REPEAT):")
            guidance_lines.append(f"   {', '.join(previous_rag_keywords[:20])}")
            guidance_lines.append("   → Use DIFFERENT, RELEVANT keywords to get NEW information")
            guidance_lines.append("   → Try alternative terms, different aspects, or related concepts")

        if previous_extras_keywords:
            guidance_lines.append("")
            guidance_lines.append("💡 Extras Keywords Already Used (DO NOT REPEAT):")
            guidance_lines.append(f"   {', '.join(previous_extras_keywords[:20])}")
            guidance_lines.append("   → Use DIFFERENT, RELEVANT keywords to get NEW hints")
            guidance_lines.append("   → Try related concepts, alternative terminology, or different aspects")

        if previous_function_calls:
            guidance_lines.append("")
            guidance_lines.append("⚙️ Functions Already Called:")
            guidance_lines.append(f"   {', '.join(previous_function_calls[:10])}")
            guidance_lines.append("   → Only call again with DIFFERENT parameters for NEW measurements")
            guidance_lines.append("   → DO NOT recalculate the same values")

        guidance_lines.append("")
        guidance_lines.append("✅ CRITICAL: When calling RAG/Extras again:")
        guidance_lines.append("   1. Identify what NEW information would fill gaps")
        guidance_lines.append("   2. Use keywords that are DIFFERENT from previous calls")
        guidance_lines.append("   3. Target specific missing information, not general exploration")
        guidance_lines.append("   4. Avoid repeating the same query - it will return the same results!")
        guidance_lines.append("━" * 60)

        return "\n".join(guidance_lines)

    def _detect_and_resolve_dependencies(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        Detect parameter dependencies in tool calls.

        Scans each tool call's parameters for references to other tool calls.
        References are strings starting with "$" followed by a tool call ID.

        Examples:
            "$call_1" - References result of tool call with id="call_1"
            "$call_2.field" - References specific field in result (future enhancement)

        Returns:
            List of tool calls with depends_on field populated
        """
        enhanced_calls = []

        for tool_call in tool_calls:
            dependencies = set()

            # Recursively scan parameters for dependencies
            def scan_for_deps(obj):
                if isinstance(obj, str):
                    # Check if string is a reference (starts with $)
                    if obj.startswith('$'):
                        # Extract the tool call ID
                        # Format: $call_1 or $call_1.field
                        ref = obj[1:]  # Remove $
                        call_id = ref.split('.')[0]  # Get ID before any dot
                        dependencies.add(call_id)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        scan_for_deps(value)
                elif isinstance(obj, list):
                    for item in obj:
                        scan_for_deps(item)

            scan_for_deps(tool_call.parameters)

            # Create enhanced tool call with dependencies
            enhanced_call = ToolCall(
                id=tool_call.id,
                type=tool_call.type,
                name=tool_call.name,
                parameters=tool_call.parameters,
                purpose=tool_call.purpose,
                depends_on=list(dependencies) if dependencies else None
            )
            enhanced_calls.append(enhanced_call)

            if dependencies:
                logger.info(f" Detected dependencies for {tool_call.id}: {dependencies}")

        return enhanced_calls

    def _topological_sort_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        Sort tool calls in dependency order using topological sort.

        Ensures that calls are executed in an order where all dependencies
        are satisfied before a call is executed.

        Returns:
            Sorted list of tool calls (dependencies first)

        Raises:
            ValueError: If circular dependencies detected
        """
        # Build adjacency list and in-degree map
        graph = {tc.id: [] for tc in tool_calls}
        in_degree = {tc.id: 0 for tc in tool_calls}
        id_to_call = {tc.id: tc for tc in tool_calls}

        for tool_call in tool_calls:
            if tool_call.depends_on:
                for dep_id in tool_call.depends_on:
                    if dep_id in graph:
                        graph[dep_id].append(tool_call.id)
                        in_degree[tool_call.id] += 1
                    else:
                        logger.warning(f" Dependency {dep_id} not found for {tool_call.id} - will be ignored")

        # Kahn's algorithm for topological sort
        queue = [tc_id for tc_id, degree in in_degree.items() if degree == 0]
        sorted_ids = []

        while queue:
            current_id = queue.pop(0)
            sorted_ids.append(current_id)

            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for circular dependencies
        if len(sorted_ids) != len(tool_calls):
            remaining = [tc_id for tc_id in in_degree if in_degree[tc_id] > 0]
            raise ValueError(f"Circular dependency detected in tool calls: {remaining}")

        # Return sorted tool calls
        return [id_to_call[tc_id] for tc_id in sorted_ids]

    def _resolve_parameter_references(self, parameters: Dict[str, Any], results_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameter references to actual results.

        Replaces strings like "$call_1" with the actual result from that tool call.
        Supports nested dictionaries and lists.

        Args:
            parameters: Original parameters (may contain references)
            results_map: Map of tool_call_id -> result

        Returns:
            Parameters with all references resolved
        """
        def resolve(obj):
            if isinstance(obj, str):
                if obj.startswith('$'):
                    # Parse reference: $call_1 or $call_1.field
                    ref = obj[1:]
                    parts = ref.split('.', 1)
                    call_id = parts[0]

                    if call_id not in results_map:
                        logger.warning(f" Cannot resolve reference {obj} - call not executed yet")
                        return obj  # Return unchanged if dependency not resolved

                    result = results_map[call_id]

                    # If there's a field selector (.field), navigate into result
                    if len(parts) > 1 and isinstance(result, dict):
                        field_path = parts[1]
                        for field in field_path.split('.'):
                            if isinstance(result, dict) and field in result:
                                result = result[field]
                            else:
                                logger.warning(f" Field {field} not found in result for {call_id}")
                                return obj

                    logger.info(f"   -> Resolved {obj} = {result}")
                    return result
                return obj
            elif isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve(item) for item in obj]
            else:
                return obj

        return resolve(parameters)

    def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call (wrapper around existing execution methods)"""
        if tool_call.name == 'query_rag':
            return self._execute_rag_tool(tool_call)
        elif tool_call.name == 'query_extras':
            return self._execute_extras_tool(tool_call)
        else:
            return self._execute_function_tool(tool_call)

    def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute all tool calls with DEPENDENCY RESOLUTION

        The agent can now structure calls where one function's output becomes another's input.
        For example:
            call_1: calculate_age(birth_date="2020-01-15")
            call_2: calculate_bmi(weight=20, height=1.1)
            call_3: analyze(age="$call_1", bmi="$call_2")  # Uses results from call_1 and call_2

        Flow:
        1. Deduplicate tool calls (prevent repeated identical calls)
        2. Detect parameter dependencies (parameters starting with "$")
        3. Build dependency graph
        4. Execute in topological order (dependencies first)
        5. Substitute results before executing dependent calls
        6. Return results in original order for conversation coherence
        """
        # CRITICAL FIX: Deduplicate tool calls before execution
        original_count = len(tool_calls)
        tool_calls = self._deduplicate_tool_calls(tool_calls)

        if len(tool_calls) == 0:
            logger.warning(" No tool calls to execute after deduplication")
            return []

        logger.info(f"Executing {len(tool_calls)} tools with dependency resolution")
        if original_count > len(tool_calls):
            logger.info(f" Deduplicated: {original_count} -> {len(tool_calls)} calls (removed {original_count - len(tool_calls)} duplicates)")

        # Step 1: Detect dependencies and build dependency graph
        tool_calls_with_deps = self._detect_and_resolve_dependencies(tool_calls)

        # Step 2: Sort by dependencies (topological sort)
        execution_order = self._topological_sort_tool_calls(tool_calls_with_deps)

        logger.info(f" Dependency-resolved execution order: {[tc.id for tc in execution_order]}")

        # Step 3: Execute in order, resolving dependencies as we go
        results_map = {}  # Map of tool_call_id -> result
        all_results = []

        for tool_call in execution_order:
            # Resolve any parameter references
            resolved_params = self._resolve_parameter_references(tool_call.parameters, results_map)

            # Update tool call with resolved parameters
            resolved_tool_call = ToolCall(
                id=tool_call.id,
                type=tool_call.type,
                name=tool_call.name,
                parameters=resolved_params,
                purpose=tool_call.purpose,
                depends_on=tool_call.depends_on
            )

            logger.info(f" Executing {tool_call.id}: {tool_call.name}")
            if tool_call.depends_on:
                logger.info(f"   -> Depends on: {tool_call.depends_on}")

            # Execute the tool
            result = self._execute_single_tool(resolved_tool_call)

            # Store result for dependency resolution
            results_map[tool_call.id] = result.result
            all_results.append(result)

            # Update tracking
            self.context.tool_results.append(result)
            self.context.total_tool_calls += 1

        return all_results

    def _deduplicate_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        Deduplicate tool calls to prevent executing identical calls multiple times.

        Two tool calls are considered identical if they have:
        - Same name
        - Same parameters (compared as JSON strings for deterministic comparison)

        This prevents the agent from getting stuck calling the same function repeatedly
        with identical parameters, which is a common LLM behavior issue.

        Returns:
            List of unique tool calls (keeps first occurrence of each duplicate)
        """
        seen = {}
        unique_calls = []
        duplicates_removed = 0

        # DIAGNOSTIC: Log all tool calls being checked
        logger.debug(f" Deduplication check for {len(tool_calls)} tool calls:")
        for i, tc in enumerate(tool_calls):
            params_json = json.dumps(tc.parameters, sort_keys=True)
            logger.debug(f"  {i+1}. {tc.name} with params: {params_json}")

        for tool_call in tool_calls:
            # Create a unique key based on name and parameters
            # Sort parameters for consistent comparison
            params_json = json.dumps(tool_call.parameters, sort_keys=True)
            key = f"{tool_call.name}||{params_json}"

            logger.debug(f" Dedup key: {key}")

            if key not in seen:
                seen[key] = tool_call
                unique_calls.append(tool_call)
                logger.debug(f"   UNIQUE - keeping this call")
            else:
                duplicates_removed += 1
                logger.warning(
                    f" DUPLICATE TOOL CALL detected and skipped: "
                    f"{tool_call.name} with parameters {params_json}"
                )
                logger.debug(f"   DUPLICATE - already seen this exact call")

        if duplicates_removed > 0:
            logger.warning(
                f" Removed {duplicates_removed} duplicate tool calls "
                f"(kept {len(unique_calls)} unique calls from {len(tool_calls)} total)"
            )

        return unique_calls

    async def _execute_tools_async(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Async execution of tool calls - runs in parallel

        This is the Phase 2 enhancement that significantly improves performance

        IMPORTANT: Deduplicates tool calls before execution to prevent
        the agent from getting stuck calling the same function repeatedly.
        """
        # CRITICAL FIX: Deduplicate tool calls before execution
        original_count = len(tool_calls)
        tool_calls = self._deduplicate_tool_calls(tool_calls)

        if len(tool_calls) == 0:
            logger.warning(" No tool calls to execute after deduplication")
            return []

        # Create tasks for all tools
        tasks = []
        for tool_call in tool_calls:
            # CRITICAL: Check by NAME first! All tools have type='function' from native APIs
            # So we must check specific names (query_rag, query_extras) before checking call_ prefix
            if tool_call.name == 'query_rag':
                task = self._execute_rag_tool_async(tool_call)
            elif tool_call.name == 'query_extras':
                task = self._execute_extras_tool_async(tool_call)
            elif tool_call.name.startswith('call_') or tool_call.type == 'function':
                task = self._execute_function_tool_async(tool_call)
            else:
                # Unknown tool type - create a coroutine that returns error
                async def unknown_tool():
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        type=tool_call.type,
                        success=False,
                        result=None,
                        message=f"Unknown tool type: {tool_call.type}"
                    )
                task = unknown_tool()

            tasks.append(task)

        # Execute all tasks in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        logger.info(f" Executed {len(tool_calls)} tools in {elapsed:.2f}s (parallel)")

        return results

    def _execute_rag_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute RAG query"""
        with TimingContext('adaptive_rag_query'):
            return self._execute_rag_tool_impl(tool_call)

    def _execute_rag_tool_impl(self, tool_call: ToolCall) -> ToolResult:
        """Execute RAG query implementation"""
        try:
            # Check if RAG engine is available
            if not self.rag_engine:
                error_msg = (
                    " RAG Engine Not Initialized\n\n"
                    "The agent requested RAG (document retrieval), but RAG is not configured.\n\n"
                    "To enable RAG:\n"
                    "1. Go to the 'RAG' tab\n"
                    "2. Upload your documents (PDFs, text files, etc.)\n"
                    "3. Click 'Build Index' to create the vector database\n"
                    "4. Ensure 'Enable RAG' is checked in RAG configuration\n\n"
                    "Without RAG, the agent will rely only on Functions and Extras for knowledge."
                )
                logger.warning(f" RAG engine not available - agent requested RAG but it's not initialized")
                logger.info(" To fix: Upload documents in RAG tab -> Build Index -> Enable RAG")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    type='rag',
                    success=False,
                    result=[],
                    message=error_msg
                )

            query = tool_call.parameters.get('query', '')

            if not query or len(query) < 5:
                logger.warning(f" RAG query rejected: Query too short or empty (length={len(query)})")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    type='rag',
                    success=False,
                    result=[],
                    message='Query too short or empty'
                )

            logger.info(f" RAG query: '{query}'")

            top_k = self.app_state.rag_config.rag_top_k
            results = self.rag_engine.query(query_text=query, k=top_k)

            if len(results) > 0:
                logger.info(f" RAG retrieved {len(results)} documents (top_k={top_k})")
            else:
                logger.warning(f" RAG found no results for query: '{query}'")

            return ToolResult(
                tool_call_id=tool_call.id,
                type='rag',
                success=len(results) > 0,
                result=results,
                message=f"Retrieved {len(results)} documents"
            )

        except Exception as e:
            logger.error(f" RAG execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                type='rag',
                success=False,
                result=[],
                message=str(e)
            )

    def _execute_function_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute function call"""
        with TimingContext('adaptive_function_call'):
            return self._execute_function_tool_impl(tool_call)

    def _execute_function_tool_impl(self, tool_call: ToolCall) -> ToolResult:
        """Execute function call implementation"""
        try:
            # Use consistent function registry instance (same as agent_system)
            # Check if function registry is available
            if not self.function_registry:
                logger.warning(f" Function registry not available")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    type='function',
                    success=False,
                    result=None,
                    message='Function registry not available'
                )

            # Extract function name (remove 'call_' prefix if present)
            func_name = tool_call.name
            if func_name.startswith('call_'):
                func_name = func_name[5:]

            parameters = tool_call.parameters

            logger.info(f" Calling function: {func_name}({', '.join(f'{k}={v}' for k, v in parameters.items())})")

            success, result, message = self.function_registry.execute_function(
                func_name, **parameters
            )

            if success:
                logger.info(f" Function {func_name} executed successfully: {result}")
            else:
                logger.warning(f" Function {func_name} failed: {message}")

            return ToolResult(
                tool_call_id=tool_call.id,
                type='function',
                success=success,
                result=result,
                message=message
            )

        except Exception as e:
            logger.error(f" Function execution failed with exception: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                type='function',
                success=False,
                result=None,
                message=str(e)
            )

    def _execute_extras_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute extras query"""
        try:
            # Check if extras manager is available
            if not self.extras_manager:
                logger.warning(f" Extras manager not available")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    type='extras',
                    success=False,
                    result=[],
                    message='Extras manager not available'
                )

            keywords = tool_call.parameters.get('keywords', [])

            logger.info(f" Querying extras with keywords: {keywords}")

            matched_extras = self.extras_manager.match_extras_by_keywords(keywords)

            if len(matched_extras) > 0:
                logger.info(f" Matched {len(matched_extras)} extras hints/patterns")
                # Log first few matched extras for debugging
                for i, extra in enumerate(matched_extras[:3]):
                    extra_id = extra.get('id', 'N/A')
                    extra_type = extra.get('type', 'unknown')
                    logger.debug(f"   - Extra [{i+1}]: {extra_id} ({extra_type})")
            else:
                logger.warning(f" No extras matched for keywords: {keywords}")

            return ToolResult(
                tool_call_id=tool_call.id,
                type='extras',
                success=len(matched_extras) > 0,
                result=matched_extras,
                message=f"Matched {len(matched_extras)} extras"
            )

        except Exception as e:
            logger.error(f" Extras execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                type='extras',
                success=False,
                result=[],
                message=str(e)
            )

    # ========================================================================
    # ASYNC VERSIONS OF TOOL EXECUTION (Phase 2)
    # ========================================================================

    async def _execute_rag_tool_async(self, tool_call: ToolCall) -> ToolResult:
        """Execute RAG query asynchronously"""
        logger.info(f"[ASYNC] Executing RAG: {tool_call.parameters.get('query', '')[:50]}...")

        # Run in thread pool since RAG engine is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default executor
            self._execute_rag_tool,
            tool_call
        )
        return result

    async def _execute_function_tool_async(self, tool_call: ToolCall) -> ToolResult:
        """Execute function call asynchronously"""
        func_name = tool_call.name
        if func_name.startswith('call_'):
            func_name = func_name[5:]

        logger.info(f"[ASYNC] Executing function: {func_name}")

        # Run in thread pool since function registry is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._execute_function_tool,
            tool_call
        )
        return result

    async def _execute_extras_tool_async(self, tool_call: ToolCall) -> ToolResult:
        """Execute extras query asynchronously"""
        keywords = tool_call.parameters.get('keywords', [])
        logger.info(f"[ASYNC] Executing extras: {keywords}")

        # Run in thread pool since extras manager is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._execute_extras_tool,
            tool_call
        )
        return result

    # ========================================================================
    # END ASYNC TOOL EXECUTION
    # ========================================================================

    def _build_agentic_initial_prompt(self) -> str:
        """
        Build initial agentic prompt using USER'S task-specific prompt as PRIMARY

        CRITICAL FIX: Now respects retry/iteration-based minimal prompt fallback.
        - First tries main_prompt
        - Falls back to minimal_prompt if main is missing OR if iteration threshold reached
        - Uses get_prompt_for_processing() to respect retry logic
        """
        from core.prompt_templates import get_agentic_extraction_prompt

        schema_instructions = format_schema_as_instructions(
            self.app_state.prompt_config.json_schema
        )

        # CRITICAL FIX: Use get_prompt_for_processing() with iteration count as retry_count
        # This ensures if iterations exceed threshold, system switches to minimal prompt
        # For adaptive agent, we use iteration count as the "retry" metric
        current_iteration = self.context.iteration if self.context else 0
        user_task_prompt = self.app_state.get_prompt_for_processing(retry_count=current_iteration)

        # Track if minimal prompt is being used
        if self.app_state.is_using_minimal_prompt:
            logger.warning(f"  Using MINIMAL prompt (iteration={current_iteration} >= max_retries={self.app_state.processing_config.max_retries})")
        else:
            logger.info(" Using MAIN prompt as primary task definition")

        # Fallback to base_prompt if neither main nor minimal is set
        if not user_task_prompt or not user_task_prompt.strip():
            user_task_prompt = self.app_state.prompt_config.base_prompt or ""
            if not user_task_prompt:
                logger.warning(" No user task-specific prompt found - using generic agentic prompt")

        # NEW: Try to format user_task_prompt with prompt variables if it has placeholders
        try:
            user_task_prompt = user_task_prompt.format(
                clinical_text=self.context.clinical_text,
                label_context=self.context.label_context or "No label provided",
                json_schema_instructions=schema_instructions,
                **self.context.prompt_variables  # NEW: Add prompt variables
            )
        except KeyError:
            # Template doesn't use these placeholders, pass as-is
            pass

        prompt = get_agentic_extraction_prompt(
            clinical_text=self.context.clinical_text,
            label_context=self.context.label_context or "No label provided",
            json_schema=json.dumps(self.app_state.prompt_config.json_schema, indent=2),
            schema_instructions=schema_instructions,
            user_task_prompt=user_task_prompt
        )

        return prompt

    def _build_system_message(self) -> str:
        """Build system message for providers that support it"""
        return """You are an autonomous extraction agent working to fulfill a specific task.

You have access to tools:
- query_rag: Retrieve guidelines, standards, and reference information from authoritative sources
- call_[function]: Perform calculations and transformations
- query_extras: Get supplementary hints, patterns, and reference information

🔴 CRITICAL: Understand the task, analyze the data, autonomously determine required tools, and execute them.

**AUTONOMOUS TASK-DRIVEN EXECUTION WORKFLOW:**

**PHASE 1 - UNDERSTAND REQUIREMENTS & ANALYZE DATA:**
1. Read the task description → Understand WHAT needs to be extracted and HOW
2. Read the clinical text → Identify WHAT data is currently available
3. Perform gap analysis between available data and required output:
   - Compare available format vs. required format
   - Identify guidelines mentioned in task that need to be retrieved
   - Identify calculations needed to complete schema fields

**PHASE 2 - AUTONOMOUSLY DETERMINE & EXECUTE REQUIRED TOOLS:**
4. Based on gap analysis, autonomously determine which tools are REQUIRED:
   - Call functions to convert/calculate values as needed
   - Call RAG to retrieve guidelines/criteria mentioned in task
   - Call extras for task-related supplementary hints
   - Call same function multiple times for serial/temporal measurements
5. Execute all required tool calls

**PHASE 3 - ASSESS & REFINE (ITERATIVE):**
6. Review tool results and assess current extraction state
7. Determine if additional tools would improve extraction:
   - Would clarify ambiguous findings?
   - Would fix inconsistencies or errors?
   - Would ascertain missing but important details?
   - Would improve completeness or quality?
8. If yes: Call additional tools and return to Phase 3
9. If no: Proceed to Phase 4

**PHASE 4 - COMPLETE EXTRACTION:**
10. Use all tool results to fill schema fields
11. Extract remaining fields directly from clinical text
12. Output final JSON matching the exact schema structure

🔴 IMPORTANT: You autonomously determine which tools are REQUIRED to fulfill the task.
Tools must serve the TASK requirements, not unrelated exploration.

** CRITICAL REQUIREMENTS - MUST FOLLOW:**

**1. USE ALL TOOL RESULTS IN YOUR FINAL OUTPUT:**
   - EVERY function result MUST appear in your JSON output
   - EVERY RAG document MUST inform your reasoning and content
   - EVERY extras hint MUST be applied to your extraction
   -  DO NOT call tools "for fun" - use every result!

**2. NO DUPLICATE CALCULATIONS:**
   - TRACK what you've already calculated
   - DO NOT call the same function with the same parameters twice
   - If you need a value you already calculated, REMEMBER it from the tool result

**3. INCORPORATE RETRIEVED INFORMATION:**
   - When RAG returns criteria/guidelines -> CITE them in your output
   - When extras returns hints -> APPLY them to your extraction
   - When functions return values -> INCLUDE the exact values in your JSON

**Key Principles:**
-  DO NOT output final JSON immediately without gathering information
-  DO NOT recalculate values you already have
-  DO NOT ignore tool results - USE EVERY ONE
-  Call tools iteratively across multiple rounds
-  Use tool results to inform next tool calls
-  Remember and reuse calculated values
-  Cite RAG sources and apply extras guidance
-  Output ONLY valid JSON - no explanations or markdown

**CRITICAL: Tool Call Format**
- Tool calls MUST start with "TOOL_CALL:" prefix
- Format: TOOL_CALL: {{"tool": "name", "parameters": {{...}}}}
- Do NOT output bare JSON without the TOOL_CALL: prefix

**FUNCTION DEPENDENCIES - ADVANCED FEATURE:**
You can now chain function calls where one function's output becomes another's input!

**How it works:**
- Use "$call_X" in parameters to reference results from other tool calls
- The system will automatically execute dependencies in the correct order
- You can reference specific fields using "$call_X.field_name"

**Example - Chained calculations:**
```
TOOL_CALL: {{"id": "call_1", "tool": "call_function_a", "parameters": {{"param": "value1"}}}}
TOOL_CALL: {{"id": "call_2", "tool": "call_function_b", "parameters": {{"param": "value2"}}}}
TOOL_CALL: {{"id": "call_3", "tool": "call_function_c", "parameters": {{"param_a": "$call_1", "param_b": "$call_2"}}}}
```

**What happens:**
1. call_1 executes -> returns result1
2. call_2 executes -> returns result2
3. call_3 executes with param_a=result1, param_b=result2 (values auto-substituted)

**When to use this:**
- When you need to combine results from multiple calculations
- When one function needs the output of another
- When building complex assessments from simple measurements

**Important:**
- Each tool call needs a unique ID (call_1, call_2, etc.)
- Dependencies are automatically detected and resolved
- Circular dependencies will raise an error
- You can reference the full result with "$call_X" or a specific field with "$call_X.field"

This allows you to structure sophisticated multi-step calculations in a single round!"""

    def _format_tool_result_for_llm(self, result: ToolResult) -> str:
        """Format tool result for LLM consumption"""
        if not result.success:
            return f"Tool execution failed: {result.message}"

        if result.type == 'rag':
            # Format RAG results
            chunks = []
            for chunk in result.result[:5]:  # Top 5
                content = chunk.get('content', '') or chunk.get('text', '')
                source = chunk.get('metadata', {}).get('source', 'unknown') if isinstance(chunk.get('metadata'), dict) else 'unknown'
                chunks.append(f"[Source: {source}]\n{content}\n")
            return "\n\n".join(chunks)

        elif result.type == 'function':
            return f"Result: {json.dumps(result.result, indent=2)}\n{result.message or ''}"

        elif result.type == 'extras':
            # Format extras
            extras_text = []
            for extra in result.result:
                content = extra.get('content', '')
                extras_text.append(content)
            return "\n\n".join(extras_text)

        else:
            return str(result.result)

    def _parse_llm_response(self, response: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Parse LLM response to determine next action

        Returns:
            (has_tool_calls, has_json_output)
        """
        # Check for tool calls
        tool_calls = response.get('tool_calls', [])
        if tool_calls:
            # Parse and store tool calls
            for tc in tool_calls:
                self.context.tool_calls_this_iteration.append(
                    ToolCall(
                        id=tc.get('id', f"call_{time.time()}"),
                        type=tc.get('type', 'function'),
                        name=tc.get('function', {}).get('name', ''),
                        parameters=json.loads(tc.get('function', {}).get('arguments', '{}')),
                        purpose=tc.get('purpose')
                    )
                )

            # Add assistant message to history
            self.context.conversation_history.append(
                ConversationMessage(
                    role='assistant',
                    content=response.get('content'),
                    tool_calls=self.context.tool_calls_this_iteration.copy()
                )
            )

            return True, False

        # Check for JSON output
        content = response.get('content', '')
        if content:
            # Try to parse JSON
            parsed_json = self._try_parse_json(content)
            if parsed_json:
                self.context.final_output = parsed_json
                # Reset failure counter on success
                self.context.consecutive_json_failures = 0
                # Add to history
                self.context.conversation_history.append(
                    ConversationMessage(role='assistant', content=content)
                )
                return False, True
            else:
                # Just thinking/analyzing OR invalid JSON was rejected
                self.context.conversation_history.append(
                    ConversationMessage(role='assistant', content=content)
                )

                # Check if this looks like a malformed tool call or invalid JSON
                if '{' in content and '}' in content:
                    # Looks like JSON but was rejected - track failure
                    self.context.consecutive_json_failures += 1
                    logger.warning(f" LLM output contains JSON but validation failed (consecutive failures: {self.context.consecutive_json_failures})")
                    # The loop will continue and prompt the LLM again

                return False, False

        return False, False

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse JSON from text with schema validation

        CRITICAL: Validates that parsed JSON matches expected schema fields
        to prevent accidentally accepting tool call JSON as final output
        """
        parsed, method = self.json_parser.parse_json_response(
            text,
            self.app_state.prompt_config.json_schema
        )

        if not parsed:
            return None

        # CRITICAL VALIDATION: Check if this is actually a tool call JSON (wrong format)
        # Tool calls have format: {"tool": "...", "parameters": {...}}
        if isinstance(parsed, dict) and 'tool' in parsed and 'parameters' in parsed:
            logger.warning(" Detected tool call JSON without TOOL_CALL: prefix - rejecting as final output")
            logger.warning(f"LLM must use format: TOOL_CALL: {{\"tool\": \"...\", \"parameters\": {{...}}}}")
            return None

        # CRITICAL VALIDATION: Verify parsed JSON has expected schema fields
        schema = self.app_state.prompt_config.json_schema
        if schema and isinstance(parsed, dict):
            # Get required fields from schema
            required_fields = [
                field for field, props in schema.items()
                if isinstance(props, dict) and props.get('required', False)
            ]

            # Check if at least one required field is present
            if required_fields:
                has_required = any(field in parsed for field in required_fields)
                if not has_required:
                    logger.warning(f" Parsed JSON missing all required fields: {required_fields}")
                    logger.warning(f"Parsed JSON keys: {list(parsed.keys())}")
                    return None

        return parsed

    def _apply_conversation_window(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """
        Apply SMART sliding window to conversation history for performance

        CRITICAL FIX: Preserves ALL tool results so LLM knows what was completed!

        CRITICAL FOR ALL PROVIDERS: Limits context size to prevent:
        1. Exponential slowdown as iterations increase
        2. Token context overflow
        3. Redundant re-processing of old messages

        SMART Strategy (FIXED - preserves completed work):
        1. Always keep: System message (tool definitions)
        2. Always keep: Initial user message (task + input text)
        3. Always keep: ALL tool result messages (completed work - CRITICAL!)
        4. Always keep: ALL assistant messages with tool_calls (what was requested)
        5. Keep: Recent other messages (user prompts, assistant thinking)
        6. Drop: Old assistant thinking messages with no tool calls

        Why this works:
        - LLM always sees what tools were called (assistant with tool_calls)
        - LLM always sees results of those tools (tool result messages)
        - LLM doesn't need old thinking messages that led to those calls
        """
        if len(messages) <= self.context.conversation_window_size:
            return messages

        # Categorize messages by importance
        system_msg = None
        initial_user_msg = None
        tool_result_messages = []  # CRITICAL: ALL tool results must be kept!
        assistant_with_tool_calls = []  # Keep: Shows what tools were called
        other_messages = []  # Can drop some of these

        for i, msg in enumerate(messages):
            if msg.role == 'system' and system_msg is None:
                system_msg = msg
            elif msg.role == 'user' and initial_user_msg is None:
                initial_user_msg = msg
            elif msg.role == 'tool':
                # CRITICAL: Always keep ALL tool results!
                # These contain completed calculations/retrievals
                tool_result_messages.append((i, msg))
            elif msg.role == 'assistant' and msg.tool_calls:
                # Keep assistant messages that made tool calls
                assistant_with_tool_calls.append((i, msg))
            else:
                # Other messages: user prompts, assistant thinking
                # These can be dropped if space is needed
                other_messages.append((i, msg))

        # Calculate available slots for "other" messages
        reserved_count = 0
        if system_msg:
            reserved_count += 1
        if initial_user_msg:
            reserved_count += 1
        reserved_count += len(tool_result_messages)  # All tool results (CRITICAL!)
        reserved_count += len(assistant_with_tool_calls)  # All tool call requests

        # How many slots left for other messages?
        # FIXED: Properly enforce window size while keeping some recent context
        available_slots = self.context.conversation_window_size - reserved_count
        if available_slots >= 5:
            # We have room for at least 5 thinking messages - great!
            other_message_limit = available_slots
        elif available_slots > 0:
            # Limited room (1-4 messages) - keep what we can
            other_message_limit = available_slots
        else:
            # Reserved messages already exceed window - keep 3 most recent for minimal context
            # This is a compromise: we MUST keep tool results, but need SOME thinking context
            other_message_limit = 3
            logger.debug(
                f" Window size exceeded by reserved messages: "
                f"{reserved_count} reserved vs {self.context.conversation_window_size} limit. "
                f"Keeping {other_message_limit} recent thinking messages anyway."
            )

        # Keep most recent "other" messages
        if len(other_messages) > other_message_limit:
            other_messages = other_messages[-other_message_limit:]

        # Reconstruct conversation in original chronological order
        all_kept = []
        if system_msg:
            all_kept.append((0, system_msg))
        if initial_user_msg:
            all_kept.append((1, initial_user_msg))
        all_kept.extend(tool_result_messages)
        all_kept.extend(assistant_with_tool_calls)
        all_kept.extend(other_messages)

        # Sort by original index to maintain conversation flow
        all_kept.sort(key=lambda x: x[0])
        windowed = [msg for idx, msg in all_kept]

        # Log windowing for performance tracking
        if len(messages) > len(windowed):
            dropped_count = len(messages) - len(windowed)
            logger.info(f" SMART windowing: {len(messages)} -> {len(windowed)} messages")
            logger.info(f"    Preserved: {len(tool_result_messages)} tool results (LLM knows what was completed)")
            logger.info(f"    Preserved: {len(assistant_with_tool_calls)} tool call requests")
            logger.info(f"    Dropped: {dropped_count} old thinking/prompt messages")

        return windowed

    def _convert_history_to_api_format(self) -> List[Dict[str, Any]]:
        """Convert conversation history to API format with windowing"""
        # PERFORMANCE: Apply conversation window for ALL providers (OpenAI, Anthropic, Google, Azure, Local)
        windowed_history = self._apply_conversation_window(self.context.conversation_history)

        # Track metrics
        self.context.total_messages_sent += len(windowed_history)

        messages = []

        for msg in windowed_history:
            if msg.role == 'system':
                messages.append({'role': 'system', 'content': msg.content})
            elif msg.role == 'user':
                messages.append({'role': 'user', 'content': msg.content})
            elif msg.role == 'assistant':
                api_msg = {'role': 'assistant'}
                if msg.content:
                    api_msg['content'] = msg.content
                if msg.tool_calls:
                    api_msg['tool_calls'] = [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.name,
                                'arguments': json.dumps(tc.parameters)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(api_msg)
            elif msg.role == 'tool':
                messages.append({
                    'role': 'tool',
                    'tool_call_id': msg.tool_call_id,
                    'content': msg.content
                })

        return messages

    def _convert_history_to_text(self) -> str:
        """Convert conversation history to text for non-tool-calling providers with windowing"""
        # PERFORMANCE: Apply conversation window (same as API format)
        windowed_history = self._apply_conversation_window(self.context.conversation_history)

        lines = []
        for msg in windowed_history:
            if msg.role == 'system':
                lines.append(f"SYSTEM: {msg.content}")
            elif msg.role == 'user':
                lines.append(f"USER: {msg.content}")
            elif msg.role == 'assistant':
                lines.append(f"ASSISTANT: {msg.content}")
            elif msg.role == 'tool':
                lines.append(f"TOOL RESULT: {msg.content}")
        return "\n\n".join(lines)

    def _convert_history_to_text_with_tools(self) -> str:
        """
        Convert conversation history to text WITH comprehensive tool descriptions

        CRITICAL for local models to understand what tools are available and how to use them
        """
        lines = []

        # Add system message with tool descriptions
        system_msg = self._build_system_message()
        lines.append(f"SYSTEM: {system_msg}")

        # Add comprehensive tool guide
        lines.append("\n" + "=" * 80)
        lines.append("📚 TOOL GUIDE - Use Tools Before Final Extraction")
        lines.append("=" * 80)

        lines.append("\n **THREE TOOL CATEGORIES:**\n")

        lines.append("1️⃣ **RAG (Retrieval-Augmented Generation)** - query_rag()")
        lines.append("   - Retrieves guidelines, diagnostic criteria, standards, and reference information")
        lines.append("   - Build queries from: domain + specificity + what you need")
        lines.append("   - Example: 'diagnostic criteria for condition X' or 'reference standards for metric Y'\n")

        lines.append("2️⃣ **FUNCTIONS (Calculations & Transformations)** - call_[function_name]()")
        lines.append("   - Performs calculations and data transformations (see full list below)")
        lines.append("   - Scan text for measurements and data, match to function parameters")
        lines.append("   - CRITICAL: Call functions BEFORE making interpretations\n")

        lines.append("3️⃣ **EXTRAS (Supplementary Hints)** - query_extras()")
        lines.append("   - Task-specific hints, patterns, reference ranges, and interpretation guides")
        lines.append("   - Build keywords from: schema field names + domain + relevant terms")
        lines.append("   - Use 3-5 specific terms relevant to your task, avoid generic words\n")

        # Add detailed tool descriptions
        lines.append("=" * 80)
        lines.append("🔧 AVAILABLE TOOLS WITH PARAMETERS:")
        lines.append("=" * 80 + "\n")

        tools = self._get_tool_schema()
        for tool in tools:
            func = tool['function']
            lines.append(f"📌 **{func['name']}**")
            lines.append(f"   {func['description']}")

            # Format parameters more clearly
            params = func['parameters'].get('properties', {})
            if params:
                lines.append("   PARAMETERS:")
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', 'No description')
                    lines.append(f"     - {param_name} ({param_type}): {param_desc}")
            lines.append("")

        # Add clear instructions with examples
        lines.append("=" * 80)
        lines.append(" HOW TO CALL TOOLS:")
        lines.append("=" * 80)
        lines.append("\n CRITICAL: Tool calls MUST start with 'TOOL_CALL:' prefix! \n")
        lines.append("REQUIRED FORMAT:")
        lines.append('TOOL_CALL: {"tool": "tool_name", "parameters": {...}}')
        lines.append("")
        lines.append(" CORRECT EXAMPLES (note the 'TOOL_CALL:' prefix):")
        lines.append('TOOL_CALL: {"tool": "query_rag", "parameters": {"query": "diagnostic criteria for condition", "purpose": "need classification thresholds"}}')
        lines.append('TOOL_CALL: {"tool": "call_calculate_bmi", "parameters": {"weight_kg": 70, "height_m": 1.75}}')
        lines.append('TOOL_CALL: {"tool": "query_extras", "parameters": {"keywords": ["assessment", "criteria", "classification", "guidelines"]}}')
        lines.append("")
        lines.append(" WRONG (missing 'TOOL_CALL:' prefix):")
        lines.append('{"tool": "query_rag", ...}  <- MISSING TOOL_CALL: prefix!')
        lines.append("")
        lines.append(" WRONG (missing closing brace):")
        lines.append('TOOL_CALL: {"tool": "query_rag", "parameters": {"query": "test"}  <- MISSING }')
        lines.append("")
        lines.append(" ITERATIVE WORKFLOW:")
        lines.append("ROUND 1 - Initial Analysis:")
        lines.append("  - Analyze task prompt + input text")
        lines.append("  - Identify key metrics/measurements/entities/information")
        lines.append("  - Call functions (calculations/transformations) + query_extras (hints)")
        lines.append("")
        lines.append("ROUND 2 - Build Context:")
        lines.append("  - Review function/extras results - REMEMBER these values!")
        lines.append("  - Build RAG keywords from what you learned")
        lines.append("  - Call query_rag (fetch guidelines/standards/criteria)")
        lines.append("")
        lines.append("ROUND 3+ - Assess & Fill Gaps:")
        lines.append("  - Assess: Do I have ALL info needed?")
        lines.append("  - If NO: Identify gaps -> Call NEW tools (not duplicates!) -> Reassess")
        lines.append("  - If YES: Output final JSON using ALL tool results")
        lines.append("")
        lines.append("=" * 80)
        lines.append(" CRITICAL REQUIREMENTS:")
        lines.append("=" * 80)
        lines.append("")
        lines.append("1. USE EVERY TOOL RESULT:")
        lines.append("   - Function results -> Include exact values in JSON output")
        lines.append("   - RAG documents -> Cite and apply criteria/guidelines")
        lines.append("   - Extras hints -> Apply patterns and guidance")
        lines.append("")
        lines.append("2. NO DUPLICATE CALCULATIONS:")
        lines.append("   - Track what you've calculated")
        lines.append("   - DO NOT call same function with same parameters twice")
        lines.append("   - Remember results from previous tool calls")
        lines.append("")
        lines.append("3. INCORPORATE ALL RETRIEVED INFO:")
        lines.append("   - Reference RAG sources in your reasoning")
        lines.append("   - Apply extras guidance to your extraction")
        lines.append("   - Include ALL calculated values in final JSON")
        lines.append("=" * 80 + "\n")

        # Add conversation history with windowing
        # PERFORMANCE: Apply windowing to preserve tool results while reducing context
        windowed_history = self._apply_conversation_window(self.context.conversation_history)

        for msg in windowed_history:
            if msg.role == 'system':
                # Already added above with tools
                continue
            elif msg.role == 'user':
                lines.append(f"\n{'=' * 80}")
                lines.append("USER REQUEST:")
                lines.append(f"{'=' * 80}")
                lines.append(msg.content)
            elif msg.role == 'assistant':
                lines.append(f"\nASSISTANT: {msg.content}")
            elif msg.role == 'tool':
                lines.append(f"\n{'=' * 40}")
                lines.append(f"TOOL RESULT ({msg.name}):")
                lines.append(f"{'=' * 40}")
                lines.append(msg.content)

        return "\n".join(lines)

    def _parse_text_response_for_tools(self, text: str) -> Dict[str, Any]:
        """
        Parse text response for tool calls or JSON (fallback for non-tool-calling providers)

        ENHANCED: Detects TOOL_CALL requests in text format with proper nested JSON handling
        """
        tool_calls = []

        # FIXED: Better pattern that handles nested JSON braces
        # Look for TOOL_CALL: followed by balanced braces
        tool_call_lines = re.findall(r'TOOL_CALL:\s*(.+?)(?=\n|$)', text, re.IGNORECASE | re.MULTILINE)

        if tool_call_lines:
            # Parse tool calls
            for i, match_text in enumerate(tool_call_lines):
                match_text = match_text.strip()

                # Try to extract complete JSON (handles nested braces)
                try:
                    # Find the complete JSON object by counting braces
                    json_str = self._extract_complete_json(match_text)

                    if not json_str:
                        logger.info(f" Skipping incomplete tool call JSON (LLM response may have been truncated): {match_text[:100]}...")
                        logger.info("   -> Continuing with other valid tool calls (this is handled gracefully)")
                        continue

                    tool_request = json.loads(json_str)
                    tool_name = tool_request.get('tool', '')
                    parameters = tool_request.get('parameters', {})

                    if not tool_name:
                        logger.warning(f"Tool call missing 'tool' field: {json_str}")
                        continue

                    tool_calls.append({
                        'id': f"call_{int(time.time() * 1000)}_{i}",
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': json.dumps(parameters)
                        }
                    })
                    logger.info(f" Parsed tool call: {tool_name}")

                except json.JSONDecodeError as e:
                    logger.info(f" Skipping malformed tool call JSON: {match_text[:100]}... | Error: {e}")
                    logger.info("   -> Continuing with other valid tool calls")
                except Exception as e:
                    logger.info(f" Skipping unparseable tool call: {match_text[:100]}... | Error: {e}")
                    logger.info("   -> Continuing with other valid tool calls")

            if tool_calls:
                logger.info(f" Successfully parsed {len(tool_calls)} tool calls from text response")
                return {
                    'content': text,
                    'tool_calls': tool_calls,
                    'finish_reason': 'tool_calls'
                }

        # Check if it looks like final JSON output
        if '{' in text and '}' in text:
            parsed = self._try_parse_json(text)
            if parsed:
                logger.info("Detected final JSON output")
                return {'content': text, 'tool_calls': [], 'finish_reason': 'stop'}

        # Otherwise, assume it's analysis/thinking
        logger.info("LLM is thinking/analyzing")
        return {'content': text, 'tool_calls': [], 'finish_reason': 'continue'}

    def _extract_complete_json(self, text: str) -> Optional[str]:
        """
        Extract complete JSON object from text by counting balanced braces

        Handles nested JSON like: {"tool": "x", "parameters": {"a": "b", "c": "d"}}
        """
        text = text.strip()
        if not text.startswith('{'):
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # Found complete JSON object
            if brace_count == 0:
                return text[:i+1]

        # Incomplete JSON
        return None

    def _build_extraction_result(self) -> Dict[str, Any]:
        """Build final extraction result with detailed tracking"""
        # Count tool usage by type
        extras_count = 0
        rag_count = 0
        function_count = 0
        extras_details = []
        rag_details = []
        function_details = []

        for result in self.context.tool_results:
            if result.type == 'extras':
                extras_count += 1
                if result.success and result.result:
                    for extra in result.result:
                        extras_details.append({
                            'id': extra.get('id', 'N/A'),
                            'type': extra.get('type', 'unknown'),
                            'content': extra.get('content', ''),
                            'relevance_score': extra.get('relevance_score', 0),
                            'matched_keywords': extra.get('matched_keywords', [])
                        })
            elif result.type == 'rag':
                rag_count += 1
                if result.success and result.result:
                    for chunk in result.result:
                        rag_details.append({
                            'content': chunk.get('content', '') or chunk.get('text', ''),
                            'source': chunk.get('metadata', {}).get('source', 'unknown') if isinstance(chunk.get('metadata'), dict) else 'unknown',
                            'score': chunk.get('score', 0)
                        })
            elif result.type == 'function':
                function_count += 1
                if result.success:
                    # Extract function name from tool call history
                    func_name = 'unknown'
                    for msg in self.context.conversation_history:
                        if msg.role == 'assistant' and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc.id == result.tool_call_id:
                                    func_name = tc.name.replace('call_', '')
                                    break

                    function_details.append({
                        'name': func_name,
                        'result': result.result,
                        'message': result.message or ''
                    })

        logger.info(f" TOOL USAGE SUMMARY: Extras={extras_count}, RAG={rag_count}, Functions={function_count}")

        return {
            'original_clinical_text': self.context.original_text,
            'clinical_text': self.context.clinical_text,
            'redacted_text': self.context.redacted_text,
            'normalized_text': self.context.normalized_text,
            'input_label_value': None,
            'label_context': self.context.label_context,
            'stage3_output': self.context.final_output or {},
            'stage4_final_output': self.context.final_output or {},

            # Usage counts for playground display (FIXED)
            'extras_used': extras_count,
            'rag_used': rag_count,
            'functions_called': function_count,

            'agentic_metadata': {
                'version': '1.0.0',
                'execution_mode': 'agentic_async',
                'iterations': self.context.iteration,
                'total_tool_calls': self.context.total_tool_calls,
                'final_state': self.context.state.value,
                'tool_results_count': len(self.context.tool_results),
                'conversation_length': len(self.context.conversation_history),
                'stall_counter': self.context.stall_counter,
                'consecutive_no_progress': self.context.consecutive_no_progress
            },
            'processing_metadata': {
                'had_label': self.context.label_context is not None,
                'final_state': self.context.state.value,
                'agentic_execution': True,

                # Detailed tracking for playground display (FIXED)
                'extras_details': extras_details,
                'rag_details': rag_details,
                'function_calls_details': function_details,

                # Tool results summary
                'tool_results_summary': {
                    'extras': extras_count,
                    'rag': rag_count,
                    'functions': function_count
                }
            }
        }

    def _build_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Build failure result"""
        return {
            'original_clinical_text': self.context.original_text if self.context else '',
            'clinical_text': self.context.clinical_text if self.context else '',
            'error': error_message,
            'stage3_output': {},
            'stage4_final_output': {},
            'agentic_metadata': {
                'version': '1.0.0',
                'execution_mode': 'agentic_async',
                'iterations': self.context.iteration if self.context else 0,
                'total_tool_calls': self.context.total_tool_calls if self.context else 0,
                'final_state': 'failed',
                'error': error_message
            },
            'processing_metadata': {
                'had_label': False,
                'final_state': 'failed',
                'agentic_execution': True
            }
        }

    def _get_label_context_string(self, label_value: Any) -> Optional[str]:
        """Get label context from mapping"""
        if label_value is None:
            return None

        label_mapping = self.app_state.data_config.label_mapping
        label_key = str(label_value)

        if label_key in label_mapping:
            return label_mapping[label_key]

        try:
            numeric_value = int(label_value)
            if numeric_value in label_mapping:
                return label_mapping[numeric_value]
        except (ValueError, TypeError):
            pass

        return None

    def _preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text with normalization and redaction

        CRITICAL ORDER: Normalize FIRST, then Redact
        - Normalization standardizes patterns (e.g., "5'11"" → "5 feet 11 inches")
        - Redaction then operates on the normalized text
        - This ensures PHI detection works on clean, standardized patterns
        """
        if not text or not text.strip():
            return ""

        processed_text = text

        # Step 1: Pattern Normalization (if enabled)
        # Standardize medical measurements, units, abbreviations, etc.
        if self.app_state.data_config.enable_pattern_normalization and self.regex_preprocessor:
            try:
                normalized = self.regex_preprocessor.preprocess(processed_text)
                if self.app_state.data_config.save_normalized_text and self.context:
                    self.context.normalized_text = normalized
                processed_text = normalized  # Continue with normalized text
                logger.info("Pattern normalization applied")
            except Exception as e:
                logger.warning(f"Regex preprocessing failed: {e}")

        # Step 2: PHI Redaction (if enabled)
        # Redact PHI from the NORMALIZED text (not original)
        # This ensures redaction operates on standardized patterns
        if self.app_state.data_config.enable_phi_redaction:
            try:
                from core.pii_redactor import create_redactor

                redactor = create_redactor(
                    entity_types=self.app_state.data_config.phi_entity_types,
                    method=self.app_state.data_config.redaction_method
                )

                # IMPORTANT: Redact the normalized text, not the original!
                redacted, redactions = redactor.redact(processed_text)
                if self.app_state.data_config.save_redacted_text and self.context:
                    self.context.redacted_text = redacted
                processed_text = redacted  # Continue with redacted text
                logger.info(f"PHI redaction applied: {len(redactions)} entities")
            except Exception as e:
                logger.warning(f"PHI redaction failed: {e}")

        return processed_text.strip()
