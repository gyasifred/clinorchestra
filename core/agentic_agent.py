#!/usr/bin/env python3
"""
Agentic Agent - Truly Agentic Continuous Loop Extraction System
Version: 1.0.0 - Agentic with Async Tool Execution
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

REVOLUTIONARY CHANGE: From rigid 4-stage pipeline to continuous agentic loop
where LLM autonomously decides what tools to call when, iterates based on
learning, and adapts strategy dynamically.

Architecture:
- Continuous loop with PAUSE/RESUME
- Native tool calling (OpenAI/Anthropic function calling)
- **ASYNC TOOL EXECUTION** - Tools run in parallel for performance
- Multiple calls to same tool with different queries
- Dynamic discovery: "That result tells me I need X"
- Context-aware chaining based on previous results

Phase 2 Features:
- Async/await for parallel tool execution
- Concurrent RAG queries
- Concurrent function calls
- Maintains conversation order while parallelizing execution
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

logger = get_logger(__name__)


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
    """Represents a single tool call"""
    id: str
    type: str  # 'rag', 'function', 'extras'
    name: str
    parameters: Dict[str, Any]
    purpose: Optional[str] = None


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
    max_iterations: int = 20
    total_tool_calls: int = 0
    max_tool_calls: int = 50
    tool_calls_this_iteration: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Preprocessing tracking
    original_text: Optional[str] = None
    redacted_text: Optional[str] = None
    normalized_text: Optional[str] = None


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

        logger.info("AgenticAgent initialized (v1.0.0 - Agentic with Async)")

    def extract(self, clinical_text: str, label_value: Optional[Any] = None) -> Dict[str, Any]:
        """
        Main agentic extraction with continuous loop

        Flow:
        1. Initialize context with clinical text
        2. Start conversation with LLM
        3. Loop:
           - LLM analyzes/continues
           - If LLM requests tools → PAUSE → Execute → RESUME
           - If LLM outputs JSON → Extract and complete
        4. Return result
        """
        try:
            # Initialize context
            label_context = self._get_label_context_string(label_value)
            preprocessed_text = self._preprocess_clinical_text(clinical_text)

            self.context = AgenticContext(
                clinical_text=preprocessed_text,
                label_context=label_context,
                state=AgenticState.IDLE,
                original_text=clinical_text
            )

            logger.info("=" * 80)
            logger.info("AGENTIC EXTRACTION STARTED (v1.0.0 - Async Tools)")
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

            # Start agentic loop
            self.context.state = AgenticState.ANALYZING
            extraction_complete = False

            while not extraction_complete and self.context.iteration < self.context.max_iterations:
                self.context.iteration += 1
                logger.info(f"=" * 60)
                logger.info(f"ITERATION {self.context.iteration}/{self.context.max_iterations}")
                logger.info(f"State: {self.context.state.value}")
                logger.info(f"=" * 60)

                # LLM generates response (may include tool calls or final JSON)
                response = self._generate_with_tools()

                if response is None:
                    logger.error("LLM returned no response")
                    break

                # Parse response
                has_tool_calls, has_json_output = self._parse_llm_response(response)

                if has_tool_calls:
                    # PAUSE - Execute tools
                    logger.info(f"LLM requested {len(self.context.tool_calls_this_iteration)} tools")
                    self.context.state = AgenticState.AWAITING_TOOL_RESULTS

                    tool_results = self._execute_tools(self.context.tool_calls_this_iteration)

                    # RESUME - Add tool results to conversation
                    logger.info(f"Tools executed, resuming with {len(tool_results)} results")
                    self.context.state = AgenticState.CONTINUING

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

                    # Loop continues - LLM can request more tools or finish

                elif has_json_output:
                    # Extraction complete
                    logger.info("LLM provided final JSON output")
                    self.context.state = AgenticState.COMPLETED
                    extraction_complete = True

                else:
                    # LLM is thinking/analyzing - continue
                    logger.info("LLM thinking, continuing conversation")
                    # Add a prompt to encourage completion
                    self.context.conversation_history.append(
                        ConversationMessage(
                            role='user',
                            content="Continue your analysis and call tools as needed, or provide the final JSON when ready."
                        )
                    )

            if self.context.iteration >= self.context.max_iterations:
                logger.warning(f"Max iterations ({self.context.max_iterations}) reached")
                self.context.state = AgenticState.FAILED
                self.context.error = "Max iterations reached"

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

    def _get_tool_schema(self) -> List[Dict[str, Any]]:
        """
        Get tool schema in OpenAI/Anthropic function calling format
        """
        tools = []

        # RAG tool
        if self.rag_engine:
            tools.append({
                'type': 'function',
                'function': {
                    'name': 'query_rag',
                    'description': 'Query retrieval system for clinical guidelines, standards, and reference information from authoritative sources (ASPEN, WHO, CDC). Can be called MULTIPLE times with different queries to refine information.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'Focused query with 4-8 specific medical keywords targeting guidelines, diagnostic criteria, or standards'
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

        # Function tool
        if self.function_registry:
            # Get all available functions
            functions = self.function_registry.get_all_functions_info()
            for func in functions:
                tools.append({
                    'type': 'function',
                    'function': {
                        'name': f"call_{func['name']}",
                        'description': f"{func.get('description', 'Medical calculation function')}. Can be called multiple times for serial measurements.",
                        'parameters': {
                            'type': 'object',
                            'properties': func.get('parameters', {}),
                            'required': list(func.get('parameters', {}).keys())
                        }
                    }
                })

        # Extras tool
        if self.extras_manager:
            tools.append({
                'type': 'function',
                'function': {
                    'name': 'query_extras',
                    'description': 'Query supplementary hints/tips/patterns that help understand the task. Use specific medical keywords to match relevant hints.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'keywords': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': '3-5 specific medical keywords related to your task (e.g., task-specific terms, clinical concepts, assessment types)'
                            }
                        },
                        'required': ['keywords']
                    }
                }
            })

        return tools

    def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute all tool calls in PARALLEL using async

        Phase 2: Tools run concurrently for better performance
        - Multiple RAG queries can run simultaneously
        - Multiple function calls can run simultaneously
        - Results returned in original order for conversation coherence
        """
        logger.info(f"Executing {len(tool_calls)} tools in PARALLEL (async)")

        # Run async execution in event loop
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in Jupyter), use nest_asyncio or run_until_complete
                results = loop.run_until_complete(self._execute_tools_async(tool_calls))
            else:
                results = loop.run_until_complete(self._execute_tools_async(tool_calls))
        except RuntimeError:
            # No event loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self._execute_tools_async(tool_calls))

        # Update tracking
        for result in results:
            self.context.tool_results.append(result)
            self.context.total_tool_calls += 1

        return results

    async def _execute_tools_async(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Async execution of tool calls - runs in parallel

        This is the Phase 2 enhancement that significantly improves performance
        """
        # Create tasks for all tools
        tasks = []
        for tool_call in tool_calls:
            if tool_call.type == 'rag' or tool_call.name == 'query_rag':
                task = self._execute_rag_tool_async(tool_call)
            elif tool_call.type == 'function' or tool_call.name.startswith('call_'):
                task = self._execute_function_tool_async(tool_call)
            elif tool_call.type == 'extras' or tool_call.name == 'query_extras':
                task = self._execute_extras_tool_async(tool_call)
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

        logger.info(f"✅ Executed {len(tool_calls)} tools in {elapsed:.2f}s (parallel)")

        return results

    def _execute_rag_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute RAG query"""
        try:
            query = tool_call.parameters.get('query', '')

            if not query or len(query) < 5:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    type='rag',
                    success=False,
                    result=[],
                    message='Query too short or empty'
                )

            logger.info(f"RAG query: {query}")

            top_k = self.app_state.rag_config.rag_top_k
            results = self.rag_engine.query(query_text=query, k=top_k)

            return ToolResult(
                tool_call_id=tool_call.id,
                type='rag',
                success=len(results) > 0,
                result=results,
                message=f"Retrieved {len(results)} documents"
            )

        except Exception as e:
            logger.error(f"RAG execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                type='rag',
                success=False,
                result=[],
                message=str(e)
            )

    def _execute_function_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute function call"""
        try:
            # Extract function name (remove 'call_' prefix if present)
            func_name = tool_call.name
            if func_name.startswith('call_'):
                func_name = func_name[5:]

            parameters = tool_call.parameters

            logger.info(f"Calling function: {func_name} with {parameters}")

            success, result, message = self.function_registry.execute_function(
                func_name, **parameters
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                type='function',
                success=success,
                result=result,
                message=message
            )

        except Exception as e:
            logger.error(f"Function execution failed: {e}", exc_info=True)
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
            keywords = tool_call.parameters.get('keywords', [])

            logger.info(f"Querying extras with keywords: {keywords}")

            matched_extras = self.extras_manager.match_extras_by_keywords(keywords)

            return ToolResult(
                tool_call_id=tool_call.id,
                type='extras',
                success=len(matched_extras) > 0,
                result=matched_extras,
                message=f"Matched {len(matched_extras)} extras"
            )

        except Exception as e:
            logger.error(f"Extras execution failed: {e}", exc_info=True)
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

        ALWAYS tries main_prompt first, falls back to minimal_prompt only if main is missing.
        """
        from core.prompt_templates import get_agentic_extraction_prompt

        schema_instructions = format_schema_as_instructions(
            self.app_state.prompt_config.json_schema
        )

        # FIXED: Always try MAIN prompt first, fall back to MINIMAL only if MAIN is missing
        user_task_prompt = None

        if self.app_state.prompt_config.main_prompt:
            user_task_prompt = self.app_state.prompt_config.main_prompt
            logger.info("✓ Using MAIN prompt as primary task definition")
        elif self.app_state.prompt_config.minimal_prompt:
            user_task_prompt = self.app_state.prompt_config.minimal_prompt
            logger.info("⚠ Using MINIMAL prompt as fallback (main_prompt not set)")
        elif self.app_state.prompt_config.base_prompt:
            user_task_prompt = self.app_state.prompt_config.base_prompt
            logger.info("⚠ Using BASE prompt as fallback (main and minimal not set)")
        else:
            user_task_prompt = ""
            logger.warning("❌ No user task-specific prompt found - using generic agentic prompt")

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
        return """You are a board-certified clinical expert performing information extraction from medical text.

You have access to tools:
- query_rag: Retrieve clinical guidelines and standards from authoritative sources
- call_[function]: Perform medical calculations (BMI, z-scores, creatinine clearance, etc.)
- query_extras: Get supplementary hints and reference ranges

**UNIVERSAL ITERATIVE WORKFLOW:**

🔴 CRITICAL: You MUST use an iterative, self-reflective approach! 🔴

**PHASE 1 - INITIAL ANALYSIS:**
1. Read the task prompt → Understand what needs to be extracted
2. Read the clinical text → Identify key metrics, measurements, words, entities
3. Build queries and execute INITIAL tool calls:
   - Call functions for calculations (BMI, z-scores, etc.)
   - Call query_extras for task-specific hints

**PHASE 2 - BUILD CONTEXT:**
4. Review function results and extras hints
5. Build RAG keywords based on what you learned (domain, criteria needed, guidelines)
6. Call query_rag to fetch clinical guidelines and standards

**PHASE 3 - ASSESS INFORMATION GAPS:**
7. Determine: Do I have ALL information needed to complete the task?
   - ✅ YES: Proceed to Phase 4
   - ❌ NO: Go to Phase 3b

**PHASE 3b - FILL GAPS:**
8. Identify what other information is needed
9. Determine which tools to call again (can call same tool with different queries)
10. Fetch additional information → Return to Phase 3

**PHASE 4 - COMPLETION:**
11. When you have all necessary information → Output final JSON extraction

**Key Principles:**
- ❌ DO NOT output final JSON immediately
- ✅ Call tools iteratively across multiple rounds
- ✅ Use tool results to inform next tool calls
- ✅ Assess information gaps after each round
- ✅ Only complete task when confident you have everything needed

**CRITICAL: Tool Call Format**
- Tool calls MUST start with "TOOL_CALL:" prefix
- Format: TOOL_CALL: {{"tool": "name", "parameters": {{...}}}}
- Do NOT output bare JSON without the TOOL_CALL: prefix"""

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
                    # Looks like JSON but was rejected - provide feedback
                    logger.info("LLM output contains JSON but was rejected (schema validation failed)")
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
            logger.warning("⚠️ Detected tool call JSON without TOOL_CALL: prefix - rejecting as final output")
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
                    logger.warning(f"⚠️ Parsed JSON missing all required fields: {required_fields}")
                    logger.warning(f"Parsed JSON keys: {list(parsed.keys())}")
                    return None

        return parsed

    def _convert_history_to_api_format(self) -> List[Dict[str, Any]]:
        """Convert conversation history to API format"""
        messages = []

        for msg in self.context.conversation_history:
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
        """Convert conversation history to text for non-tool-calling providers"""
        lines = []
        for msg in self.context.conversation_history:
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

        lines.append("\n🔍 **THREE TOOL CATEGORIES:**\n")

        lines.append("1️⃣ **RAG (Retrieval-Augmented Generation)** - query_rag()")
        lines.append("   • Retrieves clinical guidelines, diagnostic criteria, standards")
        lines.append("   • Build queries from: medical domain + specificity + what you need")
        lines.append("   • Example: 'ASPEN pediatric malnutrition severity criteria'\n")

        lines.append("2️⃣ **FUNCTIONS (Medical Calculations)** - call_[function_name]()")
        lines.append("   • Performs medical calculations (see full list below)")
        lines.append("   • Scan text for measurements, match to function parameters")
        lines.append("   • CRITICAL: Call functions BEFORE making clinical interpretations\n")

        lines.append("3️⃣ **EXTRAS (Supplementary Hints)** - query_extras()")
        lines.append("   • Task-specific hints, reference ranges, interpretation guides")
        lines.append("   • Build keywords from: schema field names + clinical domain + guidelines")
        lines.append("   • Use 3-5 specific medical terms, avoid generic words\n")

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
                    lines.append(f"     • {param_name} ({param_type}): {param_desc}")
            lines.append("")

        # Add clear instructions with examples
        lines.append("=" * 80)
        lines.append("📝 HOW TO CALL TOOLS:")
        lines.append("=" * 80)
        lines.append("\n🔴 CRITICAL: Tool calls MUST start with 'TOOL_CALL:' prefix! 🔴\n")
        lines.append("REQUIRED FORMAT:")
        lines.append('TOOL_CALL: {"tool": "tool_name", "parameters": {...}}')
        lines.append("")
        lines.append("✅ CORRECT EXAMPLES (note the 'TOOL_CALL:' prefix):")
        lines.append('TOOL_CALL: {"tool": "query_rag", "parameters": {"query": "ASPEN pediatric malnutrition criteria", "purpose": "need severity thresholds"}}')
        lines.append('TOOL_CALL: {"tool": "call_percentile_to_zscore", "parameters": {"percentile": 10}}')
        lines.append('TOOL_CALL: {"tool": "query_extras", "parameters": {"keywords": ["malnutrition", "pediatric", "z-score", "WHO", "ASPEN"]}}')
        lines.append("")
        lines.append("❌ WRONG (missing 'TOOL_CALL:' prefix):")
        lines.append('{"tool": "query_rag", ...}  ← MISSING TOOL_CALL: prefix!')
        lines.append("")
        lines.append("❌ WRONG (missing closing brace):")
        lines.append('TOOL_CALL: {"tool": "query_rag", "parameters": {"query": "test"}  ← MISSING }')
        lines.append("")
        lines.append("💡 ITERATIVE WORKFLOW:")
        lines.append("ROUND 1 - Initial Analysis:")
        lines.append("  • Analyze task prompt + clinical text")
        lines.append("  • Identify key metrics/measurements/words")
        lines.append("  • Call functions (calculations) + query_extras (hints)")
        lines.append("")
        lines.append("ROUND 2 - Build Context:")
        lines.append("  • Review function/extras results")
        lines.append("  • Build RAG keywords from what you learned")
        lines.append("  • Call query_rag (fetch guidelines/criteria)")
        lines.append("")
        lines.append("ROUND 3+ - Assess & Fill Gaps:")
        lines.append("  • Assess: Do I have ALL info needed?")
        lines.append("  • If NO: Identify gaps → Call more tools → Reassess")
        lines.append("  • If YES: Output final JSON extraction")
        lines.append("=" * 80 + "\n")

        # Add conversation history
        for msg in self.context.conversation_history:
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
                        logger.warning(f"Could not extract complete JSON from: {match_text[:100]}")
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
                    logger.info(f"✓ Parsed tool call: {tool_name}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool call (invalid JSON): {match_text[:100]}... | Error: {e}")
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {match_text[:100]}... | Error: {e}")

            if tool_calls:
                logger.info(f"✅ Successfully parsed {len(tool_calls)} tool calls from text response")
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
        """Build final extraction result"""
        return {
            'original_clinical_text': self.context.original_text,
            'clinical_text': self.context.clinical_text,
            'redacted_text': self.context.redacted_text,
            'normalized_text': self.context.normalized_text,
            'input_label_value': None,
            'label_context': self.context.label_context,
            'stage3_output': self.context.final_output or {},
            'stage4_final_output': self.context.final_output or {},
            'agentic_metadata': {
                'version': '1.0.0',
                'execution_mode': 'agentic_async',
                'iterations': self.context.iteration,
                'total_tool_calls': self.context.total_tool_calls,
                'final_state': self.context.state.value,
                'tool_results_count': len(self.context.tool_results),
                'conversation_length': len(self.context.conversation_history)
            },
            'processing_metadata': {
                'had_label': self.context.label_context is not None,
                'final_state': self.context.state.value,
                'agentic_execution': True
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
        """Preprocess clinical text"""
        if not text or not text.strip():
            return ""

        processed_text = text

        # Pattern normalization
        if self.app_state.data_config.enable_pattern_normalization and self.regex_preprocessor:
            try:
                normalized = self.regex_preprocessor.preprocess(processed_text)
                if self.app_state.data_config.save_normalized_text and self.context:
                    self.context.normalized_text = normalized
                processed_text = normalized
                logger.info("Pattern normalization applied")
            except Exception as e:
                logger.warning(f"Regex preprocessing failed: {e}")

        # PHI redaction
        if self.app_state.data_config.enable_phi_redaction:
            try:
                from core.pii_redactor import create_redactor

                redactor = create_redactor(
                    entity_types=self.app_state.data_config.phi_entity_types,
                    method=self.app_state.data_config.redaction_method
                )

                redacted, redactions = redactor.redact(processed_text)
                if self.app_state.data_config.save_redacted_text and self.context:
                    self.context.redacted_text = redacted
                processed_text = redacted
                logger.info(f"PHI redaction applied: {len(redactions)} entities")
            except Exception as e:
                logger.warning(f"PHI redaction failed: {e}")

        return processed_text.strip()
