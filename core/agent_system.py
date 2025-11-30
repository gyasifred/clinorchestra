#!/usr/bin/env python3
"""
STRUCTURED Execution Mode - Four-Stage Extraction Pipeline

Systematic 4-stage pipeline for predictable, production-ready clinical data extraction.
Uses LLM-powered autonomous task analysis, tool orchestration, and structured JSON generation.

Stages:
1. Task Analysis - Determine required tools and generate queries
2. Tool Execution - Run functions, RAG, extras in parallel (async)
3. Extraction - Generate structured JSON output
4. RAG Refinement - Optional evidence-based field enhancement

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import json
import time
import re
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from core.json_parser import JSONParser
from core.prompt_templates import (
    get_stage1_analysis_prompt_template,
    get_default_rag_refinement_prompt,
    format_schema_as_instructions,
    format_tool_outputs_for_prompt
)
from core.logging_config import get_logger, log_extraction_stage
from core.performance_monitor import get_performance_monitor, TimingContext
from core.tool_dedup_preventer import create_tool_dedup_preventer
from core.adaptive_retry import AdaptiveRetryManager, create_retry_context
from core.model_tier_helper import get_model_for_stage

logger = get_logger(__name__)
perf_monitor = get_performance_monitor(enabled=True)


class ExtractionAgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    TOOL_EXECUTION = "tool_execution"
    STAGE3_EXTRACTION = "stage3_extraction"
    STAGE4_RAG_REFINEMENT = "stage4_rag_refinement"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractionAgentContext:
    """Context for agent execution"""
    clinical_text: str
    label_value: Optional[Any]
    label_context: Optional[str]
    state: ExtractionAgentState
    task_understanding: Dict[str, Any]
    tool_requests: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    stage3_output: Dict[str, Any]
    stage4_final_output: Dict[str, Any]
    retry_count: int
    using_minimal_prompt: bool
    pause_reason: Optional[str]
    prompt_variables: Dict[str, Any] = None  # NEW: Additional columns for prompt variables
    last_raw_response: Optional[str] = None
    parsing_method_used: Optional[str] = None
    original_text: Optional[str] = None
    redacted_text: Optional[str] = None
    normalized_text: Optional[str] = None
    stage4_additional_tool_results: List[Dict[str, Any]] = None  # NEW: Track Stage 4 gap-filling tools
    

class ExtractionAgent:
    """
     UNIVERSAL AGENTIC SYSTEM - Works for ANY clinical extraction task

    This agent is task-agnostic and dynamically determines:
    - Required information based on YOUR task (defined in schema/prompts)
    - Which functions to call from registry (based on YOUR task needs)
    - Optimal RAG queries for YOUR clinical domain
    - Extras keywords matching YOUR task context
    - Execution strategy tailored to YOUR extraction goals

    Not hardcoded for any specific condition - adapts to YOUR use case!
    """
    def __init__(self, llm_manager, rag_engine, extras_manager, function_registry, 
                 regex_preprocessor, app_state):
        """Initialize agent"""
        self.llm_manager = llm_manager
        self.rag_engine = rag_engine
        self.extras_manager = extras_manager
        self.function_registry = function_registry
        self.regex_preprocessor = regex_preprocessor
        self.app_state = app_state
        
        self.context: Optional[ExtractionAgentContext] = None
        self.max_retries = 3
        self.json_parser = JSONParser()

        # Initialize tool deduplication preventer
        self.tool_dedup_preventer = None  # Created per extraction

        # Initialize adaptive retry manager
        self.retry_manager = AdaptiveRetryManager(max_retries=5)

        logger.info(" ExtractionAgent v1.0.0 initialized - STRUCTURED Mode (predictable workflows with ASYNC)")
        logger.info(" Enhanced with: Adaptive Retry + Proactive Tool Deduplication")
        
    
    def extract(self, clinical_text: str, label_value: Optional[Any] = None,
                prompt_variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main extraction method with universal agentic behavior

        Args:
            clinical_text: The main clinical text to analyze
            label_value: Optional label/diagnosis value
            prompt_variables: Optional dict of additional columns to pass as prompt variables
        """
        try:
            # Initialize context
            label_context = self._get_label_context_string(label_value)
            self.context = ExtractionAgentContext(
                clinical_text=clinical_text,
                label_value=label_value,
                label_context=label_context,
                state=ExtractionAgentState.ANALYZING,
                task_understanding={},
                tool_requests=[],
                tool_results=[],
                stage3_output={},
                stage4_final_output={},
                retry_count=0,
                using_minimal_prompt=False,
                pause_reason=None,
                prompt_variables=prompt_variables or {},  # NEW: Store prompt variables
                original_text=clinical_text
            )

            # Initialize tool deduplication preventer for this extraction
            # Use a high budget for STRUCTURED mode since tools are planned upfront
            self.tool_dedup_preventer = create_tool_dedup_preventer(max_tool_calls=200)
            
            # Preprocess the clinical text (skip if batch preprocessing was already done)
            if self.app_state.optimization_config.use_batch_preprocessing:
                # Text is already preprocessed by batch processor - use as-is
                preprocessed_text = clinical_text
                self.context.clinical_text = preprocessed_text
                # Note: redacted/normalized texts are stored in processing_tab output_handler
            else:
                # Individual preprocessing (row-by-row mode)
                preprocessed_text = self._preprocess_clinical_text(clinical_text)
                self.context.clinical_text = preprocessed_text
            
            logger.info("=" * 60)
            logger.info("STAGE 1: TASK ANALYSIS & TOOL PLANNING")
            logger.info("=" * 60)
            
            # Stage 1: LLM analyzes task and determines requirements
            self.context.state = ExtractionAgentState.ANALYZING
            with TimingContext('structured_stage1_analysis'):
                analysis_success = self._execute_stage1_analysis()

            if not analysis_success:
                return self._build_failure_result("Stage 1 analysis failed")
            
            logger.info("=" * 60)
            logger.info(f"STAGE 2: EXECUTING {len(self.context.tool_requests)} TOOL REQUESTS")
            logger.info("=" * 60)
            
            # Stage 2: Tool execution
            self.context.state = ExtractionAgentState.TOOL_EXECUTION
            with TimingContext('structured_stage2_tools'):
                self._execute_stage2_tools()

            logger.info("=" * 60)
            logger.info("STAGE 3: EXTRACTION")
            logger.info("=" * 60)

            # Stage 3: Extraction
            self.context.state = ExtractionAgentState.STAGE3_EXTRACTION
            with TimingContext('structured_stage3_extraction'):
                stage3_success = self._execute_stage3_extraction()

            if not stage3_success:
                return self._build_failure_result("Stage 3 extraction failed")

            # Stage 4: RAG refinement (optional)
            if self._should_run_rag_refinement():
                logger.info("=" * 60)
                logger.info("STAGE 4: RAG REFINEMENT")
                logger.info("=" * 60)

                self.context.state = ExtractionAgentState.STAGE4_RAG_REFINEMENT
                with TimingContext('structured_stage4_rag'):
                    self.context.stage4_final_output = self._execute_stage4_rag_refinement_with_retry()
            else:
                self.context.stage4_final_output = self.context.stage3_output
            
            self.context.state = ExtractionAgentState.COMPLETED
            return self._build_extraction_result()
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            if self.context:
                self.context.state = ExtractionAgentState.FAILED
            return self._build_failure_result(f"Extraction error: {str(e)}")
    
    @log_extraction_stage("Stage 1: Task Analysis & Tool Planning")
    def _execute_stage1_analysis(self) -> bool:
        """
        Stage 1: LLM analyzes task and determines:
        - Required information
        - Functions to call
        - RAG queries
        - Extras keywords
        """
        try:
            analysis_prompt = self._build_stage1_analysis_prompt()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Analysis attempt {attempt + 1}/{self.max_retries}")

                    # TIER 1.2: Use tiered models - fast model for Stage 1 planning
                    stage_model = get_model_for_stage(self.app_state, stage=1)

                    response = self.llm_manager.generate(
                        analysis_prompt,
                        max_tokens=self.app_state.model_config.max_tokens,
                        override_model_name=stage_model
                    )
                    
                    if not response:
                        logger.error("No response from LLM")
                        continue
                    
                    self.context.last_raw_response = response
                    
                    # Parse task understanding
                    task_understanding = self._parse_task_analysis_response(response)
                    
                    if task_understanding:
                        self.context.task_understanding = task_understanding
                        
                        # Convert to tool requests with fallback query generation
                        self._convert_task_understanding_to_tool_requests()

                        # ENHANCED: Validate and improve RAG queries
                        self._validate_and_improve_rag_queries()

                        # SOFT LIMIT: Warn if Stage 1 planned excessive tools
                        STAGE1_TOOL_BUDGET = 50  # Recommended limit for Stage 1 planning
                        if len(self.context.tool_requests) > STAGE1_TOOL_BUDGET:
                            logger.warning(
                                f"️ STAGE 1 COMPLEXITY WARNING: Planned {len(self.context.tool_requests)} tools "
                                f"(recommended: ≤{STAGE1_TOOL_BUDGET}). Consider reviewing task complexity or "
                                f"optimizing tool selection to reduce processing time and costs."
                            )

                        logger.info(f"Task analysis successful")
                        logger.info(f"Generated {len(self.context.tool_requests)} tool requests")
                        return True
                    
                    logger.warning(f"Failed to parse task analysis, attempt {attempt + 1}")
                    
                except Exception as e:
                    logger.error(f"Analysis attempt {attempt + 1} failed: {e}")
            
            # If all attempts fail, create default tool requests with smart fallbacks
            logger.warning("All analysis attempts failed, using intelligent defaults")
            self.context.task_understanding = self._create_default_task_understanding()
            self._convert_task_understanding_to_tool_requests()
            self._generate_fallback_rag_queries()  # ENHANCED: Generate smart queries
            self._generate_fallback_extras_keywords()  # ENHANCED: Generate smart extras keywords
            return True
            
        except Exception as e:
            logger.error(f"Stage 1 analysis failed: {e}", exc_info=True)
            return False
    
    def _validate_and_improve_rag_queries(self):
        """
        ENHANCED: Validate RAG queries and improve them if they're too generic or empty
        """
        rag_requests = [r for r in self.context.tool_requests if r.get('type') == 'rag']
        
        if not rag_requests:
            # No RAG queries requested, generate fallback
            logger.info("No RAG queries found, generating fallback queries")
            self._generate_fallback_rag_queries()
            return
        
        improved_count = 0
        for req in rag_requests:
            query = req.get('query', '').strip()
            
            # Check if query is too short or generic
            if len(query) < 10 or query.lower() in ['help', 'information', 'data', 'guidelines']:
                # Extract better query from context
                improved_query = self._extract_smart_query_from_context()
                if improved_query and improved_query != query:
                    req['query'] = improved_query
                    req['purpose'] = f"Auto-improved query for context-specific information"
                    improved_count += 1
                    logger.info(f"Improved RAG query: '{query}' -> '{improved_query}'")
        
        if improved_count > 0:
            logger.info(f"Improved {improved_count} RAG queries")
    
    def _generate_fallback_rag_queries(self):
        """
        ENHANCED: Generate intelligent fallback RAG queries when LLM fails
        Extract meaningful terms from schema, label_context, and text
        """
        queries = []
        
        # Extract from schema field names and descriptions
        schema_terms = self._extract_key_terms_from_schema()
        
        # Extract from label context
        label_terms = self._extract_key_terms_from_label()
        
        # Extract from clinical text
        text_terms = self._extract_key_terms_from_text()
        
        # Combine terms intelligently
        all_terms = set(schema_terms + label_terms + text_terms)
        
        if all_terms:
            # Create focused query from top terms
            query_terms = list(all_terms)[:5]  # Top 5 most relevant
            query = " ".join(query_terms)
            
            queries.append({
                'type': 'rag',
                'query': query,
                'purpose': 'Retrieve relevant guidelines and standards for extraction task'
            })
            
            logger.info(f"Generated fallback RAG query: '{query}'")
        
        # Add queries to tool requests
        for query in queries:
            if not any(r.get('query') == query.get('query') for r in self.context.tool_requests):
                self.context.tool_requests.append(query)
    
    def _generate_fallback_extras_keywords(self):
        """
        ENHANCED: Generate intelligent fallback extras keywords when LLM fails
        Extras are hints/tips that help LLM understand the task better
        """
        keywords = []
        
        # Extract from schema field names (these are task-specific concepts)
        schema = self.app_state.prompt_config.json_schema
        if schema:
            for field_name in schema.keys():
                # Split camelCase/snake_case into words
                field_words = re.findall(r'[A-Z][a-z]+|[a-z]+', field_name.replace('_', ' '))
                keywords.extend([w.lower() for w in field_words if len(w) > 3])
        
        # Extract from label context (classification/category)
        if self.context.label_context:
            label_words = re.findall(r'\b[A-Za-z]{4,}\b', self.context.label_context)
            keywords.extend([w.lower() for w in label_words[:3]])
        
        # Remove duplicates and limit to top 5
        unique_keywords = list(set(keywords))[:5]
        
        if unique_keywords:
            # Add extras request
            extras_request = {
                'type': 'extras',
                'keywords': unique_keywords
            }
            
            # Only add if not already present
            if not any(r.get('type') == 'extras' for r in self.context.tool_requests):
                self.context.tool_requests.append(extras_request)
                logger.info(f"Generated fallback extras keywords: {unique_keywords}")
    
    def _extract_smart_query_from_context(self) -> str:
        """
        ENHANCED: Extract a smart query from the current context
        """
        terms = []
        
        # Get terms from schema
        schema_terms = self._extract_key_terms_from_schema()
        terms.extend(schema_terms[:3])
        
        # Get terms from label
        label_terms = self._extract_key_terms_from_label()
        terms.extend(label_terms[:2])
        
        # Get terms from text
        text_terms = self._extract_key_terms_from_text()
        terms.extend(text_terms[:2])
        
        if terms:
            return " ".join(set(terms))
        
        return ""
    
    def _extract_key_terms_from_schema(self) -> List[str]:
        """
        ENHANCED: Extract meaningful terms from JSON schema
        """
        terms = []
        schema = self.app_state.prompt_config.json_schema
        
        if not schema:
            return terms
        
        for field_name, field_spec in schema.items():
            # Add field name (split camelCase/snake_case)
            field_words = re.findall(r'[A-Z][a-z]+|[a-z]+', field_name.replace('_', ' '))
            terms.extend([w.lower() for w in field_words if len(w) > 3])
            
            # Add description words
            if isinstance(field_spec, dict):
                desc = field_spec.get('description', '')
                if desc:
                    desc_words = re.findall(r'\b[a-zA-Z]{4,}\b', desc)
                    terms.extend([w.lower() for w in desc_words[:3]])
        
        # Remove common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 
                     'your', 'their', 'what', 'which', 'when', 'where', 'does'}
        terms = [t for t in terms if t not in stopwords]
        
        return list(set(terms))[:5]
    
    def _extract_key_terms_from_label(self) -> List[str]:
        """
        ENHANCED: Extract meaningful terms from label context
        """
        terms = []
        
        if not self.context.label_context:
            return terms
        
        # Extract capitalized terms and medical terms
        label_words = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', self.context.label_context)
        terms.extend([w.lower() for w in label_words])
        
        # Remove stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will'}
        terms = [t for t in terms if t not in stopwords]
        
        return list(set(terms))[:3]
    
    def _extract_key_terms_from_text(self) -> List[str]:
        """
        ENHANCED: Extract meaningful terms from clinical text
        Uses frequency analysis and medical term detection
        """
        terms = []
        text = self.context.clinical_text[:1000]  # First 1000 chars
        
        # Extract capitalized medical terms and longer words
        words = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', text)
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Sort by frequency and take top terms
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Filter stopwords
        stopwords = {'patient', 'history', 'noted', 'reported', 'presented', 
                     'states', 'denies', 'described', 'appeared', 'documented'}
        
        for term, freq in sorted_terms:
            if term not in stopwords and len(term) > 4:
                terms.append(term)
                if len(terms) >= 5:
                    break
        
        return terms
    
    def _convert_task_understanding_to_tool_requests(self):
        """Convert task understanding to tool requests"""
        task_understanding = self.context.task_understanding

        # Add function requests
        for func_info in task_understanding.get('functions_needed', []):
            tool_request = {
                'type': 'function',
                'name': func_info.get('name'),
                'parameters': func_info.get('parameters', {}),
                'reason': func_info.get('reason', '')
            }
            # Preserve date_context for serial measurements
            if 'date_context' in func_info:
                tool_request['date_context'] = func_info.get('date_context')

            self.context.tool_requests.append(tool_request)

        # Add RAG requests
        for query_info in task_understanding.get('rag_queries', []):
            query = query_info.get('query', '').strip()
            if query:  # Only add non-empty queries
                self.context.tool_requests.append({
                    'type': 'rag',
                    'query': query,
                    'purpose': query_info.get('purpose', '')
                })

        # Add extras request
        keywords = task_understanding.get('extras_keywords', [])
        if keywords:
            self.context.tool_requests.append({
                'type': 'extras',
                'keywords': keywords
            })

    def _deduplicate_tool_requests(self, tool_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate tool requests to prevent executing identical calls multiple times.

        Two tool requests are considered identical if they have:
        - Same type (function/rag/extras)
        - Same name
        - Same parameters (compared as JSON strings for deterministic comparison)

        This prevents the agent from getting stuck calling the same function repeatedly
        with identical parameters, which is a common LLM behavior issue.

        Returns:
            List of unique tool requests (keeps first occurrence of each duplicate)
        """
        seen = {}
        unique_requests = []
        duplicates_removed = 0

        for tool_request in tool_requests:
            # Create a unique key based on type, name, and parameters
            tool_type = tool_request.get('type', '')
            tool_name = tool_request.get('name', '')
            params = tool_request.get('parameters', {})

            # Sort parameters for consistent comparison
            params_json = json.dumps(params, sort_keys=True)
            key = f"{tool_type}||{tool_name}||{params_json}"

            if key not in seen:
                seen[key] = tool_request
                unique_requests.append(tool_request)
            else:
                duplicates_removed += 1
                logger.warning(
                    f" DUPLICATE TOOL REQUEST detected and skipped: "
                    f"{tool_type}.{tool_name} with parameters {params_json}"
                )

        if duplicates_removed > 0:
            logger.warning(
                f" Removed {duplicates_removed} duplicate tool requests "
                f"(kept {len(unique_requests)} unique requests from {len(tool_requests)} total)"
            )

        return unique_requests

    def _validate_tool_parameters(self, tool_requests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        UNIVERSAL parameter validation - works for ANY function, not task-specific.

        Validates that function calls have all required parameters with valid values.
        Filters out invalid requests to prevent execution errors.

        Returns:
            Tuple of (valid_requests, filtered_results)
            - valid_requests: List of tool requests that passed validation
            - filtered_results: List of failure results for LLM to understand what was skipped
        """
        valid_requests = []
        filtered_results = []
        invalid_count = 0

        for req in tool_requests:
            tool_type = req.get('type', '')

            # Only validate function calls (RAG and extras don't have strict requirements)
            if tool_type != 'function':
                valid_requests.append(req)
                continue

            func_name = req.get('name', '')
            provided_params = req.get('parameters', {})

            # Get function definition from registry
            try:
                func_def = self.function_registry.get_function_definition(func_name)
            except:
                logger.warning(f"Function '{func_name}' not found in registry, skipping")
                invalid_count += 1
                # Add to filtered results for LLM
                filtered_results.append({
                    'type': 'function',
                    'name': func_name,
                    'success': False,
                    'message': f"Function '{func_name}' not found in registry",
                    'parameters': provided_params,
                    'filtered_at_validation': True
                })
                continue

            if not func_def or 'parameters' not in func_def:
                # No parameter spec available - allow it through (might be simple function)
                valid_requests.append(req)
                continue

            # Check required parameters
            required_params = []
            for param_name, param_spec in func_def['parameters'].items():
                if isinstance(param_spec, dict) and param_spec.get('required', False):
                    required_params.append(param_name)

            # Find missing or invalid parameters
            missing_or_invalid = []
            for param in required_params:
                value = provided_params.get(param)

                # Check if parameter is missing or has invalid value
                if value is None:
                    missing_or_invalid.append(f"{param}=None")
                elif value == '':
                    missing_or_invalid.append(f"{param}=''")
                elif isinstance(value, (int, float)) and value == 0:
                    # Zero might be invalid for some functions (e.g., creatinine_clearance)
                    # But valid for others - we log it but allow through
                    logger.debug(f"Function '{func_name}': parameter '{param}' = 0 (may cause issues)")

            if missing_or_invalid:
                logger.warning(
                    f"Skipping function '{func_name}': "
                    f"missing/invalid required parameters: {', '.join(missing_or_invalid)}"
                )
                logger.debug(f"   Required: {required_params}")
                logger.debug(f"   Provided: {list(provided_params.keys())}")
                invalid_count += 1

                # CRITICAL: Add to filtered results so LLM sees what was skipped and why
                filtered_results.append({
                    'type': 'function',
                    'name': func_name,
                    'success': False,
                    'message': f"Missing required parameter(s): {', '.join([p.split('=')[0] for p in missing_or_invalid])}",
                    'parameters': provided_params,
                    'filtered_at_validation': True,
                    'required_parameters': required_params
                })
                continue  # Skip this function call

            # If we get here, request is valid
            valid_requests.append(req)

        if invalid_count > 0:
            logger.info(
                f"Parameter validation: {len(tool_requests)} → {len(valid_requests)} "
                f"({invalid_count} invalid requests filtered)"
            )

        return valid_requests, filtered_results

    @log_extraction_stage("Stage 2: Tool Execution")
    def _execute_stage2_tools(self):
        """
        Execute all tool requests in PARALLEL using async

        PERFORMANCE ENHANCEMENT: Tools run concurrently for 60-75% speedup
        - Multiple function calls can run simultaneously
        - Multiple RAG queries can run simultaneously
        - Multiple extras queries can run simultaneously
        """
        if not self.context.tool_requests:
            logger.info("No tool requests to execute")
            return

        # CRITICAL FIX: Deduplicate tool requests before execution using preventer
        original_count = len(self.context.tool_requests)

        if self.tool_dedup_preventer:
            # Use advanced deduplication preventer
            unique_requests, num_duplicates = self.tool_dedup_preventer.filter_duplicates(self.context.tool_requests)
            self.context.tool_requests = unique_requests

            if num_duplicates > 0:
                logger.warning(f"️ PREVENTED {num_duplicates} DUPLICATE TOOL REQUESTS IN STAGE 2")
                logger.info(f"   Original: {original_count} requests → Unique: {len(unique_requests)} requests")

            # Log budget status
            logger.info(f" {self.tool_dedup_preventer.get_budget_status()}")
        else:
            # Fallback to old deduplication
            self.context.tool_requests = self._deduplicate_tool_requests(self.context.tool_requests)

        # UNIVERSAL PARAMETER VALIDATION: Filter out invalid function calls
        # Works for ANY function - prevents execution errors from missing parameters
        # Also returns filtered results to inform LLM about skipped tools
        self.context.tool_requests, filtered_results = self._validate_tool_parameters(self.context.tool_requests)

        # Add filtered results to tool_results so LLM knows what was skipped
        if filtered_results:
            self.context.tool_results.extend(filtered_results)
            logger.info(f"Added {len(filtered_results)} filtered tool failure(s) to results for LLM feedback")

        if len(self.context.tool_requests) == 0:
            logger.warning("No valid tool requests to execute after validation")
            return

        logger.info(f"Executing {len(self.context.tool_requests)} tools in PARALLEL (async)")
        if original_count > len(self.context.tool_requests):
            logger.info(f" Deduplicated: {original_count} -> {len(self.context.tool_requests)} requests (removed {original_count - len(self.context.tool_requests)} duplicates)")

        # Run async execution - asyncio.run() handles event loop creation automatically
        try:
            # asyncio.run() creates new event loop and cleans up after
            # This is the recommended way in Python 3.7+
            results = asyncio.run(self._execute_stage2_tools_async())
        except RuntimeError as e:
            # Event loop already running (e.g., Jupyter notebook)
            if "cannot be called from a running event loop" in str(e):
                logger.warning(" Event loop already running - falling back to sequential execution")
                logger.warning("   (Async optimization disabled in this context)")
                # Fall back to sequential synchronous execution
                results = []
                for tool_request in self.context.tool_requests:
                    tool_type = tool_request.get('type')
                    if tool_type == 'function':
                        results.append(self._execute_function_tool(tool_request))
                    elif tool_type == 'rag':
                        results.append(self._execute_rag_tool(tool_request))
                    elif tool_type == 'extras':
                        results.append(self._execute_extras_tool(tool_request))
                    else:
                        results.append({
                            'type': tool_type,
                            'success': False,
                            'message': f"Unknown tool type: {tool_type}"
                        })
            else:
                # Different RuntimeError - re-raise
                raise

        # Store results in order
        self.context.tool_results.extend(results)

    async def _execute_stage2_tools_async(self) -> List[Dict[str, Any]]:
        """
        Async execution of Stage 2 tools - runs in parallel

        This provides significant performance improvement when multiple tools are requested
        """
        tasks = []

        for tool_request in self.context.tool_requests:
            tool_type = tool_request.get('type')

            if tool_type == 'function':
                task = self._execute_function_tool_async(tool_request)
            elif tool_type == 'rag':
                task = self._execute_rag_tool_async(tool_request)
            elif tool_type == 'extras':
                task = self._execute_extras_tool_async(tool_request)
            else:
                # Unknown tool type - create error result
                async def unknown_tool():
                    return {
                        'type': tool_type,
                        'success': False,
                        'message': f"Unknown tool type: {tool_type}"
                    }
                task = unknown_tool()

            tasks.append(task)

        # Execute all tasks in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        logger.info(f" Executed {len(tasks)} tools in {elapsed:.2f}s (parallel)")

        return results
    
    def _execute_function_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        try:
            func_name = tool_request.get('name')
            parameters = tool_request.get('parameters', {})
            date_context = tool_request.get('date_context', '')

            logger.info(f"Executing function: {func_name}")
            if date_context:
                logger.info(f"Date context: {date_context}")
            logger.debug(f"Parameters: {parameters}")

            # CRITICAL FIX: Use the SAME function_registry instance consistently
            # Don't switch between app_state and self - causes call_stack desync
            success, result, message = self.function_registry.execute_function(
                func_name, **parameters
            )

            result_dict = {
                'type': 'function',
                'name': func_name,
                'success': success,
                'result': result,
                'message': message,
                'parameters': parameters  # CRITICAL: Include parameters for error analysis
            }

            # Preserve date_context for serial measurements
            if date_context:
                result_dict['date_context'] = date_context

            return result_dict

        except Exception as e:
            logger.error(f"Function execution error: {e}", exc_info=True)
            result_dict = {
                'type': 'function',
                'name': tool_request.get('name'),
                'success': False,
                'result': None,
                'message': str(e),
                'parameters': tool_request.get('parameters', {})  # CRITICAL: Include attempted parameters
            }

            # Preserve date_context even on error
            if 'date_context' in tool_request:
                result_dict['date_context'] = tool_request.get('date_context')

            return result_dict
    
    def _execute_rag_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG query using correct rag_top_k from RAGConfig"""
        try:
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
                return {
                    'type': 'rag',
                    'success': False,
                    'results': [],
                    'message': error_msg
                }

            query = tool_request.get('query', '').strip()
            
            if not query or len(query) < 5:
                logger.warning(f"RAG query too short or empty: '{query}'")
                return {
                    'type': 'rag',
                    'query': query,
                    'success': False,
                    'results': [],
                    'message': 'Query too short or empty'
                }
            
            logger.info(f"Executing RAG query: {query}")

            # Use RAGConfig.rag_top_k
            top_k = self.app_state.rag_config.rag_top_k
            logger.debug(f"Using rag_top_k = {top_k} from RAGConfig")

            results = self.rag_engine.query(
                query_text=query,
                k=top_k
            )

            return {
                'type': 'rag',
                'query': query,
                'success': len(results) > 0,
                'results': results,
                'count': len(results),
                'top_k_used': top_k
            }

        except Exception as e:
            logger.error(f"RAG execution error: {e}", exc_info=True)
            return {
                'type': 'rag',
                'query': tool_request.get('query', ''),
                'success': False,
                'results': [],
                'error': str(e)
            }
    
    def _execute_extras_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute extras query - Extras are supplementary hints/tips to help LLM understand the task
        """
        try:
            if not self.extras_manager:
                return {
                    'type': 'extras',
                    'success': False,
                    'results': [],
                    'message': 'Extras manager not available'
                }
            
            keywords = tool_request.get('keywords', [])
            logger.info(f"Executing extras query with {len(keywords)} keywords: {keywords}")
            
            # ENHANCED: Match extras based on keywords (supplementary hints)
            matched_extras = self.extras_manager.match_extras_by_keywords(keywords)
            
            logger.info(f"Matched {len(matched_extras)} extras items (hints/tips)")
            
            return {
                'type': 'extras',
                'keywords': keywords,
                'success': len(matched_extras) > 0,
                'results': matched_extras,
                'count': len(matched_extras)
            }
            
        except Exception as e:
            logger.error(f"Extras execution error: {e}", exc_info=True)
            return {
                'type': 'extras',
                'keywords': tool_request.get('keywords', []),
                'success': False,
                'results': [],
                'error': str(e)
            }

    # ========================================================================
    # ASYNC TOOL EXECUTION METHODS (For parallel execution performance)
    # ========================================================================

    async def _execute_function_tool_async(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function call asynchronously"""
        func_name = tool_request.get('name', 'unknown')
        logger.info(f"[ASYNC] Executing function: {func_name}")

        # Run in thread pool since function registry is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default executor
            self._execute_function_tool,
            tool_request
        )
        return result

    async def _execute_rag_tool_async(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query asynchronously"""
        query = tool_request.get('query', '')
        logger.info(f"[ASYNC] Executing RAG: {query[:50]}...")

        # Run in thread pool since RAG engine is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._execute_rag_tool,
            tool_request
        )
        return result

    async def _execute_extras_tool_async(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute extras query asynchronously"""
        keywords = tool_request.get('keywords', [])
        logger.info(f"[ASYNC] Executing extras: {keywords}")

        # Run in thread pool since extras manager is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._execute_extras_tool,
            tool_request
        )
        return result

    @log_extraction_stage("Stage 3: JSON Extraction")
    def _execute_stage3_extraction(self) -> bool:
        """Execute Stage 3: extraction with tool results"""
        try:
            extraction_prompt = self._build_stage3_main_extraction_prompt()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Extraction attempt {attempt + 1}/{self.max_retries}")

                    # TIER 1.2: Use tiered models - accurate model for Stage 3 extraction
                    stage_model = get_model_for_stage(self.app_state, stage=3)

                    response = self.llm_manager.generate(
                        extraction_prompt,
                        max_tokens=self.app_state.model_config.max_tokens,
                        override_model_name=stage_model
                    )
                    
                    if not response:
                        logger.error("No response from LLM")
                        continue
                    
                    self.context.last_raw_response = response
                    
                    # Parse JSON output
                    parsed_output = self._parse_json_with_robust_parser(response, attempt, "stage3")
                    
                    if parsed_output:
                        self.context.stage3_output = parsed_output
                        logger.info("Extraction successful")
                        return True
                    
                    logger.warning(f"Parsing failed, attempt {attempt + 1}")
                    self.context.retry_count += 1
                    
                except Exception as e:
                    logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
                    self.context.retry_count += 1
            
            logger.error("All extraction attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"Stage 3 extraction failed: {e}", exc_info=True)
            return False
    
    def _parse_json_with_robust_parser(self, response: str, attempt: int, stage: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response using robust parser"""
        parsed_output, parse_method = self.json_parser.parse_json_response(
            response,
            self.app_state.prompt_config.json_schema
        )
        
        if parsed_output:
            self.context.parsing_method_used = parse_method
            logger.info(f"Parsed using method: {parse_method}")
        
        return parsed_output
    
    def _should_run_rag_refinement(self) -> bool:
        """Determine if RAG refinement should run"""
        if not self.app_state.prompt_config.rag_prompt:
            return False
        
        if not self.rag_engine:
            return False
        
        # Check if we have RAG results
        rag_results = [r for r in self.context.tool_results if r.get('type') == 'rag']
        has_rag_results = any(r.get('success') for r in rag_results)
        
        return has_rag_results
    
    @log_extraction_stage("Stage 4: Intelligent Refinement with Tool Calling")
    def _execute_stage4_rag_refinement_with_retry(self) -> Dict[str, Any]:
        """
        Execute Stage 4: Intelligent refinement with TOOL CALLING support

        MAJOR UPGRADE (v1.0.0):
        - LLM can now call functions, RAG, and extras during refinement
        - Two-phase refinement: (1) Gap analysis + tool calling, (2) Final refinement
        - Enables filling missing calculations, retrieving additional guidelines, etc.
        """
        try:
            # Get selected fields for refinement from RAG config
            selected_fields = self.app_state.rag_config.rag_query_fields

            if not selected_fields:
                logger.info("No fields selected for RAG refinement - using stage 3 output")
                return self.context.stage3_output

            logger.info(f"Refining selected fields: {selected_fields}")
            logger.info("=" * 60)
            logger.info("PHASE 1: Gap Analysis & Tool Calling")
            logger.info("=" * 60)

            # Phase 1: LLM analyzes extraction, identifies gaps, requests tools
            tool_requests, needs_refinement = self._execute_refinement_gap_analysis()

            # Execute any requested tools
            additional_tool_results = []
            if tool_requests:
                logger.info(f"Executing {len(tool_requests)} additional tools for refinement")
                additional_tool_results = self._execute_refinement_tools(tool_requests)

            # Store Stage 4 additional tools in context for tracking
            self.context.stage4_additional_tool_results = additional_tool_results

            logger.info("=" * 60)
            logger.info("PHASE 2: Final Refinement with Tool Results")
            logger.info("=" * 60)

            # Phase 2: Final refinement with original RAG + new tool results
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Final refinement attempt {attempt + 1}/{self.max_retries}")

                    # Build refinement prompt with all available context
                    refinement_prompt = self._build_stage4_final_refinement_prompt(
                        additional_tool_results
                    )

                    # TIER 1.2: Use tiered models - fast model for Stage 4 refinement
                    stage_model = get_model_for_stage(self.app_state, stage=4)

                    response = self.llm_manager.generate(
                        refinement_prompt,
                        max_tokens=self.app_state.model_config.max_tokens,
                        override_model_name=stage_model
                    )

                    if not response:
                        logger.warning("No refinement response")
                        continue

                    # Parse refined output
                    refined_output = self._parse_json_with_robust_parser(response, attempt, "stage4")

                    if refined_output:
                        # Merge refined fields with unrefined fields
                        merged_output = self._merge_refined_fields(
                            self.context.stage3_output,
                            refined_output,
                            selected_fields
                        )

                        logger.info(f"Refinement successful - merged {len(selected_fields)} refined fields")
                        if additional_tool_results:
                            logger.info(f"Refinement enhanced with {len(additional_tool_results)} additional tools")
                        return merged_output

                    logger.warning(f"Refinement parsing failed, attempt {attempt + 1}")

                except Exception as e:
                    logger.error(f"Refinement attempt {attempt + 1} failed: {e}")

            logger.warning("Refinement failed, using stage 3 output")
            return self.context.stage3_output

        except Exception as e:
            logger.error(f"Stage 4 refinement failed: {e}", exc_info=True)
            return self.context.stage3_output

    def _execute_refinement_gap_analysis(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Phase 1 of refinement: Analyze extraction and identify gaps/needs

        Returns:
            Tuple of (tool_requests, needs_refinement)
        """
        try:
            gap_analysis_prompt = self._build_stage4_gap_analysis_prompt()

            # TIER 1.2: Use tiered models - fast model for Stage 4 gap analysis
            stage_model = get_model_for_stage(self.app_state, stage=4)

            response = self.llm_manager.generate(
                gap_analysis_prompt,
                max_tokens=self.app_state.model_config.max_tokens,
                override_model_name=stage_model
            )

            if not response:
                logger.warning("No gap analysis response")
                return [], True

            # Parse gap analysis response
            analysis = self._parse_gap_analysis_response(response)

            if analysis:
                tool_requests = analysis.get('additional_tools_needed', [])
                needs_refinement = analysis.get('needs_refinement', True)

                # SOFT LIMIT: Warn if Stage 4 requesting excessive additional tools
                STAGE4_TOOL_BUDGET = 20  # Recommended limit for Stage 4 gap-filling
                if len(tool_requests) > STAGE4_TOOL_BUDGET:
                    logger.warning(
                        f"️ STAGE 4 COMPLEXITY WARNING: Requesting {len(tool_requests)} additional tools "
                        f"(recommended: ≤{STAGE4_TOOL_BUDGET}). Consider if all gap-filling tools are necessary "
                        f"to avoid excessive refinement complexity."
                    )

                logger.info(f"Gap analysis complete: {len(tool_requests)} additional tools requested")
                if tool_requests:
                    logger.info(f"Additional tools: {[t.get('type') for t in tool_requests]}")

                return tool_requests, needs_refinement

            return [], True

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return [], True

    def _execute_refinement_tools(self, tool_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute additional tools requested during refinement

        Supports:
        - Functions: Calculate missing values
        - RAG: Retrieve additional guidelines
        - Extras: Get supplementary hints
        """
        # CRITICAL FIX: Deduplicate tool requests before execution using preventer
        original_count = len(tool_requests)

        if self.tool_dedup_preventer:
            # Use advanced deduplication preventer
            unique_requests, num_duplicates = self.tool_dedup_preventer.filter_duplicates(tool_requests)
            tool_requests = unique_requests

            if num_duplicates > 0:
                logger.warning(f"️ PREVENTED {num_duplicates} DUPLICATE TOOL REQUESTS IN STAGE 4")
                logger.info(f"   Original: {original_count} requests → Unique: {len(unique_requests)} requests")
        else:
            # Fallback to old deduplication
            tool_requests = self._deduplicate_tool_requests(tool_requests)

        if len(tool_requests) == 0:
            logger.warning(" No refinement tools to execute after deduplication")
            return []

        if original_count > len(tool_requests):
            logger.info(f" Deduplicated refinement tools: {original_count} -> {len(tool_requests)} requests (removed {original_count - len(tool_requests)} duplicates)")

        results = []

        try:
            # Run async execution for parallel tool processing
            results = asyncio.run(self._execute_refinement_tools_async(tool_requests))
        except RuntimeError as e:
            # Fallback to sequential if event loop already running
            if "cannot be called from a running event loop" in str(e):
                logger.warning("Event loop running - using sequential tool execution")
                for tool_request in tool_requests:
                    tool_type = tool_request.get('type')
                    if tool_type == 'function':
                        results.append(self._execute_function_tool(tool_request))
                    elif tool_type == 'rag':
                        results.append(self._execute_rag_tool(tool_request))
                    elif tool_type == 'extras':
                        results.append(self._execute_extras_tool(tool_request))
            else:
                raise

        return results

    async def _execute_refinement_tools_async(self, tool_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute refinement tools in parallel"""
        tasks = []

        for tool_request in tool_requests:
            tool_type = tool_request.get('type')

            if tool_type == 'function':
                task = self._execute_function_tool_async(tool_request)
            elif tool_type == 'rag':
                task = self._execute_rag_tool_async(tool_request)
            elif tool_type == 'extras':
                task = self._execute_extras_tool_async(tool_request)
            else:
                async def unknown_tool():
                    return {
                        'type': tool_type,
                        'success': False,
                        'message': f"Unknown tool type: {tool_type}"
                    }
                task = unknown_tool()

            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def _parse_gap_analysis_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse gap analysis response"""
        try:
            parsed, method = self.json_parser.parse_json_response(
                response,
                {
                    'gap_analysis': 'string',
                    'needs_refinement': 'bool',
                    'additional_tools_needed': 'list'
                }
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse gap analysis: {e}")
            return None

    def _build_stage4_gap_analysis_prompt(self) -> str:
        """Build prompt for Phase 1: Gap analysis and tool requesting"""
        rag_context = self._format_rag_evidence_chunks()
        tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)

        # Get selected fields
        selected_fields = self.app_state.rag_config.rag_query_fields
        full_schema = self.app_state.prompt_config.json_schema
        selected_schema = {
            field: full_schema[field]
            for field in selected_fields
            if field in full_schema
        }

        selected_stage3 = {
            field: self.context.stage3_output.get(field)
            for field in selected_fields
            if field in self.context.stage3_output
        }

        # Get available functions
        available_functions = self.function_registry.get_all_functions_info()
        function_descriptions = self._build_available_tools_description(available_functions)

        prompt = f"""You are analyzing an extraction to identify gaps and determine if additional tools are needed for refinement.

ORIGINAL TEXT:
{self.context.clinical_text}

CURRENT EXTRACTION (Selected Fields for Refinement):
{json.dumps(selected_stage3, indent=2)}

TOOLS ALREADY EXECUTED (Stage 2):
{tool_outputs}

RETRIEVED EVIDENCE (Stage 2 RAG):
{rag_context}

TARGET SCHEMA (Fields Being Refined):
{json.dumps(selected_schema, indent=2)}

YOUR TASK:

1. ANALYZE CURRENT EXTRACTION:
   - Are there missing calculations or values?
   - Are there uncertainties that need more data?
   - Could additional guidelines help interpretation?

2. IDENTIFY GAPS:
   - Missing function calls (calculations not performed)
   - Missing RAG queries (guidelines not retrieved)
   - Missing context (extras hints needed)

3. REQUEST ADDITIONAL TOOLS IF NEEDED:

   You have access to:

   FUNCTIONS:
{function_descriptions}

   RAG: Retrieve additional clinical guidelines/standards

   EXTRAS: Get supplementary hints/patterns

RESPONSE FORMAT:
{{
  "gap_analysis": "Brief description of any gaps or uncertainties in current extraction",
  "needs_refinement": true/false,
  "additional_tools_needed": [
    {{
      "type": "function",
      "name": "function_name",
      "parameters": {{"param1": "value1"}},
      "reason": "Why this function helps refinement"
    }},
    {{
      "type": "rag",
      "query": "specific focused query",
      "reason": "What additional guideline/standard is needed"
    }},
    {{
      "type": "extras",
      "keywords": ["keyword1", "keyword2"],
      "reason": "What supplementary hints would help"
    }}
  ]
}}

IMPORTANT:
- Only request tools that will genuinely improve the extraction
- If extraction is complete and accurate, return empty additional_tools_needed list
- Be specific with parameters and queries

Respond with JSON only."""

        return prompt

    def _build_stage4_final_refinement_prompt(self, additional_tool_results: List[Dict[str, Any]]) -> str:
        """Build prompt for Phase 2: Final refinement with all tool results"""
        rag_context = self._format_rag_evidence_chunks()
        stage2_tools = format_tool_outputs_for_prompt(self.context.tool_results)

        # Format additional tool results
        if additional_tool_results:
            additional_tools_formatted = format_tool_outputs_for_prompt(additional_tool_results)
        else:
            additional_tools_formatted = "No additional tools were executed."

        # Get selected fields
        selected_fields = self.app_state.rag_config.rag_query_fields
        full_schema = self.app_state.prompt_config.json_schema
        selected_schema = {
            field: full_schema[field]
            for field in selected_fields
            if field in full_schema
        }
        schema_instructions = format_schema_as_instructions(selected_schema)

        selected_stage3 = {
            field: self.context.stage3_output.get(field)
            for field in selected_fields
            if field in self.context.stage3_output
        }

        # Check if user has custom refinement prompt
        refinement_template = (
            self.app_state.prompt_config.rag_prompt or get_default_rag_refinement_prompt()
        )

        # Try to use custom template with extended placeholders
        try:
            prompt = refinement_template.format(
                clinical_text=self.context.clinical_text,
                initial_extraction=json.dumps(selected_stage3, indent=2),
                rag_context=rag_context,
                label_context=self.context.label_context or "",
                stage3_json_output=json.dumps(selected_stage3, indent=2),
                retrieved_evidence_chunks=rag_context,
                schema_instructions=schema_instructions,
                json_schema_instructions=schema_instructions,
                stage2_tool_results=stage2_tools,
                additional_tool_results=additional_tools_formatted,
                **self.context.prompt_variables  # NEW: Add prompt variables
            )
        except KeyError:
            # Fallback to comprehensive default
            prompt = f"""Refine the extraction using all available evidence and tool results.

ORIGINAL TEXT:
{self.context.clinical_text}

INITIAL EXTRACTION (Stage 3 - Selected Fields):
{json.dumps(selected_stage3, indent=2)}

TOOL RESULTS FROM STAGE 2:
{stage2_tools}

ADDITIONAL TOOLS EXECUTED FOR REFINEMENT:
{additional_tools_formatted}

RETRIEVED EVIDENCE (RAG):
{rag_context}

TARGET SCHEMA:
{schema_instructions}

REFINE THE EXTRACTION:
- Use tool results to fill gaps or improve accuracy
- Incorporate retrieved evidence for interpretation
- Ensure all selected fields meet schema requirements
- Maintain consistency with clinical text

Return ONLY JSON for the selected fields."""

        return prompt


    def _build_stage1_analysis_prompt(self) -> str:
        """
        Build prompt for task analysis with ENHANCED keyword extraction guidance

        UNIVERSAL SYSTEM: This prompt is task-agnostic and works for ANY clinical extraction task.
        Examples provided are illustrative only - the system adapts to YOUR task definition.
        """
        # Get available functions (use consistent instance)
        available_functions = self.function_registry.get_all_functions_info()
        function_descriptions = self._build_available_tools_description(available_functions)

        # CRITICAL FIX: Get the FULL user task prompt, not just schema field descriptions
        full_task_prompt = self._get_full_task_prompt_for_stage1()

        # Get schema as JSON string for analysis
        schema_json = json.dumps(self.app_state.prompt_config.json_schema, indent=2)

        prompt = f"""You are a tool planning assistant. Your job is to read the TASK DESCRIPTION below and identify which tools are needed to fulfill it.

 CRITICAL INSTRUCTIONS:
- Read the TASK DESCRIPTION to understand WHAT needs to be extracted
- Analyze the CLINICAL TEXT to identify available data
- Autonomously DETERMINE which tools are REQUIRED to fulfill the task
- Call tools to transform available data into required format (e.g., percentile → z-score)
- Call tools to retrieve guidelines/criteria mentioned in task (e.g., ASPEN, WHO)
- DO NOT call tools unrelated to the task requirements (no exploration beyond task scope)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK DESCRIPTION (Follow this exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{full_task_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END TASK DESCRIPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPECTED OUTPUT SCHEMA:
{schema_json}

INPUT TEXT:
{self.context.clinical_text}

LABEL/CLASSIFICATION:
{self.context.label_context or "No label provided"}

AVAILABLE TOOLS:

FUNCTIONS:
{function_descriptions}

RAG: {"Available - retrieves relevant guidelines, standards, and reference information" if self.rag_engine else "Not available"}

EXTRAS (HINTS): {"Available - retrieves supplementary hints/tips/patterns that help understand and break down the task" if self.extras_manager else "Not available"}

YOUR TASK:

1. UNDERSTAND THE EXTRACTION GOAL:
   - What information needs to be extracted? (Review schema fields)
   - What are the key concepts in the task? (Extract from schema field names and descriptions)
   - What domain or context is this? (Extract from label/classification)

2. IDENTIFY FUNCTIONS TO CALL (WITH SMART PARAMETER EXTRACTION):

   A. SCAN FOR ALL MEASUREMENTS (INCLUDING SERIAL/TEMPORAL DATA):
   - Identify ALL numeric values, measurements, and dates in the text
   - Detect SERIAL MEASUREMENTS: Multiple instances of the same measurement type at different time points
   - Examples of serial measurements:
     * Multiple creatinine values: "Cr 1.2 (Jan), 1.5 (Mar), 1.8 (June)"
     * Multiple weights: "Weight 12.5 kg (today), 13.0 kg (3 months ago)"
     * Multiple BPs: "BP 140/90 (visit 1), 135/85 (visit 2), 130/80 (visit 3)"
     * Multiple labs over time: "HbA1c 8.5% (baseline), 7.2% (3 months), 6.8% (6 months)"

   B. CALL FUNCTIONS MULTIPLE TIMES FOR SERIAL MEASUREMENTS:
   *** CRITICAL: If there are multiple measurements of the same type at different time points,
       call the SAME function MULTIPLE times (once for each time point) ***

   Examples:
   - Text: "Creatinine 1.2 mg/dL on 1/15, 1.5 mg/dL on 3/20, 1.8 mg/dL on 6/10"
     -> Call calculate_creatinine_clearance() THREE times (once for each Cr value)

   - Text: "Weight 12.5 kg today, was 13.0 kg 3 months ago, and 13.5 kg 6 months ago"
     -> Call calculate_bmi() THREE times (once for each weight measurement)

   - Text: "BP readings: 140/90 (visit 1), 135/85 (visit 2), 130/80 (visit 3)"
     -> Call calculate_mean_arterial_pressure() THREE times (once per reading)

   C. MAP VALUES TO FUNCTION PARAMETERS:
   - Map values to function parameters based on context:
     * "45.5 kg" -> weight_kg parameter
     * "165 cm" or "5'5\"" -> height parameter (convert inches if needed)
     * "65 year old" or "age 65" -> age parameter
     * "BP 140/90" -> systolic=140, diastolic=90
     * "male" or "female" or "boy" or "girl" -> sex parameter (pass as string 'male' or 'female')
     * IMPORTANT: For growth percentile functions, sex can be passed as string ('male'/'female')
       and will be automatically converted to numbers (1=male, 2=female) by the system

   D. HANDLE UNIT CONVERSIONS:
   - If function needs kg but text has lbs -> call lbs_to_kg first
   - If function needs meters but text has cm -> call cm_to_m first

   E. CHAIN FUNCTIONS WHEN NEEDED:
   - Example: calculate_bmi needs weight_kg and height_m
   - If text has "150 cm", first call cm_to_m(150), then use result in calculate_bmi

   F. EXTRACT DATES AND TEMPORAL CONTEXT:
   - Extract date/time for each measurement: "01/15/2024", "3 months ago", "baseline", "follow-up"
   - Include date context in the function call metadata
   - Calculate intervals between measurements for trend analysis

   G. PRECISION AND RELEVANCE:
   - Extract parameters with high precision from actual text values
   - Only call functions that are truly needed for the extraction schema
   - For serial data, ALL time points are usually relevant for trend analysis

3. BUILD MULTIPLE INTELLIGENT RAG QUERIES (3-5 queries):
   - Create MULTIPLE focused queries, each targeting different aspects:

   Query Strategy:
   a) GUIDELINE QUERY: Target clinical guidelines/standards
      - Extract from: diagnosis, condition, assessment type
      - Example: "WHO pediatric growth standards malnutrition criteria"

   b) DIAGNOSTIC QUERY: Target diagnostic criteria
      - Extract from: diagnosis fields, classification fields
      - Example: "diabetes diagnostic criteria HbA1c fasting glucose"

   c) ASSESSMENT QUERY: Target assessment methods/scoring
      - Extract from: assessment fields, severity fields
      - Example: "sepsis assessment qSOFA SIRS criteria scoring"

   d) TREATMENT QUERY: Target treatment guidelines (if relevant)
      - Extract from: treatment/management fields
      - Example: "hypertension blood pressure management ACC AHA guidelines"

   e) DOMAIN-SPECIFIC QUERY: Target specialized knowledge
      - Extract from: specific medical domain in text/label
      - Example: "pediatric developmental milestones age assessment"

   - Each query should have 4-8 specific keywords
   - Use medical terminology from the input text
   - Reference known standards (WHO, CDC, ADA, ACC/AHA, ASPEN, etc.)
   - Avoid generic terms like "information", "help", "guidelines" alone

4. IDENTIFY EXTRAS KEYWORDS (5-8 SPECIFIC KEYWORDS):
   - Extras provide task-specific hints, reference ranges, criteria
   - Extract from multiple sources:
     * Schema field names: ["malnutrition", "status", "diagnosis"]
     * Label classification: ["pediatric", "diabetes", "hypertension"]
     * Medical conditions in text: ["growth", "assessment", "z-score"]
     * Assessment types: ["screening", "diagnostic", "monitoring"]
   - Use specific medical terminology (not generic words)
   - Include age group if relevant: ["pediatric", "geriatric", "adult"]
   - Include system if relevant: ["cardiac", "respiratory", "renal"]
   - Example good keywords: ["malnutrition", "pediatric", "z-score", "WHO", "assessment"]
   - Example bad keywords: ["patient", "information", "data"]

RESPONSE FORMAT (JSON only):
{{
  "required_information": ["field1", "field2", "field3"],
  "task_analysis": "Brief analysis of what needs to be extracted and clinical context",
  "functions_needed": [
    {{
      "name": "function_name",
      "parameters": {{"param1": "value1", "param2": "value2"}},
      "date_context": "date/time of this measurement (e.g., '2024-01-15', '3 months ago', 'baseline', 'follow-up visit 2')",
      "reason": "why this function is needed for extraction"
    }}
  ],
  "rag_queries": [
    {{
      "query": "specific focused query with 4-8 keywords",
      "query_type": "guideline|diagnostic|assessment|treatment|domain_specific",
      "purpose": "what specific information this retrieves"
    }},
    {{
      "query": "another focused query targeting different aspect",
      "query_type": "guideline|diagnostic|assessment|treatment|domain_specific",
      "purpose": "complementary information for extraction"
    }}
  ],
  "extras_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ILLUSTRATIVE EXAMPLES (Adapt to YOUR specific task!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1: Renal Function Assessment (Serial Measurements)
─────────────────────────────────────────────────────────
If extracting renal function data and text contains:
"Creatinine was 1.2 mg/dL on 1/15/2024, increased to 1.5 mg/dL on 3/20/2024, and now 1.8 mg/dL on 6/10/2024. Patient is 65yo male, 70kg."

Response should include:
{{
  "functions_needed": [
    {{
      "name": "calculate_creatinine_clearance",
      "parameters": {{"creatinine": 1.2, "age": 65, "weight": 70, "sex": "male"}},
      "date_context": "2024-01-15 (baseline)",
      "reason": "Calculate eGFR for baseline creatinine to assess renal function"
    }},
    {{
      "name": "calculate_creatinine_clearance",
      "parameters": {{"creatinine": 1.5, "age": 65, "weight": 70, "sex": "male"}},
      "date_context": "2024-03-20 (follow-up)",
      "reason": "Calculate eGFR for follow-up creatinine to track renal function decline"
    }},
    {{
      "name": "calculate_creatinine_clearance",
      "parameters": {{"creatinine": 1.8, "age": 65, "weight": 70, "sex": "male"}},
      "date_context": "2024-06-10 (current)",
      "reason": "Calculate eGFR for most recent creatinine to assess current renal function"
    }}
  ],
  "rag_queries": [
    {{
      "query": "CKD chronic kidney disease staging eGFR classification guidelines",
      "query_type": "guideline",
      "purpose": "Get CKD staging criteria based on eGFR values"
    }}
  ],
  "extras_keywords": ["renal", "CKD", "eGFR", "creatinine", "staging"]
}}

EXAMPLE 2: Pediatric Growth Assessment
──────────────────────────────────────
If extracting growth data and text contains:
"3-year-old with weight 12.5 kg (3rd percentile), height 92 cm (25th percentile)"

Response should include:
{{
  "functions_needed": [
    {{
      "name": "percentile_to_zscore",
      "parameters": {{"percentile": 3}},
      "reason": "Convert weight percentile to z-score for WHO classification"
    }},
    {{
      "name": "percentile_to_zscore",
      "parameters": {{"percentile": 25}},
      "reason": "Convert height percentile to z-score for growth assessment"
    }},
    {{
      "name": "calculate_bmi",
      "parameters": {{"weight_kg": 12.5, "height_cm": 92}},
      "reason": "Calculate BMI for nutritional status assessment"
    }}
  ],
  "rag_queries": [
    {{
      "query": "WHO pediatric growth standards percentile z-score classification",
      "query_type": "guideline",
      "purpose": "Get WHO growth classification criteria"
    }}
  ],
  "extras_keywords": ["growth", "pediatric", "WHO", "percentile", "z-score"]
}}

EXAMPLE 3: Cardiovascular Risk Assessment
─────────────────────────────────────────
If extracting cardiac data and text contains:
"BP 145/95, total cholesterol 240 mg/dL, HDL 35 mg/dL, age 60, smoker"

Response should include:
{{
  "functions_needed": [
    {{
      "name": "calculate_mean_arterial_pressure",
      "parameters": {{"systolic": 145, "diastolic": 95}},
      "reason": "Calculate MAP for hypertension assessment"
    }},
    {{
      "name": "calculate_framingham_risk",
      "parameters": {{"age": 60, "total_chol": 240, "hdl": 35, "smoker": true}},
      "reason": "Calculate 10-year cardiovascular risk"
    }}
  ],
  "rag_queries": [
    {{
      "query": "ACC AHA hypertension blood pressure classification guidelines",
      "query_type": "guideline",
      "purpose": "Get BP classification criteria"
    }},
    {{
      "query": "Framingham cardiovascular risk assessment interpretation",
      "query_type": "assessment",
      "purpose": "Get risk stratification guidelines"
    }}
  ],
  "extras_keywords": ["cardiovascular", "hypertension", "cholesterol", "risk", "ACC", "AHA"]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CRITICAL REQUIREMENTS (Universal Principles)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ADAPT TO YOUR TASK:
The examples above show renal, growth, and cardiac tasks. YOUR task may be completely different
(diabetes, sepsis, AKI, oncology, etc.). Analyze YOUR schema and clinical text to determine what's needed.

 FOR SERIAL/TEMPORAL DATA:
- *** Call the same function MULTIPLE times (once per time point) ***
- Include "date_context" for each function call to track temporal sequence
- When you see multiple measurements over time, calculate for ALL time points

 RAG QUERIES:
- Provide 3-5 queries targeting different aspects relevant to YOUR task
- Each query should have 4-8 specific keywords from YOUR domain
- Reference appropriate standards/organizations for YOUR domain
- Examples: WHO, CDC, ADA, ACC/AHA, ASPEN, KDIGO, ASCO (use what's relevant to YOUR task)
- Do NOT use generic queries like "help", "information", or "guidelines" alone

🛠️ FUNCTION CALLING:
- Extract parameters with precision from actual text values
- Chain functions if needed (e.g., unit conversion before calculation)
- Only call functions that help with YOUR specific extraction schema

 EXTRAS KEYWORDS:
- 5-8 keywords: specific terminology from YOUR domain, not generic words
- Extract from YOUR schema field names and clinical context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ONE-SHOT PLANNING: THIS IS YOUR ONLY CHANCE TO PLAN TOOLS!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

️  CRITICAL: Unlike iterative modes, you CANNOT call tools again after this stage.
This is your ONLY opportunity to request ALL necessary tools.

PLAN COMPREHENSIVELY:

 IDENTIFY ALL MEASUREMENTS: Scan the entire text for EVERY numeric value, lab result,
  vital sign, and measurement. Don't miss any data points.

 CALL FUNCTIONS MULTIPLE TIMES: If you see serial/temporal data (measurements at
  different time points), call the SAME function MULTIPLE times. Examples:
  - "Weight 65kg (today), 68kg (3mo ago), 70kg (6mo ago)" → call calculate_bmi 3 times
  - "Cr 1.2, 1.5, 1.8 over past 6 months" → call eGFR function 3 times
  - Do NOT skip any time points!

 EXTRACT ALL PARAMETERS: For each function call, extract PRECISE parameter values
  from the text. Don't leave parameters empty if values are available.

 PLAN RAG QUERIES STRATEGICALLY: Request 3-5 focused queries covering:
  - Clinical guidelines (WHO, CDC, ADA, ACC/AHA, ASPEN, etc.)
  - Diagnostic criteria
  - Assessment methods
  - Domain-specific standards
  Make queries specific with 4-8 keywords each.

 CHOOSE RELEVANT EXTRAS KEYWORDS: Select 5-8 specific medical terms from:
  - Schema field names
  - Clinical conditions in the text
  - Assessment types
  - Medical specialty terms

 THINK AHEAD: What calculations will Stage 3 need? What reference data will help
  interpretation? Request ALL tools that might be useful NOW.

 AVOID:
- Vague or incomplete tool requests
- Missing serial measurements
- Generic RAG queries ("help", "information")
- Skipping available functions that could help

 REMEMBER: You're setting up the extraction for success. Be thorough, be specific,
   and plan for everything you'll need in one shot!

 RESPONSE FORMAT:
Respond with ONLY the JSON object in the format shown above."""

        return prompt
    
    def _extract_task_description(self) -> str:
        """Extract task description from schema"""
        schema = self.app_state.prompt_config.json_schema

        if not schema:
            return "Extract relevant information from the input text."

        parts = []
        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                field_type = field_info.get('type', 'unknown')
                field_desc = field_info.get('description', '')
                parts.append(f"- {field_name} ({field_type}): {field_desc}")
            else:
                parts.append(f"- {field_name}: {field_info}")

        return "Extract the following:\n" + "\n".join(parts)

    def _get_full_task_prompt_for_stage1(self) -> str:
        """
        Get the FULL user task prompt for Stage 1 analysis

        CRITICAL FIX: Instead of just listing schema fields, this returns the complete
        user-defined task prompt (e.g., MALNUTRITION_MAIN_PROMPT) that contains detailed
        instructions about WHAT to extract and WHEN to use tools.

        Returns:
            Full task prompt string with all instructions, or schema-based description as fallback
        """
        # Try to get the actual task prompt from app_state
        # The main_prompt or minimal_prompt contains the full task instructions
        task_prompt = self.app_state.prompt_config.main_prompt or self.app_state.prompt_config.minimal_prompt

        if task_prompt and task_prompt.strip():
            # Found the full task prompt - this is what we want!
            # Extract just the task description portion (before placeholders)
            # The task prompt has sections like [TASK DESCRIPTION] ... [END TASK DESCRIPTION]

            # Try to extract task description section if marked
            if "[TASK DESCRIPTION" in task_prompt and "[END" in task_prompt:
                start_idx = task_prompt.find("[TASK DESCRIPTION")
                # Find the end marker (could be [END TASK DESCRIPTION] or [END])
                end_markers = ["[END TASK DESCRIPTION]", "[END]"]
                end_idx = -1
                for marker in end_markers:
                    idx = task_prompt.find(marker, start_idx)
                    if idx != -1:
                        end_idx = idx + len(marker)
                        break

                if end_idx != -1:
                    return task_prompt[start_idx:end_idx]

            # If no specific section markers, return the whole prompt (it will have placeholders)
            # Stage 1 doesn't need the clinical_text filled in - it's provided separately
            return task_prompt

        # Fallback to schema-based description if no task prompt is defined
        logger.warning("️  No user task prompt found - using schema field descriptions as fallback")
        return self._extract_task_description()

    def _build_available_tools_description(self, functions: List[Dict[str, Any]] = None) -> str:
        """Build description of available tools"""
        if functions is None:
            # Use consistent function registry instance
            functions = self.function_registry.get_all_functions_info()
        
        if not functions:
            return "No functions available."
        
        descriptions = []
        for func in functions:
            name = func.get('name', 'unknown')
            description = func.get('description', 'No description')
            parameters = func.get('parameters', {})
            
            param_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in parameters.items())
            
            descriptions.append(f"- {name}({param_str}): {description}")
        
        return "\n".join(descriptions)
    
    def _build_stage3_main_extraction_prompt(self) -> str:
        """Build prompt for stage 3 extraction with retry-based minimal prompt fallback"""
        # Format schema instructions
        schema_instructions = format_schema_as_instructions(
            self.app_state.prompt_config.json_schema
        )

        # Format tool results
        tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)

        # CRITICAL FIX: Use get_prompt_for_processing() to respect minimal prompt fallback on retries
        # This ensures if extraction fails multiple times, system switches to minimal prompt
        extraction_prompt = self.app_state.get_prompt_for_processing(self.context.retry_count)

        # Track if minimal prompt is being used
        self.context.using_minimal_prompt = self.app_state.is_using_minimal_prompt

        if self.context.using_minimal_prompt:
            logger.warning(f"  Using MINIMAL prompt (retry_count={self.context.retry_count} >= max_retries={self.app_state.processing_config.max_retries})")

        # NEW: Format extraction_prompt with prompt variables if it has placeholders
        try:
            extraction_prompt = extraction_prompt.format(
                clinical_text=self.context.clinical_text,
                label_context=self.context.label_context or "No label provided",
                rag_outputs=tool_outputs.get('rag_outputs', ''),
                function_outputs=tool_outputs.get('function_outputs', ''),
                extras_outputs=tool_outputs.get('extras_outputs', ''),
                json_schema_instructions=schema_instructions,
                **self.context.prompt_variables  # NEW: Add prompt variables
            )
            # If formatting succeeds, user template handles layout, return as-is
            return extraction_prompt
        except KeyError:
            # Template doesn't use these placeholders, use default layout
            pass

        # Default layout (backward compatible)
        prompt = f"""{extraction_prompt}

CLINICAL TEXT:
{self.context.clinical_text}

LABEL CONTEXT:
{self.context.label_context or "No label provided"}

{tool_outputs.get('rag_outputs', '')}

{tool_outputs.get('function_outputs', '')}

{tool_outputs.get('extras_outputs', '')}

TASK:
{schema_instructions}

Extract the required information and respond with ONLY a JSON object."""

        return prompt
    

    def _build_stage4_rag_refinement_prompt(self) -> str:
        """
        Build prompt for stage 4 refinement - ONLY for selected fields
        FIXED: Only include selected fields in schema instructions
        """
        rag_context = self._format_rag_evidence_chunks()
        label_context = self.context.label_context or ""
        
        refinement_template = (
            self.app_state.prompt_config.rag_prompt or get_default_rag_refinement_prompt()
        )
        
        # FIXED: Get only selected fields for refinement
        selected_fields = self.app_state.rag_config.rag_query_fields
        
        if selected_fields:
            # Create schema with only selected fields
            full_schema = self.app_state.prompt_config.json_schema
            selected_schema = {
                field: full_schema[field] 
                for field in selected_fields 
                if field in full_schema
            }
            schema_instructions = format_schema_as_instructions(selected_schema)
            
            logger.info(f"RAG refinement prompt built for {len(selected_fields)} fields: {selected_fields}")
        else:
            # Fallback to full schema if no fields selected
            schema_instructions = format_schema_as_instructions(
                self.app_state.prompt_config.json_schema
            )
            logger.warning("No fields selected for refinement - using full schema")
        
        # Extract only selected fields from stage3 output
        selected_stage3 = {
            field: self.context.stage3_output.get(field)
            for field in selected_fields
            if field in self.context.stage3_output
        }
        
        try:
            prompt = refinement_template.format(
                clinical_text=self.context.clinical_text,
                initial_extraction=json.dumps(selected_stage3, indent=2),  # Only selected fields
                rag_context=rag_context,
                label_context=label_context,
                stage3_json_output=json.dumps(selected_stage3, indent=2),  # Only selected fields
                retrieved_evidence_chunks=rag_context,
                schema_instructions=schema_instructions,
                json_schema_instructions=schema_instructions,
                **self.context.prompt_variables  # NEW: Add prompt variables
            )
        except KeyError as e:
            logger.warning(f"RAG template had unknown placeholder {e}, using fallback.")
            # Fallback with minimal placeholders
            prompt = f"""Refine the following extraction using retrieved evidence.
    
    ORIGINAL TEXT:
    {self.context.clinical_text}
    
    LABEL CONTEXT:
    {label_context}
    
    INITIAL EXTRACTION (Selected Fields Only):
    {json.dumps(selected_stage3, indent=2)}
    
    RETRIEVED EVIDENCE:
    {rag_context}
    
    REFINE THE SELECTED FIELDS:
    {schema_instructions}
    
    Return ONLY JSON for the selected fields."""
        
        return prompt
        
    
    def _format_rag_evidence_chunks(self) -> str:
        """Format RAG results for prompt"""
        rag_results = [r for r in self.context.tool_results if r.get('type') == 'rag']
        
        chunks = []
        for result in rag_results:
            if result.get('success'):
                for chunk in result.get('results', []):
                    content = chunk.get('content', '') or chunk.get('text', '')
                    if content:
                        chunks.append(content)
        
        return "\n\n".join(chunks[:5])
    
    def _parse_task_analysis_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse task analysis response"""
        try:
            parsed, method = self.json_parser.parse_json_response(
                response,
                {
                    'required_information': 'list',
                    'functions_needed': 'list',
                    'rag_queries': 'list',
                    'extras_keywords': 'list'
                }
            )
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse task analysis: {e}")
            return None

    def _merge_refined_fields(self, stage3_output: Dict[str, Any], 
                              refined_output: Dict[str, Any], 
                              selected_fields: List[str]) -> Dict[str, Any]:
        """
        Merge refined fields back with unrefined fields
        
        Args:
            stage3_output: Original Stage 3 extraction (all fields)
            refined_output: Refined output from Stage 4 (selected fields only)
            selected_fields: Fields that were selected for refinement
            
        Returns:
            Merged output with refined and unrefined fields
        """
        # Start with complete Stage 3 output
        merged = stage3_output.copy()
        
        # Update only the selected fields from refined output
        for field in selected_fields:
            if field in refined_output:
                merged[field] = refined_output[field]
                logger.debug(f"Refined field '{field}' merged into final output")
            else:
                logger.warning(f"Selected field '{field}' not found in refined output - keeping Stage 3 value")
        
        logger.info(f"Merged output: {len(selected_fields)} refined fields + {len(stage3_output) - len(selected_fields)} unrefined fields = {len(merged)} total fields")
        
        return merged
    
    def _create_default_task_understanding(self) -> Dict[str, Any]:
        """Create default task understanding"""
        return {
            'required_information': list(self.app_state.prompt_config.json_schema.keys()),
            'functions_needed': [],
            'rag_queries': [],
            'extras_keywords': []
        }
    
    def _build_extraction_result(self) -> Dict[str, Any]:
        """Build final extraction result with full tool results for playground"""
        # Count tool usage from Stage 2
        rag_results = [r for r in self.context.tool_results if r.get('type') == 'rag']
        function_results = [r for r in self.context.tool_results if r.get('type') == 'function']
        extras_results = [r for r in self.context.tool_results if r.get('type') == 'extras']

        rag_count = len([r for r in rag_results if r.get('success')])
        function_count = len([r for r in function_results if r.get('success')])
        extras_count = len([r for r in extras_results if r.get('success')])

        # Count Stage 4 additional tools (gap-filling tools)
        stage4_additional_tools = self.context.stage4_additional_tool_results or []
        stage4_rag_results = [r for r in stage4_additional_tools if r.get('type') == 'rag']
        stage4_function_results = [r for r in stage4_additional_tools if r.get('type') == 'function']
        stage4_extras_results = [r for r in stage4_additional_tools if r.get('type') == 'extras']

        stage4_rag_count = len([r for r in stage4_rag_results if r.get('success')])
        stage4_function_count = len([r for r in stage4_function_results if r.get('success')])
        stage4_extras_count = len([r for r in stage4_extras_results if r.get('success')])
    
        # Full RAG chunks
        rag_details = []
        for r in rag_results:
            if r.get('success'):
                for chunk in r.get('results', []):
                    content = chunk.get('content', '') or chunk.get('text', '')
                    rag_details.append({
                        'query': r.get('query'),
                        'top_k_used': r.get('top_k_used'),
                        'score': chunk.get('score'),
                        'source': chunk.get('metadata', {}).get('source', 'unknown') if isinstance(chunk.get('metadata'), dict) else chunk.get('source', 'unknown'),
                        'content': content[:3000] + ("..." if len(content) > 3000 else "")
                    })
    
        # Function details
        function_details = []
        for r in function_results:
            function_details.append({
                'name': r.get('name'),
                'success': r.get('success'),
                'result': r.get('result')
            })
    
        # FIXED: Full extras details with content
        extras_details = []
        for r in extras_results:
            if r.get('success'):
                for extra_item in r.get('results', []):
                    extras_details.append({
                        'id': extra_item.get('id', 'N/A'),
                        'type': extra_item.get('type', 'unknown'),
                        'content': extra_item.get('content', ''),
                        'relevance_score': extra_item.get('relevance_score', 0),
                        'matched_keywords': extra_item.get('matched_keywords', []),
                        'metadata': extra_item.get('metadata', {})
                    })

        # NEW: Stage 4 additional tool details (gap-filling tools)
        stage4_function_details = []
        for r in stage4_function_results:
            stage4_function_details.append({
                'name': r.get('name'),
                'success': r.get('success'),
                'result': r.get('result'),
                'stage': 'stage4_gap_filling'
            })

        stage4_rag_details = []
        for r in stage4_rag_results:
            if r.get('success'):
                for chunk in r.get('results', []):
                    content = chunk.get('content', '') or chunk.get('text', '')
                    stage4_rag_details.append({
                        'query': r.get('query'),
                        'top_k_used': r.get('top_k_used'),
                        'score': chunk.get('score'),
                        'source': chunk.get('metadata', {}).get('source', 'unknown') if isinstance(chunk.get('metadata'), dict) else chunk.get('source', 'unknown'),
                        'content': content[:3000] + ("..." if len(content) > 3000 else ""),
                        'stage': 'stage4_gap_filling'
                    })

        stage4_extras_details = []
        for r in stage4_extras_results:
            if r.get('success'):
                for extra_item in r.get('results', []):
                    stage4_extras_details.append({
                        'id': extra_item.get('id', 'N/A'),
                        'type': extra_item.get('type', 'unknown'),
                        'content': extra_item.get('content', ''),
                        'relevance_score': extra_item.get('relevance_score', 0),
                        'matched_keywords': extra_item.get('matched_keywords', []),
                        'metadata': extra_item.get('metadata', {}),
                        'stage': 'stage4_gap_filling'
                    })

        result = {
            'original_clinical_text': getattr(self.context, 'original_text', ''),
            'clinical_text': self.context.clinical_text,
            'redacted_text': getattr(self.context, 'redacted_text', None),
            'normalized_text': getattr(self.context, 'normalized_text', None),
            'input_label_value': self.context.label_value,
            'label_context': self.context.label_context,
            'stage3_output': self.context.stage3_output,
            'stage4_final_output': self.context.stage4_final_output,
            'extras_used': extras_count,
            'rag_used': rag_count,
            'functions_called': function_count,
            'stage4_additional_tools_used': len(stage4_additional_tools) if stage4_additional_tools else 0,  # NEW
            'used_minimal_prompt': self.context.using_minimal_prompt,
            'retry_count': self.context.retry_count,
            'parsing_method_used': self.context.parsing_method_used,
            'rag_refinement_applied': (
                self.context.state == ExtractionAgentState.COMPLETED and
                rag_count > 0 and
                self.app_state.prompt_config.rag_prompt is not None
            ),
            'processing_metadata': {
                'had_label': self.context.label_value is not None,
                'final_state': self.context.state.value,
                'task_understanding': self.context.task_understanding,
                'tool_requests': self.context.tool_requests,
                'tool_results': self.context.tool_results,  # Stage 2 tools
                'stage4_additional_tool_results': stage4_additional_tools,  # NEW: Stage 4 gap-filling tools
                'tool_results_summary': {
                    'stage2_rag': rag_count,
                    'stage2_functions': function_count,
                    'stage2_extras': extras_count,
                    'stage4_rag': stage4_rag_count,  # NEW
                    'stage4_functions': stage4_function_count,  # NEW
                    'stage4_extras': stage4_extras_count  # NEW
                },
                'rag_details': rag_details,
                'function_calls_details': function_details,
                'extras_details': extras_details,
                'stage4_rag_details': stage4_rag_details,  # NEW
                'stage4_function_calls_details': stage4_function_details,  # NEW
                'stage4_extras_details': stage4_extras_details,  # NEW
                'last_raw_response_preview': self.context.last_raw_response[:500] if self.context.last_raw_response else None,
                'text_processing': {
                    'phi_redaction_applied': self.app_state.data_config.enable_phi_redaction,
                    'pattern_normalization_applied': self.app_state.data_config.enable_pattern_normalization,
                    'redacted_text_saved': hasattr(self.context, 'redacted_text'),
                    'normalized_text_saved': hasattr(self.context, 'normalized_text')
                }
            }
        }
    
        return result
    
    def _build_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Build failure result"""
        result = {
            'original_clinical_text': getattr(self.context, 'original_text', '') if self.context else '',
            'clinical_text': self.context.clinical_text if self.context else "",
            'redacted_text': getattr(self.context, 'redacted_text', None) if self.context else None,
            'normalized_text': getattr(self.context, 'normalized_text', None) if self.context else None,
            'input_label_value': self.context.label_value if self.context else None,
            'label_context': self.context.label_context if self.context else None,
            'stage3_output': {},
            'stage4_final_output': {},
            'error': error_message,
            'extras_used': 0,
            'rag_used': 0,
            'functions_called': 0,
            'used_minimal_prompt': False,
            'retry_count': self.context.retry_count if self.context else 0,
            'parsing_method_used': None,
            'rag_refinement_applied': False,
            'processing_metadata': {
                'had_label': False,
                'final_state': 'failed',
                'tool_requests': [],
                'tool_results_summary': {'rag': 0, 'functions': 0, 'extras': 0},
                'rag_details': [],
                'function_calls_details': [],
                'extras_details': [],
                'text_processing': {
                    'phi_redaction_applied': False,
                    'pattern_normalization_applied': False,
                    'redacted_text_saved': False,
                    'normalized_text_saved': False
                }
            }
        }
        
        return result
    
    def _get_label_context_string(self, label_value: Any) -> Optional[str]:
        """
        Get label context from mapping

        FIXED: Now properly handles falsy label values (0, False, empty string)
        For binary classification, label 0 or False are valid and should have context.
        """
        # Check for None explicitly, not truthiness
        # This allows 0, False, "" as valid label values
        if label_value is None:
            return None

        label_mapping = self.app_state.data_config.label_mapping

        # Try string key first
        label_key = str(label_value)
        if label_key in label_mapping:
            return label_mapping[label_key]

        # Try numeric key if conversion is possible
        try:
            numeric_value = int(label_value)
            if numeric_value in label_mapping:
                return label_mapping[numeric_value]
        except (ValueError, TypeError):
            pass

        # Try float key
        try:
            float_value = float(label_value)
            if float_value in label_mapping:
                return label_mapping[float_value]
        except (ValueError, TypeError):
            pass

        return None
    
    def _preprocess_clinical_text(self, text: str) -> str:
        """Preprocess clinical text"""
        if not text or not text.strip():
            return ""
        
        if not hasattr(self.context, 'original_text'):
            if self.context:
                self.context.original_text = text
        
        processed_text = text
        
        if self.app_state.data_config.enable_pattern_normalization and self.regex_preprocessor:
            try:
                normalized = self.regex_preprocessor.preprocess(processed_text)
                
                if self.app_state.data_config.save_normalized_text and self.context:
                    self.context.normalized_text = normalized
                
                processed_text = normalized
                logger.info("Pattern normalization applied")
                
            except Exception as e:
                logger.warning(f"Regex preprocessing failed: {e}")
        
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
    
    def pause_execution(self, reason: str):
        """Pause agent execution"""
        if self.context:
            self.context.state = ExtractionAgentState.PAUSED
            self.context.pause_reason = reason
            logger.info(f"Agent paused: {reason}")
    
    def resume_execution(self):
        """Resume agent execution"""
        if self.context:
            self.context.state = ExtractionAgentState.ANALYZING
            self.context.pause_reason = None
            logger.info("Agent resumed")