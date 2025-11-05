#!/usr/bin/env python3
"""
Agent System - Universal Agentic Processing
Version: 1.0.2 - FIXED: RAG query building and Extras matching
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

FIXES:
1. Enhanced Stage 1 analysis prompt with better keyword extraction guidance
2. Improved RAG query building to extract meaningful terms from text
3. Added fallback query generation when LLM fails to provide good queries
4. Better tool request parsing and validation
5. Extras now correctly used as supplementary hints to help LLM understand task
"""

import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from core.json_parser import JSONParser
from core.prompt_templates import (
    get_stage1_analysis_prompt_template,
    get_default_rag_refinement_prompt,
    format_schema_as_instructions,
    format_tool_outputs_for_prompt
)
from core.logging_config import get_logger, log_extraction_stage

logger = get_logger(__name__)


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
    last_raw_response: Optional[str] = None
    parsing_method_used: Optional[str] = None
    original_text: Optional[str] = None
    redacted_text: Optional[str] = None
    normalized_text: Optional[str] = None
    

class ExtractionAgent:
    """
    Universal agentic system that dynamically determines:
    - Required information based on task
    - Functions to call from registry
    - Queries for RAG and extras
    - Execution strategy
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
        
        logger.info("ExtractionAgent initialized (v1.0.2 - RAG query & Extras fix)")
        
    
    def extract(self, clinical_text: str, label_value: Optional[Any] = None) -> Dict[str, Any]:
        """Main extraction method with universal agentic behavior"""
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
                original_text=clinical_text
            )
            
            # Preprocess the clinical text
            preprocessed_text = self._preprocess_clinical_text(clinical_text)
            self.context.clinical_text = preprocessed_text
            
            logger.info("=" * 60)
            logger.info("STAGE 1: TASK ANALYSIS & TOOL PLANNING")
            logger.info("=" * 60)
            
            # Stage 1: LLM analyzes task and determines requirements
            self.context.state = ExtractionAgentState.ANALYZING
            analysis_success = self._execute_stage1_analysis()
            
            if not analysis_success:
                return self._build_failure_result("Stage 1 analysis failed")
            
            logger.info("=" * 60)
            logger.info(f"STAGE 2: EXECUTING {len(self.context.tool_requests)} TOOL REQUESTS")
            logger.info("=" * 60)
            
            # Stage 2: Tool execution
            self.context.state = ExtractionAgentState.TOOL_EXECUTION
            self._execute_stage2_tools()
            
            logger.info("=" * 60)
            logger.info("STAGE 3: EXTRACTION")
            logger.info("=" * 60)
            
            # Stage 3: Extraction
            self.context.state = ExtractionAgentState.STAGE3_EXTRACTION
            stage3_success = self._execute_stage3_extraction()
            
            if not stage3_success:
                return self._build_failure_result("Stage 3 extraction failed")
            
            # Stage 4: RAG refinement (optional)
            if self._should_run_rag_refinement():
                logger.info("=" * 60)
                logger.info("STAGE 4: RAG REFINEMENT")
                logger.info("=" * 60)
                
                self.context.state = ExtractionAgentState.STAGE4_RAG_REFINEMENT
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
                    
                    response = self.llm_manager.generate(
                        analysis_prompt,
                        max_tokens=self.app_state.model_config.max_tokens
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
            self.context.tool_requests.append({
                'type': 'function',
                'name': func_info.get('name'),
                'parameters': func_info.get('parameters', {}),
                'reason': func_info.get('reason', '')
            })
        
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
    
    @log_extraction_stage("Stage 2: Tool Execution")
    def _execute_stage2_tools(self):
        """Execute all tool requests"""
        for tool_request in self.context.tool_requests:
            tool_type = tool_request.get('type')
            
            if tool_type == 'function':
                result = self._execute_function_tool(tool_request)
                self.context.tool_results.append(result)
            elif tool_type == 'rag':
                result = self._execute_rag_tool(tool_request)
                self.context.tool_results.append(result)
            elif tool_type == 'extras':
                result = self._execute_extras_tool(tool_request)
                self.context.tool_results.append(result)
    
    def _execute_function_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        try:
            func_name = tool_request.get('name')
            parameters = tool_request.get('parameters', {})
            
            logger.info(f"Executing function: {func_name}")
            logger.debug(f"Parameters: {parameters}")
            
            success, result, message = self.function_registry.execute_function(
                func_name, **parameters
            )
            
            return {
                'type': 'function',
                'name': func_name,
                'success': success,
                'result': result,
                'message': message
            }
            
        except Exception as e:
            logger.error(f"Function execution error: {e}", exc_info=True)
            return {
                'type': 'function',
                'name': tool_request.get('name'),
                'success': False,
                'result': None,
                'message': str(e)
            }
    
    def _execute_rag_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG query using correct rag_top_k from RAGConfig"""
        try:
            if not self.rag_engine:
                return {
                    'type': 'rag',
                    'success': False,
                    'results': [],
                    'message': 'RAG engine not available'
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
    
    @log_extraction_stage("Stage 3: JSON Extraction")
    def _execute_stage3_extraction(self) -> bool:
        """Execute Stage 3: extraction with tool results"""
        try:
            extraction_prompt = self._build_stage3_main_extraction_prompt()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Extraction attempt {attempt + 1}/{self.max_retries}")
                    
                    response = self.llm_manager.generate(
                        extraction_prompt,
                        max_tokens=self.app_state.model_config.max_tokens
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
    
    @log_extraction_stage("Stage 4: RAG Refinement")
    def _execute_stage4_rag_refinement_with_retry(self) -> Dict[str, Any]:
        """
        Execute Stage 4: RAG refinement for SELECTED fields only
        FIXED: Only refine selected fields, merge back with unrefined fields
        """
        try:
            # Get selected fields for refinement from RAG config
            selected_fields = self.app_state.rag_config.rag_query_fields
            
            if not selected_fields:
                logger.info("No fields selected for RAG refinement - using stage 3 output")
                return self.context.stage3_output
            
            logger.info(f"Refining selected fields: {selected_fields}")
            
            refinement_prompt = self._build_stage4_rag_refinement_prompt()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Refinement attempt {attempt + 1}/{self.max_retries}")
                    
                    response = self.llm_manager.generate(
                        refinement_prompt,
                        max_tokens=self.app_state.model_config.max_tokens
                    )
                    
                    if not response:
                        logger.warning("No refinement response")
                        continue
                    
                    # Parse refined output
                    refined_output = self._parse_json_with_robust_parser(response, attempt, "stage4")
                    
                    if refined_output:
                        # FIXED: Merge refined fields with unrefined fields
                        merged_output = self._merge_refined_fields(
                            self.context.stage3_output,
                            refined_output,
                            selected_fields
                        )
                        
                        logger.info(f"Refinement successful - merged {len(selected_fields)} refined fields")
                        return merged_output
                    
                    logger.warning(f"Refinement parsing failed, attempt {attempt + 1}")
                    
                except Exception as e:
                    logger.error(f"Refinement attempt {attempt + 1} failed: {e}")
            
            logger.warning("Refinement failed, using stage 3 output")
            return self.context.stage3_output
            
        except Exception as e:
            logger.error(f"Stage 4 refinement failed: {e}", exc_info=True)
            return self.context.stage3_output
            
    
    def _build_stage1_analysis_prompt(self) -> str:
        """Build prompt for task analysis with ENHANCED keyword extraction guidance"""
        # Get available functions
        available_functions = self.function_registry.get_all_functions_info()
        function_descriptions = self._build_available_tools_description(available_functions)
        
        # Get task description
        task_description = self._extract_task_description()
        
        # Get schema as JSON string for analysis
        schema_json = json.dumps(self.app_state.prompt_config.json_schema, indent=2)
        
        prompt = f"""You are an intelligent agent analyzing an extraction task to determine which tools to use.

EXTRACTION TASK:
{task_description}

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
   - Scan the input text for ALL numeric values, measurements, and dates
   - Map values to function parameters based on context:
     * "45.5 kg" → weight_kg parameter
     * "165 cm" or "5'5\"" → height parameter (convert inches if needed)
     * "65 year old" or "age 65" → age parameter
     * "BP 140/90" → systolic=140, diastolic=90
     * "male" or "female" → sex parameter
   - Handle unit conversions within parameters:
     * If function needs kg but text has lbs → call lbs_to_kg first
     * If function needs meters but text has cm → call cm_to_m first
   - Chain functions when needed:
     * Example: calculate_bmi needs weight_kg and height_m
     * If text has "150 cm", first call cm_to_m(150), then use result in calculate_bmi
   - Extract dates and calculate intervals:
     * "DOB 01/15/2020, today 10/25/2025" → can calculate age
   - IMPORTANT: Extract parameters with high precision from actual text values
   - Only call functions that are truly needed for the extraction schema

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

CRITICAL REQUIREMENTS:
- Provide 3-5 RAG queries, each targeting different aspects (guidelines, diagnostics, assessment, etc.)
- Each RAG query MUST have 4-8 specific medical keywords
- Extract function parameters with precision from actual text values
- Chain functions if needed (e.g., unit conversion before calculation)
- Extras keywords (5-8 total): specific medical terminology, not generic words
- Do NOT use generic queries like "help", "information", or "guidelines" alone
- Reference known medical standards/organizations (WHO, CDC, ADA, etc.) in RAG queries

Respond with ONLY the JSON object."""

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
    
    def _build_available_tools_description(self, functions: List[Dict[str, Any]] = None) -> str:
        """Build description of available tools"""
        if functions is None:
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
        """Build prompt for stage 3 extraction"""
        # Format schema instructions
        schema_instructions = format_schema_as_instructions(
            self.app_state.prompt_config.json_schema
        )
        
        # Format tool results
        tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)
        
        # Build full prompt
        base_prompt = self.app_state.prompt_config.base_prompt or ""
        
        prompt = f"""{base_prompt}

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
                json_schema_instructions=schema_instructions  # ADD THIS LINE - it was missing!
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
        # Count tool usage
        rag_results = [r for r in self.context.tool_results if r.get('type') == 'rag']
        function_results = [r for r in self.context.tool_results if r.get('type') == 'function']
        extras_results = [r for r in self.context.tool_results if r.get('type') == 'extras']
    
        rag_count = len([r for r in rag_results if r.get('success')])
        function_count = len([r for r in function_results if r.get('success')])
        extras_count = len([r for r in extras_results if r.get('success')])
    
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
                'tool_results': self.context.tool_results,  # FIXED: Include full tool results
                'tool_results_summary': {
                    'rag': rag_count,
                    'functions': function_count,
                    'extras': extras_count
                },
                'rag_details': rag_details,
                'function_calls_details': function_details,
                'extras_details': extras_details,  # FIXED: Full content included
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
        """Get label context from mapping"""
        if not label_value:
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