#!/usr/bin/env python3
"""
Application State Manager for ClinAnnotate
Version: 1.0.1
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

FIXED: Resolved 'PromptConfig' object has no attribute 'rag_top_k' by:
  - Removing any reliance on PromptConfig.rag_top_k
  - Ensuring RAGConfig.rag_top_k is always synced with k_value
  - Adding __post_init__ to RAGConfig for guaranteed initialization
  - Maintaining full RAG functionality with zero code conflicts
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import sqlite3
import hashlib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class StateEvent(Enum):
    """State change events for observer pattern"""
    MODEL_CONFIG_CHANGED = "model_config_changed"
    PROMPT_CONFIG_CHANGED = "prompt_config_changed"
    DATA_CONFIG_CHANGED = "data_config_changed"
    RAG_CONFIG_CHANGED = "rag_config_changed"
    PROCESSING_CONFIG_CHANGED = "processing_config_changed"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETED = "processing_completed"

@dataclass
class PromptConfig:
    """
    Prompt configuration with dual prompt support and RAG refinement prompt
    NOTE: rag_top_k is NOT part of PromptConfig — it belongs in RAGConfig
    """
    main_prompt: str = ""
    minimal_prompt: Optional[str] = None
    use_minimal: bool = False
    rag_prompt: Optional[str] = None
    rag_query_fields: List[str] = field(default_factory=list)
    json_schema: Dict[str, Any] = field(default_factory=dict)
    assembled_main_prompt: str = ""
    assembled_minimal_prompt: str = ""
    token_count_main: int = 0
    token_count_minimal: int = 0
    token_count_rag: int = 0
    
    base_prompt: str = ""
    
    def __post_init__(self):
        """Synchronize base_prompt with main_prompt if not set"""
        if not self.base_prompt and self.main_prompt:
            self.base_prompt = self.main_prompt
        elif not self.main_prompt and self.base_prompt:
            self.main_prompt = self.base_prompt

@dataclass
class DataConfig:
    """Data configuration with text processing options"""
    input_file: Optional[str] = None
    text_column: Optional[str] = None
    has_labels: bool = False
    label_column: Optional[str] = None
    label_mapping: Dict[Any, str] = field(default_factory=dict)
    deid_columns: List[str] = field(default_factory=list)
    additional_columns: List[str] = field(default_factory=list)
    enable_phi_redaction: bool = False
    phi_entity_types: List[str] = field(default_factory=list)
    redaction_method: str = "Replace with tag"
    save_redacted_text: bool = True
    enable_pattern_normalization: bool = True
    save_normalized_text: bool = False
    data_preview: Optional[pd.DataFrame] = None
    total_rows: int = 0

@dataclass
class RAGConfig:
    """RAG configuration with guaranteed rag_top_k sync"""
    enabled: bool = False
    documents: List[str] = field(default_factory=list)
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    rag_query_fields: List[str] = field(default_factory=list)
    k_value: int = 3
    initialized: bool = False
    cache_dir: str = "./rag_cache"
    rag_top_k: int = field(init=False)  # Will be set in __post_init__

    def __post_init__(self):
        """Ensure rag_top_k is always synchronized with k_value"""
        self.rag_top_k = self.k_value
        logger.debug(f"RAGConfig initialized: k_value={self.k_value}, rag_top_k={self.rag_top_k}")

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    batch_size: int = 10
    concurrent_requests: int = 4
    max_retries: int = 3
    error_strategy: str = "skip"
    auto_save_interval: int = 50
    dry_run: bool = False
    output_path: Optional[str] = None

class StateObserver:
    """Observer for state changes"""
    
    def __init__(self):
        self.callbacks = {}
    
    def subscribe(self, event: StateEvent, callback):
        """Subscribe to state changes"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def notify(self, event: StateEvent, data: Any = None):
        """Notify observers of state changes"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Observer callback failed: {e}")

class AppState:
    """Central application state manager with lazy LLM initialization"""
    
    def __init__(self):
        """Initialize application state"""
        self.model_config = None
        self.prompt_config = PromptConfig()
        self.data_config = DataConfig()
        self.rag_config = RAGConfig()  # rag_top_k auto-initialized
        self.processing_config = ProcessingConfig()
        
        self.config_valid = False
        self.prompt_valid = False
        self.data_valid = False
        
        self.is_processing = False
        self.processed_rows = 0
        self.failed_rows = 0
        self.current_progress = 0.0
        self.is_using_minimal_prompt = False
        
        self.observer = StateObserver()
        
        # Persistent components - LLM initialized on demand
        self._llm_manager = None
        self._rag_engine = None
        self._regex_preprocessor = None
        self._extras_manager = None
        self._function_registry = None
        
        # Cache management
        self._cache_db_path = Path(self.rag_config.cache_dir) / "rag_cache.db"
        self._initialize_cache()
        
        logger.info("AppState initialized (v1.0.1 - RAG fix applied)")

    def _initialize_cache(self):
        """Initialize SQLite cache for RAG documents and embeddings"""
        Path(self.rag_config.cache_dir).mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT,
                    content TEXT,
                    metadata TEXT,
                    hash TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_text TEXT,
                    embedding BLOB,
                    chunk_config TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
            """)
            conn.commit()

    def count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation)"""
        return len(text) // 4

    def set_llm_manager(self, llm_manager) -> None:
        """Set the LLM manager instance"""
        self._llm_manager = llm_manager
        logger.info("LLM Manager registered with AppState")

    def get_llm_manager(self):
        """Get or initialize LLM manager on demand (Lazy initialization)"""
        if self._llm_manager is None:
            if not self.model_config:
                raise ValueError("Model configuration not set. Please configure model first.")
            
            logger.info("Lazy initializing LLM Manager on first use...")
            
            from core.llm_manager import LLMManager
            config_dict = {
                'provider': self.model_config.provider,
                'model_name': self.model_config.model_name,
                'api_key': self.model_config.api_key,
                'temperature': self.model_config.temperature,
                'max_tokens': self.model_config.max_tokens
            }
            
            if self.model_config.provider == "azure":
                config_dict['azure_endpoint'] = self.model_config.azure_endpoint
                config_dict['azure_deployment'] = self.model_config.azure_deployment
            elif self.model_config.provider == "google":
                config_dict['google_project_id'] = self.model_config.google_project_id
            elif self.model_config.provider == "local":
                config_dict['local_model_path'] = self.model_config.local_model_path
                config_dict['quantization'] = self.model_config.quantization
                config_dict['max_seq_length'] = self.model_config.max_seq_length
                config_dict['gpu_layers'] = self.model_config.gpu_layers
            
            self._llm_manager = LLMManager(config_dict)
            logger.info("LLM Manager initialized from config")
        
        return self._llm_manager

    def set_rag_engine(self, rag_engine) -> None:
        """Set the RAG engine instance"""
        self._rag_engine = rag_engine
        logger.info("RAG Engine registered with AppState")

    def get_rag_engine(self):
        """Get RAG engine instance"""
        return self._rag_engine

    def set_regex_preprocessor(self, regex_preprocessor) -> None:
        """Set the regex preprocessor instance"""
        self._regex_preprocessor = regex_preprocessor
        logger.info("Regex Preprocessor registered with AppState")

    def get_regex_preprocessor(self):
        """Get regex preprocessor instance"""
        return self._regex_preprocessor

    def set_extras_manager(self, extras_manager) -> None:
        """Set the extras manager instance"""
        self._extras_manager = extras_manager
        logger.info("Extras Manager registered with AppState")

    def get_extras_manager(self):
        """Get extras manager instance"""
        return self._extras_manager

    def set_function_registry(self, function_registry) -> None:
        """Set the function registry instance"""
        self._function_registry = function_registry
        logger.info("Function Registry registered with AppState")

    def get_function_registry(self):
        """Get function registry instance"""
        return self._function_registry

    def set_model_config(self, model_config) -> bool:
        """Set model configuration (Don't initialize LLM automatically)"""
        try:
            self.model_config = model_config
            self.config_valid = True
            
            # Clean up old LLM manager if it exists
            if self._llm_manager is not None:
                logger.info("Cleaning up old LLM manager...")
                self._llm_manager.cleanup()
                self._llm_manager = None
            
            logger.info(f"Model config set: {model_config.provider}/{model_config.model_name} (LLM will initialize on demand)")
            self.observer.notify(StateEvent.MODEL_CONFIG_CHANGED, model_config)
            return True
        except Exception as e:
            logger.error(f"Error setting model config: {e}")
            self.config_valid = False
            return False

    def set_prompt_config(self, main_prompt: str, minimal_prompt: Optional[str] = None,
                         use_minimal: bool = False, json_schema: Dict[str, Any] = None,
                         rag_prompt: Optional[str] = None,
                         rag_query_fields: List[str] = None) -> bool:
        """Set prompt configuration including RAG refinement prompt and query fields"""
        try:
            if not main_prompt or not main_prompt.strip():
                logger.error("Main prompt is required")
                return False
            
            if json_schema is None:
                json_schema = {}
            
            if rag_query_fields is None:
                rag_query_fields = []
            
            self.prompt_config.main_prompt = main_prompt
            self.prompt_config.base_prompt = main_prompt
            self.prompt_config.minimal_prompt = minimal_prompt
            self.prompt_config.use_minimal = use_minimal
            self.prompt_config.rag_prompt = rag_prompt
            self.prompt_config.rag_query_fields = list(rag_query_fields)
            self.prompt_config.json_schema = json_schema
            
            # Assemble prompts
            self.prompt_config.assembled_main_prompt = self._assemble_prompt(main_prompt, json_schema)
            
            if use_minimal and minimal_prompt:
                self.prompt_config.assembled_minimal_prompt = self._assemble_prompt(minimal_prompt, json_schema, is_minimal=True)
            else:
                self.prompt_config.assembled_minimal_prompt = ""
            
            # Calculate token counts
            self.prompt_config.token_count_main = self.count_tokens(self.prompt_config.assembled_main_prompt)
            self.prompt_config.token_count_minimal = self.count_tokens(self.prompt_config.assembled_minimal_prompt)
            self.prompt_config.token_count_rag = self.count_tokens(rag_prompt) if rag_prompt else 0
            
            self.prompt_valid = True
            logger.info(f"Prompt config set: main={len(main_prompt)}, minimal={len(minimal_prompt) if minimal_prompt else 0}, rag={len(rag_prompt) if rag_prompt else 0}, query_fields={len(rag_query_fields)}, schema={len(json_schema)}")
            
            self.observer.notify(StateEvent.PROMPT_CONFIG_CHANGED, self.prompt_config)
            return True
            
        except Exception as e:
            logger.error(f"Error setting prompt config: {e}")
            self.prompt_valid = False
            return False

    def _assemble_prompt(self, prompt_text: str, schema: Dict[str, Any], is_minimal: bool = False) -> str:
        """Assemble prompt with JSON enforcement and schema instructions"""
        try:
            from core.prompt_templates import (
                get_json_enforcement_instructions,
                format_schema_as_instructions
            )
            
            # Get JSON enforcement instructions
            json_enforcement = get_json_enforcement_instructions()
            
            # Add schema instructions if schema is provided
            schema_str = ""
            if schema:
                schema_json = json.dumps(schema, indent=2)
                schema_str = format_schema_as_instructions(schema)
                schema_str += f"\n\n**JSON Schema:**\n```json\n"
                schema_str += schema_json
                schema_str += "\n```\n\n"
                schema_str += "**Important:**\n"
                schema_str += "- Follow the exact field names and types\n"
                schema_str += "- Include all required fields\n"
                schema_str += "- Use null for missing optional values\n"
                schema_str += "- Maintain nested structure as shown"
            
            # Ensure prompt has all placeholders before assembly
            assembled = prompt_text
            
            # Replace json_schema_instructions if it's a placeholder
            if '{json_schema_instructions}' in assembled:
                assembled = assembled.replace('{json_schema_instructions}', schema_str)
            else:
                # Append schema and enforcement
                assembled = assembled + "\n\n" + schema_str + "\n\n" + json_enforcement
            
            logger.debug(f"Assembled {'minimal' if is_minimal else 'main'} prompt: {len(assembled)} chars")
            
            return assembled
            
        except Exception as e:
            logger.error(f"Error assembling prompt: {e}")
            raise

    def get_prompt_for_processing(self, retry_count: int = 0) -> str:
        """Get appropriate prompt based on retry count"""
        if self.prompt_config.use_minimal and retry_count >= self.processing_config.max_retries:
            logger.info(f"Switching to minimal prompt after {retry_count} retries")
            self.is_using_minimal_prompt = True
            return self.prompt_config.assembled_minimal_prompt
        else:
            self.is_using_minimal_prompt = False
            return self.prompt_config.assembled_main_prompt

    def set_label_mappings(self, mappings: Dict[Any, str]) -> bool:
        """Set label mappings for all label categories"""
        try:
            if not mappings:
                logger.warning("Attempted to set empty label mappings")
                return False
            
            self.data_config.label_mapping = dict(mappings)
            logger.info(f"Set {len(mappings)} label mappings")
            return True
            
        except Exception as e:
            logger.error(f"Error setting label mappings: {e}")
            return False

    def set_data_config(self, input_file: str, text_column: str, has_labels: bool,
                       label_column: Optional[str], label_mapping: Dict[Any, str],
                       deid_columns: List[str], additional_columns: List[str],
                       enable_phi_redaction: bool, phi_entity_types: List[str],
                       redaction_method: str, save_redacted_text: bool,
                       enable_pattern_normalization: bool, save_normalized_text: bool) -> bool:
        """Set data configuration"""
        try:
            self.data_config.input_file = input_file
            self.data_config.text_column = text_column
            self.data_config.has_labels = has_labels
            self.data_config.label_column = label_column
            self.data_config.label_mapping = dict(label_mapping) if label_mapping else {}
            self.data_config.deid_columns = list(deid_columns)
            self.data_config.additional_columns = list(additional_columns)
            self.data_config.enable_phi_redaction = enable_phi_redaction
            self.data_config.phi_entity_types = list(phi_entity_types)
            self.data_config.redaction_method = redaction_method
            self.data_config.save_redacted_text = save_redacted_text
            self.data_config.enable_pattern_normalization = enable_pattern_normalization
            self.data_config.save_normalized_text = save_normalized_text
            
            self.data_valid = True
            logger.info(f"Data config set: {input_file}, labels={has_labels}, "
                       f"mappings={len(label_mapping)}, phi={enable_phi_redaction}")
            
            self.observer.notify(StateEvent.DATA_CONFIG_CHANGED, self.data_config)
            return True
            
        except Exception as e:
            logger.error(f"Error setting data config: {e}")
            self.data_valid = False
            return False

    def set_rag_config(self, enabled: bool, documents: List[str] = None,
                      embedding_model: str = None, chunk_size: int = None,
                      chunk_overlap: int = None, rag_query_fields: List[str] = None,
                      k_value: int = None, initialized: bool = None) -> bool:
        """Set RAG configuration with guaranteed rag_top_k sync"""
        try:
            self.rag_config.enabled = enabled
            
            if documents is not None:
                self.rag_config.documents = list(documents)
            
            if embedding_model is not None:
                self.rag_config.embedding_model = embedding_model
            
            if chunk_size is not None:
                self.rag_config.chunk_size = chunk_size
            
            if chunk_overlap is not None:
                self.rag_config.chunk_overlap = chunk_overlap
            
            if rag_query_fields is not None:
                self.rag_config.rag_query_fields = list(rag_query_fields)
            
            if k_value is not None:
                self.rag_config.k_value = k_value
                self.rag_config.rag_top_k = k_value  # Sync rag_top_k
            
            if initialized is not None:
                self.rag_config.initialized = initialized
            
            logger.info(f"RAG config set: enabled={enabled}, docs={len(self.rag_config.documents)}, "
                       f"model={self.rag_config.embedding_model}, k_value={self.rag_config.k_value}, "
                       f"rag_top_k={self.rag_config.rag_top_k}")
            
            self.observer.notify(StateEvent.RAG_CONFIG_CHANGED, self.rag_config)
            return True
            
        except Exception as e:
            logger.error(f"Error setting RAG config: {e}")
            return False

    def start_processing(self):
        """Start processing session"""
        self.is_processing = True
        self.processed_rows = 0
        self.failed_rows = 0
        self.current_progress = 0.0
        logger.info("Processing started")
        self.observer.notify(StateEvent.PROCESSING_STARTED)

    def update_progress(self, processed: int, failed: int, progress: float):
        """Update processing progress"""
        self.processed_rows = processed
        self.failed_rows = failed
        self.current_progress = progress
        self.observer.notify(StateEvent.PROCESSING_PROGRESS, {
            'processed': processed,
            'failed': failed,
            'progress': progress
        })

    def stop_processing(self):
        """Stop processing session"""
        self.is_processing = False
        logger.info(f"Processing stopped. Processed: {self.processed_rows}, Failed: {self.failed_rows}")
        self.observer.notify(StateEvent.PROCESSING_COMPLETED, {
            'processed': self.processed_rows,
            'failed': self.failed_rows,
            'total': self.data_config.total_rows
        })

    def cleanup(self):
        """Clean up resources"""
        if self._llm_manager:
            self._llm_manager.cleanup()
            self._llm_manager = None
        self._rag_engine = None
        self._regex_preprocessor = None
        self._extras_manager = None
        self._function_registry = None
        logger.info("AppState resources cleaned up")

    def can_start_processing(self) -> tuple[bool, str]:
        """Check if processing can start"""
        if not self.config_valid:
            return False, "Model not configured"
        if not self.prompt_valid:
            return False, "Prompt not configured"
        if not self.data_valid:
            return False, "Data not configured"
        return True, "Ready"

    def set_processing_config(self, batch_size: int, error_strategy: str, 
                            output_path: str, dry_run: bool):
        """Set processing configuration"""
        try:
            self.processing_config.batch_size = batch_size
            self.processing_config.error_strategy = error_strategy
            self.processing_config.output_path = output_path
            self.processing_config.dry_run = dry_run
            logger.info(f"Processing config set: batch={batch_size}, strategy={error_strategy}, output_path={output_path}")
            self.observer.notify(StateEvent.PROCESSING_CONFIG_CHANGED, self.processing_config)
            return True
        except Exception as e:
            logger.error(f"Error setting processing config: {e}")
            return False

    def get_configuration_summary(self) -> str:
        """Get human-readable configuration summary"""
        summary = "=== Configuration Summary ===\n\n"
        
        summary += "Model:\n"
        if self.model_config:
            summary += f"  Provider: {self.model_config.provider}\n"
            summary += f"  Model: {self.model_config.model_name}\n"
        else:
            summary += "  Not configured\n"
        summary += f"  Status: {'Valid' if self.config_valid else 'Invalid'}\n"
        summary += f"  LLM Initialized: {'Yes' if self._llm_manager else 'No (will initialize on demand)'}\n\n"
        
        summary += "Prompt:\n"
        summary += f"  Main Length: {len(self.prompt_config.main_prompt)} chars\n"
        summary += f"  Main Tokens: {self.prompt_config.token_count_main}\n"
        if self.prompt_config.use_minimal:
            summary += f"  Minimal Length: {len(self.prompt_config.minimal_prompt or '')} chars\n"
            summary += f"  Minimal Tokens: {self.prompt_config.token_count_minimal}\n"
        if self.prompt_config.rag_prompt:
            summary += f"  RAG Refinement Length: {len(self.prompt_config.rag_prompt)} chars\n"
            summary += f"  RAG Refinement Tokens: {self.prompt_config.token_count_rag}\n"
        summary += f"  Fields: {len(self.prompt_config.json_schema)}\n"
        summary += f"  Fallback Enabled: {'Yes' if self.prompt_config.use_minimal else 'No'}\n"
        summary += f"  RAG Refinement Enabled: {'Yes' if self.prompt_config.rag_prompt else 'No'}\n"
        summary += f"  Status: {'Valid' if self.prompt_valid else 'Invalid'}\n\n"
        
        summary += "Data:\n"
        summary += f"  File: {self.data_config.input_file or 'Not set'}\n"
        summary += f"  Text Column: {self.data_config.text_column or 'Not set'}\n"
        summary += f"  Has Labels: {self.data_config.has_labels}\n"
        if self.data_config.has_labels:
            summary += f"  Label Column: {self.data_config.label_column}\n"
            summary += f"  Label Mappings: {len(self.data_config.label_mapping)}\n"
        summary += f"  PHI Redaction: {self.data_config.enable_phi_redaction}\n"
        summary += f"  Pattern Normalization: {self.data_config.enable_pattern_normalization}\n"
        summary += f"  Status: {'Valid' if self.data_valid else 'Invalid'}\n\n"
        
        summary += "RAG:\n"
        summary += f"  Enabled: {self.rag_config.enabled}\n"
        summary += f"  Documents: {len(self.rag_config.documents)}\n"
        summary += f"  Embedding Model: {self.rag_config.embedding_model}\n"
        summary += f"  Chunk Size: {self.rag_config.chunk_size}\n"
        summary += f"  Chunk Overlap: {self.rag_config.chunk_overlap}\n"
        summary += f"  Query Fields: {len(self.rag_config.rag_query_fields)}\n"
        summary += f"  K Value: {self.rag_config.k_value}\n"
        summary += f"  rag_top_k: {self.rag_config.rag_top_k} (synced)\n"
        summary += f"  Initialized: {self.rag_config.initialized}\n"
        summary += f"  Cache Directory: {self.rag_config.cache_dir}\n\n"
        
        summary += "Processing:\n"
        summary += f"  Batch Size: {self.processing_config.batch_size}\n"
        summary += f"  Error Strategy: {self.processing_config.error_strategy}\n"
        summary += f"  Output Path: {self.processing_config.output_path or 'Not set'}\n"
        summary += f"  Dry Run: {self.processing_config.dry_run}\n"
        
        return summary

    def is_rag_refinement_enabled(self) -> bool:
        """Check if RAG refinement is enabled and configured"""
        return (
            self.rag_config.enabled and
            self.rag_config.initialized and
            bool(self.prompt_config.rag_prompt)
        )