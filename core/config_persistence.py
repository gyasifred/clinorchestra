#!/usr/bin/env python3
"""
Configuration Persistence Manager for ClinOrchestra
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
import sqlite3

logger = logging.getLogger(__name__)

class ConfigurationPersistenceManager:
    """Manages persistent storage of configuration state"""
    
    def __init__(self, persistence_dir: str = "./config_persistence"):
        """Initialize persistence manager"""
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        self.model_config_path = self.persistence_dir / "model_config.json"
        self.prompt_config_path = self.persistence_dir / "prompt_config.json"
        self.data_config_path = self.persistence_dir / "data_config.json"
        self.rag_config_path = self.persistence_dir / "rag_config.json"
        self.processing_config_path = self.persistence_dir / "processing_config.json"
        self.agentic_config_path = self.persistence_dir / "agentic_config.json"
        self.optimization_config_path = self.persistence_dir / "optimization_config.json"  # v1.0.0
        self.session_state_path = self.persistence_dir / "session_state.json"
        self.functions_backup_path = self.persistence_dir / "functions_backup.json"

        self.db_path = self.persistence_dir / "app_state.db"
        self._initialize_database()
        
        logger.info(f"ConfigurationPersistenceManager initialized: {self.persistence_dir}")
    
    def _initialize_database(self):
        """Initialize SQLite database for state persistence"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS state_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_type TEXT NOT NULL,
                        config_data TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        configuration_hash TEXT,
                        session_data TEXT
                    )
                """)
                
                conn.commit()
                logger.info("State persistence database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize persistence database: {e}")
    
    def save_model_config(self, model_config) -> bool:
        """Save model configuration to disk"""
        try:
            with self.lock:
                config_dict = {
                    'provider': model_config.provider,
                    'model_name': model_config.model_name,
                    'temperature': model_config.temperature,
                    'max_tokens': model_config.max_tokens,
                    'saved_at': datetime.now().isoformat()
                }
                
                if model_config.provider == "azure":
                    config_dict['azure_endpoint'] = getattr(model_config, 'azure_endpoint', '')
                    config_dict['azure_deployment'] = getattr(model_config, 'azure_deployment', '')
                elif model_config.provider == "google":
                    config_dict['google_project_id'] = getattr(model_config, 'google_project_id', '')
                elif model_config.provider == "local":
                    config_dict['local_model_path'] = getattr(model_config, 'local_model_path', '')
                    config_dict['quantization'] = getattr(model_config, 'quantization', 'none')
                    config_dict['max_seq_length'] = getattr(model_config, 'max_seq_length', 16384)
                    config_dict['gpu_layers'] = getattr(model_config, 'gpu_layers', -1)
                
                config_dict['has_api_key'] = bool(getattr(model_config, 'api_key', None))
                if config_dict['has_api_key']:
                    config_dict['api_key'] = model_config.api_key
                
                self._save_config_file(self.model_config_path, config_dict)
                self._save_config_to_db('model_config', config_dict)
                
                logger.info("Model configuration saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save model config: {e}", exc_info=True)
            return False
    
    def load_model_config(self) -> Optional[Dict[str, Any]]:
        """Load model configuration from disk"""
        try:
            return self._load_config_file(self.model_config_path)
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return None
    
    def save_prompt_config(self, prompt_config) -> bool:
        """Save prompt configuration to disk - FIXED: removed rag_query_fields"""
        try:
            with self.lock:
                config_dict = {
                    'main_prompt': prompt_config.main_prompt,
                    'minimal_prompt': prompt_config.minimal_prompt,
                    'use_minimal': prompt_config.use_minimal,
                    'rag_prompt': prompt_config.rag_prompt,
                    'json_schema': prompt_config.json_schema,
                    'token_count_main': prompt_config.token_count_main,
                    'token_count_minimal': prompt_config.token_count_minimal,
                    'token_count_rag': prompt_config.token_count_rag,
                    'saved_at': datetime.now().isoformat()
                }
                
                self._save_config_file(self.prompt_config_path, config_dict)
                self._save_config_to_db('prompt_config', config_dict)
                
                logger.info("Prompt configuration saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save prompt config: {e}", exc_info=True)
            return False
    
    def load_prompt_config(self) -> Optional[Dict[str, Any]]:
        """Load prompt configuration from disk"""
        try:
            return self._load_config_file(self.prompt_config_path)
        except Exception as e:
            logger.error(f"Failed to load prompt config: {e}")
            return None
    
    def save_data_config(self, data_config) -> bool:
        """Save data configuration to disk"""
        try:
            with self.lock:
                config_dict = {
                    'input_file': data_config.input_file,
                    'text_column': data_config.text_column,
                    'has_labels': data_config.has_labels,
                    'label_column': data_config.label_column,
                    'label_mapping': data_config.label_mapping,
                    'deid_columns': data_config.deid_columns,
                    'additional_columns': data_config.additional_columns,
                    'enable_phi_redaction': data_config.enable_phi_redaction,
                    'phi_entity_types': data_config.phi_entity_types,
                    'redaction_method': data_config.redaction_method,
                    'save_redacted_text': data_config.save_redacted_text,
                    'enable_pattern_normalization': data_config.enable_pattern_normalization,
                    'save_normalized_text': data_config.save_normalized_text,
                    'total_rows': data_config.total_rows,
                    'saved_at': datetime.now().isoformat()
                }
                
                self._save_config_file(self.data_config_path, config_dict)
                self._save_config_to_db('data_config', config_dict)
                
                logger.info("Data configuration saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save data config: {e}", exc_info=True)
            return False
    
    def load_data_config(self) -> Optional[Dict[str, Any]]:
        """Load data configuration from disk"""
        try:
            return self._load_config_file(self.data_config_path)
        except Exception as e:
            logger.error(f"Failed to load data config: {e}")
            return None
    
    def save_rag_config(self, rag_config) -> bool:
        """Save RAG configuration to disk - FIXED: includes rag_query_fields"""
        try:
            with self.lock:
                config_dict = {
                    'enabled': rag_config.enabled,
                    'documents': rag_config.documents,
                    'embedding_model': rag_config.embedding_model,
                    'chunk_size': rag_config.chunk_size,
                    'chunk_overlap': rag_config.chunk_overlap,
                    'rag_query_fields': rag_config.rag_query_fields,  # FIXED: Moved here
                    'k_value': rag_config.k_value,
                    'initialized': rag_config.initialized,
                    'cache_dir': rag_config.cache_dir,
                    'config_hash': str(hash(tuple(sorted(rag_config.documents)) if rag_config.documents else ())),
                    'saved_at': datetime.now().isoformat()
                }
                
                self._save_config_file(self.rag_config_path, config_dict)
                self._save_config_to_db('rag_config', config_dict)
                
                logger.info("RAG configuration saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save RAG config: {e}", exc_info=True)
            return False
    
    def load_rag_config(self) -> Optional[Dict[str, Any]]:
        """Load RAG configuration from disk"""
        try:
            return self._load_config_file(self.rag_config_path)
        except Exception as e:
            logger.error(f"Failed to load RAG config: {e}")
            return None
    
    def save_processing_config(self, processing_config) -> bool:
        """Save processing configuration to disk"""
        try:
            with self.lock:
                config_dict = {
                    'batch_size': processing_config.batch_size,
                    'concurrent_requests': processing_config.concurrent_requests,
                    'max_retries': processing_config.max_retries,
                    'error_strategy': processing_config.error_strategy,
                    'auto_save_interval': processing_config.auto_save_interval,
                    'dry_run': processing_config.dry_run,
                    'output_path': processing_config.output_path,
                    'saved_at': datetime.now().isoformat()
                }
                
                self._save_config_file(self.processing_config_path, config_dict)
                self._save_config_to_db('processing_config', config_dict)
                
                logger.info("Processing configuration saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save processing config: {e}", exc_info=True)
            return False
    
    def load_processing_config(self) -> Optional[Dict[str, Any]]:
        """Load processing configuration from disk"""
        try:
            return self._load_config_file(self.processing_config_path)
        except Exception as e:
            logger.error(f"Failed to load processing config: {e}")
            return None

    def save_agentic_config(self, agentic_config) -> bool:
        """Save agentic configuration to disk"""
        try:
            with self.lock:
                config_dict = {
                    'enabled': agentic_config.enabled,
                    'max_iterations': agentic_config.max_iterations,
                    'max_tool_calls': agentic_config.max_tool_calls,
                    'iteration_logging': agentic_config.iteration_logging,
                    'tool_call_logging': agentic_config.tool_call_logging,
                    'saved_at': datetime.now().isoformat()
                }

                self._save_config_file(self.agentic_config_path, config_dict)
                self._save_config_to_db('agentic_config', config_dict)

                logger.info(f"Agentic configuration saved (enabled={config_dict['enabled']})")
                return True

        except Exception as e:
            logger.error(f"Failed to save agentic config: {e}", exc_info=True)
            return False

    def load_agentic_config(self) -> Optional[Dict[str, Any]]:
        """Load agentic configuration from disk"""
        try:
            return self._load_config_file(self.agentic_config_path)
        except Exception as e:
            logger.error(f"Failed to load agentic config: {e}")
            return None

    def save_optimization_config(self, optimization_config) -> bool:
        """Save optimization configuration to disk (v1.0.0)"""
        try:
            with self.lock:
                config_dict = {
                    'llm_cache_enabled': optimization_config.llm_cache_enabled,
                    'llm_cache_db_path': optimization_config.llm_cache_db_path,
                    'llm_cache_bypass': optimization_config.llm_cache_bypass,
                    'performance_monitoring_enabled': optimization_config.performance_monitoring_enabled,
                    'use_parallel_processing': optimization_config.use_parallel_processing,
                    'use_batch_preprocessing': optimization_config.use_batch_preprocessing,
                    'max_parallel_workers': optimization_config.max_parallel_workers,
                    'use_model_profiles': optimization_config.use_model_profiles,
                    'use_gpu_faiss': optimization_config.use_gpu_faiss,
                    'saved_at': datetime.now().isoformat()
                }

                self._save_config_file(self.optimization_config_path, config_dict)
                self._save_config_to_db('optimization_config', config_dict)

                logger.info(f"Optimization configuration saved")
                return True

        except Exception as e:
            logger.error(f"Failed to save optimization config: {e}", exc_info=True)
            return False

    def load_optimization_config(self) -> Optional[Dict[str, Any]]:
        """Load optimization configuration from disk (v1.0.0)"""
        try:
            return self._load_config_file(self.optimization_config_path)
        except Exception as e:
            logger.error(f"Failed to load optimization config: {e}")
            return None

    def save_session_state(self, session_data: Dict[str, Any]) -> bool:
        """Save session state data"""
        try:
            with self.lock:
                session_dict = {
                    'session_data': session_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                self._save_config_file(self.session_state_path, session_dict)
                
                logger.info("Session state saved")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save session state: {e}", exc_info=True)
            return False
    
    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """Load session state data"""
        try:
            data = self._load_config_file(self.session_state_path)
            return data.get('session_data') if data else None
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None
    
    def backup_function_registry(self, functions_data: List[Dict[str, Any]]) -> bool:
        """Backup function registry data"""
        try:
            with self.lock:
                backup_dict = {
                    'functions': functions_data,
                    'backup_time': datetime.now().isoformat(),
                    'count': len(functions_data)
                }
                
                self._save_config_file(self.functions_backup_path, backup_dict)
                
                logger.info(f"Function registry backed up: {len(functions_data)} functions")
                return True
                
        except Exception as e:
            logger.error(f"Failed to backup function registry: {e}", exc_info=True)
            return False
    
    def restore_function_registry(self) -> Optional[List[Dict[str, Any]]]:
        """Restore function registry data"""
        try:
            data = self._load_config_file(self.functions_backup_path)
            if data and 'functions' in data:
                logger.info(f"Function registry restored: {data.get('count', 0)} functions")
                return data['functions']
            return None
        except Exception as e:
            logger.error(f"Failed to restore function registry: {e}")
            return None
    
    def get_configuration_history(self, config_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration history for a specific type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config_data, timestamp FROM state_history "
                    "WHERE config_type = ? ORDER BY timestamp DESC LIMIT ?",
                    (config_type, limit)
                )
                
                history = []
                for row in cursor.fetchall():
                    try:
                        config_data = json.loads(row[0])
                        history.append({
                            'config': config_data,
                            'timestamp': row[1]
                        })
                    except json.JSONDecodeError:
                        continue
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get configuration history: {e}")
            return []
    
    def clean_old_configurations(self, days_to_keep: int = 30) -> bool:
        """Clean old configuration entries"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM state_history WHERE timestamp < datetime(?, 'unixepoch')",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned {deleted_count} old configuration entries")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clean old configurations: {e}")
            return False
    
    def export_all_configurations(self) -> Optional[Dict[str, Any]]:
        """Export all configurations to a single dictionary"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'model_config': self.load_model_config(),
                'prompt_config': self.load_prompt_config(),
                'data_config': self.load_data_config(),
                'rag_config': self.load_rag_config(),
                'processing_config': self.load_processing_config(),
                'session_state': self.load_session_state(),
                'function_registry': self.restore_function_registry()
            }
            
            logger.info("All configurations exported")
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export configurations: {e}")
            return None
    
    def import_all_configurations(self, import_data: Dict[str, Any]) -> bool:
        """Import all configurations from a dictionary"""
        try:
            success_count = 0
            
            if import_data.get('function_registry'):
                self.backup_function_registry(import_data['function_registry'])
                success_count += 1
            
            logger.info(f"Imported {success_count} configuration types")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to import configurations: {e}")
            return False
    
    def _save_config_file(self, file_path: Path, config_dict: Dict[str, Any]):
        """Save configuration to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def _load_config_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file"""
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_config_to_db(self, config_type: str, config_dict: Dict[str, Any]):
        """Save configuration to database for history tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "UPDATE state_history SET is_active = 0 WHERE config_type = ?",
                    (config_type,)
                )
                
                cursor.execute(
                    "INSERT INTO state_history (config_type, config_data, is_active) VALUES (?, ?, 1)",
                    (config_type, json.dumps(config_dict, default=str))
                )
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save config to database: {e}")
    
    def get_persistence_status(self) -> Dict[str, Any]:
        """Get status of persistent storage"""
        try:
            status = {
                'persistence_dir': str(self.persistence_dir),
                'db_path': str(self.db_path),
                'files_exist': {
                    'model_config': self.model_config_path.exists(),
                    'prompt_config': self.prompt_config_path.exists(),
                    'data_config': self.data_config_path.exists(),
                    'rag_config': self.rag_config_path.exists(),
                    'processing_config': self.processing_config_path.exists(),
                    'session_state': self.session_state_path.exists(),
                    'functions_backup': self.functions_backup_path.exists()
                },
                'db_exists': self.db_path.exists()
            }
            
            if self.db_path.exists():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM state_history")
                    status['db_entries'] = cursor.fetchone()[0]
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get persistence status: {e}")
            return {'error': str(e)}
    
    def reset_all_persistence(self) -> bool:
        """Reset all persistent data (use with caution)"""
        try:
            with self.lock:
                for file_path in [
                    self.model_config_path,
                    self.prompt_config_path,
                    self.data_config_path,
                    self.rag_config_path,
                    self.processing_config_path,
                    self.session_state_path,
                    self.functions_backup_path
                ]:
                    if file_path.exists():
                        file_path.unlink()
                
                if self.db_path.exists():
                    self.db_path.unlink()
                    self._initialize_database()
                
                logger.info("All persistent data reset")
                return True
                
        except Exception as e:
            logger.error(f"Failed to reset persistence: {e}")
            return False

    def load_all_configs(self, app_state) -> bool:
        """Load all configurations into the app state - FIXED"""
        loaded_any = False
        
        try:
            # Load model config
            model_config_data = self.load_model_config()
            if model_config_data:
                try:
                    from core.model_config import ModelConfig
                    model_config = ModelConfig(
                        provider=model_config_data.get('provider', 'openai'),
                        model_name=model_config_data.get('model_name', 'gpt-4o'),
                        api_key=model_config_data.get('api_key', None),
                        temperature=model_config_data.get('temperature', 0.0),
                        max_tokens=model_config_data.get('max_tokens', 4000),
                        model_type=model_config_data.get('model_type', 'chat'),
                        azure_endpoint=model_config_data.get('azure_endpoint', None),
                        azure_deployment=model_config_data.get('azure_deployment', None),
                        google_project_id=model_config_data.get('google_project_id', None),
                        local_model_path=model_config_data.get('local_model_path', None),
                        quantization=model_config_data.get('quantization', '4bit'),
                        max_seq_length=model_config_data.get('max_seq_length', 16384),
                        gpu_layers=model_config_data.get('gpu_layers', -1)
                    )
                    success = app_state.set_model_config(model_config)
                    if success:
                        logger.info("Model configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore model config: {e}")
            
            # Load prompt config (WITHOUT rag_query_fields)
            prompt_config_data = self.load_prompt_config()
            if prompt_config_data:
                try:
                    success = app_state.set_prompt_config(
                        main_prompt=prompt_config_data.get('main_prompt', ''),
                        minimal_prompt=prompt_config_data.get('minimal_prompt'),
                        use_minimal=prompt_config_data.get('use_minimal', False),
                        json_schema=prompt_config_data.get('json_schema', {}),
                        rag_prompt=prompt_config_data.get('rag_prompt')
                    )
                    if success:
                        logger.info("Prompt configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore prompt config: {e}")
            
            # Load data config
            data_config_data = self.load_data_config()
            if data_config_data:
                try:
                    success = app_state.set_data_config(
                        input_file=data_config_data.get('input_file', ''),
                        text_column=data_config_data.get('text_column', ''),
                        has_labels=data_config_data.get('has_labels', False),
                        label_column=data_config_data.get('label_column'),
                        label_mapping=data_config_data.get('label_mapping', {}),
                        deid_columns=data_config_data.get('deid_columns', []),
                        additional_columns=data_config_data.get('additional_columns', []),
                        enable_phi_redaction=data_config_data.get('enable_phi_redaction', False),
                        phi_entity_types=data_config_data.get('phi_entity_types', []),
                        redaction_method=data_config_data.get('redaction_method', 'Replace with tag'),
                        save_redacted_text=data_config_data.get('save_redacted_text', True),
                        enable_pattern_normalization=data_config_data.get('enable_pattern_normalization', True),
                        save_normalized_text=data_config_data.get('save_normalized_text', False)
                    )
                    if success:
                        logger.info("Data configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore data config: {e}")
            
            # Load RAG config (WITH rag_query_fields)
            rag_config_data = self.load_rag_config()
            if rag_config_data:
                try:
                    success = app_state.set_rag_config(
                        enabled=rag_config_data.get('enabled', False),
                        documents=rag_config_data.get('documents', []),
                        embedding_model=rag_config_data.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
                        chunk_size=rag_config_data.get('chunk_size', 512),
                        chunk_overlap=rag_config_data.get('chunk_overlap', 50),
                        rag_query_fields=rag_config_data.get('rag_query_fields', []),  # FIXED: Moved here
                        k_value=rag_config_data.get('k_value', 3),
                        initialized=rag_config_data.get('initialized', False)
                    )
                    if success:
                        logger.info("RAG configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore RAG config: {e}")
            
            # Load processing config
            processing_config_data = self.load_processing_config()
            if processing_config_data:
                try:
                    success = app_state.set_processing_config(
                        batch_size=processing_config_data.get('batch_size', 10),
                        error_strategy=processing_config_data.get('error_strategy', 'skip'),
                        output_path=processing_config_data.get('output_path', ''),
                        dry_run=processing_config_data.get('dry_run', False),
                        max_retries=processing_config_data.get('max_retries', 3),
                        concurrent_requests=processing_config_data.get('concurrent_requests', 4),
                        auto_save_interval=processing_config_data.get('auto_save_interval', 50)
                    )
                    if success:
                        logger.info("Processing configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore processing config: {e}")

            # Load agentic config
            agentic_config_data = self.load_agentic_config()
            if agentic_config_data:
                try:
                    success = app_state.set_agentic_config(
                        enabled=agentic_config_data.get('enabled', False),
                        max_iterations=agentic_config_data.get('max_iterations', 3),  # Minimum 3 for ADAPTIVE mode
                        max_tool_calls=agentic_config_data.get('max_tool_calls', 50),
                        iteration_logging=agentic_config_data.get('iteration_logging', True),
                        tool_call_logging=agentic_config_data.get('tool_call_logging', True)
                    )
                    if success:
                        logger.info("Agentic configuration restored")
                        loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore agentic config: {e}")

            # Load optimization config (v1.0.0)
            optimization_config_data = self.load_optimization_config()
            if optimization_config_data:
                try:
                    from core.app_state import OptimizationConfig
                    app_state.optimization_config = OptimizationConfig(
                        llm_cache_enabled=optimization_config_data.get('llm_cache_enabled', True),
                        llm_cache_db_path=optimization_config_data.get('llm_cache_db_path', 'cache/llm_responses.db'),
                        performance_monitoring_enabled=optimization_config_data.get('performance_monitoring_enabled', True),
                        use_parallel_processing=optimization_config_data.get('use_parallel_processing', True),
                        use_batch_preprocessing=optimization_config_data.get('use_batch_preprocessing', True),
                        max_parallel_workers=optimization_config_data.get('max_parallel_workers', 5),
                        use_model_profiles=optimization_config_data.get('use_model_profiles', True),
                        use_gpu_faiss=optimization_config_data.get('use_gpu_faiss', False)
                    )
                    logger.info("Optimization configuration restored")
                    loaded_any = True
                except Exception as e:
                    logger.warning(f"Failed to restore optimization config: {e}")

            return loaded_any
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            return False

    def save_all_configs(self, app_state) -> bool:
        """Save all configurations from the app state"""
        success_count = 0
        
        try:
            if app_state.model_config:
                if self.save_model_config(app_state.model_config):
                    success_count += 1
            
            if app_state.prompt_valid:
                if self.save_prompt_config(app_state.prompt_config):
                    success_count += 1
            
            if app_state.data_valid:
                if self.save_data_config(app_state.data_config):
                    success_count += 1
            
            if self.save_rag_config(app_state.rag_config):
                success_count += 1
            
            if self.save_processing_config(app_state.processing_config):
                success_count += 1

            if self.save_agentic_config(app_state.agentic_config):
                success_count += 1

            if self.save_optimization_config(app_state.optimization_config):
                success_count += 1

            logger.info(f"Saved {success_count} configurations")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False

    def get_last_saved_info(self) -> Dict[str, str]:
        """Get information about when configurations were last saved"""
        info = {}
        
        try:
            configs = [
                ('model', self.model_config_path),
                ('prompt', self.prompt_config_path),
                ('data', self.data_config_path),
                ('rag', self.rag_config_path),
                ('processing', self.processing_config_path),
                ('agentic', self.agentic_config_path)
            ]
            
            for config_name, config_path in configs:
                if config_path.exists():
                    try:
                        config_data = self._load_config_file(config_path)
                        if config_data and 'saved_at' in config_data:
                            saved_at = datetime.fromisoformat(config_data['saved_at'])
                            info[config_name] = saved_at.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            info[config_name] = 'Unknown'
                    except Exception:
                        info[config_name] = 'Error'
                else:
                    info[config_name] = 'Never'
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get last saved info: {e}")
            return {}


# Global persistence manager instance
_persistence_manager = None

def get_persistence_manager() -> ConfigurationPersistenceManager:
    """Get global persistence manager instance"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = ConfigurationPersistenceManager()
    return _persistence_manager