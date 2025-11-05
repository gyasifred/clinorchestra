#!/usr/bin/env python3
"""
Output Handler - Fixed to use only JSON keys as column names
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.bool, bool)):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


class OutputHandler:
    """
    Handles output formatting and saving.
    Only uses JSON keys as column names.
    """
    
    def __init__(self, data_config, prompt_config):
        self.data_config = data_config
        self.prompt_config = prompt_config
        self.records = []
        
        logger.info("OutputHandler initialized")
    
    def add_record(self, row: pd.Series, llm_output: Dict[str, Any], 
                   processed_text: Optional[str] = None,
                   redacted_text: Optional[str] = None,
                   normalized_text: Optional[str] = None,
                   label_context: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Add a processing record.
        Only JSON keys become columns.
        """
        record = {}
        
        # 1. Add identifier columns (DEID columns)
        for col in self.data_config.deid_columns:
            if col in row.index:
                record[col] = row[col]
        
        # 2. Add original text column
        text_col = self.data_config.text_column
        record[text_col] = row[text_col]
        
        # 3. Add optional processed text columns
        if self.data_config.enable_phi_redaction and self.data_config.save_redacted_text:
            if redacted_text is not None:
                record[f"{text_col}_redacted"] = redacted_text
        
        if self.data_config.enable_pattern_normalization and self.data_config.save_normalized_text:
            if normalized_text is not None:
                record[f"{text_col}_normalized"] = normalized_text
        
        # 4. Add label information (if applicable)
        if self.data_config.has_labels:
            label_col = self.data_config.label_column
            if label_col in row.index:
                record['input_label_value'] = row[label_col]
            
            if label_context:
                record['label_context_used'] = label_context
        
        # 5. CRITICAL: Add LLM output fields - ONLY JSON keys as columns
        for key, value in llm_output.items():
            if isinstance(value, (dict, list)):
                record[key] = json.dumps(value, cls=NpEncoder)
            elif pd.isna(value) or value is None:
                record[key] = None
            else:
                record[key] = value
        
        # 6. Add additional columns from original data
        for col in self.data_config.additional_columns:
            if col in row.index:
                record[col] = row[col]
        
        # 7. Add processing metadata
        if metadata:
            record.update(metadata)
        else:
            record['processing_timestamp'] = datetime.now().isoformat()
        
        self.records.append(record)
        
        logger.debug(f"Record added with {len(record)} columns")
    
    def get_column_names(self) -> List[str]:
        """Get all column names that will appear in the output"""
        if not self.records:
            return []
        
        all_columns = set()
        for record in self.records:
            all_columns.update(record.keys())
        
        return sorted(list(all_columns))
    
    def save_to_csv(self, output_path: str, filename: Optional[str] = None) -> tuple[bool, str]:
        """Save all records to CSV file"""
        try:
            if not self.records:
                return False, "No records to save"
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"clinannotate_results_{timestamp}.csv"
            
            output_file = output_dir / filename
            
            df = pd.DataFrame(self.records)
            
            columns = self._get_preferred_column_order(df.columns.tolist())
            df = df[columns]
            
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved {len(df)} records to {output_file}")
            logger.info(f"Output columns ({len(columns)}): {', '.join(columns[:10])}...")
            
            return True, str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return False, f"Error: {str(e)}"
    
    def _get_preferred_column_order(self, columns: List[str]) -> List[str]:
        """Order columns in a logical sequence"""
        ordered = []
        
        # 1. Identifier columns first
        for col in self.data_config.deid_columns:
            if col in columns:
                ordered.append(col)
                columns.remove(col)
        
        # 2. Original text column
        text_col = self.data_config.text_column
        if text_col in columns:
            ordered.append(text_col)
            columns.remove(text_col)
        
        # 3. Processed text columns
        processed_cols = [
            f"{text_col}_redacted",
            f"{text_col}_normalized"
        ]
        for col in processed_cols:
            if col in columns:
                ordered.append(col)
                columns.remove(col)
        
        # 4. Label columns
        label_cols = ['input_label_value', 'label_context_used']
        for col in label_cols:
            if col in columns:
                ordered.append(col)
                columns.remove(col)
        
        # 5. JSON output columns (from schema) - preserve original names
        schema_keys = list(self.prompt_config.json_schema.keys())
        for key in schema_keys:
            if key in columns:
                ordered.append(key)
                columns.remove(key)
        
        # 6. Additional columns from original data
        for col in self.data_config.additional_columns:
            if col in columns:
                ordered.append(col)
                columns.remove(col)
        
        # 7. Metadata columns
        metadata_cols = [
            'processing_timestamp',
            'llm_provider',
            'llm_model',
            'llm_temperature',
            'prompt_type',
            'retry_count',
            'extras_used',
            'rag_used',
            'functions_called',
            'rag_refinement_applied',
            'processing_time_seconds'
        ]
        for col in metadata_cols:
            if col in columns:
                ordered.append(col)
                columns.remove(col)
        
        # 8. Any remaining columns
        ordered.extend(sorted(columns))
        
        return ordered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get output statistics"""
        if not self.records:
            return {
                'total_records': 0,
                'total_columns': 0,
                'columns': []
            }
        
        df = pd.DataFrame(self.records)
        
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats['numeric_columns'] = numeric_cols
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def save_statistics(self, output_path: str, filename: str = "statistics.json") -> tuple[bool, str]:
        """Save statistics to JSON file"""
        try:
            stats = self.get_statistics()
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            stats_file = output_dir / filename
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, cls=NpEncoder)
            
            logger.info(f"Statistics saved to {stats_file}")
            
            return True, str(stats_file)
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
            return False, f"Error: {str(e)}"
    
    def clear(self):
        """Clear all records"""
        self.records = []
        logger.info("Output records cleared")