#!/usr/bin/env python3
"""
Data Processor - CSV processing and batch management
"""

import pandas as pd
import numpy as np
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
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


def preprocess_clinical_text(text: str, max_chars: int = 60000) -> str:
    """
    Preprocess clinical text by removing special tokens and normalizing.
    From your existing code.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove special tokens
    special_tokens = [
        '<s>', '</s>', '<pad>', '</pad>', '<eos>', '<bos>', '<|endoftext|>', '<|padding|>',
        '<|begin_of_text|>', '<|end_of_text|>', '<|start_header_id|>', '<|end_header_id|>',
        '<|eot_id|>', '<|assistant|>', '<|user|>', '<|system|>',
        '[INST]', '[/INST]', '<|im_start|>', '<|im_end|>'
    ]
    
    for token in special_tokens:
        text = text.replace(token, ' ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*\.\s*\.', ' ', text)
    text = re.sub(r'\s*-\s*-\s*-', ' ', text)
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text.strip()


class DataProcessor:
    """
    Process CSV data for clinical assessment.
    Handles batch processing, data validation, and output formatting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor with configuration.
        
        Args:
            config: Configuration dictionary with file paths, columns, etc.
        """
        self.input_file = config.get('input_file')
        self.text_column = config.get('text_column')
        self.deid_columns = config.get('deid_columns', [])
        self.additional_columns = config.get('additional_columns', [])
        self.batch_size = config.get('batch_size', 10)
        self.output_path = config.get('output_path', './output')
        self.dry_run = config.get('dry_run', False)
        self.auto_save_interval = config.get('auto_save_interval', 50)
        
        self.df = None
        self.total_rows = 0
        self.processed_records = []
        
        logger.info(f"DataProcessor initialized: {self.input_file}")
    
    def load_data(self) -> Tuple[bool, str]:
        """Load and validate data from CSV file."""
        try:
            if not self.input_file:
                return False, "Input file not specified"
            
            file_path = Path(self.input_file)
            if not file_path.exists():
                return False, f"File not found: {self.input_file}"
            
            # Load CSV
            self.df = pd.read_csv(self.input_file)
            
            if self.df.empty:
                return False, "CSV file is empty"
            
            # Validate required columns
            if self.text_column not in self.df.columns:
                return False, f"Text column '{self.text_column}' not found in CSV"
            
            # Drop rows with missing text
            initial_count = len(self.df)
            self.df = self.df.dropna(subset=[self.text_column])
            dropped = initial_count - len(self.df)
            
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with missing text")
            
            # Preprocess text column
            self.df[self.text_column] = self.df[self.text_column].astype(str).apply(
                preprocess_clinical_text
            )
            
            # Filter out very short texts
            self.df = self.df[self.df[self.text_column].str.len() > 100].reset_index(drop=True)
            
            self.total_rows = len(self.df)
            
            # Apply dry run limit
            if self.dry_run:
                self.df = self.df.head(5)
                self.total_rows = len(self.df)
                logger.info("DRY RUN mode: Processing only first 5 rows")
            
            logger.info(f"Data loaded: {self.total_rows} rows ready for processing")
            return True, f"Successfully loaded {self.total_rows} rows"
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False, f"Error loading data: {str(e)}"
    
    def get_batches(self) -> List[pd.DataFrame]:
        """Split data into batches for processing."""
        if self.df is None:
            return []
        
        batches = []
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df.iloc[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches (size: {self.batch_size})")
        return batches
    
    def get_row_data(self, row: pd.Series) -> Dict[str, Any]:
        """Extract data from a single row for processing."""
        return {
            'text': row[self.text_column],
            'deid': {col: row.get(col) for col in self.deid_columns if col in row.index},
            'additional': {col: row.get(col) for col in self.additional_columns if col in row.index},
            'row_index': row.name
        }
    
    def add_result(self, row: pd.Series, llm_output: Dict[str, Any], 
                   metadata: Dict[str, Any]):
        """
        Add a processing result to the collection.
        
        Args:
            row: Original DataFrame row
            llm_output: LLM generated output (parsed JSON)
            metadata: Processing metadata (timestamps, model info, etc.)
        """
        record = {}
        
        # 1. Add DEID columns
        for col in self.deid_columns:
            if col in row.index:
                record[col] = row[col]
        
        # 2. Add original text
        record[self.text_column] = row[self.text_column]
        
        # 3. Add LLM output fields (flatten if needed)
        for key, value in llm_output.items():
            if isinstance(value, (dict, list)):
                # Store complex types as JSON strings
                record[key] = json.dumps(value, cls=NpEncoder)
            else:
                record[key] = value
        
        # 4. Add additional columns
        for col in self.additional_columns:
            if col in row.index:
                record[col] = row[col]
        
        # 5. Add metadata
        record.update(metadata)
        
        self.processed_records.append(record)
    
    def save_results(self, output_filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Save processed results to CSV.
        
        Args:
            output_filename: Optional custom filename
            
        Returns:
            Success status and output file path or error message
        """
        try:
            # Create output directory
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"clinical_assessment_results_{timestamp}.csv"
            
            output_file = output_dir / output_filename
            
            # Convert to DataFrame and save
            results_df = pd.DataFrame(self.processed_records)
            results_df.to_csv(output_file, index=False)
            
            logger.info(f"Results saved: {output_file}")
            return True, str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            return False, f"Error saving results: {str(e)}"
    
    def save_incremental(self, checkpoint_number: int) -> Tuple[bool, str]:
        """Save incremental checkpoint during processing."""
        try:
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = output_dir / f"checkpoint_{checkpoint_number}_{timestamp}.csv"
            
            results_df = pd.DataFrame(self.processed_records)
            results_df.to_csv(checkpoint_file, index=False)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return True, str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def save_failed_rows(self, failed_rows: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Save failed rows to separate CSV for review."""
        try:
            if not failed_rows:
                return True, "No failed rows to save"
            
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            failed_file = output_dir / f"failed_rows_{timestamp}.csv"
            
            failed_df = pd.DataFrame(failed_rows)
            failed_df.to_csv(failed_file, index=False)
            
            logger.info(f"Failed rows saved: {failed_file}")
            return True, str(failed_file)
            
        except Exception as e:
            logger.error(f"Failed to save failed rows: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def save_statistics(self, stats: Dict[str, Any]) -> Tuple[bool, str]:
        """Save processing statistics to JSON."""
        try:
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_file = output_dir / f"processing_stats_{timestamp}.json"
            
            # Add dataset statistics
            if self.processed_records:
                results_df = pd.DataFrame(self.processed_records)
                stats['dataset_statistics'] = {
                    'total_records': len(results_df),
                    'columns': list(results_df.columns),
                    'column_count': len(results_df.columns)
                }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, cls=NpEncoder)
            
            logger.info(f"Statistics saved: {stats_file}")
            return True, str(stats_file)
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_processing_summary(self) -> str:
        """Generate human-readable processing summary."""
        if not self.processed_records:
            return "No records processed yet"
        
        summary = f"""Processing Summary
{'='*50}

Input File: {Path(self.input_file).name}
Total Rows Processed: {len(self.processed_records)}
Text Column: {self.text_column}
DEID Columns: {len(self.deid_columns)}
Additional Columns: {len(self.additional_columns)}

Output Columns: {len(self.processed_records[0].keys()) if self.processed_records else 0}
"""
        return summary


class BatchProcessor:
    """Process data in batches with progress tracking."""
    
    def __init__(self, data_processor: DataProcessor, llm_manager, 
                 concurrent_requests: int = 4, error_strategy: str = "skip",
                 max_retries: int = 2):
        """
        Initialize batch processor.
        
        Args:
            data_processor: DataProcessor instance
            llm_manager: LLM manager instance
            concurrent_requests: Number of parallel requests
            error_strategy: How to handle errors (skip, retry, halt)
            max_retries: Maximum retry attempts
        """
        self.data_processor = data_processor
        self.llm_manager = llm_manager
        self.concurrent_requests = concurrent_requests
        self.error_strategy = error_strategy
        self.max_retries = max_retries
        
        self.processed_count = 0
        self.failed_count = 0
        self.failed_rows = []
        
        logger.info("BatchProcessor initialized")
    
    def process_batch(self, batch: pd.DataFrame, prompt_template: str, 
                     progress_callback=None) -> Tuple[int, int]:
        """
        Process a batch of rows.
        
        Args:
            batch: DataFrame batch to process
            prompt_template: Prompt template with {clinical_text} placeholder
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        batch_success = 0
        batch_failed = 0
        
        for idx, row in batch.iterrows():
            try:
                # Extract row data
                row_data = self.data_processor.get_row_data(row)
                clinical_text = row_data['text']
                
                # Build prompt
                prompt = prompt_template.replace('{clinical_text}', clinical_text)
                
                # Generate with LLM
                response = self.llm_manager.generate(prompt)
                
                # Parse JSON output
                llm_output = json.loads(response)
                
                # Add metadata
                metadata = {
                    'processing_timestamp': datetime.now().isoformat(),
                    'llm_provider': self.llm_manager.provider,
                    'llm_model': self.llm_manager.model_name,
                    'row_index': idx
                }
                
                # Store result
                self.data_processor.add_result(row, llm_output, metadata)
                
                batch_success += 1
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process row {idx}: {str(e)}")
                
                # Handle based on error strategy
                if self.error_strategy == "halt":
                    raise
                elif self.error_strategy == "retry":
                    # Retry logic would go here
                    pass
                
                # Record failure
                failed_row = row.to_dict()
                failed_row['error'] = str(e)
                failed_row['row_index'] = idx
                self.failed_rows.append(failed_row)
                
                batch_failed += 1
                self.failed_count += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(self.processed_count, self.failed_count)
        
        return batch_success, batch_failed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed': self.processed_count,
            'failed': self.failed_count,
            'success_rate': (self.processed_count / (self.processed_count + self.failed_count) * 100) 
                           if (self.processed_count + self.failed_count) > 0 else 0
        }