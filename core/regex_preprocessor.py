#!/usr/bin/env python3
"""
Regex Preprocessor - Text preprocessing before LLM
Version: 1.0.0
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegexPattern:
    """Regex pattern definition"""
    name: str
    pattern: str
    replacement: str
    description: str
    enabled: bool = True


class RegexPreprocessor:
    """Preprocess text with regex patterns before LLM"""
    
    def __init__(self, storage_path: str = "./patterns"):
        """Initialize preprocessor"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.patterns: List[RegexPattern] = []
        
        self._load_all_patterns()
        self._register_builtin_patterns()
        
        logger.info(f"RegexPreprocessor initialized with {len(self.patterns)} patterns")
    
    def _register_builtin_patterns(self):
        """Register common medical text patterns with FIXED escape sequences"""
        
        builtins = [
            RegexPattern(
                name="standardize_dosage",
                pattern=r'(\d+)\s+(mg|mcg|g|ml|units)',
                replacement=r'\1\2',
                description="Remove space between dose and unit: '20 mg' -> '20mg'",
                enabled=True
            ),
            RegexPattern(
                name="standardize_bp",
                pattern=r'BP[\s:]*(\d+)\s*/\s*(\d+)',
                replacement=r'BP: \1/\2',
                description="Standardize blood pressure: 'BP 120 / 80' -> 'BP: 120/80'",
                enabled=True
            ),
            RegexPattern(
                name="standardize_lab_format",
                pattern=r'([A-Za-z0-9]+)\s*[:=]\s*(\d+\.?\d*)\s*([a-zA-Z/]+)',
                replacement=r'\1: \2 \3',
                description="Standardize lab format: 'Glucose=120mg/dL' -> 'Glucose: 120 mg/dL'",
                enabled=True
            ),
            RegexPattern(
                name="remove_extra_whitespace",
                pattern=r'\s+',
                replacement=r' ',
                description="Replace multiple spaces with single space",
                enabled=True
            ),
            RegexPattern(
                name="standardize_negation",
                pattern=r'\b(no|not|negative for|denies)\s+([a-zA-Z\s]+)',
                replacement=r'No \2',
                description="Standardize negation: 'negative for fever' -> 'No fever'",
                enabled=True
            )
        ]
        
        for pattern in builtins:
            if not any(p.name == pattern.name for p in self.patterns):
                # FIXED: Validate pattern before adding
                if self._validate_pattern_syntax(pattern.pattern):
                    self.patterns.append(pattern)
                    self._save_pattern(pattern)
                else:
                    logger.error(f"Pattern '{pattern.name}' failed validation: bad escape sequence")
    
    def _validate_pattern_syntax(self, pattern: str) -> bool:
        """Validate regex pattern syntax"""
        try:
            re.compile(pattern)
            return True
        except re.error as e:
            logger.error(f"Pattern validation failed: {e}")
            return False
    
    def add_pattern(self, name: str, pattern: str, replacement: str, 
                   description: str, enabled: bool = True) -> Tuple[bool, str]:
        """Add a regex pattern with validation"""
        try:
            # FIXED: Validate pattern first
            if not self._validate_pattern_syntax(pattern):
                return False, f"Invalid regex pattern syntax"
            
            # Check for duplicates
            if any(p.name == name for p in self.patterns):
                return False, f"Pattern '{name}' already exists"
            
            regex_pattern = RegexPattern(
                name=name,
                pattern=pattern,
                replacement=replacement,
                description=description,
                enabled=enabled
            )
            
            self.patterns.append(regex_pattern)
            self._save_pattern(regex_pattern)
            
            logger.info(f"Added pattern: {name}")
            return True, f"Pattern '{name}' added"
            
        except re.error as e:
            return False, f"Invalid regex: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            return False, f"Error: {str(e)}"
    
    def update_pattern(self, pattern_id: str, name: str, pattern: str, replacement: str,
                      description: str, enabled: bool) -> Tuple[bool, str]:
        """Update an existing pattern"""
        try:
            # FIXED: Validate pattern first
            if not self._validate_pattern_syntax(pattern):
                return False, f"Invalid regex pattern syntax"
            
            # Find and update pattern
            for i, p in enumerate(self.patterns):
                if p.name == pattern_id or p.name == name:
                    self.patterns[i] = RegexPattern(
                        name=name,
                        pattern=pattern,
                        replacement=replacement,
                        description=description,
                        enabled=enabled
                    )
                    self._save_pattern(self.patterns[i])
                    logger.info(f"Updated pattern: {name}")
                    return True, f"Pattern '{name}' updated"
            
            return False, f"Pattern not found"
            
        except Exception as e:
            logger.error(f"Failed to update pattern: {e}")
            return False, f"Error: {str(e)}"
    
    def remove_pattern(self, name: str) -> Tuple[bool, str]:
        """Remove a pattern"""
        try:
            self.patterns = [p for p in self.patterns if p.name != name]
            
            pattern_file = self.storage_path / f"{name}.json"
            if pattern_file.exists():
                pattern_file.unlink()
            
            logger.info(f"Removed pattern: {name}")
            return True, f"Pattern '{name}' removed"
            
        except Exception as e:
            logger.error(f"Failed to remove pattern: {e}")
            return False, f"Error: {str(e)}"
    
    def enable_pattern(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a pattern"""
        for pattern in self.patterns:
            if pattern.name == name:
                pattern.enabled = enabled
                self._save_pattern(pattern)
                return True
        return False
    
    def preprocess(self, text: str, pattern_names: Optional[List[str]] = None) -> str:
        """Preprocess text with regex patterns"""
        if not text:
            return text
        
        processed_text = text
        
        # Select patterns to apply
        if pattern_names:
            patterns_to_apply = [p for p in self.patterns if p.name in pattern_names]
        else:
            patterns_to_apply = [p for p in self.patterns if p.enabled]
        
        # Apply patterns in order
        for pattern in patterns_to_apply:
            try:
                processed_text = re.sub(
                    pattern.pattern,
                    pattern.replacement,
                    processed_text,
                    flags=re.IGNORECASE
                )
            except Exception as e:
                logger.error(f"Pattern '{pattern.name}' failed: {e}")
                continue
        
        return processed_text.strip()
    
    def test_pattern(self, pattern_name: str, test_text: str) -> Tuple[bool, str, str]:
        """Test a pattern on sample text"""
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                try:
                    result = re.sub(
                        pattern.pattern,
                        pattern.replacement,
                        test_text,
                        flags=re.IGNORECASE
                    )
                    return True, result, "Test successful"
                except Exception as e:
                    return False, test_text, f"Error: {str(e)}"
        
        return False, test_text, f"Pattern '{pattern_name}' not found"
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all patterns"""
        return [
            {
                'id': p.name,  # FIXED: Added id field for UI
                'name': p.name,
                'pattern': p.pattern,
                'replacement': p.replacement,
                'description': p.description,
                'enabled': p.enabled
            }
            for p in self.patterns
        ]
    
    def get_pattern(self, name: str) -> Optional[Dict[str, Any]]:
        """Get pattern details"""
        for pattern in self.patterns:
            if pattern.name == name:
                return {
                    'id': pattern.name,  # FIXED: Added id field for UI
                    'name': pattern.name,
                    'pattern': pattern.pattern,
                    'replacement': pattern.replacement,
                    'description': pattern.description,
                    'enabled': pattern.enabled
                }
        return None
    
    def _save_pattern(self, pattern: RegexPattern):
        """Save pattern to file"""
        pattern_file = self.storage_path / f"{pattern.name}.json"
        
        pattern_data = {
            'name': pattern.name,
            'pattern': pattern.pattern,
            'replacement': pattern.replacement,
            'description': pattern.description,
            'enabled': pattern.enabled
        }
        
        with open(pattern_file, 'w') as f:
            json.dump(pattern_data, f, indent=2)
    
    def _load_all_patterns(self):
        """Load all patterns from storage"""
        for pattern_file in self.storage_path.glob("*.json"):
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                
                # FIXED: Validate pattern before loading
                if not self._validate_pattern_syntax(data['pattern']):
                    logger.error(f"Pattern '{data['name']}' failed: bad escape \\U at position 0")
                    continue
                
                pattern = RegexPattern(
                    name=data['name'],
                    pattern=data['pattern'],
                    replacement=data['replacement'],
                    description=data['description'],
                    enabled=data.get('enabled', True)
                )
                
                self.patterns.append(pattern)
            except Exception as e:
                logger.error(f"Failed to load {pattern_file}: {e}")
    
    def export_patterns(self) -> str:
        """Export all patterns as JSON"""
        patterns_data = [
            {
                'name': p.name,
                'pattern': p.pattern,
                'replacement': p.replacement,
                'description': p.description,
                'enabled': p.enabled
            }
            for p in self.patterns
        ]
        return json.dumps(patterns_data, indent=2)
    
    def import_patterns(self, json_str: str) -> Tuple[bool, int, str]:
        """Import patterns from JSON"""
        try:
            patterns_data = json.loads(json_str)
            count = 0
            
            for data in patterns_data:
                # FIXED: Validate before importing
                if not self._validate_pattern_syntax(data['pattern']):
                    logger.warning(f"Skipping invalid pattern '{data['name']}'")
                    continue
                
                success, _ = self.add_pattern(
                    data['name'],
                    data['pattern'],
                    data['replacement'],
                    data['description'],
                    data.get('enabled', True)
                )
                if success:
                    count += 1
            
            return True, count, f"Imported {count} patterns"
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False, 0, f"Error: {str(e)}"