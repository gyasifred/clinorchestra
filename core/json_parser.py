#!/usr/bin/env python3
"""
JSON Parser with robust fallback strategies for LLM responses
FIXED VERSION 1.0.1 - Corrected method signatures for proper usage

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.1 - FIXED

FIXES:
1. Made parse_json_response an instance method (not static)
2. Added backward compatibility with static method wrapper
3. Consistent method signatures across all parsing functions
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, List
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class JSONExtractionResult:
    """Result container for JSON extraction attempts"""
    def __init__(self, success: bool, data: Optional[Dict] = None, 
                 error: Optional[str] = None, method: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.method = method

class JSONParser:
    """
    Robust JSON parser with multiple fallback strategies for LLM responses
    FIXED: Instance methods for proper usage in agent_system
    """
    
    def __init__(self):
        """Initialize JSON parser"""
        pass
    
    @staticmethod
    def aggressive_json_sanitizer(json_str: str) -> str:
        """
        AGGRESSIVE JSON sanitization - fixes all common LLM JSON issues
        
        Fixes:
        - Markdown code blocks (```json, ```)
        - Extra text before/after JSON
        - Multiline strings (collapses to single line)
        - Trailing commas
        - Unbalanced braces and brackets
        - Escape sequence issues
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Sanitized JSON string
        """
        if not json_str:
            return json_str
        
        # Remove markdown code blocks
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE | re.IGNORECASE)
        json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\s*```\s*$', '', json_str, flags=re.MULTILINE)
        
        # Extract from first { to last }
        first_brace = json_str.find('{')
        if first_brace > 0:
            json_str = json_str[first_brace:]
        
        last_brace = json_str.rfind('}')
        if last_brace > 0 and last_brace < len(json_str) - 1:
            json_str = json_str[:last_brace + 1]
        
        # Fix multiline strings - collapse them to single line
        def fix_multiline_strings(match):
            key = match.group(1)
            value = match.group(2)
            # Collapse newlines and multiple spaces
            value = value.replace('\n', ' ').replace('\r', ' ')
            value = re.sub(r'\s+', ' ', value)
            # Fix escape sequences
            value = value.replace('\\"', '"').replace('"', '\\"')
            # Remove null bytes and other control characters
            value = value.replace('\t', ' ').replace('\x00', '')
            return f'"{key}": "{value}"'
        
        json_str = re.sub(
            r'"([^"]+)":\s*"((?:[^"\\]|\\.)*)(?:"(?=\s*[,}\]]))',
            fix_multiline_strings,
            json_str,
            flags=re.DOTALL
        )
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix double backslashes
        json_str = json_str.replace('\\\\', '\\')
        
        # Balance braces and brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    @staticmethod
    def extract_json_from_response(text: str) -> Optional[str]:
        """
        Extract JSON from LLM response with advanced pattern matching
        
        Tracks opening/closing braces while respecting strings and escapes
        
        Args:
            text: Raw response text
            
        Returns:
            Extracted JSON string or None
        """
        text = JSONParser.aggressive_json_sanitizer(text)
        
        # Find first opening brace
        start = text.find('{')
        if start < 0:
            return None
        
        # Track depth and string state
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            # Handle escapes
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            # Toggle string state
            if char == '"':
                in_string = not in_string
                continue
            
            # Only count braces outside strings
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    # Found matching closing brace
                    if depth == 0:
                        return text[start:i+1]
        
        # Return partial if we have opening brace but incomplete
        return text[start:] if depth > 0 else None
    
    @staticmethod
    def repair_truncated_json(json_str: str) -> str:
        """
        Repair truncated or incomplete JSON structures
        
        Args:
            json_str: Potentially truncated JSON
            
        Returns:
            Repaired JSON string
        """
        if not json_str:
            return json_str
        
        # First sanitize
        json_str = JSONParser.aggressive_json_sanitizer(json_str)
        
        # Count quotes (excluding escaped quotes)
        quote_count = 0
        escape_next = False
        for char in json_str:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                quote_count += 1
        
        # Close unclosed string
        if quote_count % 2 != 0:
            json_str += '"'
        
        # Balance braces and brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Close arrays first, then objects
        json_str += ']' * max(0, open_brackets - close_brackets)
        json_str += '}' * max(0, open_braces - close_braces)
        
        return json_str
    
    def parse_json_response(self, response: str, schema: Optional[Dict] = None) -> Tuple[Optional[Dict], str]:
        """
        FIXED: Instance method for parsing JSON with multiple fallback strategies
        
        Args:
            response: Raw text response from LLM
            schema: Optional schema for validation (not strictly enforced)
            
        Returns:
            Tuple of (parsed_dict, method_used)
        """
        result = self.parse_json_strict(response)
        return (result.data, result.method) if result.success else (None, None)
    
    @staticmethod
    def parse_json_strict(response: str) -> JSONExtractionResult:
        """
        Strict JSON parsing with multiple fallback strategies
        ENHANCED with aggressive sanitization from working code
        """
        if not response:
            return JSONExtractionResult(
                success=False, 
                error="Empty response",
                method="strict"
            )
        
        # Method 1: Direct parse
        try:
            parsed = json.loads(response)
            return JSONExtractionResult(
                success=True,
                data=parsed,
                method="direct"
            )
        except json.JSONDecodeError:
            pass
        
        # Method 2: Aggressive sanitize + parse
        try:
            sanitized = JSONParser.aggressive_json_sanitizer(response)
            parsed = json.loads(sanitized)
            return JSONExtractionResult(
                success=True,
                data=parsed,
                method="aggressive_sanitize"
            )
        except json.JSONDecodeError as e:
            logger.debug(f"Aggressive sanitization failed: {e}")
        
        # Method 3: Extract + aggressive sanitize + parse
        try:
            extracted = JSONParser.extract_json_from_response(response)
            if extracted:
                sanitized = JSONParser.aggressive_json_sanitizer(extracted)
                parsed = json.loads(sanitized)
                return JSONExtractionResult(
                    success=True,
                    data=parsed,
                    method="extract_sanitize"
                )
        except json.JSONDecodeError as e:
            logger.debug(f"Extract + sanitize failed: {e}")
        
        # Method 4: Extract + repair + parse
        try:
            extracted = JSONParser.extract_json_from_response(response)
            if extracted:
                repaired = JSONParser.repair_truncated_json(extracted)
                parsed = json.loads(repaired)
                return JSONExtractionResult(
                    success=True,
                    data=parsed,
                    method="extract_repair"
                )
        except json.JSONDecodeError as e:
            logger.debug(f"Extract + repair failed: {e}")
        
        # Method 5: Full pipeline (sanitize + extract + repair + parse)
        try:
            sanitized = JSONParser.aggressive_json_sanitizer(response)
            extracted = JSONParser.extract_json_from_response(sanitized)
            if extracted:
                repaired = JSONParser.repair_truncated_json(extracted)
                parsed = json.loads(repaired)
                return JSONExtractionResult(
                    success=True,
                    data=parsed,
                    method="full_pipeline"
                )
        except json.JSONDecodeError as e:
            logger.debug(f"Full pipeline failed: {e}")
        
        # Method 6: Character cleaning + sanitize + parse
        try:
            # Remove non-printable characters
            cleaned = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' 
                            for char in response)
            sanitized = JSONParser.aggressive_json_sanitizer(cleaned)
            parsed = json.loads(sanitized)
            return JSONExtractionResult(
                success=True,
                data=parsed,
                method="char_clean"
            )
        except json.JSONDecodeError as e:
            logger.debug(f"Character cleaning failed: {e}")
        
        # Method 7: Salvage attempt (find last complete structure)
        try:
            extracted = JSONParser.extract_json_from_response(response)
            if extracted:
                # Find last complete field
                last_complete = extracted.rfind('",')
                if last_complete > 0:
                    truncated = extracted[:last_complete + 2]
                    repaired = JSONParser.repair_truncated_json(truncated)
                    parsed = json.loads(repaired)
                    return JSONExtractionResult(
                        success=True,
                        data=parsed,
                        method="salvage"
                    )
        except json.JSONDecodeError as e:
            logger.debug(f"Salvage attempt failed: {e}")
        
        # Method 8: Markdown table extraction fallback
        markdown_result = JSONParser.extract_from_markdown(response)
        if markdown_result.success:
            return markdown_result
        
        # Method 9: Key-value pair extraction fallback
        kv_result = JSONParser.extract_key_value_pairs(response)
        if kv_result.success:
            return kv_result
        
        # All methods failed
        return JSONExtractionResult(
            success=False,
            error="No valid JSON found in response after all 9 parsing methods",
            method="strict"
        )
    
    @staticmethod
    def extract_from_markdown(response: str) -> JSONExtractionResult:
        """Extract data from markdown tables or structured text"""
        try:
            html = markdown.markdown(response, extensions=['tables'])
            soup = BeautifulSoup(html, 'html.parser')
            
            extracted_data = {}
            
            # Try to extract from tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) >= 2:
                    headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
                    
                    for row in rows[1:]:
                        cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                        if len(cells) == len(headers) and len(headers) == 2:
                            key = cells[0].lower().replace(' ', '_')
                            value = cells[1]
                            extracted_data[key] = JSONParser._convert_value(value)
            
            if extracted_data:
                logger.info("Extracted data from markdown structure")
                return JSONExtractionResult(
                    success=True,
                    data=extracted_data,
                    method="markdown"
                )
                
        except Exception as e:
            logger.debug(f"Markdown extraction failed: {e}")
        
        return JSONExtractionResult(
            success=False,
            error="Markdown extraction failed",
            method="markdown"
        )
    
    @staticmethod
    def extract_key_value_pairs(response: str) -> JSONExtractionResult:
        """Extract key-value pairs from unstructured text"""
        try:
            extracted_data = {}
            
            # Pattern for key: value pairs
            kv_pattern = r'(?:^|\n)\s*([a-zA-Z_][a-zA-Z0-9_\s]*?):\s*(.+?)(?=\n\s*[a-zA-Z_]|$)'
            matches = re.findall(kv_pattern, response, re.MULTILINE | re.DOTALL)
            
            for key, value in matches:
                clean_key = key.strip().lower().replace(' ', '_')
                clean_value = value.strip()
                
                if len(clean_key) < 2 or len(clean_value) < 1:
                    continue
                
                extracted_data[clean_key] = JSONParser._convert_value(clean_value)
            
            if extracted_data and len(extracted_data) >= 2:
                logger.info("Extracted data from key-value pairs")
                return JSONExtractionResult(
                    success=True,
                    data=extracted_data,
                    method="key_value"
                )
                
        except Exception as e:
            logger.debug(f"Key-value extraction failed: {e}")
        
        return JSONExtractionResult(
            success=False,
            error="Key-value extraction failed",
            method="key_value"
        )
    
    @staticmethod
    def _convert_value(value: str) -> Any:
        """Convert string value to appropriate type"""
        if not value or value.lower() in ['null', 'none', 'n/a', 'unknown']:
            return None
        
        # Try boolean
        if value.lower() in ['true', 'yes', 'y']:
            return True
        if value.lower() in ['false', 'no', 'n']:
            return False
        
        # Try number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value.strip('"\'')


# Backward compatibility wrapper
def parse_json_with_progressive_fallbacks(json_str: str, schema: Optional[Dict] = None, 
                                         attempt_repair: bool = True) -> Optional[Dict]:
    """Legacy function for backward compatibility"""
    parser = JSONParser()
    result, _ = parser.parse_json_response(json_str, schema)
    return result
    

# Main class alias for backward compatibility
class EnhancedJSONParser:
    """Alias for JSONParser class"""
    
    @staticmethod
    def parse_json_strict(response: str) -> JSONExtractionResult:
        return JSONParser.parse_json_strict(response)
    
    @staticmethod
    def extract_json_only(text: str) -> Optional[str]:
        return JSONParser.extract_json_only(text)
    
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        return JSONParser.clean_json_string(json_str)
    
    @staticmethod
    def aggressive_json_sanitizer(json_str: str) -> str:
        return JSONParser.aggressive_json_sanitizer(json_str)
    
    @staticmethod
    def extract_json_from_response(text: str) -> Optional[str]:
        return JSONParser.extract_json_from_response(text)
    
    @staticmethod
    def repair_truncated_json(json_str: str) -> str:
        return JSONParser.repair_truncated_json(json_str)
    
    def parse_with_fallbacks(self, response: str, schema: Optional[Dict] = None, 
                            attempt_repair: bool = True) -> JSONExtractionResult:
        return JSONParser.parse_with_fallbacks(response, schema, attempt_repair)