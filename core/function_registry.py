#!/usr/bin/env python3
"""
Function Registry - Parameter validation and transformation
Version: 1.0.0
"""
import sys
import json
from typing import Dict, Any, Tuple, List, Callable, Optional
from pathlib import Path
import inspect
from core.logging_config import get_logger

logger = get_logger(__name__)

class FunctionRegistry:
    """Registry for custom functions with robust parameter validation"""
    
    def __init__(self, storage_path: str = "./functions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.functions: Dict[str, Dict[str, Any]] = {}
        self.namespace = self._create_namespace()

        # PERFORMANCE: Result caching for repeated function calls
        self.result_cache: Dict[str, Tuple[bool, Any, str]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self._load_all_functions()
        self._register_builtin_functions()

        logger.info(f"FunctionRegistry initialized with {len(self.functions)} functions")
    
    def _create_namespace(self) -> Dict[str, Any]:
        """Create execution namespace with comprehensive support for growth calculators"""
        return {
            '__builtins__': {
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'dict': dict,
                'float': float,
                'int': int,
                'len': len,
                'list': list,
                'max': max,
                'min': min,
                'round': round,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'zip': zip,
                'enumerate': enumerate,
                'range': range,
                'sorted': sorted,
                'reversed': reversed,
                '__import__': __import__,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'Exception': Exception,
                'print': print,
            },
            'json': json,
            'math': __import__('math'),
            're': __import__('re'),
            'pandas': __import__('pandas'),
            'numpy': __import__('numpy'),
            'scipy': __import__('scipy'),
            'dataclasses': __import__('dataclasses'),
            'enum': __import__('enum'),
            'datetime': __import__('datetime'),
            'dateutil': __import__('dateutil'),
        }
    
    def register_function(self, name: str, code: str, description: str,
                         parameters: Dict[str, Any], returns: str) -> Tuple[bool, str]:
        """Register a function with comprehensive validation"""
        try:
            # Validate code syntax
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return False, f"Syntax error in function code: {str(e)}"
            
            # Execute the function code
            local_namespace = self.namespace.copy()
            exec(code, local_namespace)
            
            # Extract function name from code
            func_name = self._extract_function_name(code)
            if not func_name:
                return False, "Could not extract function name from code"
            
            # Get the compiled function
            compiled_func = local_namespace.get(func_name)
            if not compiled_func:
                return False, f"Function {func_name} not found after compilation"
            
            # Validate function signature against parameters
            try:
                sig = inspect.signature(compiled_func)
                sig_params = list(sig.parameters.keys())
                expected_params = list(parameters.keys())
                
                if sig_params != expected_params:
                    logger.warning(f"Parameter mismatch for {name}: signature={sig_params}, expected={expected_params}")
            except Exception as e:
                logger.warning(f"Could not validate signature for {name}: {e}")
            
            # Store function metadata
            self.functions[name] = {
                'name': name,
                'code': code,
                'description': description,
                'parameters': parameters,
                'returns': returns,
                'compiled': compiled_func,
                'signature': str(inspect.signature(compiled_func)) if hasattr(compiled_func, '__call__') else None
            }
            
            # Persist to disk
            self._save_function(name)
            
            logger.info(f"Registered function: {name}")
            return True, f"Function '{name}' registered successfully"
            
        except SyntaxError as e:
            return False, f"Syntax error in function code: {str(e)}"
        except Exception as e:
            logger.error(f"Function registration failed: {e}", exc_info=True)
            return False, f"Registration error: {str(e)}"
    
    def execute_function(self, name: str, **kwargs) -> Tuple[bool, Any, str]:
        """
        Execute a registered function with parameter validation and conversion
        FIXED: Apply transformations BEFORE validation
        PERFORMANCE: Cache results to avoid redundant calculations
        """
        if name not in self.functions:
            return False, None, f"Function '{name}' not found"

        # PERFORMANCE: Create cache key from function name and sorted parameters
        try:
            # Sort kwargs for consistent cache keys
            cache_key = f"{name}:{json.dumps(kwargs, sort_keys=True)}"

            # Check cache first
            if cache_key in self.result_cache:
                self.cache_hits += 1
                cached_result = self.result_cache[cache_key]
                logger.debug(f"Cache HIT for {name}({kwargs}) - returning cached result")
                return cached_result

            self.cache_misses += 1

        except (TypeError, ValueError) as e:
            # If kwargs not JSON-serializable, skip caching
            logger.debug(f"Could not create cache key for {name}: {e}")
            cache_key = None

        try:
            func = self.functions[name]['compiled']
            func_info = self.functions[name]

            # Get function signature for validation
            try:
                sig = inspect.signature(func)
                sig_params = sig.parameters
            except Exception as e:
                logger.warning(f"Could not get signature for {name}: {e}")
                sig_params = {}

            # CRITICAL FIX: Apply transformations FIRST, then validate
            transformed_kwargs = self._apply_parameter_transformations(kwargs.copy(), name)

            # Then validate and convert the transformed parameters
            validated_kwargs = self._validate_and_convert_parameters(
                transformed_kwargs,
                func_info.get('parameters', {}),
                sig_params,
                name
            )

            # Execute function
            result = func(**validated_kwargs)

            execution_result = (True, result, "Execution successful")

            # PERFORMANCE: Cache the result if cache_key was created
            if cache_key is not None:
                self.result_cache[cache_key] = execution_result
                logger.debug(f"Cached result for {name}({kwargs})")

            logger.info(f"Function '{name}' executed successfully")
            return execution_result

        except TypeError as e:
            error_msg = f"Invalid arguments for function '{name}': {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Function '{name}' execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def _validate_and_convert_parameters(self, 
                                       kwargs: Dict[str, Any], 
                                       param_specs: Dict[str, Any],
                                       sig_params: Dict[str, Any],
                                       func_name: str) -> Dict[str, Any]:
        """
        Validate and convert function parameters to proper types
        """
        validated = {}
        
        # Get actual parameter names from signature if available
        if sig_params:
            expected_param_names = list(sig_params.keys())
        else:
            expected_param_names = list(param_specs.keys())
        
        # Convert and validate each parameter
        for param_name in expected_param_names:
            if param_name in kwargs:
                raw_value = kwargs[param_name]
                
                # Skip if value indicates missing data
                if self._is_missing_value(raw_value):
                    logger.info(f"Skipping {func_name} - parameter {param_name} is missing/unknown")
                    continue
                
                # Get expected type from parameter spec
                param_spec = param_specs.get(param_name, {})
                expected_type = param_spec.get('type', 'string')
                
                # Convert value
                converted_value = self._convert_parameter_value(raw_value, expected_type, param_name)
                if converted_value is not None:
                    validated[param_name] = converted_value
                
            elif sig_params and param_name in sig_params:
                # Check if parameter has default value
                param = sig_params[param_name]
                if param.default is not inspect.Parameter.empty:
                    # Parameter has default, skip
                    continue
                else:
                    # Required parameter missing
                    logger.warning(f"Required parameter {param_name} missing for {func_name}")
        
        logger.debug(f"Validated parameters for {func_name}: {validated}")
        return validated
    
    def _is_missing_value(self, value: Any) -> bool:
        """Check if value indicates missing/unknown data"""
        if value is None:
            return True
        
        if isinstance(value, str):
            missing_indicators = [
                'not documented', 'unknown', 'missing', 'n/a', 'na',
                'not available', 'not provided', 'not specified',
                'not mentioned', 'not noted', 'not stated'
            ]
            return value.lower().strip() in missing_indicators
        
        return False
    
    def _convert_parameter_value(self, value: Any, expected_type: str, param_name: str) -> Any:
        """Convert parameter value to expected type"""
        try:
            if expected_type == 'number':
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Handle common number formats
                    cleaned = value.strip().replace(',', '')
                    if cleaned:
                        return float(cleaned)
                    else:
                        return None
                else:
                    return float(value)
                    
            elif expected_type == 'boolean':
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', 'yes', '1', 'y', 'on']
                else:
                    return bool(value)
                    
            elif expected_type == 'string':
                return str(value) if value is not None else None
                
            else:
                # Default to string
                return str(value) if value is not None else None
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Parameter conversion failed for {param_name}: {e}")
            return None
    
    def _apply_parameter_transformations(self, params: Dict[str, Any], func_name: str) -> Dict[str, Any]:
        """
        Apply function-specific parameter transformations
        FIXED: Now handles height_cm -> height_m conversion properly and BMI calculations
        """
        transformed = params.copy()
        
        # BMI calculation transformations - FIXED
        if func_name == 'calculate_bmi':
            # If height_cm is provided but not height_m, keep height_cm (function handles conversion)
            # If both are provided, let the function handle it
            # The function will prioritize height_m if both are given
            
            # Ensure weight parameter name is correct
            if 'weight' in transformed and 'weight_kg' not in transformed:
                transformed['weight_kg'] = transformed['weight']
                transformed.pop('weight', None)
        
        # Growth percentile calculations
        elif func_name in ['calculate_cdc_growth_percentile', 'calculate_who_growth_percentile', 'calculate_growth_percentile']:
            # Ensure age is in months
            if 'age' in transformed and 'age_months' not in transformed:
                transformed['age_months'] = transformed['age']
                transformed.pop('age', None)

            # Handle sex parameter - convert string to number
            # CDC/WHO convention: 1 = male, 2 = female
            if 'sex' in transformed:
                sex_value = transformed['sex']
                if isinstance(sex_value, str):
                    sex_lower = sex_value.lower().strip()
                    if sex_lower in ['male', 'm', 'boy', 'man']:
                        transformed['sex'] = 1
                        logger.info(f"Converted sex '{sex_value}' to 1 (male)")
                    elif sex_lower in ['female', 'f', 'girl', 'woman']:
                        transformed['sex'] = 2
                        logger.info(f"Converted sex '{sex_value}' to 2 (female)")
                    else:
                        logger.warning(f"Unknown sex value '{sex_value}', cannot convert")

            # Handle weight/height parameters
            if 'weight' in transformed and 'weight_kg' not in transformed:
                transformed['weight_kg'] = transformed['weight']
                transformed.pop('weight', None)

            if 'height' in transformed and 'height_cm' not in transformed:
                transformed['height_cm'] = transformed['height']
                transformed.pop('height', None)
        
        # Z-score calculations
        elif func_name in ['calculate_cdc_z_score', 'calculate_who_z_score']:
            # Similar transformations as percentile calculations
            if 'age' in transformed and 'age_months' not in transformed:
                transformed['age_months'] = transformed['age']
                transformed.pop('age', None)
            
            if 'height_cm' in transformed and 'height_m' not in transformed:
                height_cm = transformed.get('height_cm')
                if height_cm and height_cm > 0:
                    transformed['height_m'] = height_cm / 100.0
                    logger.info(f"Converted height {height_cm}cm to {transformed['height_m']}m")
                    # Keep height_cm as well in case function needs it
        
        logger.debug(f"Transformed parameters for {func_name}: {transformed}")
        return transformed
    
    def _register_builtin_functions(self):
        """Register built-in utility functions"""
        builtin_functions = [
            {
                'name': 'calculate_age_months',
                'description': 'Calculate age in months from birth date',
                'code': '''
def calculate_age_months(birth_date, reference_date=None):
    """Calculate age in months between birth date and reference date"""
    from datetime import datetime
    
    if reference_date is None:
        reference_date = datetime.now()
    
    if isinstance(birth_date, str):
        try:
            birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
        except ValueError:
            try:
                birth_date = datetime.strptime(birth_date, '%m/%d/%Y')
            except ValueError:
                return None
    
    if isinstance(reference_date, str):
        try:
            reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        except ValueError:
            try:
                reference_date = datetime.strptime(reference_date, '%m/%d/%Y')
            except ValueError:
                return None
    
    # Calculate months difference
    months = (reference_date.year - birth_date.year) * 12 + (reference_date.month - birth_date.month)
    
    # Adjust for day of month
    if reference_date.day < birth_date.day:
        months -= 1
    
    return max(0, months)
                ''',
                'parameters': {
                    'birth_date': {
                        'type': 'string',
                        'description': 'Birth date in YYYY-MM-DD or MM/DD/YYYY format'
                    },
                    'reference_date': {
                        'type': 'string',
                        'description': 'Reference date (optional, defaults to today)'
                    }
                },
                'returns': 'Age in months (number)'
            },
            {
                'name': 'calculate_bmi',
                'description': 'Calculate Body Mass Index',
                'code': '''
def calculate_bmi(weight_kg, height_m=None, height_cm=None):
    """Calculate BMI from weight in kg and height in meters or centimeters"""
    if not weight_kg or weight_kg <= 0:
        return None
    
    # Prioritize height_m if both are provided
    if height_m and height_m > 0:
        pass  # Use height_m as is
    elif height_cm and height_cm > 0:
        # Convert height_cm to height_m
        height_m = height_cm / 100.0
    else:
        return None
    
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)
                ''',
                'parameters': {
                    'weight_kg': {
                        'type': 'number',
                        'description': 'Weight in kilograms'
                    },
                    'height_m': {
                        'type': 'number',
                        'description': 'Height in meters (optional if height_cm provided)'
                    },
                    'height_cm': {
                        'type': 'number',
                        'description': 'Height in centimeters (optional if height_m provided)'
                    }
                },
                'returns': 'BMI value (number)'
            }
        ]
        
        for func_def in builtin_functions:
            success, message = self.register_function(
                func_def['name'],
                func_def['code'],
                func_def['description'],
                func_def['parameters'],
                func_def['returns']
            )
            if success:
                logger.debug(f"Registered builtin function: {func_def['name']}")
            else:
                logger.warning(f"Failed to register builtin function {func_def['name']}: {message}")
    
    def list_functions(self) -> List[str]:
        """List all registered function names"""
        return list(self.functions.keys())
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get function metadata"""
        if name not in self.functions:
            return None
        
        func_info = self.functions[name].copy()
        func_info.pop('compiled', None)  # Don't include compiled function
        return func_info
    
    def get_all_functions_info(self) -> List[Dict[str, Any]]:
        """Get metadata for all functions"""
        return [self.get_function_info(name) for name in self.list_functions()]
    
    def remove_function(self, name: str) -> Tuple[bool, str]:
        """Remove a registered function"""
        if name not in self.functions:
            return False, f"Function '{name}' not found"
        
        try:
            # Don't allow removal of builtin functions
            builtin_names = ['calculate_age_months', 'calculate_bmi']
            if name in builtin_names:
                return False, f"Cannot remove builtin function '{name}'"
            
            # Remove from memory
            del self.functions[name]
            
            # Remove from disk
            func_file = self.storage_path / f"{name}.json"
            if func_file.exists():
                func_file.unlink()
            
            logger.info(f"Removed function: {name}")
            return True, f"Function '{name}' removed successfully"
            
        except Exception as e:
            error_msg = f"Failed to remove function '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code"""
        import re
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        return match.group(1) if match else None
    
    def _save_function(self, name: str):
        """Save function to disk"""
        func_file = self.storage_path / f"{name}.json"
        
        # Get function data without compiled function
        func_data = self.functions[name].copy()
        func_data.pop('compiled', None)
        
        try:
            with open(func_file, 'w', encoding='utf-8') as f:
                json.dump(func_data, f, indent=2)
            logger.debug(f"Saved function to {func_file}")
        except Exception as e:
            logger.error(f"Failed to save function {name}: {e}")
    
    def _load_all_functions(self):
        """Load all functions from storage directory"""
        if not self.storage_path.exists():
            return

        for func_file in self.storage_path.glob("*.json"):
            try:
                with open(func_file, 'r', encoding='utf-8') as f:
                    func_data = json.load(f)

                # Check if code is in JSON or separate .py file
                if 'code' in func_data:
                    # Old format: code embedded in JSON
                    code = func_data['code']
                else:
                    # New format: code in separate .py file
                    py_file = func_file.with_suffix('.py')
                    if py_file.exists():
                        with open(py_file, 'r', encoding='utf-8') as pf:
                            code = pf.read()
                    else:
                        logger.warning(f"No code found for {func_file} (no 'code' field and no .py file)")
                        continue

                # Register the function
                success, message = self.register_function(
                    func_data['name'],
                    code,
                    func_data.get('description', ''),
                    func_data.get('parameters', {}),
                    func_data.get('returns', {})
                )

                if not success:
                    logger.warning(f"Failed to load function from {func_file}: {message}")

            except Exception as e:
                logger.error(f"Failed to load function from {func_file}: {e}")
    
    def export_functions(self) -> str:
        """Export all functions as JSON string"""
        export_data = []
        for name, func in self.functions.items():
            func_copy = func.copy()
            func_copy.pop('compiled', None)
            export_data.append(func_copy)
        return json.dumps(export_data, indent=2)
    
    def import_functions(self, json_str: str) -> Tuple[bool, int, str]:
        """Import functions from JSON string"""
        try:
            functions = json.loads(json_str)
            count = 0
            errors = []
            
            for func in functions:
                success, message = self.register_function(
                    func['name'],
                    func['code'],
                    func['description'],
                    func['parameters'],
                    func['returns']
                )
                if success:
                    count += 1
                else:
                    errors.append(f"{func['name']}: {message}")
            
            if errors:
                error_summary = "\n".join(errors[:5])
                if len(errors) > 5:
                    error_summary += f"\n... and {len(errors) - 5} more errors"
                return True, count, f"Imported {count}/{len(functions)} functions. Errors:\n{error_summary}"
            
            return True, count, f"Successfully imported {count} functions"
            
        except json.JSONDecodeError as e:
            return False, 0, f"Invalid JSON: {str(e)}"
        except Exception as e:
            logger.error(f"Function import failed: {e}", exc_info=True)
            return False, 0, f"Import error: {str(e)}"
    
    def validate_function_code(self, code: str) -> Tuple[bool, str]:
        """Validate function code without registering"""
        try:
            # Try to compile the code
            compile(code, '<string>', 'exec')
            
            # Check if function name can be extracted
            func_name = self._extract_function_name(code)
            if not func_name:
                return False, "Could not extract function name. Ensure code starts with 'def function_name(...)'"
            
            return True, "Code is valid"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def clear_all_functions(self) -> Tuple[bool, str]:
        """Clear all registered functions except builtins"""
        try:
            builtin_names = ['calculate_age_months', 'calculate_bmi']
            
            # Get non-builtin functions
            functions_to_remove = [name for name in self.functions.keys() if name not in builtin_names]
            count = len(functions_to_remove)
            
            # Remove non-builtin functions
            for name in functions_to_remove:
                del self.functions[name]
            
            # Clear from disk
            for func_file in self.storage_path.glob("*.json"):
                func_name = func_file.stem
                if func_name not in builtin_names:
                    func_file.unlink()
            
            logger.info(f"Cleared {count} custom functions (kept {len(builtin_names)} builtins)")
            return True, f"Cleared {count} custom functions successfully"
            
        except Exception as e:
            error_msg = f"Failed to clear functions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def get_function_signature(self, name: str) -> Optional[str]:
        """Get function signature string"""
        if name not in self.functions:
            return None
        
        stored_sig = self.functions[name].get('signature')
        if stored_sig:
            return f"{name}{stored_sig}"
        
        func = self.functions[name]['compiled']
        try:
            sig = inspect.signature(func)
            return f"{name}{sig}"
        except Exception:
            return f"{name}(...)"
    
    def test_function(self, name: str, **test_args) -> Tuple[bool, Any, str]:
        """Test a function with provided arguments"""
        logger.info(f"Testing function '{name}' with args: {test_args}")
        return self.execute_function(name, **test_args)