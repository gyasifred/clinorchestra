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

# Global registry instance for call_function helper to use
_global_registry: Optional['FunctionRegistry'] = None

def set_global_registry(registry: 'FunctionRegistry') -> None:
    """Set the global function registry instance for call_function helper"""
    global _global_registry
    _global_registry = registry
    logger.debug("[REGISTRY] Global function registry set")

def get_global_registry() -> Optional['FunctionRegistry']:
    """Get the global function registry instance"""
    return _global_registry

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

        # RECURSIVE CALLS: Use thread-local storage for call tracking
        # This prevents race conditions when functions execute in parallel (ThreadPoolExecutor)
        import threading
        self._thread_local = threading.local()
        self.max_call_depth = 10  # Maximum nested function calls

        self._load_all_functions()
        self._register_builtin_functions()

        # CRITICAL: Set this instance as the global registry for call_function helper
        set_global_registry(self)

        logger.info(f"FunctionRegistry initialized with {len(self.functions)} functions")

    def _get_call_depth(self) -> int:
        """Get call depth for current thread"""
        if not hasattr(self._thread_local, 'call_depth'):
            self._thread_local.call_depth = 0
        return self._thread_local.call_depth

    def _set_call_depth(self, value: int) -> None:
        """Set call depth for current thread"""
        self._thread_local.call_depth = value

    def _get_call_stack(self) -> List[str]:
        """Get call stack for current thread"""
        if not hasattr(self._thread_local, 'call_stack'):
            self._thread_local.call_stack = []
        return self._thread_local.call_stack
    
    def _create_namespace(self) -> Dict[str, Any]:
        """Create execution namespace with comprehensive support for growth calculators"""
        # Ensure project root is in sys.path for 'from core.' imports
        import os
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Build namespace with error handling for optional modules
        namespace = {
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
            'datetime': __import__('datetime'),
            'dataclasses': __import__('dataclasses'),
            'enum': __import__('enum'),
        }

        # Add optional modules with error handling
        optional_modules = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scipy': 'scipy',
            'dateutil': 'dateutil',
        }

        for key, module_name in optional_modules.items():
            try:
                namespace[key] = __import__(module_name)
            except ImportError:
                logger.warning(f"Optional module '{module_name}' not available in function namespace")

        return namespace

    def _create_function_execution_namespace(self) -> Dict[str, Any]:
        """
        Create execution namespace for a function with access to other registered functions.

        This allows functions to call other registered functions, enabling composition.
        For example:
            def calculate_bmi_for_age(weight_kg, height_m, birth_date):
                # Can call other registered functions
                age_months = call_function('calculate_age_months', birth_date=birth_date)
                bmi = call_function('calculate_bmi', weight_kg=weight_kg, height_m=height_m)
                return {'age_months': age_months, 'bmi': bmi}
        """
        def call_function(func_name: str, **func_kwargs):
            """
            Helper function that allows registered functions to call other registered functions.

            CRITICAL FIX: Uses global registry to ensure consistent call_stack tracking
            across STRUCTURED and ADAPTIVE workflows.

            Args:
                func_name: Name of the function to call
                **func_kwargs: Arguments to pass to the function

            Returns:
                The result of the function call

            Raises:
                ValueError: If function not found or execution fails
            """
            # Use global registry to ensure same instance across all function calls
            registry = get_global_registry()
            if not registry:
                raise RuntimeError("Global function registry not set. Cannot call functions.")

            success, result, message = registry.execute_function(func_name, **func_kwargs)
            if not success:
                raise ValueError(f"Function '{func_name}' failed: {message}")
            return result

        # Return namespace with helper function
        return {
            'call_function': call_function,
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
                'enabled': True,  # New: functions are enabled by default
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

    def update_function(self, func_id: str, name: str, code: str, description: str,
                       parameters: Dict[str, Any], returns: str) -> Tuple[bool, str]:
        """
        Update an existing function (re-register with new code/metadata)

        Args:
            func_id: Function ID (currently same as name)
            name: Function name
            code: Function code
            description: Function description
            parameters: Function parameters spec
            returns: Return value description

        Returns:
            Tuple of (success, message)
        """
        # For now, func_id is the same as name (functions are keyed by name)
        # Simply re-register the function with updated details
        return self.register_function(name, code, description, parameters, returns)

    def execute_function(self, name: str, **kwargs) -> Tuple[bool, Any, str]:
        """
        Execute a registered function with parameter validation and conversion
        FIXED: Apply transformations BEFORE validation
        PERFORMANCE: Cache results to avoid redundant calculations
        RECURSIVE CALLS: Functions can call other registered functions
        THREAD-SAFE: Uses thread-local storage for call tracking (parallel execution safe)
        """
        if name not in self.functions:
            return False, None, f"Function '{name}' not found"

        # Check if function is enabled
        if not self.functions[name].get('enabled', True):
            return False, None, f"Function '{name}' is disabled"

        # Get thread-local state
        call_depth = self._get_call_depth()
        call_stack = self._get_call_stack()

        # RECURSIVE CALLS: Check call depth to prevent infinite recursion
        if call_depth >= self.max_call_depth:
            error_msg = f"Maximum call depth ({self.max_call_depth}) exceeded. Call stack: {' -> '.join(call_stack)}"
            logger.error(error_msg)
            return False, None, error_msg

        # RECURSIVE CALLS: Check for DIRECT recursion (function calling itself)
        # Only flag as circular if the IMMEDIATE parent is the same function
        # This allows calling the same function multiple times with different parameters
        if call_stack and call_stack[-1] == name:
            error_msg = f"Direct recursion detected: {name} calling itself"
            logger.error(f"[RECURSION] {error_msg}")
            logger.error(f"[RECURSION] Full call stack: {' -> '.join(call_stack)} -> {name}")
            logger.error(f"[RECURSION] Current depth: {call_depth}")
            logger.error(f"[RECURSION] Parameters: {json.dumps(kwargs, indent=2)}")

            # SAFEGUARD: If call_depth is 0 but stack is not empty, clear stale entries
            if call_depth == 0 and call_stack:
                logger.warning(f"[RECURSION] Detected stale call_stack (depth=0 but stack={call_stack}). Clearing stack.")
                call_stack.clear()
                # After clearing, allow this call to proceed
                logger.info(f"[RECURSION] Stack cleared. Allowing {name} to execute.")
            else:
                return False, None, error_msg

        # PERFORMANCE: Create cache key from function name and sorted parameters
        try:
            # Sort kwargs for consistent cache keys
            cache_key = f"{name}:{json.dumps(kwargs, sort_keys=True)}"

            # Check cache first (skip for recursive calls to track proper depth)
            if cache_key in self.result_cache and call_depth == 0:
                self.cache_hits += 1
                cached_result = self.result_cache[cache_key]
                logger.debug(f"Cache HIT for {name}({kwargs}) - returning cached result")
                return cached_result

            if call_depth == 0:
                self.cache_misses += 1

        except (TypeError, ValueError) as e:
            # If kwargs not JSON-serializable, skip caching
            logger.debug(f"Could not create cache key for {name}: {e}")
            cache_key = None

        # RECURSIVE CALLS: Track call depth and stack (thread-local)
        self._set_call_depth(call_depth + 1)
        call_stack.append(name)
        indent = "  " * call_depth

        try:
            func = self.functions[name]['compiled']
            func_info = self.functions[name]

            logger.debug(f"{indent}-> Entering {name}() [depth={call_depth + 1}]")

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

            # RECURSIVE CALLS: Create enhanced namespace with access to other functions
            enhanced_namespace = self._create_function_execution_namespace()

            # CRITICAL FIX: Re-exec code to ensure helper functions are available
            # Many functions have helper functions defined in the same code block
            # Example: interpret_zscore_malnutrition calls zscore_to_percentile
            # These helpers are only available if we re-exec the code
            exec_globals = {**self.namespace, **enhanced_namespace}

            try:
                # Execute the function code in the enhanced namespace
                exec(self.functions[name]['code'], exec_globals)

                # Extract the main function
                func_name_from_code = self._extract_function_name(self.functions[name]['code'])
                if func_name_from_code and func_name_from_code in exec_globals:
                    result = exec_globals[func_name_from_code](**validated_kwargs)
                else:
                    # Fallback to compiled function if extraction fails
                    logger.warning(f"Could not extract {func_name_from_code} from namespace, using compiled function")
                    result = func(**validated_kwargs)
            except Exception as e:
                # If re-exec fails, try the compiled function as last resort
                logger.warning(f"Re-exec failed for {name}, trying compiled function: {e}")
                result = func(**validated_kwargs)

            execution_result = (True, result, "Execution successful")

            logger.debug(f"{indent}<- Exiting {name}() [depth={call_depth + 1}] = {result}")

            # PERFORMANCE: Cache the result if cache_key was created (only for top-level calls)
            if cache_key is not None and call_depth + 1 == 1:
                self.result_cache[cache_key] = execution_result
                logger.debug(f"Cached result for {name}({kwargs})")

            logger.info(f"Function '{name}' executed successfully")
            return execution_result

        except TypeError as e:
            error_msg = f"Invalid arguments for function '{name}': {str(e)}"
            logger.error(f"{indent}[FAIL] {error_msg}")
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Function '{name}' execution failed: {str(e)}"
            logger.error(f"{indent}[FAIL] {error_msg}", exc_info=True)
            return False, None, error_msg
        finally:
            # RECURSIVE CALLS: Always cleanup call depth and stack (thread-local)
            self._set_call_depth(call_depth)  # Restore original depth
            if call_stack and call_stack[-1] == name:
                call_stack.pop()
    
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
        """Get metadata for all enabled functions"""
        # Filter to only return enabled functions for LLM tool selection
        enabled_functions = [name for name in self.list_functions()
                           if self.functions[name].get('enabled', True)]
        return [self.get_function_info(name) for name in enabled_functions]
    
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

    def enable_function(self, name: str, enabled: bool = True) -> Tuple[bool, str]:
        """Enable or disable a function"""
        if name not in self.functions:
            return False, f"Function '{name}' not found"

        try:
            self.functions[name]['enabled'] = enabled
            self._save_function(name)

            status = "enabled" if enabled else "disabled"
            logger.info(f"Function '{name}' {status}")
            return True, f"Function '{name}' {status} successfully"

        except Exception as e:
            error_msg = f"Failed to enable/disable function '{name}': {str(e)}"
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

                # Backward compatibility: restore enabled field if it was in stored data
                if success and 'enabled' in func_data:
                    self.functions[func_data['name']]['enabled'] = func_data['enabled']
                # If not present in stored data, it will default to True (set in register_function)

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