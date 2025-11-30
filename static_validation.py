#!/usr/bin/env python3
"""
Static Code Analysis for Multi-Instance Task Isolation Architecture

Validates code structure, parameter flows, and potential errors without running imports.

Author: Frederick Gyasi (gyasi@musc.edu)
Version: 1.0.0
"""

import ast
import sys
from pathlib import Path

print("="*80)
print("Static Code Analysis - Multi-Instance Architecture")
print("="*80)

errors = []
warnings = []

# Test 1: Parse session_manager.py
print("\n[1/5] Analyzing core/session_manager.py...")
try:
    with open('core/session_manager.py', 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    print("  ✓ Syntax valid")

    # Check for required classes
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    required_classes = ['TaskContext', 'SessionState', 'SessionManager']
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ Class '{cls}' found")
        else:
            errors.append(f"Missing class '{cls}' in session_manager.py")

    # Check for required functions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    required_functions = ['get_session_manager', 'create_session', 'get_task_context', 'switch_task']
    for func in required_functions:
        if func in functions:
            print(f"  ✓ Function '{func}' found")
        else:
            errors.append(f"Missing function '{func}' in session_manager.py")

except SyntaxError as e:
    errors.append(f"Syntax error in session_manager.py: {e}")
except Exception as e:
    errors.append(f"Error analyzing session_manager.py: {e}")

# Test 2: Parse app_state_proxy.py
print("\n[2/5] Analyzing core/app_state_proxy.py...")
try:
    with open('core/app_state_proxy.py', 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    print("  ✓ Syntax valid")

    # Check for AppStateProxy class
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if 'AppStateProxy' in classes:
        print("  ✓ Class 'AppStateProxy' found")
    else:
        errors.append("Missing class 'AppStateProxy' in app_state_proxy.py")

    # Check for required magic methods
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'AppStateProxy':
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

    required_methods = ['__init__', '__getattribute__', '__setattr__', '_set_current_app_state']
    for method in required_methods:
        if method in methods:
            print(f"  ✓ Method '{method}' found")
        else:
            errors.append(f"Missing method '{method}' in AppStateProxy")

except SyntaxError as e:
    errors.append(f"Syntax error in app_state_proxy.py: {e}")
except Exception as e:
    errors.append(f"Error analyzing app_state_proxy.py: {e}")

# Test 3: Parse annotate.py
print("\n[3/5] Analyzing annotate.py...")
try:
    with open('annotate.py', 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    print("  ✓ Syntax valid")

    # Check for required imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.append(f"{node.module}.{alias.name}")

    required_imports = [
        'core.session_manager.get_session_manager',
        'core.app_state_proxy.AppStateProxy'
    ]
    for imp in required_imports:
        if imp in imports:
            print(f"  ✓ Import '{imp}' found")
        else:
            errors.append(f"Missing import '{imp}' in annotate.py")

    # Check for create_main_interface function
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if 'create_main_interface' in functions:
        print("  ✓ Function 'create_main_interface' found")
    else:
        errors.append("Missing function 'create_main_interface' in annotate.py")

except SyntaxError as e:
    errors.append(f"Syntax error in annotate.py: {e}")
except Exception as e:
    errors.append(f"Error analyzing annotate.py: {e}")

# Test 4: Check parameter flow in annotate.py
print("\n[4/5] Validating parameter flows in annotate.py...")
try:
    with open('annotate.py', 'r') as f:
        content = f.read()

    # Check for switch_task callback definition
    if 'def switch_task(task_name: str, sess_id: str, prev_task: str):' in content:
        print("  ✓ switch_task callback signature correct")
    else:
        errors.append("switch_task callback signature incorrect or missing")

    # Check for task_selector.change wiring
    if 'task_selector.change(' in content:
        print("  ✓ task_selector.change() wiring found")
    else:
        errors.append("task_selector.change() wiring missing")

    # Check for proxy creation
    if 'app_state_proxy = AppStateProxy(initial_app_state)' in content:
        print("  ✓ AppStateProxy instantiation found")
    else:
        errors.append("AppStateProxy instantiation missing or incorrect")

    # Check for proxy usage in tabs
    proxy_usages = [
        'create_config_tab(app_state_proxy)',
        'create_prompt_tab(app_state_proxy)',
        'create_data_tab(app_state_proxy)',
        'create_processing_tab(app_state_proxy)'
    ]
    for usage in proxy_usages:
        if usage in content:
            print(f"  ✓ Proxy passed to {usage.split('(')[0]}")
        else:
            warnings.append(f"Proxy might not be passed to {usage.split('(')[0]}")

    # Check for proxy update in switch_task
    if 'app_state_proxy._set_current_app_state(new_app_state)' in content:
        print("  ✓ Proxy update in switch_task found")
    else:
        errors.append("Proxy update in switch_task missing")

except Exception as e:
    errors.append(f"Error validating parameter flows: {e}")

# Test 5: Check for common runtime errors
print("\n[5/5] Checking for potential runtime errors...")
try:
    with open('annotate.py', 'r') as f:
        content = f.read()

    # Check for scope issues
    if 'session_manager = get_session_manager()' in content:
        print("  ✓ session_manager initialization found")
    else:
        errors.append("session_manager initialization missing")

    if 'session_id = session_manager.create_session()' in content:
        print("  ✓ session_id creation found")
    else:
        errors.append("session_id creation missing")

    # Check for closure access (functions accessing outer scope variables)
    if 'def get_status_for_task' in content and 'session_manager.get_task_context' in content:
        print("  ✓ get_status_for_task accesses session_manager (closure)")
    else:
        warnings.append("get_status_for_task might not access session_manager properly")

    if 'def switch_task' in content and 'session_manager.switch_task' in content:
        print("  ✓ switch_task accesses session_manager (closure)")
    else:
        errors.append("switch_task doesn't access session_manager")

    # Check return tuple consistency
    if 'return (' in content and 'task_name,' in content and 'new_status,' in content:
        print("  ✓ switch_task return tuple appears consistent")
    else:
        warnings.append("switch_task return tuple might be inconsistent")

except Exception as e:
    errors.append(f"Error checking for runtime errors: {e}")

# Summary
print("\n" + "="*80)
if errors:
    print("❌ VALIDATION FAILED")
    print("="*80)
    print(f"\n{len(errors)} ERROR(S) FOUND:\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print("✅ VALIDATION PASSED")
    print("="*80)
    print("\nNo errors detected!")

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):\n")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

print("\n" + "="*80)
print("Static Analysis Summary:")
print(f"  • Files analyzed: 3")
print(f"  • Errors found: {len(errors)}")
print(f"  • Warnings: {len(warnings)}")
print("="*80)

if errors:
    sys.exit(1)
else:
    print("\n✅ Multi-Instance Architecture code structure is valid!")
    print("   Safe to proceed with execution.")
    sys.exit(0)
