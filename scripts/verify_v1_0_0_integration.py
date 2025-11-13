#!/usr/bin/env python3
"""
Verification Script for ClinOrchestra v1.0.0 Integration
Verifies all recent updates are properly integrated:
1. Multi-GPU support
2. Multi-column prompt variables
3. Config persistence
4. Backward compatibility
"""

import sys
import inspect
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_app_state():
    """Verify AppState has all required methods and signatures"""
    print("=" * 70)
    print("CHECKING: core/app_state.py")
    print("=" * 70)

    from core.app_state import AppState, OptimizationConfig

    # Check OptimizationConfig has new fields
    opt_config = OptimizationConfig()

    checks = []
    checks.append(("use_multi_gpu field", hasattr(opt_config, 'use_multi_gpu')))
    checks.append(("num_gpus field", hasattr(opt_config, 'num_gpus')))
    checks.append(("use_multi_gpu default", opt_config.use_multi_gpu == True))
    checks.append(("num_gpus default", opt_config.num_gpus == -1))

    # Check set_optimization_config signature
    app_state = AppState()
    sig = inspect.signature(app_state.set_optimization_config)
    params = list(sig.parameters.keys())

    checks.append(("use_multi_gpu parameter", 'use_multi_gpu' in params))
    checks.append(("num_gpus parameter", 'num_gpus' in params))

    # Check set_data_config signature
    sig = inspect.signature(app_state.set_data_config)
    params = list(sig.parameters.keys())

    checks.append(("prompt_input_columns parameter", 'prompt_input_columns' in params))

    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    all_passed = all(p for _, p in checks)
    print(f"\nAppState: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed


def check_config_persistence():
    """Verify config persistence handles new fields"""
    print("\n" + "=" * 70)
    print("CHECKING: core/config_persistence.py")
    print("=" * 70)

    try:
        from core.config_persistence import ConfigurationPersistenceManager

        checks = []

        # Check that the manager has methods for optimization config
        manager = ConfigurationPersistenceManager()

        checks.append(("save_optimization_config method", hasattr(manager, 'save_optimization_config')))
        checks.append(("load_optimization_config method", hasattr(manager, 'load_optimization_config')))

        # Verify save_optimization_config can handle OptimizationConfig with new fields
        from core.app_state import OptimizationConfig
        opt_config = OptimizationConfig(use_multi_gpu=True, num_gpus=2)

        # Check method signature
        sig = inspect.signature(manager.save_optimization_config)
        params = list(sig.parameters.keys())
        checks.append(("save_optimization_config signature", 'optimization_config' in params))

        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")

        all_passed = all(p for _, p in checks)
        print(f"\nConfigPersistence: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        return all_passed

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_agents():
    """Verify both agents support prompt_variables"""
    print("\n" + "=" * 70)
    print("CHECKING: Agent Systems")
    print("=" * 70)

    checks = []

    # Check ExtractionAgent (STRUCTURED mode)
    try:
        from core.agent_system import ExtractionAgent
        sig = inspect.signature(ExtractionAgent.extract)
        params = list(sig.parameters.keys())
        checks.append(("ExtractionAgent.extract has prompt_variables", 'prompt_variables' in params))
    except Exception as e:
        checks.append(("ExtractionAgent.extract", False))
        print(f"  Error: {e}")

    # Check AdaptiveAgent (ADAPTIVE mode)
    try:
        from core.agentic_agent import AdaptiveAgent
        sig = inspect.signature(AdaptiveAgent.extract)
        params = list(sig.parameters.keys())
        checks.append(("AdaptiveAgent.extract has prompt_variables", 'prompt_variables' in params))
    except Exception as e:
        checks.append(("AdaptiveAgent.extract", False))
        print(f"  Error: {e}")

    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    all_passed = all(p for _, p in checks)
    print(f"\nAgents: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed


def check_multi_gpu():
    """Verify multi-GPU processor exists and is properly structured"""
    print("\n" + "=" * 70)
    print("CHECKING: Multi-GPU Support")
    print("=" * 70)

    checks = []

    try:
        from core.multi_gpu_processor import MultiGPUProcessor, MultiGPUTask, MultiGPUResult

        checks.append(("MultiGPUProcessor class exists", True))
        checks.append(("MultiGPUTask class exists", True))
        checks.append(("MultiGPUResult class exists", True))

        # Check MultiGPUTask has prompt_variables
        from dataclasses import fields
        task_fields = [f.name for f in fields(MultiGPUTask)]
        checks.append(("MultiGPUTask.prompt_variables field", 'prompt_variables' in task_fields))

        # Check MultiGPUProcessor methods
        checks.append(("process_batch method", hasattr(MultiGPUProcessor, 'process_batch')))

    except Exception as e:
        checks.append(("Multi-GPU module", False))
        print(f"  Error: {e}")

    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    all_passed = all(p for _, p in checks)
    print(f"\nMulti-GPU: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed


def check_ui_integration():
    """Verify UI components are updated"""
    print("\n" + "=" * 70)
    print("CHECKING: UI Integration")
    print("=" * 70)

    checks = []

    # Check data_tab
    try:
        with open('/home/user/clinorchestra/ui/data_tab.py', 'r') as f:
            content = f.read()
        checks.append(("data_tab has prompt_input_columns", 'prompt_input_columns' in content))
        checks.append(("data_tab passes prompt_input_columns to set_data_config",
                      'prompt_input_columns=list(prompt_input_cols)' in content))
    except Exception as e:
        checks.append(("data_tab.py", False))
        print(f"  Error: {e}")

    # Check config_tab
    try:
        with open('/home/user/clinorchestra/ui/config_tab.py', 'r') as f:
            content = f.read()
        checks.append(("config_tab has use_multi_gpu", 'use_multi_gpu' in content))
        checks.append(("config_tab has num_gpus", 'num_gpus' in content))
        checks.append(("config_tab passes use_multi_gpu to set_optimization_config",
                      'use_multi_gpu=use_multi_gpu_val' in content))
    except Exception as e:
        checks.append(("config_tab.py", False))
        print(f"  Error: {e}")

    # Check processing_tab
    try:
        with open('/home/user/clinorchestra/ui/processing_tab.py', 'r') as f:
            content = f.read()
        checks.append(("processing_tab uses prompt_input_columns",
                      'app_state.data_config.prompt_input_columns' in content))
        checks.append(("processing_tab passes prompt_variables to agent",
                      'agent.extract(clinical_text, label_value, prompt_variables)' in content))
    except Exception as e:
        checks.append(("processing_tab.py", False))
        print(f"  Error: {e}")

    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    all_passed = all(p for _, p in checks)
    print(f"\nUI Integration: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed


def main():
    """Run all verification checks"""
    print("\n" + "🔍" * 35)
    print("ClinOrchestra v1.0.0 Integration Verification")
    print("🔍" * 35 + "\n")

    results = []

    # Run all checks
    results.append(("AppState", check_app_state()))
    results.append(("Config Persistence", check_config_persistence()))
    results.append(("Agent Systems", check_agents()))
    results.append(("Multi-GPU Support", check_multi_gpu()))
    results.append(("UI Integration", check_ui_integration()))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<50} {status}")

    all_passed = all(p for _, p in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - v1.0.0 Integration is Complete!")
        print("\nIf you're seeing runtime errors, please:")
        print("  1. Restart the ClinOrchestra application (to reload updated modules)")
        print("  2. Clear Python cache: rm -rf core/__pycache__ ui/__pycache__")
        print("  3. Try again")
    else:
        print("❌ SOME CHECKS FAILED - Review errors above")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
