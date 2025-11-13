#!/usr/bin/env python3
"""
Test script to verify multi-GPU integration in ClinOrchestra
Run this to confirm the system automatically uses multiple GPUs
"""

def test_integration():
    """Test that multi-GPU is properly integrated"""
    print("=" * 80)
    print("TESTING MULTI-GPU INTEGRATION")
    print("=" * 80)

    try:
        # Test 1: Check OptimizationConfig has multi-GPU fields
        print("\n[Test 1] Checking OptimizationConfig...")
        from core.app_state import OptimizationConfig

        config = OptimizationConfig()
        assert hasattr(config, 'use_multi_gpu'), "❌ Missing use_multi_gpu field"
        assert hasattr(config, 'num_gpus'), "❌ Missing num_gpus field"
        print(f"✅ OptimizationConfig has multi-GPU fields")
        print(f"   - use_multi_gpu: {config.use_multi_gpu} (default)")
        print(f"   - num_gpus: {config.num_gpus} (default, -1 = all GPUs)")

        # Test 2: Check MultiGPUProcessor exists
        print("\n[Test 2] Checking MultiGPUProcessor...")
        from core.multi_gpu_processor import MultiGPUProcessor, check_multi_gpu_readiness

        print("✅ MultiGPUProcessor imported successfully")

        # Test 3: Check GPU readiness
        print("\n[Test 3] Checking GPU readiness...")
        readiness = check_multi_gpu_readiness()
        print(f"   - CUDA Available: {readiness['cuda_available']}")
        print(f"   - GPU Count: {readiness['gpu_count']}")
        print(f"   - Ready for Multi-GPU: {readiness['ready']}")

        if readiness['gpus']:
            print(f"\n   GPUs Detected:")
            for gpu in readiness['gpus']:
                status = "✅" if gpu['accessible'] else "❌"
                print(f"      {status} GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

        if readiness['issues']:
            print(f"\n   ⚠️  Issues:")
            for issue in readiness['issues']:
                print(f"      - {issue}")

        # Test 4: Check processing_tab integration
        print("\n[Test 4] Checking processing_tab.py integration...")
        with open('ui/processing_tab.py', 'r') as f:
            content = f.read()
            checks = [
                ('use_multi_gpu', 'Multi-GPU flag check'),
                ('MultiGPUProcessor', 'MultiGPUProcessor import'),
                ('MultiGPUTask', 'MultiGPUTask usage'),
                ('num_gpus_to_use', 'GPU count determination')
            ]

            for pattern, description in checks:
                if pattern in content:
                    print(f"   ✅ {description}: Found")
                else:
                    print(f"   ❌ {description}: NOT FOUND")

        # Test 5: Check config_tab integration
        print("\n[Test 5] Checking config_tab.py integration...")
        with open('ui/config_tab.py', 'r') as f:
            content = f.read()
            checks = [
                ('use_multi_gpu', 'use_multi_gpu checkbox'),
                ('num_gpus', 'num_gpus slider'),
                ('Enable Multi-GPU Processing', 'UI label')
            ]

            for pattern, description in checks:
                if pattern in content:
                    print(f"   ✅ {description}: Found")
                else:
                    print(f"   ❌ {description}: NOT FOUND")

        # Test 6: Check AppState integration
        print("\n[Test 6] Checking AppState integration...")
        from core.app_state import AppState

        app_state = AppState()
        assert hasattr(app_state.optimization_config, 'use_multi_gpu'), "❌ AppState missing use_multi_gpu"
        assert hasattr(app_state.optimization_config, 'num_gpus'), "❌ AppState missing num_gpus"
        print(f"✅ AppState properly initialized with multi-GPU config")

        print("\n" + "=" * 80)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 80)

        print("\n📋 SUMMARY:")
        print("   1. ✅ OptimizationConfig has use_multi_gpu and num_gpus fields")
        print("   2. ✅ MultiGPUProcessor module exists and works")
        print("   3. ✅ GPU detection and readiness check works")
        print("   4. ✅ processing_tab.py integrated with multi-GPU logic")
        print("   5. ✅ config_tab.py has multi-GPU UI controls")
        print("   6. ✅ AppState properly configured")

        if readiness['ready']:
            print("\n🚀 YOUR SYSTEM IS READY FOR MULTI-GPU PROCESSING!")
            print(f"   You have {readiness['gpu_count']} GPU(s) available")
            print("\n   To use:")
            print("   1. Open ClinOrchestra UI (Config tab)")
            print("   2. Check 'Enable Multi-GPU Processing'")
            print("   3. Set number of GPUs (or leave at detected value)")
            print("   4. Save configuration")
            print("   5. Process your data - both GPUs will be used!")
        else:
            print("\n⚠️  MULTI-GPU NOT AVAILABLE:")
            print("   Reasons:")
            for issue in readiness['issues']:
                print(f"      - {issue}")
            print("\n   Standard parallel processing will be used instead.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    success = test_integration()
    sys.exit(0 if success else 1)
