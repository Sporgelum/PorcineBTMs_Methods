#!/usr/bin/env python3
"""
PyTorch Compatibility Test — validate GPU setup and device functionality.
=========================================================================

This script checks:
  1. PyTorch + CUDA version compatibility
  2. GPU device detection and properties
  3. Tensor allocation and GPU compute on each device
  4. Auto-batch sizing function (gpu_utils)
  5. Basic MINE forward pass with autocast (mixed precision)

Run this before submitting full pipeline jobs to catch environment issues early.
"""

import sys
import torch
import traceback
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mine_network.gpu_utils import compute_optimal_batch_size, get_gpu_info
from mine_network.mine_estimator import estimate_mi_batch


def test_pytorch_version():
    """Check PyTorch + CUDA version."""
    print("\n" + "=" * 80)
    print("TEST 1: PyTorch & CUDA Version")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print("✓ PASS: PyTorch + CUDA functional")
        return True
    else:
        print("✗ FAIL: CUDA not available (this is OK if testing on CPU-only system)")
        return torch.cuda.is_available()


def test_gpu_detection():
    """Test GPU detection and properties."""
    print("\n" + "=" * 80)
    print("TEST 2: GPU Detection & Properties")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("✗ SKIP: No CUDA devices found")
        return True  # Don't fail on CPU-only systems
    
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    all_ok = True
    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            gpu_name = props.name
            total_vram_gb = props.total_memory / 1e9
            compute_cap = f"{props.major}.{props.minor}"
            print(f"  Device {i}: {gpu_name} | {total_vram_gb:.1f} GB | Compute {compute_cap}")
        except Exception as e:
            print(f"  Device {i}: ✗ FAILED - {e}")
            all_ok = False
    
    if all_ok:
        print("✓ PASS: All GPU devices detected")
    return all_ok


def test_gpu_allocation():
    """Test tensor allocation on each GPU."""
    print("\n" + "=" * 80)
    print("TEST 3: GPU Memory Allocation & Compute")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("✓ SKIP: No CUDA devices (CPU-only system)")
        return True
    
    all_ok = True
    for i in range(torch.cuda.device_count()):
        try:
            device = f"cuda:{i}"
            # Allocate small tensor
            x = torch.randn(10000, 10000, device=device, dtype=torch.float32)
            # Force GPU sync
            torch.cuda.synchronize(i)
            # Do a computation
            y = x.mean()
            torch.cuda.synchronize(i)
            print(f"  Device {i}: ✓ allocation & compute OK (memory used: ~380 MB)")
        except Exception as e:
            print(f"  Device {i}: ✗ FAILED - {e}")
            traceback.print_exc()
            all_ok = False
    
    if all_ok:
        print("✓ PASS: All devices support compute")
    return all_ok


def test_batch_sizing():
    """Test auto-batch sizing on each GPU."""
    print("\n" + "=" * 80)
    print("TEST 4: Auto-Batch Sizing (gpu_utils)")
    print("=" * 80)
    
    test_cases = [
        ("auto", "auto-detect first GPU"),
        ("cpu", "CPU fallback"),
        ("cuda:0", "first GPU explicit"),
    ]
    
    if torch.cuda.is_available():
        test_cases.append((f"cuda:{torch.cuda.device_count()-1}", "last GPU"))
    
    all_ok = True
    for device_str, desc in test_cases:
        try:
            batch_size = compute_optimal_batch_size(device_str, verbose=False)
            print(f"  {device_str:12s} ({desc:25s}): batch_size = {batch_size:6d}")
            # Sanity check: batch size should be between 256 and 16000
            if not (256 <= batch_size <= 16000):
                print(f"    ✗ WARNING: batch_size out of expected range [256, 16000]")
                all_ok = False
        except Exception as e:
            print(f"  {device_str:12s}: ✗ FAILED - {e}")
            all_ok = False
    
    if all_ok:
        print("✓ PASS: Auto-sizing works on all devices")
    return all_ok


def test_gpu_info():
    """Test get_gpu_info utility."""
    print("\n" + "=" * 80)
    print("TEST 5: GPU Info Query (get_gpu_info)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("✓ SKIP: No CUDA devices")
        return True
    
    all_ok = True
    for i in range(torch.cuda.device_count()):
        try:
            gpu_name, compute_cap, vram_gb = get_gpu_info(f"cuda:{i}")
            print(f"  Device {i}: {gpu_name:30s} | Compute {compute_cap} | {vram_gb:6.1f} GB")
        except Exception as e:
            print(f"  Device {i}: ✗ FAILED - {e}")
            all_ok = False
    
    if all_ok:
        print("✓ PASS: GPU info retrieval works")
    return all_ok


def test_mine_forward_pass():
    """Test basic MINE forward pass (no mixed precision)."""
    print("\n" + "=" * 80)
    print("TEST 6: MINE Forward Pass (float32)")
    print("=" * 80)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        # Small batch: 2 pairs, 50 samples
        gene_i = torch.randn(2, 50, device=device)
        gene_j = torch.randn(2, 50, device=device)
        
        # Very short training for speed
        mi_values, diags = estimate_mi_batch(
            gene_i, gene_j,
            hidden_dim=32,
            n_epochs=5,
            lr=1e-3,
            mixed_precision=False,
        )
        
        print(f"  Input: 2 pairs × 50 samples | Device: {device}")
        print(f"  Output MI: {mi_values} nats")
        print(f"  Loss curve (final 3 epochs): {diags['loss_curve'][-3:]}")
        print("✓ PASS: MINE forward pass OK")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        traceback.print_exc()
        return False


def test_mine_mixed_precision():
    """Test MINE with mixed precision (float16 forward)."""
    print("\n" + "=" * 80)
    print("TEST 7: MINE Forward Pass (float16 mixed precision)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("✓ SKIP: No CUDA (mixed precision only for GPU)")
        return True
    
    try:
        device = "cuda:0"
        # Small batch: 2 pairs, 50 samples
        gene_i = torch.randn(2, 50, device=device)
        gene_j = torch.randn(2, 50, device=device)
        
        # Very short training for speed
        mi_values, diags = estimate_mi_batch(
            gene_i, gene_j,
            hidden_dim=32,
            n_epochs=5,
            lr=1e-3,
            mixed_precision=True,  # Enable autocast
        )
        
        print(f"  Input: 2 pairs × 50 samples | Device: {device} | Mixed precision: ON")
        print(f"  Output MI: {mi_values} nats")
        print(f"  Loss curve (final 3 epochs): {diags['loss_curve'][-3:]}")
        print("✓ PASS: MINE mixed precision OK")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "PyTorch + GPU Environment Validation".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    tests = [
        ("PyTorch & CUDA Version", test_pytorch_version),
        ("GPU Detection", test_gpu_detection),
        ("GPU Memory Allocation", test_gpu_allocation),
        ("Auto-Batch Sizing", test_batch_sizing),
        ("GPU Info Query", test_gpu_info),
        ("MINE Forward (float32)", test_mine_forward_pass),
        ("MINE Forward (float16)", test_mine_mixed_precision),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    print("=" * 80)
    
    if all(results.values()):
        print("\n✓ All tests passed! Your environment is ready for MINE pipeline.")
        return 0
    else:
        print("\n✗ Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
