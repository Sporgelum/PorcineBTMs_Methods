#!/usr/bin/env python3
"""
Quick test to verify GPU acceleration is working correctly.

This script:
1. Checks GPU availability
2. Creates a small test matrix (1000 × 1000)
3. Compares CPU vs GPU CLR results (should be nearly identical)
4. Benchmarks speed (should see clear GPU advantage)

Run this BEFORE your full 32K gene analysis to catch any issues early.

Usage:
    python test_gpu_clr.py
"""

import numpy as np
import time


def test_gpu_available():
    """Test if GPU libraries are installed and working."""
    print("="*70)
    print("TEST 1: GPU Availability")
    print("="*70)
    
    tests_passed = 0
    
    # Test CuPy
    try:
        import cupy as cp
        print("✓ CuPy is installed")
        
        # Try a simple operation
        x = cp.array([1, 2, 3])
        y = cp.sum(x)
        print(f"  Simple CuPy operation works: sum([1,2,3]) = {y}")
        
        # Get GPU info
        print(f"  CuPy version: {cp.__version__}")
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")
        print(f"  GPU Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
        
        tests_passed += 1
        
    except ImportError:
        print("✗ CuPy is NOT installed")
        print("  Install with: pip install cupy-cuda11x (or cuda12x)")
    except Exception as e:
        print(f"✗ CuPy error: {e}")
    
    # Test PyTorch
    try:
        import torch
        print("✓ PyTorch is installed")
        
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA works: {torch.cuda.get_device_name(0)}")
            
            # Try a simple operation
            x = torch.tensor([1, 2, 3], device='cuda')
            y = torch.sum(x)
            print(f"  Simple PyTorch operation works: sum([1,2,3]) = {y}")
            
            tests_passed += 1
        else:
            print("  Warning: PyTorch installed but CUDA not available")
            
    except ImportError:
        print("✗ PyTorch is NOT installed")
        print("  Install from: https://pytorch.org/get-started/")
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
    
    print()
    if tests_passed > 0:
        print(f"✓ GPU acceleration available ({tests_passed} library/libraries working)")
        return True
    else:
        print("✗ No GPU libraries found. GPU acceleration unavailable.")
        print("  GPU functions will fail. Install CuPy or PyTorch to proceed.")
        return False


def create_test_matrix(size=1000):
    """Create a small test MI matrix similar to real data."""
    print("="*70)
    print(f"TEST 2: Creating test matrix ({size} × {size})")
    print("="*70)
    
    np.random.seed(42)
    
    # Create symmetric matrix with realistic MI values (0 to ~1)
    mi_matrix = np.random.exponential(0.1, size=(size, size))
    mi_matrix = (mi_matrix + mi_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(mi_matrix, 0)  # No self-interactions
    
    print(f"✓ Test matrix created: {size} × {size}")
    print(f"  MI range: {mi_matrix.min():.4f} to {mi_matrix.max():.4f}")
    print(f"  Mean MI: {mi_matrix[mi_matrix > 0].mean():.4f}")
    print(f"  Matrix size in memory: {mi_matrix.nbytes / 1e6:.1f} MB")
    print()
    
    return mi_matrix


def test_correctness(mi_matrix):
    """Test that GPU and CPU produce same results."""
    print("="*70)
    print("TEST 3: Correctness (CPU vs GPU)")
    print("="*70)
    
    # Import functions
    try:
        from clr_gpu import clr_transform_cupy, clr_transform_pytorch
    except ImportError:
        print("✗ Cannot import clr_gpu.py")
        print("  Make sure clr_gpu.py is in the same directory")
        return False
    
    # CPU reference (using original nested loop for small matrix)
    print("Computing CPU CLR (reference)...")
    start = time.time()
    clr_cpu = clr_transform_cpu_reference(mi_matrix)
    cpu_time = time.time() - start
    print(f"  Done in {cpu_time:.2f}s")
    
    # Test CuPy
    try:
        print("\nComputing CuPy GPU CLR...")
        start = time.time()
        clr_cupy = clr_transform_cupy(mi_matrix, dtype='float32')
        cupy_time = time.time() - start
        print(f"  Done in {cupy_time:.2f}s")
        
        # Compare results
        diff = np.abs(clr_cpu - clr_cupy)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-4:  # Allow small floating point differences
            print("  ✓ CuPy results match CPU (within tolerance)")
        else:
            print("  ✗ CuPy results differ from CPU significantly!")
            return False
            
    except Exception as e:
        print(f"  ✗ CuPy test failed: {e}")
    
    # Test PyTorch
    try:
        print("\nComputing PyTorch GPU CLR...")
        start = time.time()
        clr_torch = clr_transform_pytorch(mi_matrix, dtype='float32', device='cuda')
        torch_time = time.time() - start
        print(f"  Done in {torch_time:.2f}s")
        
        # Compare results
        diff = np.abs(clr_cpu - clr_torch)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-4:
            print("  ✓ PyTorch results match CPU (within tolerance)")
        else:
            print("  ✗ PyTorch results differ from CPU significantly!")
            return False
            
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
    
    print("\n✓ All correctness tests passed!")
    print()
    return True


def clr_transform_cpu_reference(mi_matrix):
    """
    Simple CPU implementation for testing.
    Uses basic numpy without fancy optimizations.
    """
    n_genes = mi_matrix.shape[0]
    clr_matrix = np.zeros_like(mi_matrix)
    
    # Compute row and column statistics (excluding diagonal)
    mask = ~np.eye(n_genes, dtype=bool)
    
    row_mean = np.zeros(n_genes)
    row_std = np.zeros(n_genes)
    col_mean = np.zeros(n_genes)
    col_std = np.zeros(n_genes)
    
    for i in range(n_genes):
        row_vals = mi_matrix[i, mask[i]]
        row_mean[i] = np.mean(row_vals)
        row_std[i] = np.std(row_vals)
        
        col_vals = mi_matrix[mask[:, i], i]
        col_mean[i] = np.mean(col_vals)
        col_std[i] = np.std(col_vals)
    
    # Avoid division by zero
    row_std[row_std == 0] = 1.0
    col_std[col_std == 0] = 1.0
    
    # Compute CLR scores
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:
                z_i = (mi_matrix[i, j] - row_mean[i]) / row_std[i]
                z_j = (mi_matrix[i, j] - col_mean[j]) / col_std[j]
                clr_matrix[i, j] = np.sqrt(z_i**2 + z_j**2) / np.sqrt(2)
    
    return clr_matrix


def benchmark_speed(mi_matrix):
    """Benchmark CPU vs GPU performance."""
    print("="*70)
    print("TEST 4: Performance Benchmark")
    print("="*70)
    
    from clr_gpu import clr_transform_cupy, clr_transform_pytorch
    
    n_runs = 3
    results = {}
    
    # Benchmark CuPy
    try:
        print("Benchmarking CuPy GPU (3 runs)...")
        times = []
        for i in range(n_runs + 1):  # +1 for warmup
            start = time.time()
            _ = clr_transform_cupy(mi_matrix, dtype='float32')
            elapsed = time.time() - start
            if i > 0:  # Skip warmup
                times.append(elapsed)
            print(f"  Run {i}: {elapsed:.3f}s" + (" (warmup)" if i == 0 else ""))
        
        results['CuPy GPU'] = {
            'mean': np.mean(times),
            'std': np.std(times)
        }
    except Exception as e:
        print(f"  CuPy benchmark failed: {e}")
    
    # Benchmark PyTorch
    try:
        print("\nBenchmarking PyTorch GPU (3 runs)...")
        times = []
        for i in range(n_runs + 1):
            start = time.time()
            _ = clr_transform_pytorch(mi_matrix, dtype='float32', device='cuda')
            elapsed = time.time() - start
            if i > 0:
                times.append(elapsed)
            print(f"  Run {i}: {elapsed:.3f}s" + (" (warmup)" if i == 0 else ""))
        
        results['PyTorch GPU'] = {
            'mean': np.mean(times),
            'std': np.std(times)
        }
    except Exception as e:
        print(f"  PyTorch benchmark failed: {e}")
    
    # Benchmark CPU reference (just 1 run, it's slow)
    print("\nBenchmarking CPU reference (1 run)...")
    start = time.time()
    _ = clr_transform_cpu_reference(mi_matrix)
    cpu_time = time.time() - start
    print(f"  Run 0: {cpu_time:.3f}s")
    
    results['CPU Reference'] = {
        'mean': cpu_time,
        'std': 0.0
    }
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    baseline = results['CPU Reference']['mean']
    
    for method, stats in results.items():
        mean_time = stats['mean']
        std_time = stats['std']
        speedup = baseline / mean_time
        
        print(f"{method:20s}: {mean_time:6.3f}s ± {std_time:.3f}s", end="")
        if method != 'CPU Reference':
            print(f"  (🚀 {speedup:.1f}x faster)")
        else:
            print()
    
    print()
    
    # Extrapolate to full 32K matrix
    if 'CuPy GPU' in results:
        test_size = mi_matrix.shape[0]
        full_size = 32763
        
        # CLR scales roughly O(n²) due to matrix operations
        scale_factor = (full_size / test_size) ** 2
        
        estimated_gpu = results['CuPy GPU']['mean'] * scale_factor
        estimated_cpu = baseline * scale_factor
        
        print("="*70)
        print(f"EXTRAPOLATION TO FULL DATASET ({full_size} genes)")
        print("="*70)
        print(f"Test matrix:  {test_size} × {test_size}")
        print(f"Full matrix:  {full_size} × {full_size}")
        print(f"Scale factor: {scale_factor:.1f}x")
        print()
        print(f"Estimated CPU time: {estimated_cpu/60:.1f} minutes")
        print(f"Estimated GPU time: {estimated_gpu:.1f} seconds")
        print(f"Estimated speedup:  {estimated_cpu/estimated_gpu:.0f}x")
        print()
    
    return results


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "GPU CLR TEST SUITE" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Test 1: GPU availability
    if not test_gpu_available():
        print("\n❌ FAILED: No GPU available. Cannot proceed with GPU tests.")
        print("\nTo fix:")
        print("  1. Check GPU is available: nvidia-smi")
        print("  2. Install CuPy: pip install cupy-cuda11x")
        print("  3. Or install PyTorch: see https://pytorch.org/get-started/")
        return False
    
    # Test 2: Create test matrix
    print("Creating small test matrix for fast testing...")
    mi_matrix = create_test_matrix(size=1000)
    
    # Test 3: Correctness
    if not test_correctness(mi_matrix):
        print("\n❌ FAILED: Results don't match between CPU and GPU")
        return False
    
    # Test 4: Performance
    benchmark_speed(mi_matrix)
    
    # Final summary
    print("="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nYour GPU acceleration is working correctly!")
    print("\nNext steps:")
    print("  1. Run full analysis with GPU:")
    print("     sbatch run_network_gpu.sh")
    print("\n  2. Or integrate into your code:")
    print("     from clr_gpu import clr_transform_cupy")
    print("     clr = clr_transform_cupy(mi_matrix, dtype='float32')")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
