"""
GPU-Accelerated CLR Transformation for Gene Networks

This module provides GPU implementations of the Context Likelihood of Relatedness (CLR)
transformation using CuPy and PyTorch. These can provide 10-50x speedup over CPU 
vectorized implementations.

Author: Educational implementation for gene network analysis
Date: March 2026
"""

import numpy as np
import warnings


def clr_transform_cupy(mi_matrix, dtype='float32'):
    """
    GPU-accelerated CLR transformation using CuPy (NumPy-compatible GPU arrays).
    
    This is the most straightforward GPU implementation - nearly identical to the
    vectorized CPU version, but runs on GPU.
    
    Parameters:
    -----------
    mi_matrix : numpy.ndarray
        Mutual information matrix (n_genes × n_genes)
    dtype : str
        Data type for GPU arrays ('float32' saves memory, 'float64' more precise)
        
    Returns:
    --------
    clr_matrix : numpy.ndarray
        CLR-transformed matrix on CPU (same shape as input)
        
    Performance:
    ------------
    - Expected: 10-50x faster than CPU vectorized (depends on GPU model)
    - Memory: Requires ~4-8 GB GPU RAM for 32,763 genes
    - Best for: Large matrices (>10,000 genes)
    
    Example:
    --------
    >>> mi = compute_mi_matrix_parallel(expr_discrete, gene_names, n_jobs=24)
    >>> clr = clr_transform_cupy(mi, dtype='float32')
    >>> print(f"CLR range: {clr.min():.4f} to {clr.max():.4f}")
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy not installed. Install with:\n"
            "  pip install cupy-cuda11x  # for CUDA 11.x\n"
            "  pip install cupy-cuda12x  # for CUDA 12.x\n"
            "Check your CUDA version with: nvcc --version"
        )
    
    print("Transferring MI matrix to GPU...")
    n_genes = mi_matrix.shape[0]
    
    # Transfer to GPU with specified precision
    mi_gpu = cp.array(mi_matrix, dtype=dtype)
    
    print("Computing statistics on GPU...")
    # Mask diagonal (self-interactions)
    mask = cp.eye(n_genes, dtype=bool)
    mi_masked = cp.where(mask, cp.nan, mi_gpu)
    
    # Compute row and column statistics (ignoring diagonal)
    row_mean = cp.nanmean(mi_masked, axis=1)
    row_std = cp.nanstd(mi_masked, axis=1)
    col_mean = cp.nanmean(mi_masked, axis=0)
    col_std = cp.nanstd(mi_masked, axis=0)
    
    # Handle zero standard deviations
    row_std = cp.where(row_std == 0, 1.0, row_std)
    col_std = cp.where(col_std == 0, 1.0, col_std)
    
    print("Computing z-scores on GPU...")
    # Compute z-scores using broadcasting
    # z_i = (MI - row_mean) / row_std for each row
    # z_j = (MI - col_mean) / col_std for each column
    z_rows = (mi_gpu - row_mean[:, cp.newaxis]) / row_std[:, cp.newaxis]
    z_cols = (mi_gpu - col_mean[cp.newaxis, :]) / col_std[cp.newaxis, :]
    
    print("Computing CLR scores on GPU...")
    # CLR formula: sqrt(z_i^2 + z_j^2) / sqrt(2)
    clr_gpu = cp.sqrt(z_rows**2 + z_cols**2) / cp.sqrt(2.0)
    
    # Set diagonal to zero (no self-regulation)
    clr_gpu[mask] = 0.0
    
    # Handle any NaN or Inf values
    clr_gpu = cp.nan_to_num(clr_gpu, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("Transferring CLR matrix back to CPU...")
    # Transfer back to CPU
    clr_matrix = cp.asnumpy(clr_gpu)
    
    # Clean up GPU memory
    del mi_gpu, mi_masked, z_rows, z_cols, clr_gpu, mask
    cp.get_default_memory_pool().free_all_blocks()
    
    return clr_matrix


def clr_transform_pytorch(mi_matrix, dtype='float32', device='cuda'):
    """
    GPU-accelerated CLR transformation using PyTorch.
    
    PyTorch is more commonly used in bioinformatics (for deep learning), so you
    might already have it installed. It's also more flexible for custom operations.
    
    Parameters:
    -----------
    mi_matrix : numpy.ndarray
        Mutual information matrix (n_genes × n_genes)
    dtype : str
        Data type ('float32' or 'float64')
    device : str
        Device to use ('cuda', 'cuda:0', 'cuda:1', etc., or 'cpu' for testing)
        
    Returns:
    --------
    clr_matrix : numpy.ndarray
        CLR-transformed matrix on CPU
        
    Performance:
    ------------
    - Similar to CuPy (10-50x faster than CPU vectorized)
    - Slightly more overhead due to autograd system
    - Better integration if you're doing deep learning
    
    Example:
    --------
    >>> mi = compute_mi_matrix_parallel(expr_discrete, gene_names, n_jobs=24)
    >>> clr = clr_transform_pytorch(mi, dtype='float32', device='cuda:0')
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install with:\n"
            "  pip install torch  # CPU version\n"
            "  # For GPU, see: https://pytorch.org/get-started/locally/\n"
        )
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Transferring MI matrix to {device.upper()}...")
    n_genes = mi_matrix.shape[0]
    
    # Convert dtype string to torch dtype
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    
    # Transfer to GPU with no gradient tracking (saves memory)
    with torch.no_grad():
        mi_gpu = torch.from_numpy(mi_matrix).to(device=device, dtype=torch_dtype)
        
        print("Computing statistics on GPU...")
        # Create mask for diagonal
        mask = torch.eye(n_genes, device=device, dtype=torch.bool)
        mi_masked = mi_gpu.clone()
        mi_masked[mask] = float('nan')
        
        # Compute row and column statistics (excluding diagonal)
        # Use nanmean and nanstd-equivalent (variance with nan handling)
        row_mean = torch.nanmean(mi_masked, dim=1, keepdim=True)
        
        # Compute row variance manually (excluding NaNs/diagonal)
        row_var = torch.nanmean((mi_masked - row_mean)**2, dim=1, keepdim=True)
        row_std = torch.sqrt(row_var)
        
        col_mean = torch.nanmean(mi_masked, dim=0, keepdim=True)
        
        # Compute col variance manually (excluding NaNs/diagonal)
        col_var = torch.nanmean((mi_masked - col_mean)**2, dim=0, keepdim=True)
        col_std = torch.sqrt(col_var)
        
        # Handle zero standard deviations
        row_std = torch.where(row_std == 0, torch.ones_like(row_std), row_std)
        col_std = torch.where(col_std == 0, torch.ones_like(col_std), col_std)
        
        print("Computing z-scores on GPU...")
        # Compute z-scores using broadcasting
        # row_mean and row_std are (n_genes, 1), will broadcast across columns
        z_rows = (mi_gpu - row_mean) / row_std
        # col_mean and col_std are (1, n_genes), will broadcast across rows
        z_cols = (mi_gpu - col_mean) / col_std
        
        print("Computing CLR scores on GPU...")
        # CLR formula
        clr_gpu = torch.sqrt(z_rows**2 + z_cols**2) / torch.sqrt(torch.tensor(2.0, device=device))
        
        # Set diagonal to zero
        clr_gpu[mask] = 0.0
        
        # Handle NaN/Inf
        clr_gpu = torch.nan_to_num(clr_gpu, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("Transferring CLR matrix back to CPU...")
        # Transfer back to CPU
        clr_matrix = clr_gpu.cpu().numpy()
    
    # Clean up GPU memory
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    return clr_matrix


def get_gpu_info():
    """
    Check GPU availability and print useful information.
    
    Returns:
    --------
    dict : GPU information or None if no GPU available
    """
    gpu_info = {
        'cupy_available': False,
        'pytorch_available': False,
        'cuda_available': False,
        'gpu_devices': []
    }
    
    # Check CuPy
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['cuda_available'] = True
        
        # Get device info
        n_devices = cp.cuda.runtime.getDeviceCount()
        for i in range(n_devices):
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            gpu_info['gpu_devices'].append({
                'id': i,
                'name': props['name'].decode(),
                'memory_gb': props['totalGlobalMem'] / 1e9,
                'compute_capability': f"{props['major']}.{props['minor']}"
            })
    except ImportError:
        pass
    except Exception as e:
        print(f"CuPy error: {e}")
    
    # Check PyTorch
    try:
        import torch
        gpu_info['pytorch_available'] = True
        
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            n_devices = torch.cuda.device_count()
            
            # Only add if not already added by CuPy
            if not gpu_info['gpu_devices']:
                for i in range(n_devices):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['gpu_devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / 1e9,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
    except ImportError:
        pass
    except Exception as e:
        print(f"PyTorch error: {e}")
    
    return gpu_info


def print_gpu_info():
    """Print formatted GPU information."""
    info = get_gpu_info()
    
    print("\n" + "="*60)
    print("GPU ACCELERATION STATUS")
    print("="*60)
    
    print(f"CuPy available: {info['cupy_available']}")
    print(f"PyTorch available: {info['pytorch_available']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['gpu_devices']:
        print(f"\nDetected {len(info['gpu_devices'])} GPU(s):")
        for gpu in info['gpu_devices']:
            print(f"  [{gpu['id']}] {gpu['name']}")
            print(f"      Memory: {gpu['memory_gb']:.1f} GB")
            print(f"      Compute Capability: {gpu['compute_capability']}")
    else:
        print("\nNo GPUs detected. GPU acceleration unavailable.")
        print("\nTo use GPU acceleration:")
        print("  1. Ensure NVIDIA GPU with CUDA support is available")
        print("  2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  3. Install CuPy: pip install cupy-cuda11x (or cuda12x)")
        print("     OR PyTorch with CUDA: see https://pytorch.org/get-started/")
    
    print("="*60 + "\n")
    
    return info


def benchmark_clr_methods(mi_matrix, n_runs=3):
    """
    Benchmark different CLR implementations on your data.
    
    This will help you understand the actual speedup on your specific GPU.
    
    Parameters:
    -----------
    mi_matrix : numpy.ndarray
        Your MI matrix
    n_runs : int
        Number of runs to average (first run is warmup)
    """
    import time
    
    results = {}
    
    print("\n" + "="*60)
    print("CLR PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Matrix size: {mi_matrix.shape[0]} × {mi_matrix.shape[1]}")
    print(f"Runs per method: {n_runs}")
    print("="*60 + "\n")
    
    # Test CPU vectorized (if available)
    print("Testing CPU vectorized...")
    try:
        from generate_net_python import clr_transform
        times = []
        for i in range(n_runs + 1):
            start = time.time()
            clr = clr_transform(mi_matrix)
            elapsed = time.time() - start
            if i > 0:  # Skip warmup
                times.append(elapsed)
            print(f"  Run {i}: {elapsed:.2f}s" + (" (warmup)" if i == 0 else ""))
        results['CPU (original)'] = {'mean': np.mean(times), 'std': np.std(times)}
    except Exception as e:
        print(f"  CPU test failed: {e}")
    
    # Test CuPy
    print("\nTesting CuPy GPU...")
    try:
        times = []
        for i in range(n_runs + 1):
            start = time.time()
            clr = clr_transform_cupy(mi_matrix, dtype='float32')
            elapsed = time.time() - start
            if i > 0:
                times.append(elapsed)
            print(f"  Run {i}: {elapsed:.2f}s" + (" (warmup)" if i == 0 else ""))
        results['CuPy GPU'] = {'mean': np.mean(times), 'std': np.std(times)}
    except Exception as e:
        print(f"  CuPy test failed: {e}")
    
    # Test PyTorch
    print("\nTesting PyTorch GPU...")
    try:
        times = []
        for i in range(n_runs + 1):
            start = time.time()
            clr = clr_transform_pytorch(mi_matrix, dtype='float32', device='cuda')
            elapsed = time.time() - start
            if i > 0:
                times.append(elapsed)
            print(f"  Run {i}: {elapsed:.2f}s" + (" (warmup)" if i == 0 else ""))
        results['PyTorch GPU'] = {'mean': np.mean(times), 'std': np.std(times)}
    except Exception as e:
        print(f"  PyTorch test failed: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results:
        baseline = results.get('CPU (original)', {}).get('mean', None)
        
        for method, stats in results.items():
            mean_time = stats['mean']
            std_time = stats['std']
            speedup = baseline / mean_time if baseline else 1.0
            
            print(f"{method:20s}: {mean_time:6.2f}s ± {std_time:.2f}s", end="")
            if baseline and method != 'CPU (original)':
                print(f"  (🚀 {speedup:.1f}x speedup)")
            else:
                print()
    else:
        print("No successful runs.")
    
    print("="*60 + "\n")
    
    return results


# Educational example of how GPU parallelism works
def explain_gpu_advantage():
    """
    Print an educational explanation of why GPUs are faster for CLR.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              WHY GPU ACCELERATION WORKS FOR CLR              ║
╚══════════════════════════════════════════════════════════════╝

CLR Transformation: sqrt(z_i² + z_j²) / sqrt(2)

For 32,763 genes, this means:
  • 32,763² = 1,073,251,369 score calculations
  • Each calculation: 4-5 floating point operations
  • Total: ~5 billion operations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CPU (24 cores):
  • 24 cores doing sequential calculations
  • ~45 million calculations per core
  • Bottleneck: Limited parallelism

GPU (e.g., NVIDIA A100):
  • 6,912 CUDA cores working simultaneously
  • Specialized tensor cores for matrix operations
  • Massive memory bandwidth (1.6 TB/s vs ~100 GB/s CPU)
  • Thousands of calculations at once

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expected Performance:
  • CPU vectorized:      ~3 minutes (baseline)
  • CuPy on V100:        ~10-15 seconds (12-18x faster)
  • CuPy on A100:        ~5-8 seconds (22-36x faster)
  • CuPy on RTX 3090:    ~8-12 seconds (15-22x faster)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Insight:
  Matrix operations like CLR are "embarrassingly parallel" - 
  millions of independent calculations that can happen simultaneously.
  This is exactly what GPUs are designed for!

╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    """
    Run this script directly to check GPU availability and learn about GPU CLR.
    
    Usage:
        python clr_gpu.py
    """
    print("\n🚀 GPU-Accelerated CLR Transformation for Gene Networks\n")
    
    # Show GPU info
    print_gpu_info()
    
    # Explain why GPU helps
    explain_gpu_advantage()
    
    # Offer to run benchmark if data available
    print("\nTo benchmark on your data:")
    print("  from clr_gpu import benchmark_clr_methods")
    print("  from generate_net_python import load_data, compute_mi_matrix_parallel")
    print("  expr, expr_discrete, genes = load_data()")
    print("  mi = compute_mi_matrix_parallel(expr_discrete, genes, n_jobs=24)")
    print("  benchmark_clr_methods(mi, n_runs=3)")
    print()
