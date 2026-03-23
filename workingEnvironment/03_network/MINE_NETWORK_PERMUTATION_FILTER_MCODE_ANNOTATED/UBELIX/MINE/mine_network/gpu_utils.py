"""
GPU utilities — auto-detect VRAM and compute safe batch sizes.
================================================================

This module provides functions to:
  1. Query GPU device properties (VRAM, compute capability, name)
  2. Auto-compute optimal batch sizes based on available VRAM
  3. Gracefully fall back to sensible defaults on CPU or old hardware

Rationale: With H200 (141GB) vs RTX 2080 Ti (11GB), fixed batch_pairs is wasteful.
This function scales automatically to maximize GPU utilization (80% threshold).
"""

import logging
import torch
from typing import Tuple


logger = logging.getLogger(__name__)


def compute_optimal_batch_size(
    device_str: str,
    safety_threshold: float = 0.8,
    bytes_per_pair: float = 2.54e6,
    fallback: int = 512,
    verbose: bool = True,
) -> int:
    """
    Auto-detect GPU VRAM and compute safe batch size for MINE training.

    Given a CUDA device, query its total VRAM and divide by empirically-measured
    memory per gene pair (~2.54 MB). Uses a safety threshold (default 80%) to leave
    headroom for PyTorch runtime overhead, optimizer state, and gradient tensors.

    Parameters
    ----------
    device_str : str
        CUDA device string: "cuda", "cuda:0", "cuda:1", "auto", or "cpu".
        If "auto", uses cuda:0 if available, else CPU.
    safety_threshold : float
        Fraction of VRAM to use (0.8 = 80%, leaves 20% for overhead).
    bytes_per_pair : float
        Empirically measured memory per gene pair in forward+backward pass.
        Default 2.54 MB from profiling RTX 2080 Ti with batch_pairs=512.
        Scales linearly with batch size.
    fallback : int
        Default batch size if memory query fails (CPU, old drivers, etc.).
        Default 512 (original hardcoded value).
    verbose : bool
        Print GPU info to logger.

    Returns
    -------
    int
        Safe batch size (clamped between 256 and 16000).
    """
    try:
        # Normalize device string
        if device_str in ("cpu", "CPU"):
            if verbose:
                logger.info(
                    f"CPU detected; using fallback batch_size={fallback}"
                )
            return fallback

        if device_str == "auto":
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device_str == "cpu":
                if verbose:
                    logger.info("No GPU available; using CPU with batch_size=512")
                return fallback

        # Extract device index from "cuda:N" format
        if ":" in device_str:
            device_idx = int(device_str.split(":")[-1])
        else:
            device_idx = 0

        # Query GPU properties
        props = torch.cuda.get_device_properties(device_idx)
        total_vram_bytes = props.total_memory
        gpu_name = props.name
        major, minor = props.major, props.minor

        # Compute safe VRAM budget
        safe_vram_bytes = total_vram_bytes * safety_threshold
        batch_size = int(safe_vram_bytes / bytes_per_pair)

        # Clamp to reasonable range
        batch_size = max(256, min(batch_size, 16000))

        if verbose:
            logger.info(
                f"GPU auto-sizing | Device: {device_idx} ({gpu_name}) | "
                f"Total VRAM: {total_vram_bytes / 1e9:.1f} GB | "
                f"Safe threshold ({safety_threshold*100:.0f}%): {safe_vram_bytes / 1e9:.1f} GB | "
                f"Computed batch_size: {batch_size} pairs"
            )

        return batch_size

    except Exception as e:
        if verbose:
            logger.warning(
                f"Failed to auto-detect batch size (reason: {e}); "
                f"using fallback batch_size={fallback}"
            )
        return fallback


def get_gpu_info(device_str: str = "cuda:0") -> Tuple[str, str, float]:
    """
    Query GPU name, compute capability, and total VRAM.

    Parameters
    ----------
    device_str : str
        CUDA device string (e.g., "cuda:0").

    Returns
    -------
    Tuple[str, str, float]
        (gpu_name, compute_capability, total_vram_gb)

    Raises
    ------
    RuntimeError
        If device is not available or not a CUDA device.
    """
    try:
        if ":" in device_str:
            device_idx = int(device_str.split(":")[-1])
        else:
            device_idx = 0

        props = torch.cuda.get_device_properties(device_idx)
        gpu_name = props.name
        compute_cap = f"{props.major}.{props.minor}"
        total_vram_gb = props.total_memory / 1e9

        return gpu_name, compute_cap, total_vram_gb

    except Exception as e:
        raise RuntimeError(f"Failed to query GPU info: {e}")
