"""
Batched MINE — Mutual Information Neural Estimation for gene pairs.
====================================================================

This module implements the core MI estimator that *replaces* histogram MI.

Background (Section 2 of the user design)
------------------------------------------
For a gene pair (g_i, g_j) in study s with n_s samples:

    x ∈ ℝ^{n_s}  — expression of gene g_i
    z ∈ ℝ^{n_s}  — expression of gene g_j

We feed paired samples (x_k, z_k) into a small neural network T_θ and
optimise the **Donsker–Varadhan (DV)** objective:

    I(X; Z) ≥ E_{P_XZ}[T_θ] − log E_{P_X ⊗ P_Z}[e^{T_θ}]

where P_XZ is the joint distribution and P_X ⊗ P_Z is the product of
marginals.  The product-of-marginals samples are created by independently
shuffling Z across samples (breaking the pairing).

**MINE gives a continuous MI estimate, not a p-value** (Section 2, key
point).  The permutation significance layer is in ``permutation.py``.

Efficiency: batched weight tensors (Section 4a–b)
--------------------------------------------------
Training one network per gene pair is too slow for hundreds of thousands of
candidates.  Instead, we represent **B independent T_k networks** as
batched weight tensors of shape (B, H, D):

    W1 : (B, H, 2)     — first layer weights for all B networks
    W2 : (B, H, H)     — second layer
    W3 : (B, 1, H)     — output layer

A single ``torch.bmm`` call applies all B networks in one GPU kernel.
With B = 512 and 200 epochs, a batch takes ~5 seconds on a consumer GPU.

EMA bias correction (Paper §3.2)
---------------------------------
The log-mean-exp term in the DV bound causes biased gradients.
We use exponential moving average (EMA) bias correction:

    EMA ← (1 − α) · EMA + α · E_Q[e^T]
    gradient uses E_Q[e^T] / EMA  instead of log E_Q[e^T]

The EMA tensor is cloned before each in-place update to avoid PyTorch
autograd errors (``RuntimeError: in-place operation``).

Architecture (Section 6)
-------------------------
::

    class MINE(nn.Module):
        Input : (x_k, z_k) ∈ ℝ²
        Hidden: 2 → H → H  (ELU activations)
        Output: (H → 1) scalar T_θ

Small sizes (H = 32–64) suffice for 2-D inputs; larger H wastes compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext


def _make_torch_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    """Create a per-device RNG generator for reproducible torch.rand calls."""
    if seed is None:
        return None
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(int(seed))
    return gen


def _autocast_ctx(enabled: bool, device: torch.device):
    """Return a compatible autocast context manager for the current torch version."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        from torch.amp import autocast
        return autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        from torch.cuda.amp import autocast
        return autocast(dtype=torch.float16)


class BatchedMINE(nn.Module):
    """
    B independent statistics networks trained in parallel.

    Each network maps ℝ² → ℝ through two hidden layers with ELU:

        T_k(x, z) = W3_k · ELU(W2_k · ELU(W1_k · [x, z] + b1_k) + b2_k) + b3_k

    All B share one forward pass via batched matrix multiplications.

    Parameters
    ----------
    batch_size : int
        Number of independent networks (one per gene pair in the batch).
    hidden_dim : int
        Width of both hidden layers.
    """

    def __init__(self, batch_size: int, hidden_dim: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # Xavier-scale initialisation
        s1 = (2.0 / (2 + hidden_dim)) ** 0.5
        s2 = (2.0 / (2 * hidden_dim)) ** 0.5
        s3 = (2.0 / (hidden_dim + 1)) ** 0.5

        self.W1 = nn.Parameter(torch.randn(batch_size, hidden_dim, 2) * s1)
        self.b1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(batch_size, hidden_dim, hidden_dim) * s2)
        self.b2 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(batch_size, 1, hidden_dim) * s3)
        self.b3 = nn.Parameter(torch.zeros(batch_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all B networks simultaneously.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N_samples, 2)
            Paired gene expression values.

        Returns
        -------
        torch.Tensor, shape (B, N_samples)
            T_k scores for every sample in every pair.
        """
        # Layer 1: (B, N, 2) @ (B, 2, H) → (B, N, H)
        h = torch.bmm(x, self.W1.transpose(1, 2)) + self.b1.unsqueeze(1)
        h = F.elu(h)
        # Layer 2: (B, N, H) @ (B, H, H) → (B, N, H)
        h = torch.bmm(h, self.W2.transpose(1, 2)) + self.b2.unsqueeze(1)
        h = F.elu(h)
        # Layer 3: (B, N, H) @ (B, H, 1) → (B, N, 1) → squeeze → (B, N)
        out = torch.bmm(h, self.W3.transpose(1, 2)) + self.b3.unsqueeze(1)
        return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Core batch MI estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mi_batch(
    gene_i_data: torch.Tensor,
    gene_j_data: torch.Tensor,
    hidden_dim: int = 64,
    n_epochs: int = 200,
    lr: float = 1e-3,
    ema_alpha: float = 0.01,
    grad_clip: float = 1.0,
    n_eval_shuffles: int = 5,
    mixed_precision: bool = False,
    rand_generator: torch.Generator | None = None,
) -> np.ndarray:
    """
    Train B MINE networks and estimate MI for B gene pairs simultaneously.

    This is the core computational primitive.  All B networks share a single
    forward/backward pass per epoch, making GPU utilisation efficient.

    The training loop:
      1. Build joint input: (x_k, z_k) for real pairing.
      2. Build marginal input: (x_k, z_π(k)) where π is a random permutation
         (breaks the joint structure → product of marginals).
      3. Compute DV bound with EMA bias correction.
      4. Backprop and update weights.

    After training, the final MI is averaged over ``n_eval_shuffles`` marginal
    shuffles to reduce variance.

    Parameters
    ----------
    gene_i_data : torch.Tensor, shape (B, N_samples)
        Z-scored expression of gene i for each of the B pairs.
    gene_j_data : torch.Tensor, shape (B, N_samples)
        Z-scored expression of gene j for each of the B pairs.
    hidden_dim : int
        Hidden-layer width.
    n_epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    ema_alpha : float
        EMA decay for bias correction.
    grad_clip : float
        Maximum gradient norm.
    n_eval_shuffles : int
        Marginal shuffles for final MI evaluation.
    mixed_precision : bool
        Enable float16 forward pass via torch.autocast for memory savings.
        Backward pass remains float32 for numerical stability.

    Returns
    -------
    np.ndarray, shape (B,)
        MI estimates in nats, clamped ≥ 0.
    """
    B, N = gene_i_data.shape
    device = gene_i_data.device

    model = BatchedMINE(B, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = torch.ones(B, device=device)
    loss_curve = []  # per-epoch mean loss for diagnostics
    mi_curve_train = []  # per-epoch mean training MI objective
    use_mixed_precision = bool(mixed_precision and device.type == "cuda")

    for _ in range(n_epochs):
        optimizer.zero_grad()

        # Joint: real pairing (x_k, z_k)
        joint = torch.stack([gene_i_data, gene_j_data], dim=2)  # (B, N, 2)

        # Marginal: shuffle z independently per pair → product of marginals
        perm = torch.argsort(
            torch.rand(B, N, device=device, generator=rand_generator),
            dim=1,
        )
        gj_shuf = torch.gather(gene_j_data, 1, perm)
        marginal = torch.stack([gene_i_data, gj_shuf], dim=2)  # (B, N, 2)

        # Forward pass with optional mixed precision (CUDA only)
        with _autocast_ctx(use_mixed_precision, device):
            T_joint = model(joint)
            T_marginal = model(marginal)

        # DV bound with EMA bias correction (always float32 for stability)
        joint_mean = T_joint.mean(dim=1)       # (B,)
        et = T_marginal.exp().mean(dim=1)      # (B,)

        # Clone EMA before in-place update (avoids autograd error)
        ema_snap = ema.detach().clone()
        with torch.no_grad():
            ema.copy_((1 - ema_alpha) * ema + ema_alpha * et.detach())

        mi = joint_mean - (et / ema_snap).log()
        loss = -mi.mean()
        loss_curve.append(float(loss.detach()))
        mi_curve_train.append(float(mi.mean().detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # Final evaluation: average over several marginal shuffles
    with torch.no_grad():
        mi_acc = torch.zeros(B, device=device)
        for _ in range(n_eval_shuffles):
            joint = torch.stack([gene_i_data, gene_j_data], dim=2)
            perm = torch.argsort(
                torch.rand(B, N, device=device, generator=rand_generator),
                dim=1,
            )
            gj_shuf = torch.gather(gene_j_data, 1, perm)
            marginal = torch.stack([gene_i_data, gj_shuf], dim=2)

            with _autocast_ctx(use_mixed_precision, device):
                T_j = model(joint)
                T_m = model(marginal)

            mi_acc += T_j.mean(dim=1) - T_m.exp().mean(dim=1).log()
        mi_acc /= n_eval_shuffles

    mi_result = mi_acc.clamp(min=0).cpu().numpy()
    diagnostics = {
        "loss_curve": loss_curve,
        "mi_curve_train": mi_curve_train,
        "final_mi_mean": float(mi_result.mean()),
        "final_mi_std": float(mi_result.std()),
        "final_mi_max": float(mi_result.max()),
    }
    return mi_result, diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# High-level: estimate MI for a list of candidate gene pairs
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mi_for_pairs(
    expr_matrix: np.ndarray,
    pair_indices: np.ndarray,
    mine_cfg,
    device: torch.device,
    verbose: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Estimate MI for all candidate gene pairs using batched MINE.

    Splits the candidate list into chunks of ``mine_cfg.batch_pairs`` and
    processes each chunk in a single batched forward/backward pass.

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_samples)
        Z-scored expression (float32).
    pair_indices : np.ndarray, shape (n_pairs, 2)
        Each row (i, j) is a candidate gene pair.
    mine_cfg : MINEConfig
        MINE hyper-parameters.
    device : torch.device
        Compute device.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        MI estimates in nats.
    """
    n_pairs = len(pair_indices)
    batch_size = mine_cfg.batch_pairs
    n_batches = (n_pairs + batch_size - 1) // batch_size
    mi_all = np.zeros(n_pairs, dtype=np.float32)
    all_diagnostics = []
    rand_generator = _make_torch_generator(device, seed)

    expr_t = torch.from_numpy(expr_matrix).float().to(device)

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_pairs)
        actual_B = end - start

        idx_i = pair_indices[start:end, 0]
        idx_j = pair_indices[start:end, 1]

        gi = expr_t[idx_i]  # (actual_B, N_samples)
        gj = expr_t[idx_j]

        mi_batch, batch_diag = estimate_mi_batch(
            gi, gj,
            hidden_dim=mine_cfg.hidden_dim,
            n_epochs=mine_cfg.n_epochs,
            lr=mine_cfg.learning_rate,
            ema_alpha=mine_cfg.ema_alpha,
            grad_clip=mine_cfg.gradient_clip,
            n_eval_shuffles=mine_cfg.n_eval_shuffles,
            mixed_precision=mine_cfg.mixed_precision,
            rand_generator=rand_generator,
        )
        mi_all[start:end] = mi_batch[:actual_B]
        batch_diag["batch_id"] = b
        batch_diag["n_pairs"] = actual_B
        all_diagnostics.append(batch_diag)

        if verbose and (b + 1) % max(1, n_batches // 10) == 0:
            pct = (b + 1) / n_batches * 100
            last_loss = batch_diag["loss_curve"][-1] if batch_diag["loss_curve"] else float("nan")
            print(f"  MINE progress: {b+1}/{n_batches} batches ({pct:.0f}%) "
                  f"| last_loss={last_loss:.4f} | MI_mean={batch_diag['final_mi_mean']:.4f}")

    return mi_all, all_diagnostics
