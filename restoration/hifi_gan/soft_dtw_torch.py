"""
Pure PyTorch implementation of SoftDTW for GPU efficiency.

This implementation:
- No Numba compilation (instant startup)
- Fully GPU-parallelized (faster on GPU)
- Differentiable through PyTorch autograd
- Memory-efficient with in-place operations

Based on "Soft-DTW: a Differentiable Loss Function for Time-Series"
(Cuturi & Blondel, 2017)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorized pairwise squared Euclidean distances."""
    # torch.cdist computes ||x - y||_2; square without in-place ops to keep grads
    return torch.cdist(x, y, p=2).pow(2)


def _softdtw_forward(
    D: torch.Tensor,
    gamma: float,
    bandwidth: int = -1,
) -> torch.Tensor:
    """
    Dynamic-programming Soft-DTW using diagonal sweeps for better parallelism.
    """
    B, N, M = D.shape
    device = D.device
    dtype = D.dtype
    R = torch.full((B, N + 2, M + 2), float("inf"), device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    diag_limit = N + M
    use_band = bandwidth is not None and bandwidth >= 0

    for diag in range(2, diag_limit + 1):
        row_start = max(1, diag - M)
        row_end = min(N, diag - 1)
        if row_end < row_start:
            continue

        rows = torch.arange(row_start, row_end + 1, device=device, dtype=torch.long)
        cols = diag - rows

        valid = (cols >= 1) & (cols <= M)
        if use_band:
            valid &= torch.abs(rows - cols) <= bandwidth
        if not torch.any(valid):
            continue

        rows = rows[valid]
        cols = cols[valid]

        prev = torch.stack(
            (
                R[:, rows - 1, cols - 1],
                R[:, rows - 1, cols],
                R[:, rows, cols - 1],
            ),
            dim=-1,
        )
        prev_min = prev.min(dim=-1, keepdim=True).values
        soft = -gamma * torch.logsumexp(-(prev - prev_min) / gamma, dim=-1) + prev_min.squeeze(-1)
        R[:, rows, cols] = D[:, rows - 1, cols - 1] + soft

    return R[:, N, M]


class SoftDTWTorch(nn.Module):
    """
    Pure PyTorch implementation of Soft-DTW.
    
    Advantages over Numba version:
    - No compilation time (instant startup)
    - GPU-parallelized (faster for batch processing)
    - Native PyTorch autograd (cleaner gradients)
    - Better memory efficiency
    
    Parameters
    ----------
    gamma : float
        Smoothing parameter for soft-min. Lower = closer to hard DTW.
    normalize : bool
        If True, normalizes by sequence lengths (diagonal normalization).
    bandwidth : int | None
        If set, uses Sakoe-Chiba band (restricts paths within bandwidth).
        This reduces complexity from O(N*M) to O(N*bandwidth).
    """
    
    def __init__(self, gamma: float = 1.0, normalize: bool = False, bandwidth: int | None = None):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize
        self.bandwidth = bandwidth
        self._softdtw_impl = _softdtw_forward
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-DTW distance between sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Reference sequences, shape (B, N, D) or (N, D)
        y : torch.Tensor
            Query sequences, shape (B, M, D) or (M, D)
        
        Returns
        -------
        torch.Tensor
            Soft-DTW distances, shape (B,) or scalar
        """
        # Handle non-batched input
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        
        # Compute pairwise distances
        D = _pairwise_distances(x, y)  # (B, N, M)
        bandwidth = -1 if self.bandwidth is None else int(self.bandwidth)
        gamma_val = float(self.gamma)
        
        # Compute soft-DTW
        result = self._softdtw_impl(D, gamma_val, bandwidth)
        if self.normalize:
            # Diagonal normalization: DTW(x,y) - 0.5*(DTW(x,x) + DTW(y,y))
            dist_xx = self._softdtw_impl(_pairwise_distances(x, x), gamma_val, bandwidth)
            dist_yy = self._softdtw_impl(_pairwise_distances(y, y), gamma_val, bandwidth)
            result = result - 0.5 * (dist_xx + dist_yy)
        
        return result.squeeze(0) if squeeze else result


class SoftDTWTorchFast(SoftDTWTorch):
    """
    Optimized PyTorch Soft-DTW that tries to compile the diagonal sweep kernel.
    
    Falls back to the base implementation if torch.compile is unavailable.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        normalize: bool = False,
        bandwidth: int | None = None,
        use_compile: bool = True,
    ):
        super().__init__(gamma=gamma, normalize=normalize, bandwidth=bandwidth)
        self._compiled_impl = _maybe_compile() if use_compile else None
        if self._compiled_impl is not None:
            self._softdtw_impl = self._compiled_impl


def _maybe_compile():
    """Compile the core kernel with torch.compile when available."""
    if hasattr(torch, "compile"):
        try:
            return torch.compile(_softdtw_forward)
        except Exception:
            return None
    return None


__all__ = ["SoftDTWTorch", "SoftDTWTorchFast"]
