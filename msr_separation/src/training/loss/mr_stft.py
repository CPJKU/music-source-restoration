"""
Multi-Resolution Short-Time Fourier Transform (MR-STFT) loss implementation
based on the BS-Roformer architecture.

This module provides the MR-STFT loss implementation extracted from BS-Roformer,
which includes multi-resolution STFT with different window sizes and proper
complex number handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import List, Tuple, Optional, Callable

from einops import rearrange


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss extracted from BS-Roformer implementation.
    
    This loss function computes STFT losses across multiple resolutions
    with different window sizes, following the BS-Roformer approach.
    """
    
    def __init__(
        self,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_window_fn: Callable = torch.hann_window,
        multi_stft_n_fft: int = 2048
    ):
        """
        Initialize MR-STFT loss based on BS-Roformer implementation.
        
        Args:
            multi_stft_resolution_loss_weight: Weight for the multi-resolution loss
            multi_stft_resolutions_window_sizes: Tuple of window sizes for different resolutions
            multi_stft_hop_size: Hop size for STFT
            multi_stft_normalized: Whether to normalize the STFT
            multi_stft_window_fn: Window function to use
            multi_stft_n_fft: Base FFT size
        """
        super().__init__()
        
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_hop_size = multi_stft_hop_size
        self.multi_stft_normalized = multi_stft_normalized
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_n_fft = multi_stft_n_fft
        
        # STFT kwargs
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss following BS-Roformer approach.
        
        Args:
            pred: Predicted audio [B, S, C, T] or [B, C, T]
            target: Target audio [B, S, C, T] or [B, C, T]
            
        Returns:
            Multi-resolution STFT loss value
        """
        device = pred.device
        
        # Handle different input shapes - flatten to [B*S, C, T] if needed
        if pred.dim() == 4:  # [B, S, C, T] - multi-source
            B, S, C, T = pred.shape
            pred_flat = pred.view(B * S, C, T)  # [B*S, C, T]
            target_flat = target.view(B * S, C, T)  # [B*S, C, T]
        else:  # [B, C, T] - single source
            pred_flat = pred
            target_flat = target
        
        multi_stft_resolution_loss = 0.0
        
        # Compute multi-resolution STFT loss for each window size
        for window_size in self.multi_stft_resolutions_window_sizes:
            # STFT parameters for this resolution
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            # Ensure STFT runs in fp32 precision for numerical stability
            with autocast(enabled=False):
                recon_Y = torch.stft(rearrange(pred_flat, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target_flat, '... s t -> (... s) t'), **res_stft_kwargs)
            
            # Compute L1 loss between complex STFT representations
            resolution_loss = F.l1_loss(recon_Y, target_Y)
            multi_stft_resolution_loss += resolution_loss
        
        # Apply weight
        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        
        return weighted_multi_resolution_loss


def get_loss_func(
    multi_stft_resolution_loss_weight: float = 1.0,
    multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
    multi_stft_hop_size: int = 147,
    multi_stft_normalized: bool = False,
    multi_stft_window_fn: Callable = torch.hann_window,
    multi_stft_n_fft: int = 2048
):
    """
    Factory function for BS-Roformer MR-STFT loss.
    
    Args:
        multi_stft_resolution_loss_weight: Weight for the multi-resolution loss
        multi_stft_resolutions_window_sizes: Tuple of window sizes for different resolutions
        multi_stft_hop_size: Hop size for STFT
        multi_stft_normalized: Whether to normalize the STFT
        multi_stft_window_fn: Window function to use
        multi_stft_n_fft: Base FFT size
    
    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    
    def loss_func(output, target):
        """
        Calculate BS-Roformer MR-STFT loss for training.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
        
        Returns:
            Dictionary containing 'loss' key with MR-STFT loss value
        """
        pred = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]
        
        # Initialize BS-Roformer MR-STFT loss
        mr_stft_loss_fn = MultiResolutionSTFTLoss(
            multi_stft_resolution_loss_weight=multi_stft_resolution_loss_weight,
            multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
            multi_stft_hop_size=multi_stft_hop_size,
            multi_stft_normalized=multi_stft_normalized,
            multi_stft_window_fn=multi_stft_window_fn,
            multi_stft_n_fft=multi_stft_n_fft
        )
        
        # Compute MR-STFT loss
        mr_stft_loss = mr_stft_loss_fn(pred, target_wav)
        
        loss_dict = {
            'loss': mr_stft_loss,
        }
        return loss_dict
    
    return loss_func


if __name__ == '__main__':
    # Test the BS-Roformer MR-STFT loss
    import torch
    
    # Create test data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples)
    target = torch.randn(batch_size, n_sources, n_channels, n_samples)
    
    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}
    
    # Test loss function
    loss_fn = get_loss_func()
    loss_dict = loss_fn(output_dict, target_dict)
    
    print(f"BS-Roformer MR-STFT Loss: {loss_dict['loss'].item():.4f}")
