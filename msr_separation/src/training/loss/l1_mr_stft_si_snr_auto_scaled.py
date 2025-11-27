"""
Combined loss with automatic L1 scaling: L1 + MR-STFT + SI-SNR + Multi-Mel-SNR + LogWMSE.

This module provides a loss function that combines L1, MR-STFT, SI-SNR, Multi-Mel-SNR, and LogWMSE losses
with automatic dynamic scaling of the L1 loss based on the mean amplitude of predictions.
The L1 weight is computed as: (1e5 / mean_amplitude).clamp(max=1e6)

Formula: l1_weight * l1_loss + mr_stft_loss + si_snr_weight * si_snr_loss + multi_mel_snr_weight * multi_mel_snr_loss + log_wmse_weight * log_wmse_loss
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Callable, Tuple
from torch.cuda.amp import autocast
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from einops import rearrange
from torch_log_wmse import LogWMSE

from src.training.loss.mr_stft import MultiResolutionSTFTLoss
from src.training.loss.multi_mel_snr import get_loss_func as get_multi_mel_snr_loss_func


def get_loss_func(
    l1_base_weight: float = 1e5,
    l1_max_weight: float = 1e6,
    si_snr_weight: float = 0.1,
    multi_mel_snr_weight: float = 0.0,
    multi_mel_snr_sample_rate: int = 48000,
    log_wmse_weight: float = 0.0,
    log_wmse_sample_rate: int = 48000,
    log_wmse_audio_length: float = 10, 
    low_amplitude_penalty_weight: float = 0.0,
    low_amplitude_ratio: float = 0.4,
    low_amplitude_target_floor: float = 1e-4,
    multi_stft_resolution_loss_weight: float = 1.0,
    multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
    multi_stft_hop_size: int = 147,
    multi_stft_normalized: bool = False,
    multi_stft_window_fn: Callable = torch.hann_window,
    multi_stft_n_fft: int = 2048,
    phased_fine_tuning_crossover_bin: int = None  # Bin index to mask frequencies >= this during phase 1
):
    """
    Factory function for combined L1 + MR-STFT + SI-SNR + Multi-Mel-SNR + LogWMSE loss with automatic L1 scaling.
    
    Args:
        l1_base_weight: Base weight for L1 scaling (default: 1e5)
        l1_max_weight: Maximum allowed L1 weight (default: 1e6)
        si_snr_weight: Weight for SI-SNR loss component (default: 0.1)
        multi_mel_snr_weight: Weight for Multi-Mel-SNR loss component (default: 0.0)
        multi_mel_snr_sample_rate: Sample rate for Multi-Mel-SNR computation (default: 48000)
        log_wmse_weight: Weight for LogWMSE loss component (default: 0.0)
        log_wmse_sample_rate: Sample rate for LogWMSE computation (default: 48000)
        log_wmse_audio_length: Audio length in seconds for LogWMSE (default: 10)
        low_amplitude_penalty_weight: Weight for the low-amplitude hinge penalty (default: 0.0 disables it)
        low_amplitude_ratio: Minimum fraction of target amplitude predictions should match when target is active
        low_amplitude_target_floor: Minimum target amplitude considered "active" for the penalty
        multi_stft_resolution_loss_weight: Weight for the multi-resolution loss
        multi_stft_resolutions_window_sizes: Tuple of window sizes for different resolutions
        multi_stft_hop_size: Hop size for STFT
        multi_stft_normalized: Whether to normalize the STFT
        multi_stft_window_fn: Window function to use
        multi_stft_n_fft: Base FFT size
    
    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    
    # Initialize MR-STFT loss module
    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        multi_stft_resolution_loss_weight=multi_stft_resolution_loss_weight,
        multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
        multi_stft_hop_size=multi_stft_hop_size,
        multi_stft_normalized=multi_stft_normalized,
        multi_stft_window_fn=multi_stft_window_fn,
        multi_stft_n_fft=multi_stft_n_fft
    )
    
    # Initialize Multi-Mel-SNR loss function if weight > 0
    if multi_mel_snr_weight > 0.0:
        multi_mel_snr_loss_fn = get_multi_mel_snr_loss_func(sample_rate=multi_mel_snr_sample_rate)
    else:
        multi_mel_snr_loss_fn = None
    
    # Initialize LogWMSE loss function if weight > 0
    if log_wmse_weight > 0.0:
        if log_wmse_audio_length is None:
            raise ValueError("log_wmse_audio_length must be provided when log_wmse_weight > 0")
        log_wmse_loss_fn = LogWMSE(
            audio_length=log_wmse_audio_length,
            sample_rate=log_wmse_sample_rate,
            return_as_loss=True,
            bypass_filter=False
        )
    else:
        log_wmse_loss_fn = None
    
    def loss_func(output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss with automatic L1 scaling.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
                   May also contain 'phase' key for phased fine-tuning (1 or 2)
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
        
        Returns:
            Dictionary containing loss components matching MultiLossManager format:
                - 'loss': Total combined loss
                - 'l1_loss': Unweighted L1 loss
                - 'l1_loss_weighted': Weighted L1 loss (with dynamic scaling)
                - 'l1_weight': Dynamic L1 weight used (for logging)
                - 'mean_amp': Mean absolute amplitude of predictions (for wandb logging)
                - 'mr_stft_loss': Unweighted MR-STFT loss
                - 'mr_stft_loss_weighted': Weighted MR-STFT loss
                - 'si_snr_loss': Unweighted SI-SNR loss (negated SI-SNR)
                - 'si_snr_loss_weighted': Weighted SI-SNR loss
                - 'multi_mel_snr_loss': Unweighted Multi-Mel-SNR loss (negated Multi-Mel-SNR)
                - 'multi_mel_snr_loss_weighted': Weighted Multi-Mel-SNR loss
                - 'log_wmse_loss': Unweighted LogWMSE loss
                - 'log_wmse_loss_weighted': Weighted LogWMSE loss
        """
        pred = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]

        # Retrieve optional source mask indicating active sources
        source_mask = target.get('source_mask', None)
        if source_mask is None:
            source_mask = torch.ones(
                pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype
            )
        else:
            source_mask = source_mask.to(device=pred.device, dtype=pred.dtype)
            if source_mask.dim() > 2:
                source_mask = source_mask.view(source_mask.shape[0], source_mask.shape[1])

        # Expand mask for broadcasting over channels/time
        expand_dims = pred.dim() - source_mask.dim()
        source_mask_expanded = source_mask.view(*source_mask.shape, *([1] * expand_dims))
        mask_sum = source_mask.sum()
        active_sources = mask_sum.clamp_min(1.0)
        
        # Check if we're in phase 1 of phased fine-tuning (mask high frequencies)
        current_phase = output.get('phase', None)
        mask_high_freq = (current_phase == 1) and (phased_fine_tuning_crossover_bin is not None)
        
        # Compute unweighted L1 loss (masked to active sources)
        # For phase 1, we can optionally filter high frequencies, but since model output is already masked,
        # the loss will naturally focus on lower frequencies
        abs_diff = torch.abs(pred - target_wav) * source_mask_expanded
        l1_denominator = (active_sources * pred.shape[2] * pred.shape[3]).clamp_min(1.0)
        l1_loss = abs_diff.sum() / l1_denominator
        
        # Compute unweighted MR-STFT loss (before applying multi_stft_resolution_loss_weight)
        # We need to compute this directly to get the unweighted version
        device = pred.device
        if pred.dim() == 4:  # [B, S, C, T]
            B, S, C, T = pred.shape
            pred_flat = pred.view(B * S, C, T)
            target_flat = target_wav.view(B * S, C, T)
            mask_flat = source_mask.view(B * S)
            mask_channels = mask_flat.repeat_interleave(C)
            mask_channels = mask_channels[:, None, None]
        else:
            pred_flat = pred
            target_flat = target_wav
            mask_flat = source_mask.view(-1)
            mask_channels = mask_flat[:, None, None]
        
        mr_stft_loss_unweighted = 0.0
        for window_size in multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=multi_stft_window_fn(window_size, device=device),
                hop_length=multi_stft_hop_size,
                normalized=multi_stft_normalized
            )
            
            with autocast(enabled=False):
                recon_Y = torch.stft(rearrange(pred_flat, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target_flat, '... s t -> (... s) t'), **res_stft_kwargs)
            
            # Phase 1: Mask high frequencies in loss computation (both pred and target)
            if mask_high_freq and phased_fine_tuning_crossover_bin is not None:
                # recon_Y and target_Y shape: [batch*channels, freq_bins, time_frames] (complex)
                # Mask frequencies >= crossover_bin (adjust for this STFT's n_fft)
                stft_n_fft_used = max(window_size, multi_stft_n_fft)
                stft_freq_bins = stft_n_fft_used // 2 + 1
                
                # Calculate crossover bin for this STFT resolution
                # phased_fine_tuning_crossover_bin is for model's STFT (n_fft=2229, 1115 bins)
                # We need to scale it to this STFT's resolution
                # Frequency resolution scales with n_fft, so:
                # crossover_bin_here = crossover_bin * (stft_n_fft_used / model_n_fft)
                # But we don't know model_n_fft here, so we use proportional scaling based on freq bins
                # Model: 1025 bins for n_fft=2229, Loss: stft_freq_bins for n_fft=stft_n_fft_used
                # Approximate: crossover_bin_here ≈ crossover_bin * (stft_freq_bins / 1025)
                # More accurate: use frequency-based scaling
                # Since freq_res = sample_rate / n_fft, and we want same frequency:
                # bin_model * (sample_rate / n_fft_model) = bin_loss * (sample_rate / n_fft_loss)
                # bin_loss = bin_model * (n_fft_loss / n_fft_model)
                # For n_fft_model=2229, n_fft_loss varies by window_size
                # Use proportional scaling: crossover_bin_here = crossover_bin * (stft_n_fft_used / 2229)
                model_n_fft = 2229  # Model's n_fft
                crossover_bin_here = int(phased_fine_tuning_crossover_bin * (stft_n_fft_used / model_n_fft))
                crossover_bin_here = min(crossover_bin_here, stft_freq_bins - 1)  # Ensure within bounds
                
                if crossover_bin_here < stft_freq_bins:
                    # Zero out frequencies >= crossover_bin
                    recon_Y[:, crossover_bin_here:, :] = 0
                    target_Y[:, crossover_bin_here:, :] = 0
            
            diff = torch.abs(recon_Y - target_Y) * mask_channels
            resolution_denominator = (
                mask_channels.sum() * diff.shape[1] * diff.shape[2]
            ).clamp_min(1.0)
            resolution_loss = diff.sum() / resolution_denominator
            mr_stft_loss_unweighted += resolution_loss
        
        # Compute weighted MR-STFT loss
        mr_stft_loss_weighted = mr_stft_loss_unweighted * multi_stft_resolution_loss_weight
        
        # Compute SI-SNR loss (reuse flattened tensors from MR-STFT computation)
        # Calculate SI-SNR for all sources
        batch_si_snrs = si_snr(pred_flat, target_flat)  # [B*S]
        si_snr_denominator = mask_flat.sum().clamp_min(1.0)
        masked_si_snr = (batch_si_snrs * mask_flat.unsqueeze(-1)).sum() / si_snr_denominator
        si_snr_loss_unweighted = -masked_si_snr  # Negated SI-SNR as loss
        si_snr_loss_weighted = si_snr_weight * si_snr_loss_unweighted
        
        # Compute Multi-Mel-SNR loss if enabled
        if multi_mel_snr_weight > 0.0 and multi_mel_snr_loss_fn is not None:
            multi_mel_snr_loss_dict = multi_mel_snr_loss_fn(output, target)
            multi_mel_snr_loss_unweighted = multi_mel_snr_loss_dict['loss']
            multi_mel_snr_loss_weighted = multi_mel_snr_weight * multi_mel_snr_loss_unweighted
        else:
            multi_mel_snr_loss_unweighted = torch.tensor(0.0, device=pred.device)
            multi_mel_snr_loss_weighted = torch.tensor(0.0, device=pred.device)
        
        # Compute LogWMSE loss if enabled
        if log_wmse_weight > 0.0 and log_wmse_loss_fn is not None:
            B, S, C, T = pred.shape
            
            # Move LogWMSE to correct device if needed and ensure training mode
            # Note: .to() returns the same module if already on device, preserving computation graph
            log_wmse_loss_fn.to(device=pred.device)
            log_wmse_loss_fn.train()  # Ensure training mode for gradient computation
            
            # Compute unprocessed audio (mix) by summing target sources
            # target_wav: [B, S, C, T] -> unprocessed: [B, C, T]
            unprocessed_audio = target_wav.sum(dim=1)  # Sum over source dimension
            
            # Rearrange pred and target to match logWMSE format
            # pred: [B, S, C, T] -> processed_audio: [B, C, S, T]
            processed_audio = pred.permute(0, 2, 1, 3)  # [B, C, S, T]
            target_audio = target_wav.permute(0, 2, 1, 3)  # [B, C, S, T]
            
            # Mask inactive stems: set to zero for inactive sources
            # source_mask: [B, S] -> [B, 1, S, 1] for broadcasting
            source_mask_for_wmse = source_mask.view(B, 1, S, 1).to(device=pred.device, dtype=pred.dtype)
            processed_audio_masked = processed_audio * source_mask_for_wmse
            target_audio_masked = target_audio * source_mask_for_wmse
            
            # Compute LogWMSE loss
            log_wmse_loss_unweighted = log_wmse_loss_fn(
                unprocessed_audio,  # [B, C, T]
                processed_audio_masked,  # [B, C, S, T]
                target_audio_masked  # [B, C, S, T]
            )
            
            # Average over active stems only
            # The loss is already computed per batch, but we need to account for inactive stems
            # Since we masked inactive stems, we can just use the loss as-is
            log_wmse_loss_weighted = log_wmse_weight * log_wmse_loss_unweighted
        else:
            log_wmse_loss_unweighted = torch.tensor(0.0, device=pred.device)
            log_wmse_loss_weighted = torch.tensor(0.0, device=pred.device)
        
        # Dynamic L1 weight based on mean amplitude of prediction
        with torch.no_grad():
            mean_amp_numerator = (pred.abs() * source_mask_expanded).sum()
            mean_amp = (mean_amp_numerator / l1_denominator).clamp(min=1e-8)
            l1_weight = (float(l1_base_weight) / mean_amp).clamp(max=float(l1_max_weight))
        
        # Compute weighted L1 loss
        l1_loss_weighted = l1_weight * l1_loss
        
        # Low amplitude penalty: hinge on mean abs amplitude per source when target is active
        
        if low_amplitude_penalty_weight > 0.0:
            pred_amp = pred.abs().mean(dim=(2, 3))
            target_amp = target_wav.abs().mean(dim=(2, 3))
            target_active_mask = (target_amp >= low_amplitude_target_floor).to(pred.dtype)
            penalty_mask = target_active_mask * source_mask
            penalty_denominator = penalty_mask.sum().clamp_min(1.0)
            required_amp = target_amp * float(low_amplitude_ratio)
            amp_deficit = (required_amp - pred_amp).clamp_min(0.0)
            low_amp_penalty = (amp_deficit * penalty_mask).sum() / penalty_denominator
            low_amp_penalty_weighted = float(low_amplitude_penalty_weight) * low_amp_penalty
        else: 
            low_amp_penalty_weighted = torch.tensor([0.0], device=pred.device)
            low_amp_penalty = torch.tensor([0.0], device=pred.device)

        # Combine losses
        total_loss = (
            l1_loss_weighted
            + mr_stft_loss_weighted
            + si_snr_loss_weighted
            + multi_mel_snr_loss_weighted
            + log_wmse_loss_weighted
            + low_amp_penalty_weighted
        )
        
        loss_dict = {
            'loss': total_loss,
            'l1_loss': l1_loss,
            'l1_loss_weighted': l1_loss_weighted,
            'l1_weight': l1_weight,  # Extra logging key for the dynamic weight
            'mean_amp': mean_amp,  # Mean absolute amplitude of predictions (for wandb logging)
            'mr_stft_loss': mr_stft_loss_unweighted,
            'mr_stft_loss_weighted': mr_stft_loss_weighted,
            'si_snr_loss': si_snr_loss_unweighted,
            'si_snr_loss_weighted': si_snr_loss_weighted,
            'multi_mel_snr_loss': multi_mel_snr_loss_unweighted,
            'multi_mel_snr_loss_weighted': multi_mel_snr_loss_weighted,
            'log_wmse_loss': log_wmse_loss_unweighted,
            'log_wmse_loss_weighted': log_wmse_loss_weighted,
            'low_amp_penalty': low_amp_penalty,
            'low_amp_penalty_weighted': low_amp_penalty_weighted,
        }
        
        return loss_dict
    
    return loss_func


if __name__ == '__main__':
    # Test the combined loss function
    import torch
    
    # Create test data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1
    target = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1
    
    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}
    
    # Test loss function
    loss_fn = get_loss_func()
    loss_dict = loss_fn(output_dict, target_dict)
    
    print("Combined Loss Results:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value:.4f}")

