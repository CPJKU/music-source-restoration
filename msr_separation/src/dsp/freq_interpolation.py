"""
Frequency-domain interpolation utilities for aligning STFT bins across different sampling rates.

This module provides functions to interpolate STFT representations to match
frequency bin centers from a different sampling rate configuration, enabling
pretrained models to work with different sampling rates while maintaining
bin alignment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def interpolate_stft_to_target_bins(
    stft_complex: torch.Tensor,
    source_n_fft: int,
    target_n_fft: int,
    source_sr: int,
    target_sr: int,
    method: str = 'linear'
) -> torch.Tensor:
    """
    Interpolate STFT from source bin centers to target bin centers.
    
    This function interpolates the STFT representation to match the frequency
    bin centers of a target configuration, enabling pretrained models trained
    on one sampling rate to work with audio at a different sampling rate.
    
    IMPORTANT: Only interpolates the overlapping frequency range (0 to min(Nyquist)).
    If source has higher frequencies than target, they are preserved separately.
    
    Args:
        stft_complex: Complex STFT tensor [..., freq_bins, time_frames]
        source_n_fft: FFT size used for source STFT
        target_n_fft: FFT size for target bin centers
        source_sr: Sample rate of source audio
        target_sr: Sample rate of target configuration (usually 44100 for pretrained models)
        method: Interpolation method ('linear' or 'cubic')
    
    Returns:
        Interpolated complex STFT tensor [..., target_freq_bins, time_frames]
        Note: If source has frequencies beyond target's Nyquist, only the overlapping
        range (0 to target Nyquist) is interpolated and returned.
    """
    device = stft_complex.device
    dtype = stft_complex.dtype
    
    # Calculate frequency bin centers
    source_freq_bins = source_n_fft // 2 + 1
    target_freq_bins = target_n_fft // 2 + 1
    
    # Frequency resolution
    source_freq_res = source_sr / source_n_fft
    target_freq_res = target_sr / target_n_fft
    
    # Calculate Nyquist frequencies
    source_nyquist = source_sr / 2
    target_nyquist = target_sr / 2
    
    # Determine overlapping frequency range
    overlap_freq = min(source_nyquist, target_nyquist)
    
    # Find which bins to interpolate (only the overlapping range)
    # For source: bins covering 0 to overlap_freq
    source_max_bin = int(overlap_freq / source_freq_res) + 1
    source_max_bin = min(source_max_bin, source_freq_bins)
    
    # For target: all bins (0 to target_nyquist)
    # target_max_bin = target_freq_bins (all bins)
    
    # Only interpolate the overlapping frequency range
    # Extract the overlapping range from source STFT
    stft_overlap = stft_complex[..., :source_max_bin, :]  # [..., source_max_bin, time]
    
    # Get magnitude and phase for overlapping range only
    magnitude = torch.abs(stft_overlap)
    phase = torch.angle(stft_overlap)
    
    # Now interpolate only this overlapping range to target bins
    # Note: We're interpolating source_max_bin bins to target_freq_bins bins
    
    # Interpolate magnitude and phase separately
    # For magnitude, use linear or cubic interpolation
    # For phase, we need to handle unwrapping to avoid phase jumps
    
    # Reshape for interpolation: [..., freq, time] -> [..., time, freq] -> [..., time, 1, freq]
    # F.interpolate with mode='linear' expects 3D input [N, C, L] for 1D interpolation
    # We need to handle batch and channel dimensions properly
    original_shape = magnitude.shape
    
    # Flatten all dimensions except the last two (freq, time)
    if magnitude.dim() > 2:
        # Reshape to [batch*..., freq, time]
        batch_dims = magnitude.shape[:-2]
        magnitude_flat = magnitude.view(-1, magnitude.shape[-2], magnitude.shape[-1])
        phase_flat = phase.view(-1, phase.shape[-2], phase.shape[-1])
    else:
        magnitude_flat = magnitude.unsqueeze(0)  # Add batch dimension
        phase_flat = phase.unsqueeze(0)
        batch_dims = None
    
    # Transpose to [batch, time, freq] then add channel dimension: [batch, 1, time, freq]
    # For linear interpolation, we need [batch, channels, length]
    magnitude_reshaped = magnitude_flat.transpose(-2, -1).unsqueeze(1)  # [batch, 1, time, freq]
    phase_reshaped = phase_flat.transpose(-2, -1).unsqueeze(1)
    
    # Interpolate magnitude
    if method == 'linear':
        # For linear mode, we need 3D input [N, C, L], so we need to interpolate along the last dimension
        # Reshape to [batch*time, 1, freq] for 1D interpolation
        batch_size, channels, time_frames, freq_bins = magnitude_reshaped.shape
        magnitude_reshaped_1d = magnitude_reshaped.permute(0, 2, 1, 3).contiguous().view(batch_size * time_frames, channels, freq_bins)
        
        magnitude_interp_1d = F.interpolate(
            magnitude_reshaped_1d,
            size=target_freq_bins,
            mode='linear',
            align_corners=False
        )  # [batch*time, 1, target_freq]
        
        # Reshape back: [batch*time, 1, target_freq] -> [batch, time, 1, target_freq] -> [batch, 1, time, target_freq]
        magnitude_interp = magnitude_interp_1d.view(batch_size, time_frames, channels, target_freq_bins).permute(0, 2, 1, 3)
        magnitude_interp = magnitude_interp.squeeze(1).transpose(-2, -1)  # [batch, target_freq, time]
    elif method == 'cubic':
        # Cubic interpolation requires 4D input [N, C, H, W]
        # We need to add batch dimension if needed
        if magnitude_reshaped.dim() == 3:
            magnitude_reshaped = magnitude_reshaped.unsqueeze(0)
            phase_reshaped = phase_reshaped.unsqueeze(0)
            needs_squeeze = True
        else:
            needs_squeeze = False
        
        magnitude_interp = F.interpolate(
            magnitude_reshaped,
            size=target_freq_bins,
            mode='bicubic',
            align_corners=False
        )
        
        if needs_squeeze:
            magnitude_interp = magnitude_interp.squeeze(0)
        
        magnitude_interp = magnitude_interp.squeeze(-2).transpose(-2, -1)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # For phase, unwrap first, interpolate, then wrap
    # Unwrap phase along frequency dimension
    # torch.unwrap doesn't exist, so we use numpy unwrap
    phase_np = phase_flat.detach().cpu().numpy()
    phase_unwrapped_np = np.unwrap(phase_np, axis=-2)
    phase_unwrapped = torch.from_numpy(phase_unwrapped_np).to(device).to(phase.dtype)
    
    # Interpolate unwrapped phase (same reshaping as magnitude)
    phase_unwrapped_reshaped = phase_unwrapped.transpose(-2, -1).unsqueeze(1)  # [batch, 1, time, freq]
    
    if method == 'linear':
        # Same reshaping as magnitude
        batch_size, channels, time_frames, freq_bins = phase_unwrapped_reshaped.shape
        phase_unwrapped_reshaped_1d = phase_unwrapped_reshaped.permute(0, 2, 1, 3).contiguous().view(batch_size * time_frames, channels, freq_bins)
        
        phase_unwrapped_interp_1d = F.interpolate(
            phase_unwrapped_reshaped_1d,
            size=target_freq_bins,
            mode='linear',
            align_corners=False
        )
        
        phase_unwrapped_interp = phase_unwrapped_interp_1d.view(batch_size, time_frames, channels, target_freq_bins).permute(0, 2, 1, 3)
        phase_unwrapped_interp = phase_unwrapped_interp.squeeze(1).transpose(-2, -1)  # [batch, target_freq, time]
    elif method == 'cubic':
        if phase_unwrapped_reshaped.dim() == 3:
            phase_unwrapped_reshaped = phase_unwrapped_reshaped.unsqueeze(0)
            needs_squeeze = True
        else:
            needs_squeeze = False
        
        phase_unwrapped_interp = F.interpolate(
            phase_unwrapped_reshaped,
            size=target_freq_bins,
            mode='bicubic',
            align_corners=False
        )
        
        if needs_squeeze:
            phase_unwrapped_interp = phase_unwrapped_interp.squeeze(0)
        
        phase_unwrapped_interp = phase_unwrapped_interp.squeeze(-2).transpose(-2, -1)
    
    # Wrap phase back to [-pi, pi]
    phase_interp = torch.atan2(
        torch.sin(phase_unwrapped_interp),
        torch.cos(phase_unwrapped_interp)
    )
    
    # Reconstruct complex STFT
    stft_interp = magnitude_interp * torch.exp(1j * phase_interp)
    
    # Reshape back to original batch dimensions if needed
    if batch_dims is not None:
        # Reshape from [batch*..., target_freq, time] back to [..., target_freq, time]
        stft_interp = stft_interp.view(*batch_dims, target_freq_bins, stft_interp.shape[-1])
    else:
        # Remove the batch dimension we added
        stft_interp = stft_interp.squeeze(0)
    
    return stft_interp.to(dtype)


def interpolate_stft_from_target_bins(
    stft_complex: torch.Tensor,
    source_n_fft: int,
    target_n_fft: int,
    source_sr: int,
    target_sr: int,
    method: str = 'linear'
) -> torch.Tensor:
    """
    Interpolate STFT from target bin centers back to source bin centers.
    
    This is the inverse operation of interpolate_stft_to_target_bins,
    used to convert model output back to the original frequency bin structure.
    
    Args:
        stft_complex: Complex STFT tensor at target bin centers [..., target_freq_bins, time_frames]
        source_n_fft: FFT size for source (original) bin centers
        target_n_fft: FFT size used for target STFT
        source_sr: Sample rate of source audio
        target_sr: Sample rate of target configuration
        method: Interpolation method ('linear' or 'cubic')
    
    Returns:
        Interpolated complex STFT tensor [..., source_freq_bins, time_frames]
    """
    # Same as forward interpolation, just swap source and target
    return interpolate_stft_to_target_bins(
        stft_complex,
        source_n_fft=target_n_fft,
        target_n_fft=source_n_fft,
        source_sr=target_sr,
        target_sr=source_sr,
        method=method
    )

