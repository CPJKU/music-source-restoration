"""
Multi-Mel-SNR loss implementation.

This module provides a loss function based on negated Multi-Mel-SNR for training.
Multi-Mel-SNR computes SNR across multiple mel-spectrogram configurations and averages them.
"""

import torch
import torchaudio
import math
from typing import Dict


def get_loss_func(sample_rate: int = 48000):
    """
    Get Multi-Mel-SNR loss function for training.
    
    This function returns a loss function that uses negated Multi-Mel-SNR as the loss.
    The loss is only computed on active stems (using source_mask).
    
    Args:
        sample_rate: Sample rate of the audio (default: 48000)
    
    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    # Mel configurations (same as in metric)
    configs = [
        (512, 256, 80),  # (n_fft, hop_length, n_mels)
        (1024, 512, 128),
        (2048, 1024, 192)
    ]
    
    def loss_func(output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate Multi-Mel-SNR loss for training.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
                   Optionally contains 'source_mask' key [batch_size, n_sources] 
                   to filter only active stems (1 = active, 0 = inactive)
        
        Returns:
            Dictionary containing 'loss' key with negated Multi-Mel-SNR value
        """
        output_wav = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]
        B, S, C, T = output_wav.shape
        device = output_wav.device
        
        # Retrieve optional source mask indicating active sources
        source_mask = target.get('source_mask', None)
        if source_mask is None:
            source_mask = torch.ones(B, S, device=device, dtype=output_wav.dtype)
        else:
            source_mask = source_mask.to(device=device, dtype=output_wav.dtype)
            if source_mask.dim() > 2:
                source_mask = source_mask.view(B, S)
        
        # Filter to only active stems
        active_mask = source_mask.bool()  # [B, S]
        
        # Collect active stems across all batches
        active_outputs = []
        active_targets = []
        
        for b in range(B):
            # Get active sources for this batch
            batch_active_sources = active_mask[b]  # [S]
            if batch_active_sources.any():
                # Extract only active sources: [num_active, C, T]
                batch_output = output_wav[b][batch_active_sources]  # [num_active, C, T]
                batch_target = target_wav[b][batch_active_sources]  # [num_active, C, T]
                active_outputs.append(batch_output)
                active_targets.append(batch_target)
        
        # If we have no active stems, return zero loss
        if len(active_outputs) == 0:
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        # Concatenate all active stems: [total_active, C, T]
        output_active = torch.cat(active_outputs, dim=0)  # [total_active, C, T]
        target_active = torch.cat(active_targets, dim=0)  # [total_active, C, T]
        
        # Reshape to [total_active*C, T] to process each channel independently
        output_flat = output_active.view(-1, T)  # [total_active*C, T]
        target_flat = target_active.view(-1, T)  # [total_active*C, T]
        
        N, T_flat = output_flat.shape
        
        # Check for NaN/inf in inputs and replace with zeros
        output_flat = torch.where(torch.isfinite(output_flat), output_flat, torch.zeros_like(output_flat))
        target_flat = torch.where(torch.isfinite(target_flat), target_flat, torch.zeros_like(target_flat))
        
        # Scale-invariant normalization per sample
        # alpha = <ref, pred> / <pred, pred>
        dot_ref_pred = (target_flat * output_flat).sum(dim=1)  # [N]
        dot_pred_pred = (output_flat * output_flat).sum(dim=1) + 1e-8  # [N]
        alpha = dot_ref_pred / dot_pred_pred  # [N]
        # Check for NaN/inf in alpha and handle
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        output_scaled = alpha.unsqueeze(1) * output_flat  # [N, T]
        
        # Compute SNR for each mel configuration and average
        all_snrs = []
        for n_fft, hop, n_mels in configs:
            # Create mel transform with correct device
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, f_min=0, f_max=24000, power=2.0
            ).to(device)
            
            # Compute mel spectrograms
            M_ref = mel_transform(target_flat)  # [N, n_mels, time_frames]
            M_pred = mel_transform(output_scaled)  # [N, n_mels, time_frames]
            
            # Check for NaN/inf in mel spectrograms and replace with zeros
            M_ref = torch.where(torch.isfinite(M_ref), M_ref, torch.zeros_like(M_ref))
            M_pred = torch.where(torch.isfinite(M_pred), M_pred, torch.zeros_like(M_pred))
            
            # Compute SNR: 10 * log10(sum(M_ref^2) / sum((M_ref - M_pred)^2))
            M_ref_sq = M_ref.pow(2)  # [N, n_mels, time_frames]
            M_diff_sq = (M_ref - M_pred).pow(2)  # [N, n_mels, time_frames]
            
            ref_power = M_ref_sq.sum(dim=(1, 2))  # [N]
            diff_power = M_diff_sq.sum(dim=(1, 2)) + 1e-8  # [N]
            
            # Check for NaN/inf in power values and handle
            ref_power = torch.where(torch.isfinite(ref_power), ref_power, torch.zeros_like(ref_power))
            diff_power = torch.where(torch.isfinite(diff_power), diff_power, torch.ones_like(diff_power) * 1e-8)
            
            # Compute SNR ratio with proper handling of edge cases
            snr_ratio = ref_power / diff_power
            snr = 10 * torch.log10(snr_ratio)  # [N]
            
            # Check for NaN/inf in SNR values and replace with a very negative value (-80 dB)
            snr = torch.where(torch.isfinite(snr), snr, torch.full_like(snr, -80.0))
            
            all_snrs.append(snr)
        
        # Average across all mel configurations
        # Stack to [num_configs, N] then mean along first dimension
        stacked_snrs = torch.stack(all_snrs, dim=0)  # [num_configs, N]
        avg_snr = stacked_snrs.mean(dim=0)  # [N]
        
        # Final check for NaN/inf in averaged SNR
        avg_snr = torch.where(torch.isfinite(avg_snr), avg_snr, torch.full_like(avg_snr, -80.0))
        
        # Use negated Multi-Mel-SNR as loss (higher Multi-Mel-SNR = lower loss)
        multi_mel_snr_loss = -avg_snr.mean()
        
        loss_dict = {
            'loss': multi_mel_snr_loss,  # Main loss for backpropagation
        }
        return loss_dict
    
    return loss_func


if __name__ == '__main__':
    # Test the loss function
    import torch
    
    # Create test data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1
    target = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1
    
    # Test with all sources active
    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}
    
    loss_fn = get_loss_func(sample_rate=48000)
    loss_dict = loss_fn(output_dict, target_dict)
    
    print("Multi-Mel-SNR Loss (all sources active):")
    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    
    # Test with some sources inactive
    source_mask = torch.ones(batch_size, n_sources)
    source_mask[0, 1] = 0  # First batch, second source inactive
    source_mask[1, 3] = 0  # Second batch, fourth source inactive
    
    target_dict_masked = {'waveform': target, 'source_mask': source_mask}
    loss_dict_masked = loss_fn(output_dict, target_dict_masked)
    
    print("\nMulti-Mel-SNR Loss (with source mask):")
    print(f"  Loss: {loss_dict_masked['loss'].item():.4f}")

