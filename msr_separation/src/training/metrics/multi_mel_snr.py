"""
Multi-Mel-SNR (Multi-Mel Spectrogram Signal-to-Noise Ratio) metric implementation.

This module provides Multi-Mel-SNR evaluation for source separation models.
The metric computes SNR across multiple mel-spectrogram configurations and averages them.
"""

import torch
import torch.nn as nn
import torchaudio
import math
from typing import Dict


def multi_mel_snr(reference, prediction, sr=48000):
    """Compute Multi-Mel-SNR between reference and prediction."""
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()

    # Check for NaN/inf in inputs and replace with zeros
    reference = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
    prediction = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))

    # Scale-invariant normalization
    dot_ref_pred = torch.dot(reference, prediction)
    dot_pred_pred = torch.dot(prediction, prediction) + 1e-8
    alpha = dot_ref_pred / dot_pred_pred
    # Check for NaN/inf in alpha and handle
    if not torch.isfinite(alpha):
        alpha = torch.tensor(0.0)
    prediction = alpha * prediction

    # Three mel configurations
    configs = [
        (512, 256, 80),  # (n_fft, hop_length, n_mels)
        (1024, 512, 128),
        (2048, 1024, 192)
    ]

    snrs = []
    for n_fft, hop, n_mels in configs:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=0, f_max=24000, power=2.0
        )
        M_ref = mel(reference)
        M_pred = mel(prediction)
        
        # Check for NaN/inf in mel spectrograms and replace with zeros
        M_ref = torch.where(torch.isfinite(M_ref), M_ref, torch.zeros_like(M_ref))
        M_pred = torch.where(torch.isfinite(M_pred), M_pred, torch.zeros_like(M_pred))
        
        ref_power = M_ref.pow(2).sum()
        diff_power = (M_ref - M_pred).pow(2).sum() + 1e-8
        
        # Check for NaN/inf in power values
        if not torch.isfinite(ref_power):
            ref_power = torch.tensor(0.0)
        if not torch.isfinite(diff_power):
            diff_power = torch.tensor(1e-8)
        
        snr_ratio = (ref_power / diff_power).clamp_min(1e-8)
        snr = 10 * torch.log10(snr_ratio)
        
        # Check for NaN/inf in SNR and replace with -80 dB
        if not torch.isfinite(snr):
            snr = torch.tensor(-80.0)
        
        snrs.append(snr.item())

    # Filter out any NaN/inf values from snrs list
    finite_snrs = [s for s in snrs if math.isfinite(s)]
    if finite_snrs:
        return sum(finite_snrs) / len(finite_snrs)
    else:
        return -80.0  # Default to very negative SNR if all values are NaN/inf


class MultiMelSNRMetric(nn.Module):
    """
    Multi-Mel-SNR metric for evaluating source separation quality.
    
    This metric calculates the Multi-Mel Signal-to-Noise Ratio
    across multiple mel-spectrogram configurations to measure the quality
    of separated audio sources.
    """
    
    def __init__(self, sample_rate=48000):
        """
        Initialize Multi-Mel-SNR metric.
        
        Args:
            sample_rate: Sample rate of the audio (default: 48000)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.multi_mel_snr_values = []
        
        # Mel configurations (transforms will be created on-the-fly with correct device)
        self.configs = [
            (512, 256, 80),  # (n_fft, hop_length, n_mels)
            (1024, 512, 128),
            (2048, 1024, 192)
        ]
        
    def update(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]):
        """
        Update metric with a batch of predictions and targets.
        
        Args:
            output_dict: Dictionary containing 'waveform' key with separated sources
                        [batch_size, n_sources, n_channels, n_samples]
            target_dict: Dictionary containing 'waveform' key with target sources
                        [batch_size, n_sources, n_channels, n_samples]
                        Optionally contains 'source_mask' key [batch_size, n_sources] 
                        to filter only active stems (1 = active, 0 = inactive)
        """
        output_wav = output_dict['waveform']  # [B, S, C, T]
        target_wav = target_dict['waveform']  # [B, S, C, T]
        B, S, C, T = output_wav.shape

        # Filter to only active stems if source_mask is provided
        source_mask = target_dict.get('source_mask', None)
        if source_mask is not None:
            # source_mask is [B, S] where 1 means present, 0 means not present
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
            
            # If we have any active stems, compute the metric
            if len(active_outputs) > 0:
                # Concatenate all active stems: [total_active, C, T]
                output_active = torch.cat(active_outputs, dim=0)  # [total_active, C, T]
                target_active = torch.cat(active_targets, dim=0)  # [total_active, C, T]
                
                # Reshape to [total_active*C, T] to process each channel independently
                output_flat = output_active.view(-1, T)  # [total_active*C, T]
                target_flat = target_active.view(-1, T)  # [total_active*C, T]
            else:
                # No active stems in this batch, skip
                return
        else:
            # No source_mask provided, process all stems
            output_flat = output_wav.view(B * S * C, T)  # [B*S*C, T]
            target_flat = target_wav.view(B * S * C, T)  # [B*S*C, T]

        # Compute Multi-Mel-SNR for each channel
        batch_multi_mel_snrs = self._compute_batch_multi_mel_snr(target_flat, output_flat)
        self.multi_mel_snr_values.extend(batch_multi_mel_snrs.cpu().tolist())
    
    def _compute_batch_multi_mel_snr(self, reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute Multi-Mel-SNR for a batch of reference and prediction pairs.
        
        Args:
            reference: [N, T] reference waveforms
            prediction: [N, T] prediction waveforms
        
        Returns:
            [N] tensor of Multi-Mel-SNR values
        """
        device = reference.device
        N, T = reference.shape
        
        # Check for NaN/inf in inputs and replace with zeros
        reference = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
        prediction = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))
        
        # Scale-invariant normalization per sample
        # alpha = <ref, pred> / <pred, pred>
        dot_ref_pred = (reference * prediction).sum(dim=1)  # [N]
        dot_pred_pred = (prediction * prediction).sum(dim=1) + 1e-8  # [N]
        alpha = dot_ref_pred / dot_pred_pred  # [N]
        # Check for NaN/inf in alpha and handle
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        prediction_scaled = alpha.unsqueeze(1) * prediction  # [N, T]
        
        # Compute SNR for each mel configuration and average
        all_snrs = []
        for n_fft, hop, n_mels in self.configs:
            # Create mel transform with correct device
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, f_min=0, f_max=24000, power=2.0
            ).to(device)
            
            # Compute mel spectrograms
            M_ref = mel_transform(reference)  # [N, n_mels, time_frames]
            M_pred = mel_transform(prediction_scaled)  # [N, n_mels, time_frames]
            
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
            # If ref_power is zero or very small, set SNR to a very negative value instead of -inf
            snr_ratio = ref_power / diff_power
            # Clamp to ensure we don't get log10(0) = -inf
            # snr_ratio = snr_ratio.clamp_min(1e-8)
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
        
        return avg_snr
    
    def compute(self) -> Dict[str, float]:
        """
        Compute the Multi-Mel-SNR metric.
        
        Returns:
            Dictionary containing Multi-Mel-SNR score
        """
        if not self.multi_mel_snr_values:
            return {"multi_mel_snr": 0.0}  # Return 0 instead of -inf when no data
        
        # Convert to tensor and filter out NaN/inf values
        values_tensor = torch.tensor(self.multi_mel_snr_values)
        finite_mask = torch.isfinite(values_tensor)
        
        if finite_mask.any():
            # Compute mean only over finite values
            avg_multi_mel_snr = values_tensor[finite_mask].mean().item()
            # Final safety check
            if not math.isfinite(avg_multi_mel_snr):
                avg_multi_mel_snr = -80.0  # Default to very negative SNR if still NaN/inf
        else:
            # All values are NaN/inf, return default
            avg_multi_mel_snr = -80.0
        
        return {"multi_mel_snr": avg_multi_mel_snr}
    
    def reset(self):
        """Reset the metric state."""
        self.multi_mel_snr_values = []


def get_multi_mel_snr_metric(sample_rate=48000):
    """
    Factory function to create Multi-Mel-SNR metric.
    
    Args:
        sample_rate: Sample rate of the audio (default: 48000)
    
    Returns:
        MultiMelSNRMetric instance
    """
    return MultiMelSNRMetric(sample_rate=sample_rate)


def get_metric_func(sample_rate=48000):
    """Get metric function that returns Multi-Mel-SNR for per-batch metrics."""

    def metric_func(output, target):
        output_wav = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]

        # Reshape to [B*S*C, T]
        B, S, C, T = output_wav.shape
        output_flat = output_wav.view(B * S * C, T)
        target_flat = target_wav.view(B * S * C, T)

        # Create a temporary metric instance for computation and move to correct device
        device = target_flat.device
        temp_metric = MultiMelSNRMetric(sample_rate=sample_rate).to(device)
        avg_multi_mel_snr = temp_metric._compute_batch_multi_mel_snr(target_flat, output_flat).mean()
        return {'multi_mel_snr': avg_multi_mel_snr.item()}
    
    return metric_func


if __name__ == "__main__":
    """
    Example usage of Multi-Mel-SNR metric.
    """
    print("Testing Multi-Mel-SNR metric...")
    
    # Create a metric instance
    metric = MultiMelSNRMetric(sample_rate=48000)
    
    # Create dummy data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    sample_rate = 48000
    
    # Generate test signals (sine waves with some noise)
    t = torch.linspace(0, 1.0, n_samples)
    
    # Create target (reference) signal
    target_wav = torch.zeros(batch_size, n_sources, n_channels, n_samples)
    for b in range(batch_size):
        for s in range(n_sources):
            for c in range(n_channels):
                freq = 440.0 * (s + 1)  # Different frequency for each source
                target_wav[b, s, c] = torch.sin(2 * torch.pi * freq * t) + 0.1 * torch.randn(n_samples)
    
    # Create prediction (output) signal with some noise added
    output_wav = target_wav + 0.2 * torch.randn_like(target_wav)
    
    output_dict = {'waveform': output_wav}
    target_dict = {'waveform': target_wav}
    
    # Test update
    metric.update(output_dict, target_dict)
    print(f"Multi-Mel-SNR values collected: {len(metric.multi_mel_snr_values)}")
    
    # Test compute
    scores = metric.compute()
    print(f"Multi-Mel-SNR score: {scores['multi_mel_snr']:.3f} dB")
    
    # Test reset
    metric.reset()
    print(f"After reset - Multi-Mel-SNR values: {len(metric.multi_mel_snr_values)}")
    
    # Test compute with no data
    scores = metric.compute()
    print(f"Multi-Mel-SNR score with no data: {scores['multi_mel_snr']:.3f} dB")
    
    # Test with the original function for comparison
    print("\nTesting original function for comparison...")
    ref_single = target_wav[0, 0, 0].numpy()
    pred_single = output_wav[0, 0, 0].numpy()
    original_score = multi_mel_snr(ref_single, pred_single, sr=sample_rate)
    print(f"Original function score: {original_score:.3f} dB")
    
    # Test get_metric_func
    print("\nTesting get_metric_func...")
    metric_func = get_metric_func(sample_rate=sample_rate)
    batch_score = metric_func(output_dict, target_dict)
    print(f"Per-batch metric score: {batch_score['multi_mel_snr']:.3f} dB")
    
    print("\n✅ All tests passed!")
