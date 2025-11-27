"""
SI-SNR (Scale-Invariant Signal-to-Noise Ratio) metric implementation.

This module provides SI-SNR evaluation for source separation models.
"""

import torch
import torch.nn as nn
from typing import Dict
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr


class SI_SNRMetric(nn.Module):
    """
    SI-SNR metric for evaluating source separation quality.
    
    This metric calculates the Scale-Invariant Signal-to-Noise Ratio
    to measure the quality of separated audio sources.
    """
    
    def __init__(self):
        super().__init__()
        self.si_snr_values = []
        
    def update(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]):
        """
        Update metric with a batch of predictions and targets.
        
        Args:
            output_dict: Dictionary containing 'waveform' key with separated sources
                        [batch_size, n_sources, n_channels, n_samples]
            target_dict: Dictionary containing 'waveform' key with target sources
                        [batch_size, n_sources, n_channels, n_samples]
        """
        output_wav = output_dict['waveform']  # [B, S, C, T]
        target_wav = target_dict['waveform']  # [B, S, C, T]

        # 🔥 VECTORIZED: Reshape to [B*S, C, T] → single si_snr call
        B, S, C, T = output_wav.shape
        output_flat = output_wav.view(B * S, C, T)  # [B*S, C, T]
        target_flat = target_wav.view(B * S, C, T)  # [B*S, C, T]

        # One call! Returns [B*S] (auto channel avg)
        batch_si_snrs = si_snr(output_flat, target_flat)  # [B*S]
        self.si_snr_values.extend(batch_si_snrs.cpu().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute the SI-SNR metric.
        
        Returns:
            Dictionary containing SI-SNR score
        """
        if not self.si_snr_values:
            return {"si_snr": 0.0}  # Return 0 instead of -inf when no data
        
        avg_si_snr = torch.tensor(self.si_snr_values).mean().item()
        return {"si_snr": avg_si_snr}
    
    def reset(self):
        """Reset the metric state."""
        self.si_snr_values = []


def get_si_snr_metric():
    """
    Factory function to create SI-SNR metric.
    
    Returns:
        SI_SNRMetric instance
    """
    return SI_SNRMetric()


def get_metric_func():
    """Get metric function that returns SI-SNR for per-batch metrics."""

    def metric_func(output, target):
        output_wav = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]

        # 🔥 VECTORIZED: Same trick!
        B, S, C, T = output_wav.shape
        output_flat = output_wav.view(B * S, C, T)
        target_flat = target_wav.view(B * S, C, T)

        avg_si_snr = si_snr(output_flat, target_flat).mean()
        return {'si_snr': avg_si_snr}
    
    return metric_func
