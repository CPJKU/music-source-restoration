"""Evaluation utilities for FinallyGAN training.

This module provides audio quality metrics (SI-SNR, SNR, SDR) and audio saving utilities
adapted from the latent flow matching training pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchaudio

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_si_snr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-noise ratio in dB.
    
    Parameters
    ----------
    reference : torch.Tensor
        Reference audio signal
    estimate : torch.Tensor
        Estimated audio signal
    eps : float, optional
        Small constant for numerical stability
        
    Returns
    -------
    float
        SI-SNR value in dB
    """
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    ref_energy = torch.sum(ref ** 2)
    if ref_energy < eps:
        return float("nan")
    proj = torch.sum(est * ref) * ref / (ref_energy + eps)
    noise = est - proj
    ratio = torch.sum(proj ** 2) / (torch.sum(noise ** 2) + eps)
    return float(10.0 * torch.log10(ratio + eps))


def compute_snr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Signal-to-noise ratio in dB.
    
    Parameters
    ----------
    reference : torch.Tensor
        Reference audio signal
    estimate : torch.Tensor
        Estimated audio signal
    eps : float, optional
        Small constant for numerical stability
        
    Returns
    -------
    float
        SNR value in dB
    """
    signal_power = torch.sum(reference ** 2)
    if signal_power < eps:
        return float("nan")
    noise = estimate - reference
    noise_power = torch.sum(noise ** 2)
    ratio = signal_power / (noise_power + eps)
    return float(10.0 * torch.log10(ratio + eps))


def compute_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Signal-to-distortion ratio in dB.
    
    Parameters
    ----------
    reference : torch.Tensor
        Reference audio signal
    estimate : torch.Tensor
        Estimated audio signal
    eps : float, optional
        Small constant for numerical stability
        
    Returns
    -------
    float
        SDR value in dB
    """
    signal_power = torch.sum(reference ** 2)
    if signal_power < eps:
        return float("nan")
    distortion = reference - estimate
    distortion_power = torch.sum(distortion ** 2)
    ratio = signal_power / (distortion_power + eps)
    return float(10.0 * torch.log10(ratio + eps))


def compute_multi_mel_snr(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int = 48000, eps: float = 1e-8) -> float:
    """Multi-Mel-SNR using scale-invariant projection and multiple mel configurations.
    
    Parameters
    ----------
    reference : torch.Tensor
        Reference audio signal (1D tensor)
    estimate : torch.Tensor
        Estimated audio signal (1D tensor)
    sample_rate : int
        Sample rate in Hz
    eps : float, optional
        Small constant for numerical stability
    
    Returns
    -------
    float
        Multi-Mel-SNR in dB (average across 3 mel configurations)
    """
    # Scale-invariant normalization
    alpha = torch.dot(reference, estimate) / (torch.dot(estimate, estimate) + eps)
    estimate_scaled = alpha * estimate
    
    # Three mel configurations: (n_fft, hop_length, n_mels)
    configs = [
        (512, 256, 80),
        (1024, 512, 128),
        (2048, 1024, 192)
    ]
    
    snrs = []
    for n_fft, hop_length, n_mels in configs:
        try:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0,
                f_max=sample_rate // 2,
                power=2.0
            )
            M_ref = mel_transform(reference)
            M_est = mel_transform(estimate_scaled)
            
            snr = 10.0 * torch.log10(M_ref.pow(2).sum() / ((M_ref - M_est).pow(2).sum() + eps))
            snrs.append(snr.item())
        except Exception:
            # Skip this config if it fails (e.g., audio too short for n_fft)
            continue
    
    if not snrs:
        return float("nan")
    
    return float(np.mean(snrs))


def save_audio_file(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio waveform to file.
    
    Parameters
    ----------
    path : Path
        Output file path
    waveform : torch.Tensor
        Audio waveform tensor
    sample_rate : int
        Sample rate in Hz
    """
    waveform = waveform.detach().cpu().float()
    
    # Ensure shape is [T, C] or [T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:  # [C, T]
        waveform = waveform.T
    
    # Try soundfile first (best quality)
    if HAS_SOUNDFILE:
        try:
            sf.write(str(path), waveform.numpy(), sample_rate)
            return
        except Exception:
            pass
    
    # Fallback to scipy wavfile
    if HAS_SCIPY:
        try:
            wav = np.clip(waveform.numpy(), -1.0, 1.0)
            wavfile.write(str(path), sample_rate, (wav * 32767).astype(np.int16))
            return
        except Exception as exc:
            print(f"[save_audio] Failed to save {path}: {exc}")
    
    print(f"[save_audio] No audio writing library available (soundfile or scipy)")


def evaluate_audio_batch(
    clean_audio: torch.Tensor,
    restored_audio: torch.Tensor,
    degraded_audio: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute audio quality metrics for a batch.
    
    Parameters
    ----------
    clean_audio : torch.Tensor
        Clean reference audio [B, C, T]
    restored_audio : torch.Tensor
        Restored audio [B, C, T]
    degraded_audio : torch.Tensor, optional
        Degraded input audio [B, C, T]
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names to mean values across batch
    """
    metrics: Dict[str, list[float]] = {
        "si_snr_restored": [],
        "snr_restored": [],
        "sdr_restored": [],
        "multi_mel_snr_restored": [],
    }
    
    if degraded_audio is not None:
        metrics["si_snr_input"] = []
        metrics["snr_input"] = []
        metrics["sdr_input"] = []
        metrics["multi_mel_snr_input"] = []
    
    batch_size = clean_audio.size(0)
    # Get sample rate from tensor metadata or use default
    sample_rate = 48000  # Default for this project
    
    for idx in range(batch_size):
        clean = clean_audio[idx].view(-1)
        restored = restored_audio[idx].view(-1)
        
        # Align lengths
        length = min(clean.numel(), restored.numel())
        if length > 0:
            metrics["si_snr_restored"].append(compute_si_snr(clean[:length], restored[:length]))
            metrics["snr_restored"].append(compute_snr(clean[:length], restored[:length]))
            metrics["sdr_restored"].append(compute_sdr(clean[:length], restored[:length]))
            metrics["multi_mel_snr_restored"].append(compute_multi_mel_snr(clean[:length], restored[:length], sample_rate))
        
        if degraded_audio is not None:
            degraded = degraded_audio[idx].view(-1)
            length = min(clean.numel(), degraded.numel())
            if length > 0:
                metrics["si_snr_input"].append(compute_si_snr(clean[:length], degraded[:length]))
                metrics["snr_input"].append(compute_snr(clean[:length], degraded[:length]))
                metrics["sdr_input"].append(compute_sdr(clean[:length], degraded[:length]))
                metrics["multi_mel_snr_input"].append(compute_multi_mel_snr(clean[:length], degraded[:length], sample_rate))
    
    # Compute means
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = float(np.nanmean(values))
    
    return result


def save_validation_samples(
    step: int,
    output_dir: Path,
    clean_audio: torch.Tensor,
    restored_audio: torch.Tensor,
    degraded_audio: torch.Tensor,
    sample_rate: int,
    num_samples: int = 4,
    use_wandb: bool = False,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Save validation audio samples and compute metrics.
    
    Parameters
    ----------
    step : int
        Training step number
    output_dir : Path
        Base output directory
    clean_audio : torch.Tensor
        Clean reference audio [B, C, T]
    restored_audio : torch.Tensor
        Restored audio [B, C, T]
    degraded_audio : torch.Tensor
        Degraded input audio [B, C, T]
    sample_rate : int
        Sample rate in Hz
    num_samples : int, optional
        Number of samples to save
    use_wandb : bool, optional
        Whether to create W&B audio objects
        
    Returns
    -------
    Tuple[Dict[str, float], Dict[str, object]]
        (metrics_dict, wandb_audio_dict)
    """
    save_dir = output_dir / f"step_{step:07d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = evaluate_audio_batch(clean_audio, restored_audio, degraded_audio)
    
    # Save audio files
    wandb_audio: Dict[str, object] = {}
    count = min(num_samples, clean_audio.size(0))
    
    for idx in range(count):
        sample_id = f"sample{idx}"
        
        # Save all versions
        for tag, audio in [
            ("clean", clean_audio[idx]),
            ("restored", restored_audio[idx]),
            ("degraded", degraded_audio[idx]),
        ]:
            audio_path = save_dir / f"{sample_id}_{tag}.wav"
            save_audio_file(audio_path, audio, sample_rate)
            
            # Create W&B audio object
            if use_wandb and HAS_WANDB and wandb.run is not None:
                try:
                    key = f"audio/{sample_id}_{tag}"
                    wandb_audio[key] = wandb.Audio(
                        str(audio_path),
                        caption=f"{sample_id}_{tag}",
                        sample_rate=sample_rate,
                    )
                except Exception as exc:
                    print(f"[wandb_audio] Failed to create audio object for {sample_id}_{tag}: {exc}")
    
    return metrics, wandb_audio


__all__ = [
    "compute_si_snr",
    "compute_snr",
    "compute_sdr",
    "compute_multi_mel_snr",
    "save_audio_file",
    "evaluate_audio_batch",
    "save_validation_samples",
]
