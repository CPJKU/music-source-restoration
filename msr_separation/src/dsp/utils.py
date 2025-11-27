"""Generic DSP helpers."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import librosa
import torch
import torchaudio.functional as taF
from .lufs import measure_lufs


def ensure_sample_rate(audio: np.ndarray, src_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Return audio resampled to the desired sample rate if needed."""
    if src_sr == target_sr:
        return np.asarray(audio, dtype=np.float32), src_sr
    if librosa is None:  # pragma: no cover - runtime guard
        raise RuntimeError("librosa is required for resampling but is not installed")
    resampled = librosa.resample(np.asarray(audio, dtype=np.float32).T, orig_sr=src_sr, target_sr=target_sr, res_type="kaiser_best")
    return resampled.T.astype(np.float32), target_sr


def fade_edges(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply linear fade in/out to avoid clicks when segmenting."""
    frames = np.asarray(audio, dtype=np.float32)
    if fade_samples <= 0:
        return frames
    if frames.ndim == 1:
        frames = frames[:, np.newaxis]

    fade_samples = min(fade_samples, frames.shape[0] // 2)
    if fade_samples == 0:
        return frames

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = fade_in[::-1]
    frames[:fade_samples] *= fade_in[:, np.newaxis]
    frames[-fade_samples:] *= fade_out[:, np.newaxis]
    return frames


def batch_has_signal_lufs(
    audio_batch: "torch.Tensor",
    sample_rate: int,
    *,
    min_lufs: float = -70.0,
    min_peak: float = 1e-4,
) -> "torch.Tensor":
    """
    Return a boolean mask indicating which items in a batch contain sufficient signal.

    Parameters
    ----------
    audio_batch:
        Tensor shaped (batch, channels, frames) holding audio in -1..1 range.
    sample_rate:
        Sample rate associated with the waveforms.
    min_lufs:
        Minimum integrated LUFS required to consider the clip non-silent.
    min_peak:
        Absolute peak threshold used as a quick guard for numerical silence.

    Returns
    -------
    torch.Tensor:
        Boolean tensor of shape (batch,) where True indicates sufficient signal.
    """
    if torch is None:  # pragma: no cover - runtime guard
        raise RuntimeError("PyTorch is required for batch_has_signal_lufs but is not installed")
    if audio_batch.ndim != 3:
        raise ValueError(f"Expected audio tensor of shape (B, C, L); received {audio_batch.shape}")

    batch_size = audio_batch.size(0)
    if batch_size == 0:
        return torch.zeros(0, dtype=torch.bool, device=audio_batch.device)
    if audio_batch.size(-1) == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=audio_batch.device)

    if taF is None:  # pragma: no cover - optional dependency fallback
        return _batch_has_signal_lufs_numpy(audio_batch, sample_rate, min_lufs=min_lufs, min_peak=min_peak)

    waveforms = audio_batch.detach().to(dtype=torch.float32)
    peaks = waveforms.abs().flatten(1).amax(dim=1)
    peak_mask = peaks >= float(min_peak)

    signal_mask = torch.zeros(batch_size, dtype=torch.bool, device=audio_batch.device)
    if peak_mask.any():
        loudness_input = waveforms[peak_mask].contiguous() # waveforms[peak_mask].contiguous().to(device="cpu", dtype=torch.float32)
        loudness = taF.loudness(loudness_input, sample_rate)
        if loudness.ndim == 0:
            loudness = loudness.unsqueeze(0)
        # loudness = loudness.to(signal_mask.device)
        signal_mask[peak_mask] = torch.isfinite(loudness) & (loudness > float(min_lufs))

    return signal_mask


def _batch_has_signal_lufs_numpy(
    audio_batch: "torch.Tensor",
    sample_rate: int,
    *,
    min_lufs: float,
    min_peak: float,
) -> "torch.Tensor":
    original_device = audio_batch.device
    audio_cpu = audio_batch.detach().to("cpu")

    mask_values: list[bool] = []
    for clip in audio_cpu:
        clip_np = clip.numpy().astype(np.float32, copy=False)
        peak = float(np.max(np.abs(clip_np))) if clip_np.size else 0.0

        if peak < float(min_peak):
            mask_values.append(False)
            continue

        measurement = measure_lufs(clip_np, sample_rate)
        integrated = measurement.integrated_lufs
        has_signal = np.isfinite(integrated) and integrated > float(min_lufs)
        mask_values.append(has_signal)

    return torch.tensor(mask_values, dtype=torch.bool, device=original_device)


__all__ = ["ensure_sample_rate", "fade_edges", "batch_has_signal_lufs"]
