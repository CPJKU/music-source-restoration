"""LUFS and loudness utilities with optional pyloudnorm dependency."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyloudnorm as pyln
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyln = None


DEFAULT_BLOCK_SIZE = 0.400  # seconds


@dataclass
class LoudnessMeasurement:
    """Container for loudness related metrics."""

    integrated_lufs: float
    loudness_range: float
    short_term_lufs: Optional[np.ndarray] = None

    def to_json(self) -> str:
        """Return a stable JSON representation for metadata storage."""
        payload: Dict[str, float | Sequence[float]] = {
            "integrated_lufs": float(self.integrated_lufs),
            "loudness_range": float(self.loudness_range),
        }
        if self.short_term_lufs is not None:
            payload["short_term_lufs"] = [float(v) for v in self.short_term_lufs]
        return json.dumps(payload, sort_keys=True)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.shape[-1] > arr.shape[0]:
        # expecting (time, channels)
        arr = arr.T
    return arr


def measure_lufs(audio: np.ndarray, sample_rate: int, block_size: float = DEFAULT_BLOCK_SIZE) -> LoudnessMeasurement:
    """Measure integrated LUFS, loudness range, and optional short-term profile."""
    frames = _to_float32(audio)

    if pyln is not None:
        meter = pyln.Meter(sample_rate, block_size=block_size, filter_class="K-weighting")
        integrated = float(meter.integrated_loudness(frames))
        lra = float(meter.loudness_range(frames))
        short_term = np.array(meter.short_term_loudness(frames), dtype=np.float32)
        return LoudnessMeasurement(integrated_lufs=integrated, loudness_range=lra, short_term_lufs=short_term)

    # Fallback: approximate LUFS via RMS in LU weighting (simplified).
    rms = np.sqrt(np.mean(np.square(frames), axis=0)).mean()
    rms_db = -np.inf if rms == 0 else 20.0 * np.log10(rms)
    # Heuristic offsets so downstream normalization keeps operating range similar.
    integrated = rms_db - 0.691
    return LoudnessMeasurement(integrated_lufs=integrated, loudness_range=0.0, short_term_lufs=None)


def true_peak_db(audio: np.ndarray) -> float:
    """Approximate the true peak (dBFS) using 4x oversampling."""
    frames = _to_float32(audio).astype(np.float64)
    if frames.size == 0:
        return -np.inf

    upsample_factor = 4
    # Zero-order hold oversampling for simplicity when scipy is absent.
    expanded = np.repeat(frames, upsample_factor, axis=0)
    peak = np.max(np.abs(expanded), initial=0.0)
    if peak <= 0.0:
        return -np.inf
    return 20.0 * np.log10(peak)


def normalize_to_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float) -> tuple[np.ndarray, float]:
    """Normalize audio to a target LUFS, returning adjusted audio and applied gain (dB)."""
    measurement = measure_lufs(audio, sample_rate)
    current = measurement.integrated_lufs
    if not np.isfinite(current):
        return np.asarray(audio, dtype=np.float32), 0.0

    gain_db = float(target_lufs - current)
    gain = 10.0 ** (gain_db / 20.0)
    normalized = np.asarray(audio, dtype=np.float32) * gain
    return normalized, gain_db


__all__ = [
    "LoudnessMeasurement",
    "measure_lufs",
    "normalize_to_lufs",
    "true_peak_db",
]
