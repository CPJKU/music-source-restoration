"""
Torch-based offline mastering chain built from torch_audiomentations transforms.

Pipeline stages:
1) Linear-phase high-pass to clean sub frequencies.
2) Loudness normalization to a target LUFS (multiple backends supported).
3) True-peak limiting with oversampling and lookahead.
4) Optional mid/side width adjustment.

All transforms operate on tensors shaped (batch, channels, time).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from .music_effect_pipeline import Compressor

try:  # pragma: no cover - optional dependency
    from torch_audiomentations import Compose
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Compose = None  # type: ignore[assignment]


class LoudnessNormalize(nn.Module):
    """
    Normalize integrated loudness to a target LUFS value.

    backends:
        - 'torchaudio': uses torchaudio.functional.loudness (GPU-capable).
        - 'pyloudnorm': converts to numpy and uses pyloudnorm.Meter.
        - 'loudness_py': delegates to the Brecht De Man reference implementation.
        - 'auto': pick the first available backend in the order above.
    """

    def __init__(self, target_lufs: float = -14.0, backend: str = "auto", sample_rate: int = 48_000) -> None:
        super().__init__()
        self.target_lufs = float(target_lufs)
        self.backend = backend
        self.sample_rate = int(sample_rate)

    @torch.no_grad()
    def _lufs_torchaudio(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        import torchaudio.functional as taf

        # original_device = samples.device
        # original_dtype = samples.dtype
        # loudness_input = samples.detach().to(device="cpu", dtype=torch.float32)
        # loudness = taf.loudness(loudness_input, sample_rate=sample_rate)
        # if loudness.ndim == 0:
        #     loudness = loudness.unsqueeze(0)
        # return loudness.to(device=original_device, dtype=original_dtype)

        return taf.loudness(samples, sample_rate=sample_rate)

    @torch.no_grad()
    def _lufs_pyloudnorm(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        import pyloudnorm as pyln

        batch, _, _ = samples.shape
        values = []
        meter = pyln.Meter(sample_rate)
        for idx in range(batch):
            data = samples[idx].T.detach().cpu().numpy()
            values.append(float(meter.integrated_loudness(data)))
        return torch.tensor(values, dtype=samples.dtype, device=samples.device)

    @torch.no_grad()
    def _lufs_loudness_py(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            import loudness as ldn  # type: ignore[import]
        except Exception:  # pragma: no cover - best effort import name
            from loudness import loudness as ldn  # type: ignore[import]

        batch, _, _ = samples.shape
        values = []
        for idx in range(batch):
            data = samples[idx].T.detach().cpu().numpy()
            values.append(float(ldn.loudness(data, sample_rate)))
        return torch.tensor(values, dtype=samples.dtype, device=samples.device)

    def _select_backend(self) -> str:
        backend = self.backend
        if backend != "auto":
            return backend

        try:
            import torchaudio.functional  # noqa: F401

            return "torchaudio"
        except Exception:  # pragma: no cover - optional dependency
            pass

        try:
            import pyloudnorm  # noqa: F401

            return "pyloudnorm"
        except Exception:  # pragma: no cover - optional dependency
            pass

        return "loudness_py"

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")

        backend = self._select_backend()
        if backend == "torchaudio":
            lufs = self._lufs_torchaudio(samples, self.sample_rate)
        elif backend == "pyloudnorm":
            lufs = self._lufs_pyloudnorm(samples, self.sample_rate)
        else:
            lufs = self._lufs_loudness_py(samples, self.sample_rate)

        # Handle non-finite loudness values (NaN, inf, -inf) e.g. silent inputs
        mask = ~torch.isfinite(lufs)
        if mask.any():
            lufs = lufs.clone()
            lufs[mask] = float(self.target_lufs)

        gain_db = self.target_lufs - lufs
        gain = (10.0 ** (gain_db / 20.0)).view(-1, 1, 1)
        return samples * gain


class LinearPhaseHPF(nn.Module):
    """Linear-phase high-pass FIR filter implemented with reflection padding."""

    def __init__(self, cutoff_hz: float = 25.0, taps: int = 1025, sample_rate: int = 48_000) -> None:
        super().__init__()
        if taps % 2 == 0:
            raise ValueError("taps must be odd for linear-phase response")
        self.cutoff_hz = float(cutoff_hz)
        self.taps = int(taps)
        self.sample_rate = int(sample_rate)
        self.register_buffer("_kernel", torch.zeros(1), persistent=False)
        self._kernel_sr: Optional[int] = None

    def _design_kernel(self, sample_rate: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._kernel_sr == sample_rate and self._kernel.numel() == self.taps:
            return self._kernel

        fc = self.cutoff_hz / sample_rate
        n = torch.arange(self.taps, device=device, dtype=dtype) - (self.taps - 1) / 2
        h_lp = torch.where(
            n == 0,
            torch.tensor(2 * fc, device=device, dtype=dtype),
            torch.sin(2 * math.pi * fc * n) / (math.pi * n),
        )
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * (n + (self.taps - 1) / 2) / (self.taps - 1))
        h_lp = h_lp * window
        h_lp = h_lp / h_lp.sum().clamp(min=1e-12)

        delta = torch.zeros_like(h_lp)
        delta[(self.taps - 1) // 2] = 1.0
        h_hp = delta - h_lp

        kernel = h_hp.view(1, 1, self.taps)
        self._kernel = kernel.to(device=device, dtype=dtype)
        self._kernel_sr = sample_rate
        return self._kernel

    def forward(self, samples: Tensor) -> Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")

        sample_rate = self.sample_rate
        batch, channels, frames = samples.shape
        kernel = self._design_kernel(sample_rate, samples.device, samples.dtype)
        pad = (self.taps - 1) // 2
        x = samples.view(batch * channels, 1, frames)
        x = F.pad(x, (pad, pad), mode="reflect")
        y = F.conv1d(x, kernel, stride=1, padding=0, groups=1)
        return y.view(batch, channels, frames)


class MidSideWidth(nn.Module):
    """Mid/side stereo width adjustment. Mono inputs are returned unchanged."""

    def __init__(self, width_db: float = 0.0) -> None:
        super().__init__()
        self.width_db = float(width_db)

    def forward(self, samples: Tensor, sample_rate: Optional[int] = None) -> Tensor:  # noqa: D401 - signature compatibility
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")
        if samples.size(1) < 2:
            return samples

        gain = 10.0 ** (self.width_db / 20.0)
        left = samples[:, 0:1, :]
        right = samples[:, 1:2, :]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * gain
        left_out = mid + side
        right_out = mid - side
        if samples.size(1) > 2:
            return torch.cat([left_out, right_out, samples[:, 2:, :]], dim=1)
        return torch.cat([left_out, right_out], dim=1)


class TruePeakLimiter(nn.Module):
    """Lookahead limiter that respects a true-peak ceiling via oversampling."""

    def __init__(
        self,
        tp_db: float = -1.0,
        oversample: int = 4,
        sample_rate: int = 48_000,
        lookahead_ms: float = 1.0,
        release_ms: float = 50.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if oversample < 2:
            raise ValueError("oversample must be >= 2")
        self.tp_db = float(tp_db)
        self.oversample = int(oversample)
        self.sample_rate = int(sample_rate)
        self.lookahead_ms = float(lookahead_ms)
        self.release_ms = float(release_ms)
        self.eps = float(eps)

    def _resampler(self, src: int, dst: int, device: torch.device, dtype: torch.dtype) -> torchaudio.transforms.Resample:
        return torchaudio.transforms.Resample(src, dst).to(device=device, dtype=dtype)

    def forward(self, samples: Tensor) -> Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")

        sample_rate = self.sample_rate
        batch, channels, frames = samples.shape
        device, dtype = samples.device, samples.dtype
        up_rate = sample_rate * self.oversample

        up = self._resampler(sample_rate, up_rate, device, dtype)
        down = self._resampler(up_rate, sample_rate, device, dtype)

        reshaped = samples.reshape(batch * channels, 1, frames)
        upsampled = up(reshaped)
        up_frames = upsampled.size(-1)

        magnitude = upsampled.abs()
        lookahead = max(1, int(self.lookahead_ms * 1e-3 * up_rate))
        envelope = F.max_pool1d(magnitude, kernel_size=lookahead, stride=1, padding=lookahead // 2)

        threshold = 10.0 ** (self.tp_db / 20.0)
        gain = torch.clamp(threshold / (envelope + self.eps), max=1.0)

        release = max(1, int(self.release_ms * 1e-3 * up_rate))
        kernel = torch.ones(1, 1, release, device=device, dtype=dtype) / release
        gain = F.pad(gain, (release - 1, 0), mode="replicate")
        gain = F.conv1d(gain, kernel)

        if gain.size(-1) != up_frames:
            gain = F.pad(gain, (0, up_frames - gain.size(-1))) if gain.size(-1) < up_frames else gain[..., :up_frames]
        limited = upsampled * gain

        restored = down(limited)
        restored = restored.view(batch, channels, -1)
        if restored.size(-1) != frames:
            restored = (
                F.pad(restored, (0, frames - restored.size(-1)))
                if restored.size(-1) < frames
                else restored[..., :frames]
            )
        return restored


class PeakNormalizer(nn.Module):
    """Scale audio so the absolute peak does not exceed a threshold."""

    def __init__(self, ceiling: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        if ceiling <= 0:
            raise ValueError("ceiling must be positive")
        self.ceiling = float(ceiling)
        self.eps = float(eps)

    def forward(self, samples: Tensor) -> Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")
        peak = samples.abs().amax(dim=(1, 2), keepdim=True)
        if torch.all(peak <= self.ceiling + self.eps):
            return samples
        gain = torch.clamp(self.ceiling / (peak + self.eps), max=1.0)
        return samples * gain


def build_mastering_compose(
    sample_rate: int,
    target_lufs: float = -14.0,
    tp_db: float = -1.0,
    hpf_hz: float = 25.0,
    hpf_taps: int = 1025,
    width_db: float = 0.0,
    oversample: int = 4,
    lookahead_ms: float = 1.0,
    release_ms: float = 50.0,
    peak_ceiling: float = 1.0,
) -> Compose:
    """Create a torch_audiomentations.Compose mastering chain."""

    if Compose is None:  # pragma: no cover - runtime guard
        raise RuntimeError("torch_audiomentations is required for build_mastering_compose but is not installed")

    transforms = [
        LinearPhaseHPF(cutoff_hz=hpf_hz, taps=hpf_taps, sample_rate=sample_rate),
        Compressor(sample_rate=sample_rate, p=0.1),
        LoudnessNormalize(target_lufs=target_lufs, sample_rate=sample_rate, backend="auto"),
        TruePeakLimiter(
            tp_db=tp_db,
            oversample=oversample,
            sample_rate=sample_rate,
            lookahead_ms=lookahead_ms,
            release_ms=release_ms,
        ),
    ]
    if abs(width_db) > 1e-9:
        transforms.append(MidSideWidth(width_db=width_db))

    transforms.append(PeakNormalizer(ceiling=peak_ceiling))

    return Compose(transforms=transforms, shuffle=False, output_type="tensor")


__all__ = [
    "LoudnessNormalize",
    "LinearPhaseHPF",
    "MidSideWidth",
    "TruePeakLimiter",
    "PeakNormalizer",
    "build_mastering_compose",
]
