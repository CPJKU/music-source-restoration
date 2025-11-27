"""
Diffusion-based music restoration utilities.

This module provides:
- STFT helpers that operate on (batch, channels, time) tensors.
- Representation transforms between complex spectrograms and a log-magnitude /
  instantaneous-frequency factorisation.
- A lightweight multi-band decomposition (PQMF-style) to focus diffusion
  modelling on spectral regions.
- Dataset wrapper that prepares 10 second stereo crops from SonicMaster entries.
- Diffusion model wrapper around diffusers.UNet2DModel with optional band
  conditioning.
- Training and inference pipelines with per-step consistency projection and a
  CombinedMusicLoss auxiliary objective.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

from restoration.sonicmaster_finally_gan.utils import get_cosine_schedule_with_warmup
# from pure.train.loss import CombinedMusicLoss

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------


@dataclass
class STFTCfg:
    """Configuration for STFT / iSTFT operations."""

    n_fft: int
    hop_length: int
    win_length: Optional[int] = None
    center: bool = True
    pad_mode: str = "reflect"
    eps: float = 1e-7

    def window(self, device: torch.device) -> Tensor:
        """Return Hann window on the requested device."""
        size = self.win_length or self.n_fft
        return torch.hann_window(size, periodic=True, device=device)


def _flatten_channels(wav: Tensor) -> Tuple[Tensor, int, int]:
    """Reshape (B, C, T) -> (B*C, T) while remembering sizes."""
    if wav.dim() != 3:
        raise ValueError(f"Expected input with shape [B, C, T], got {wav.shape}")
    b, c, t = wav.shape
    return wav.reshape(b * c, t), b, c


def stft(wav: Tensor, cfg: STFTCfg) -> Tensor:
    """Compute complex STFT for a batch of multi-channel waveforms."""
    flat, batch, channels = _flatten_channels(wav)
    window = cfg.window(flat.device)
    spec = torch.stft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length or cfg.n_fft,
        window=window,
        center=cfg.center,
        pad_mode=cfg.pad_mode,
        return_complex=True,
    )
    freq, frames = spec.shape[-2:]
    spec = spec.view(batch, channels, freq, frames)
    return spec


def istft(spec: Tensor, cfg: STFTCfg, length: Optional[int] = None) -> Tensor:
    """Inverse STFT that restores (batch, channels, time)."""
    if spec.dim() != 4:
        raise ValueError(f"Expected spectrogram with shape [B, C, F, K], got {spec.shape}")
    batch, channels, _, _ = spec.shape
    window = cfg.window(spec.device)
    flat = spec.view(batch * channels, spec.shape[-2], spec.shape[-1])
    wav = torch.istft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length or cfg.n_fft,
        window=window,
        length=length,
        center=cfg.center,
    )
    wav = wav.view(batch, channels, -1)
    return wav


def wrap_phase(delta: Tensor) -> Tensor:
    """Wrap phase differences into [-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def pad_to_multiple(tensor: Tensor, multiple: int = 8, dim: int = -1) -> Tuple[Tensor, int]:
    """Pad tensor along specified dimension to be divisible by multiple.
    
    Returns:
        Tuple of (padded_tensor, original_size)
    """
    size = tensor.shape[dim]
    remainder = size % multiple
    if remainder == 0:
        return tensor, size
    
    pad_size = multiple - remainder
    # Create padding specification (last dim first in pad specification)
    padding = [0, 0] * tensor.ndim
    # Convert dim to positive index
    pos_dim = dim if dim >= 0 else tensor.ndim + dim
    # Pad on the right side of the specified dimension
    padding[2 * (tensor.ndim - pos_dim - 1) + 1] = pad_size
    
    padded = torch.nn.functional.pad(tensor, padding)
    return padded, size


def unpad_to_size(tensor: Tensor, size: int, dim: int = -1) -> Tensor:
    """Remove padding added by pad_to_multiple."""
    indices = [slice(None)] * tensor.ndim
    indices[dim] = slice(0, size)
    return tensor[tuple(indices)]


def complex_to_rep(spec: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Convert complex STFT to real-valued representation: [log|X|, IF_t, IF_f].

    Args:
        spec: Complex tensor [B, C, F, K].
        eps: Clamp to avoid log(0).

    Returns:
        Real tensor [B, 3*C, F, K].
    """
    if not torch.is_complex(spec):
        raise ValueError("complex_to_rep expects a complex STFT tensor.")
    mag = spec.abs().clamp_min(eps)
    logmag = torch.log(mag)
    phase = torch.angle(spec)

    if_t = torch.zeros_like(phase)
    if_t[..., 1:] = wrap_phase(phase[..., 1:] - phase[..., :-1])

    if_f = torch.zeros_like(phase)
    if_f[:, :, 1:, :] = wrap_phase(phase[:, :, 1:, :] - phase[:, :, :-1, :])

    rep = torch.cat([logmag, if_t, if_f], dim=1)
    return rep


def rep_to_complex(rep: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Invert representation generated by complex_to_rep.

    Args:
        rep: Real tensor [B, 3*C, F, K].
        eps: Clamp magnitude for stability.

    Returns:
        Complex STFT tensor [B, C, F, K].
    """
    if rep.dim() != 4 or rep.shape[1] % 3 != 0:
        raise ValueError(f"rep_to_complex expects channel dimension divisible by 3; got {rep.shape}")
    channels = rep.shape[1] // 3
    logmag, if_t, if_f = torch.chunk(rep, 3, dim=1)
    # Clamp logmag to prevent extremely large magnitudes that cause NaN
    logmag = logmag.clamp(-10.0, 10.0)
    mag = logmag.exp().clamp_min(eps)

    phase = torch.cumsum(if_t, dim=-1)
    phase = phase + torch.cumsum(if_f, dim=2)
    complex_spec = torch.polar(mag, phase)
    return complex_spec


def project_consistency(spec: Tensor, cfg: STFTCfg, length: Optional[int] = None) -> Tensor:
    """Project arbitrary STFT to the consistent manifold via iSTFT -> STFT."""
    wav = istft(spec, cfg, length=length)
    proj = stft(wav, cfg)
    return proj


# ---------------------------------------------------------------------------
# Light-weight PQMF implementation (frequency mask splitter)
# ---------------------------------------------------------------------------


class PQMF(nn.Module):
    """
    Lightweight analysis/synthesis filter bank.

    The implementation uses frequency-domain masking for perfect reconstruction.
    Each band uses mutually exclusive frequency bins, so synthesis simply sums
    the sub-band waveforms.
    """

    def __init__(
        self,
        bands: int = 8,
        sample_rate: int = 48000,
        min_hz: float = 20.0,
    ) -> None:
        super().__init__()
        if bands < 2:
            raise ValueError("PQMF requires at least two bands.")
        self.bands = bands
        self.sample_rate = sample_rate
        nyquist = sample_rate * 0.5
        # Log-spaced edges with gentle widening for high frequencies.
        edges = np.geomspace(max(min_hz, 1.0), nyquist, bands + 1, dtype=np.float32)
        edges[0] = 0.0
        edges[-1] = nyquist
        self.register_buffer("band_edges", torch.tensor(edges, dtype=torch.float32), persistent=False)

    def _frequency_masks(self, fft_size: int, device: torch.device) -> List[Tensor]:
        freqs = torch.linspace(0.0, self.sample_rate * 0.5, steps=fft_size, device=device)
        masks: List[Tensor] = []
        for b in range(self.bands):
            low = self.band_edges[b]
            high = self.band_edges[b + 1]
            mask = (freqs >= low) & (freqs < high if b < self.bands - 1 else freqs <= high)
            masks.append(mask.to(dtype=torch.float32))
        # Ensure partition of unity
        total = torch.stack(masks, dim=0).sum(dim=0).clamp_min(1.0)
        masks = [m / total for m in masks]
        return masks

    def analysis(self, wav: Tensor) -> Tensor:
        """
        Decompose waveform into sub-bands.

        Args:
            wav: [B, C, T] waveform.

        Returns:
            Tensor of shape [B, bands, C, T].
        """
        if wav.dim() != 3:
            raise ValueError(f"Expected waveform with shape [B, C, T], got {wav.shape}")
        B, C, T = wav.shape
        fft = torch.fft.rfft(wav, dim=-1)
        masks = self._frequency_masks(fft.shape[-1], wav.device)
        subbands: List[Tensor] = []
        for mask in masks:
            shaped = mask.view(1, 1, -1)
            sub_fft = fft * shaped
            sub = torch.fft.irfft(sub_fft, n=T, dim=-1)
            subbands.append(sub)
        stacked = torch.stack(subbands, dim=1)
        return stacked

    def synthesis(self, subbands: Tensor) -> Tensor:
        """
        Reconstruct waveform from sub-bands.

        Args:
            subbands: Tensor [B, bands, C, T].

        Returns:
            Waveform tensor [B, C, T].
        """
        if subbands.dim() != 4:
            raise ValueError(f"Expected subbands [B, bands, C, T], got {subbands.shape}")
        return subbands.sum(dim=1)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def _ensure_stereo(wav: Tensor) -> Tensor:
    """Guarantee 2-channel layout."""
    if wav.shape[0] == 1:
        return wav.repeat(2, 1)
    if wav.shape[0] >= 2:
        return wav[:2]
    raise ValueError(f"Unexpected channel dimension for waveform: {wav.shape}")


def _to_tensor(audio: Union[np.ndarray, Tensor, Sequence[float]]) -> Tensor:
    if isinstance(audio, Tensor):
        tensor = audio.to(torch.float32)
    else:
        tensor = torch.tensor(np.asarray(audio), dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.shape[0] < tensor.shape[1]:
        tensor = tensor.transpose(0, 1)
    return tensor.contiguous()


def _extract_audio(item: Dict[str, Union[Dict[str, Union[np.ndarray, float]], np.ndarray, Tensor]], key: str) -> Tuple[Tensor, int]:
    """
    Extract waveform tensor and sampling rate from an HF-style item.

    The dataset stores audio as AudioDecoder objects that can be accessed like dicts
    with 'array' and 'sampling_rate' keys.
    """
    if key not in item:
        raise KeyError(f"Expected key '{key}' in dataset item.")
    
    payload = item[key]
    
    # The AudioDecoder object can be accessed like a dictionary
    if isinstance(payload, dict) or hasattr(payload, '__getitem__'):
        try:
            # Try to access as dict-like object (AudioDecoder)
            array = payload["array"]
            sr = int(payload["sampling_rate"])
            tensor = _to_tensor(array)
        except (KeyError, TypeError):
            # Fallback for other dict structures
            array = payload.get("array", None)
            if array is None:
                raise KeyError(f"Audio dict for key '{key}' must include 'array'.")
            sr = int(payload.get("sampling_rate", 48000))
            tensor = _to_tensor(array)
    else:
        # Direct array (unlikely with AudioDecoder)
        tensor = _to_tensor(payload)
        sr_key = f"{key.replace('_flac', '')}_sr"
        sr = int(item.get(sr_key, 48000))
    
    return tensor, sr


class TenSecondSegments(Dataset):
    """Random 10-second 48 kHz stereo segments from SonicMaster entries."""

    def __init__(
        self,
        dataset: HFDataset,
        target_sr: int = 48000,
        segment_seconds: float = 10.0,
        degraded_key: str = "input_flac",
        clean_key: str = "gt_flac",
        pad_value: float = 0.0,
    ) -> None:
        if isinstance(dataset, IterableDataset):
            raise ValueError("TenSecondSegments requires random-access dataset. Please materialise to disk.")
        self.dataset = dataset
        self.target_sr = target_sr
        self.segment_samples = int(segment_seconds * target_sr)
        self.degraded_key = degraded_key
        self.clean_key = clean_key
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.dataset)

    def _resample(self, wav: Tensor, orig_sr: int) -> Tensor:
        if orig_sr == self.target_sr:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, self.target_sr)

    def _pad_to(self, wav: Tensor, length: int) -> Tensor:
        if wav.shape[-1] >= length:
            return wav
        pad = length - wav.shape[-1]
        return F.pad(wav, (0, pad), value=self.pad_value)

    def _crop_pair(self, degraded: Tensor, clean: Tensor) -> Tuple[Tensor, Tensor]:
        target = self.segment_samples
        max_len = max(degraded.shape[-1], clean.shape[-1])
        degraded = self._pad_to(degraded, max_len)
        clean = self._pad_to(clean, max_len)
        if max_len > target:
            start = random.randint(0, max_len - target)
        else:
            start = 0
        degraded = degraded[..., start:start + target]
        clean = clean[..., start:start + target]
        degraded = self._pad_to(degraded, target)
        clean = self._pad_to(clean, target)
        return degraded, clean

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        item = self.dataset[int(index)]
        degraded, degraded_sr = _extract_audio(item, self.degraded_key)
        clean, clean_sr = _extract_audio(item, self.clean_key)
        degraded = _ensure_stereo(self._resample(degraded, degraded_sr))
        clean = _ensure_stereo(self._resample(clean, clean_sr))
        degraded, clean = self._crop_pair(degraded, clean)
        return {
            "degraded": degraded,
            "clean": clean,
        }


# ---------------------------------------------------------------------------
# Diffusion model components
# ---------------------------------------------------------------------------


@dataclass
class BandSpec:
    name: str
    n_fft: int
    hop: int


def default_bandplan(num_bands: int) -> List[BandSpec]:
    """Heuristic STFT plan from low to high bands.
    
    Note: All bands use the same n_fft and hop_length to ensure consistent
    tensor dimensions for batch processing through the UNet. The PQMF analysis
    already separates the frequency content appropriately.
    Using smaller n_fft (2048) to reduce memory usage.
    """
    if num_bands == 8:
        # Use consistent n_fft=2048 and hop=512 across all bands
        return [
            BandSpec("low0", 2048, 512),
            BandSpec("low1", 2048, 512),
            BandSpec("mid2", 2048, 512),
            BandSpec("mid3", 2048, 512),
            BandSpec("mid4", 2048, 512),
            BandSpec("mid5", 2048, 512),
            BandSpec("high6", 2048, 512),
            BandSpec("high7", 2048, 512),
        ]
    if num_bands == 4:
        return [
            BandSpec("low0", 2048, 512),
            BandSpec("mid1", 2048, 512),
            BandSpec("mid2", 2048, 512),
            BandSpec("high3", 2048, 512),
        ]
    if num_bands == 2:
        return [
            BandSpec("low0", 2048, 512),
            BandSpec("high1", 2048, 512),
        ]
    raise ValueError(f"No default band plan available for {num_bands} bands.")


class STFTDiffuser(nn.Module):
    """Diffusion UNet operating in STFT representation space."""

    def __init__(
        self,
        rep_channels: int,
        cond_channels: Optional[int] = None,
        num_bands: int = 1,
        band_embed_dim: int = 64,
        base_channels: Sequence[int] = (64, 128, 256),  # Reduced from (160, 320, 640, 640)
        layers_per_block: int = 2,
    ) -> None:
        super().__init__()
        cond_channels = cond_channels or rep_channels
        in_channels = rep_channels + cond_channels
        self.band_embed = nn.Embedding(num_embeddings=num_bands, embedding_dim=band_embed_dim) if num_bands > 1 else None
        if self.band_embed is not None:
            self.band_proj = nn.Conv2d(in_channels + band_embed_dim, in_channels, kernel_size=1)
        else:
            self.band_proj = None
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=rep_channels,
            layers_per_block=layers_per_block,
            block_out_channels=tuple(base_channels),
            down_block_types=("DownBlock2D",) * len(base_channels),
            up_block_types=("UpBlock2D",) * len(base_channels),
            attention_head_dim=8,  # Single value instead of tuple
        )

    def forward(
        self,
        noisy_z: Tensor,
        timesteps: Tensor,
        cond_z: Optional[Tensor] = None,
        band_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if cond_z is None:
            raise ValueError("STFTDiffuser expects conditioning tensor.")
        x = torch.cat([noisy_z, cond_z], dim=1)
        if self.band_embed is not None and band_ids is not None:
            embed = self.band_embed(band_ids).view(-1, self.band_embed.embedding_dim, 1, 1)
            embed = embed.expand(-1, -1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, embed], dim=1)
            x = self.band_proj(x)
        return self.unet(x, timesteps).sample


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------


def load_sonicmaster_split(
    split: str,
    *,
    cache_dir: Optional[str] = None,
    saved_dir: Optional[str] = None,
    max_size: Optional[int] = None,
    num_proc: int = 16,
) -> HFDataset:
    """
    Load SonicMasterDataset split (non-streaming) and persist to disk if needed.
    """
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", cache_dir or "~/.cache/huggingface/datasets")).expanduser()
    saved_root = Path(saved_dir).expanduser() if saved_dir else cache_root / "saved" / "SonicMasterDataset"
    split_dir = saved_root / "train_val_test" / split
    split_dir.parent.mkdir(parents=True, exist_ok=True)

    if split_dir.exists():
        dataset = load_from_disk(str(split_dir))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
    else:
        dataset = load_dataset(
            "amaai-lab/SonicMasterDataset",
            split=split,
            streaming=False,
            cache_dir=str(cache_root),
            verification_mode="no_checks",
            download_mode="reuse_cache_if_exists",
            num_proc=num_proc,
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(split_dir))
    if max_size is not None:
        indices = list(range(min(max_size, len(dataset))))
        dataset = dataset.select(indices)
    return dataset


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_x0(noisy: Tensor, pred_noise: Tensor, timesteps: Tensor, scheduler: DDPMScheduler) -> Tensor:
    """Closed-form x0 prediction."""
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy.device)
    a_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return (noisy - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)


def build_band_stft_cfgs(plan: Sequence[BandSpec]) -> List[STFTCfg]:
    return [STFTCfg(n_fft=spec.n_fft, hop_length=spec.hop) for spec in plan]


def train_diffusion_restorer(
    *,
    split: str = "train",
    cache_dir: Optional[str] = None,
    saved_dir: Optional[str] = None,
    batch_size: int = 2,
    max_steps: int = 200_000,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    log_interval: int = 100,
    eval_interval: Optional[int] = None,
    gradient_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
    lambda_max: float = 0.2,
    lambda_exponent: float = 3.0,
    num_bands: int = 8,
    segment_seconds: float = 10.0,
    target_sr: int = 48000,
    guidance_weight: float = 1.5,
    warmup_steps: int = 4_000,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    best_checkpoint_path: Optional[str] = None,
    val_split: Optional[str] = "validation",
    val_batch_size: Optional[int] = None,
    max_val_batches: Optional[int] = 2,
    use_wandb: bool = True,
    wandb_project: str = "diffusion-restorer",
    wandb_run_name: Optional[str] = None,
    wandb_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train the diffusion restorer on SonicMasterDataset 10-second segments.

    Returns:
        Dictionary with the final losses.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataset = load_sonicmaster_split(split, cache_dir=cache_dir, saved_dir=saved_dir)
    segment_ds = TenSecondSegments(dataset, target_sr=target_sr, segment_seconds=segment_seconds)
    dataloader = DataLoader(
        segment_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_dataloader: Optional[DataLoader] = None
    val_iterator: Optional[Iterator[Dict[str, Tensor]]] = None
    val_batch_size_resolved = val_batch_size or batch_size
    if eval_interval and val_split:
        val_dataset = load_sonicmaster_split(val_split, cache_dir=cache_dir, saved_dir=saved_dir)
        val_segment_ds = TenSecondSegments(
            val_dataset,
            target_sr=target_sr,
            segment_seconds=segment_seconds,
        )
        val_dataloader = DataLoader(
            val_segment_ds,
            batch_size=val_batch_size_resolved,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_iterator = iter(val_dataloader)

    checkpoint_path_resolved = Path(checkpoint_path).expanduser() if checkpoint_path else None
    best_checkpoint_path_resolved = Path(best_checkpoint_path).expanduser() if best_checkpoint_path else None
    if best_checkpoint_path_resolved is None and checkpoint_path_resolved is not None:
        suffix = checkpoint_path_resolved.suffix
        if suffix:
            derived_name = f"{checkpoint_path_resolved.stem}_best{suffix}"
        else:
            derived_name = f"{checkpoint_path_resolved.name}_best"
        best_checkpoint_path_resolved = checkpoint_path_resolved.with_name(derived_name)

    pqmf = PQMF(bands=num_bands, sample_rate=target_sr).to(device)
    bandplan = default_bandplan(num_bands)
    stft_cfgs = build_band_stft_cfgs(bandplan)

    channels = 2
    rep_channels = channels * 3
    model = STFTDiffuser(rep_channels=rep_channels, cond_channels=rep_channels, num_bands=num_bands).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    # CombinedMusicLoss with baseline normalization enabled
    # music_loss = CombinedMusicLoss(sample_rate=target_sr, normalize_baseline=True).to(device)
    loss = nn.MSELoss()

    print("CombinedMusicLoss initialized with baseline normalization.")

    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or set use_wandb=False.")
        wandb_config = {
            "split": split,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "eval_interval": eval_interval,
            "gradient_clip": gradient_clip,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lambda_max": lambda_max,
            "lambda_exponent": lambda_exponent,
            "num_bands": num_bands,
            "segment_seconds": segment_seconds,
            "target_sr": target_sr,
            "guidance_weight": guidance_weight,
            "warmup_steps": warmup_steps,
            "val_split": val_split,
            "val_batch_size": val_batch_size or batch_size,
            "max_val_batches": max_val_batches,
            "checkpoint_path": str(checkpoint_path_resolved) if checkpoint_path_resolved else None,
            "best_checkpoint_path": str(best_checkpoint_path_resolved) if best_checkpoint_path_resolved else None,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            dir=wandb_dir,
            config=wandb_config,
        )

    def _to_wandb_audio(tensor: Tensor) -> np.ndarray:
        """Convert a (channels, time) tensor to W&B-friendly (time, channels) numpy array."""
        wav = tensor.detach().cpu().numpy()
        if wav.ndim == 2:
            wav = wav.transpose(1, 0)
        else:
            wav = np.squeeze(wav, axis=0) if wav.ndim > 0 else wav
        return wav

    final_stats: Dict[str, float] = {}
    best_val_metric = math.inf
    step = 0
    accum_step = 0
    model.train()

    def _checkpoint_payload() -> Dict[str, Any]:
        stats_snapshot = dict(final_stats)
        stats_snapshot["best_val_loss_music_normalized"] = float(best_val_metric)
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "stats": stats_snapshot,
            "bandplan": [spec.__dict__ for spec in bandplan],
            "config": {
                "num_bands": num_bands,
                "target_sr": target_sr,
                "segment_seconds": segment_seconds,
            },
        }

    def _save_checkpoint(path: Path, label: str) -> None:
        payload = _checkpoint_payload()
        payload["label"] = label
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"Saved {label} checkpoint to {path}")
        if wandb_run is not None:
            wandb.log({f"checkpoint/{label}_path": str(path)}, step=step)

    while step < max_steps:
        for batch in dataloader:
            degraded = batch["degraded"].to(device)  # [B, 2, T]
            clean = batch["clean"].to(device)

            B, _, T = clean.shape

            # PQMF decomposition
            degraded_bands = pqmf.analysis(degraded)
            clean_bands = pqmf.analysis(clean)

            target_reps: List[Tensor] = []
            cond_reps: List[Tensor] = []
            cfg_index: List[int] = []
            original_time_size = None
            original_freq_size = None
            for b_idx in range(num_bands):
                cfg = stft_cfgs[b_idx]
                Yb = stft(degraded_bands[:, b_idx], cfg)  # [B, 2, F, K]
                Xb = stft(clean_bands[:, b_idx], cfg)
                cond_rep_b = complex_to_rep(Yb)
                target_rep_b = complex_to_rep(Xb)
                # Pad both time and frequency dimensions to be divisible by 8 for UNet
                cond_rep_b, orig_time = pad_to_multiple(cond_rep_b, multiple=8, dim=-1)
                cond_rep_b, orig_freq = pad_to_multiple(cond_rep_b, multiple=8, dim=-2)
                target_rep_b, _ = pad_to_multiple(target_rep_b, multiple=8, dim=-1)
                target_rep_b, _ = pad_to_multiple(target_rep_b, multiple=8, dim=-2)
                if original_time_size is None:
                    original_time_size = orig_time
                    original_freq_size = orig_freq
                cond_reps.append(cond_rep_b)
                target_reps.append(target_rep_b)
                cfg_index.extend([b_idx] * B)
            target_rep = torch.cat(target_reps, dim=0)
            cond_rep = torch.cat(cond_reps, dim=0)
            band_ids = torch.tensor(cfg_index, device=device, dtype=torch.long)

            noise = torch.randn_like(target_rep)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (target_rep.shape[0],), device=device, dtype=torch.long)
            noisy_rep = scheduler.add_noise(target_rep, noise, timesteps)

            cond_keep = torch.rand(target_rep.shape[0], device=device) >= 0.1
            cond = cond_rep.clone()
            cond[~cond_keep] = 0.0

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_noise = model(noisy_rep, timesteps, cond, band_ids=band_ids)
                loss_diff = F.mse_loss(pred_noise, noise)

            pred_rep = predict_x0(noisy_rep, pred_noise, timesteps, scheduler)

            # Per-band projection + Reconstruction
            recon_bands: List[Tensor] = []
            offset = 0
            assert original_time_size is not None and original_freq_size is not None, "original sizes must be set"
            for b_idx in range(num_bands):
                cfg = stft_cfgs[b_idx]
                band_slice = slice(offset, offset + B)
                offset += B
                # Unpad both time and frequency dimensions before converting back to complex
                pred_rep_band = unpad_to_size(pred_rep[band_slice], original_time_size, dim=-1)
                pred_rep_band = unpad_to_size(pred_rep_band, original_freq_size, dim=-2)
                complex_band = rep_to_complex(pred_rep_band)
                complex_band = project_consistency(complex_band, cfg, length=T)
                recon = istft(complex_band, cfg, length=T)
                recon_bands.append(recon)
            recon_stack = torch.stack(recon_bands, dim=1)  # [B, bands, 2, T]

            # Full-band projection
            recon_full = pqmf.synthesis(recon_stack)
            # Clamp reconstructed audio to reasonable range to prevent NaN in loss
            recon_full = recon_full.clamp(-10.0, 10.0)
            recon_bands = pqmf.analysis(recon_full)

            # Flatten channels for music loss: [B, C, T] -> [B*C, T]
            recon_flat = recon_full.view(-1, recon_full.shape[-1])
            clean_flat = clean.view(-1, clean.shape[-1])
            
            # Check for NaN/Inf before computing loss
            if torch.isnan(recon_flat).any() or torch.isinf(recon_flat).any():
                # Skip music loss if reconstruction has NaN/Inf
                loss_music = torch.tensor(0.0, device=device, dtype=recon_flat.dtype)
                music_loss_normalized = loss_music
                baseline_val = 0.0
            else:
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    music_losses = music_loss(recon_flat, clean_flat)
                    loss_music = music_losses["total"]
                    # Use normalized loss from CombinedMusicLoss (baseline-subtracted)
                    music_loss_normalized = music_losses.get("total_normalized", music_losses["total"])
                    baseline_val = music_losses.get("baseline", torch.tensor(0.0)).item()
            
            alphas_cumprod = scheduler.alphas_cumprod.to(device)
            a_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            w_t = torch.pow(a_bar, lambda_exponent).mean()
            
            # Progressive music loss weighting: ramp up over first 30% of training
            music_loss_progress = min(1.0, step / (max_steps * 0.3))
            lambda_music = lambda_max * music_loss_progress
            
            # Combined loss with progressive weighting (using normalized loss)
            total_loss = loss_diff + (lambda_music * w_t) * music_loss_normalized
            
            # Scale loss by accumulation steps for proper gradient averaging
            scaled_loss = total_loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(scaled_loss).backward()
            
            accum_step += 1
            
            # Only update weights every gradient_accumulation_steps
            if accum_step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if gradient_clip is not None and gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if step % log_interval == 0:
                music_loss_val = loss_music.item()
                music_loss_norm_val = music_loss_normalized.item()
                # Compute relative loss (how many times worse than perfect reconstruction)
                music_loss_rel = (music_loss_val / baseline_val) if baseline_val > 0 else 0.0
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss_diff": loss_diff.item(),
                            "train/loss_music": music_loss_val,
                            "train/loss_music_normalized": music_loss_norm_val,
                            "train/loss_music_relative": music_loss_rel,
                            "train/music_baseline": baseline_val,
                            "train/w_t": w_t.item(),
                            "train/loss_total": total_loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/lambda_music": lambda_music,
                            "train/music_loss_progress": music_loss_progress,
                        },
                        step=step,
                    )
                print(
                    f"[step {step}] diff={loss_diff.item():.5f} "
                    f"music={music_loss_val:.5f} (norm={music_loss_norm_val:.5f}, rel={music_loss_rel:.2f}x, baseline={baseline_val:.3f}) "
                    f"λ={lambda_music:.3f} wt={w_t.item():.5f} total={total_loss.item():.5f}"
                )
            final_stats = {
                "loss_diff": float(loss_diff.item()),
                "loss_music": float(loss_music.item()),
                "w_t": float(w_t.item()),
                "loss_total": float(total_loss.item()),
            }

            step += 1
            if step >= max_steps:
                break

        if eval_interval and step % eval_interval == 0:
            if val_dataloader is None:
                model.eval()
                model.train()
                continue

            model.eval()
            val_loss_diff: List[float] = []
            val_loss_music: List[float] = []
            val_loss_music_norm: List[float] = []
            val_baselines: List[float] = []
            val_w_t: List[float] = []
            val_audio_payload: Dict[str, Any] = {}

            total_val_batches = len(val_dataloader)
            batches_to_run = total_val_batches if max_val_batches is None else min(max_val_batches, total_val_batches)
            batches_processed = 0

            with torch.no_grad():
                while batches_processed < batches_to_run:
                    try:
                        batch = next(val_iterator) if val_iterator is not None else None
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        batch = next(val_iterator)

                    if batch is None:
                        break

                    degraded = batch["degraded"].to(device)
                    clean = batch["clean"].to(device)

                    B, _, T = clean.shape
                    degraded_bands = pqmf.analysis(degraded)
                    clean_bands = pqmf.analysis(clean)

                    target_reps: List[Tensor] = []
                    cond_reps: List[Tensor] = []
                    cfg_index: List[int] = []
                    original_time_size = None
                    original_freq_size = None
                    for b_idx in range(num_bands):
                        cfg = stft_cfgs[b_idx]
                        Yb = stft(degraded_bands[:, b_idx], cfg)
                        Xb = stft(clean_bands[:, b_idx], cfg)
                        cond_rep_b = complex_to_rep(Yb)
                        target_rep_b = complex_to_rep(Xb)
                        cond_rep_b, orig_time = pad_to_multiple(cond_rep_b, multiple=8, dim=-1)
                        cond_rep_b, orig_freq = pad_to_multiple(cond_rep_b, multiple=8, dim=-2)
                        target_rep_b, _ = pad_to_multiple(target_rep_b, multiple=8, dim=-1)
                        target_rep_b, _ = pad_to_multiple(target_rep_b, multiple=8, dim=-2)
                        if original_time_size is None:
                            original_time_size = orig_time
                            original_freq_size = orig_freq
                        cond_reps.append(cond_rep_b)
                        target_reps.append(target_rep_b)
                        cfg_index.extend([b_idx] * B)

                    target_rep = torch.cat(target_reps, dim=0)
                    cond_rep = torch.cat(cond_reps, dim=0)
                    band_ids = torch.tensor(cfg_index, device=device, dtype=torch.long)

                    timesteps = torch.randint(
                        0,
                        scheduler.num_train_timesteps,
                        (target_rep.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    noise = torch.randn_like(target_rep)
                    noisy_rep = scheduler.add_noise(target_rep, noise, timesteps)

                    pred_noise = model(noisy_rep, timesteps, cond_rep, band_ids=band_ids)
                    loss_diff_val = F.mse_loss(pred_noise, noise)

                    pred_rep = predict_x0(noisy_rep, pred_noise, timesteps, scheduler)

                    recon_bands: List[Tensor] = []
                    offset = 0
                    assert (
                        original_time_size is not None and original_freq_size is not None
                    ), "original sizes must be set during validation"
                    for b_idx in range(num_bands):
                        cfg = stft_cfgs[b_idx]
                        band_slice = slice(offset, offset + B)
                        offset += B
                        pred_rep_band = unpad_to_size(pred_rep[band_slice], original_time_size, dim=-1)
                        pred_rep_band = unpad_to_size(pred_rep_band, original_freq_size, dim=-2)
                        complex_band = rep_to_complex(pred_rep_band)
                        complex_band = project_consistency(complex_band, cfg, length=T)
                        recon = istft(complex_band, cfg, length=T)
                        recon_bands.append(recon)

                    recon_stack = torch.stack(recon_bands, dim=1)
                    recon_full = pqmf.synthesis(recon_stack)
                    recon_full = recon_full.clamp(-10.0, 10.0)

                    recon_flat = recon_full.view(-1, recon_full.shape[-1])
                    clean_flat = clean.view(-1, clean.shape[-1])

                    music_losses = music_loss(recon_flat, clean_flat)
                    loss_music_val = music_losses["total"]
                    loss_music_norm_val = music_losses.get("total_normalized", loss_music_val)
                    baseline_val = music_losses.get("baseline", torch.tensor(0.0)).item()

                    alphas_cumprod = scheduler.alphas_cumprod.to(device)
                    a_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    w_t_val = torch.pow(a_bar, lambda_exponent).mean()

                    val_loss_diff.append(float(loss_diff_val.item()))
                    val_loss_music.append(float(loss_music_val.item()))
                    val_loss_music_norm.append(float(loss_music_norm_val.item()))
                    val_baselines.append(float(baseline_val))
                    val_w_t.append(float(w_t_val.item()))

                    if wandb_run is not None and not val_audio_payload:
                        val_audio_payload = {
                            "val/audio_degraded": wandb.Audio(_to_wandb_audio(degraded[0]), sample_rate=target_sr),
                            "val/audio_restored": wandb.Audio(_to_wandb_audio(recon_full[0]), sample_rate=target_sr),
                        }

                    batches_processed += 1

            mean_loss_diff = float(np.mean(val_loss_diff)) if val_loss_diff else 0.0
            mean_loss_music = float(np.mean(val_loss_music)) if val_loss_music else 0.0
            mean_loss_music_norm = float(np.mean(val_loss_music_norm)) if val_loss_music_norm else 0.0
            mean_baseline = float(np.mean(val_baselines)) if val_baselines else 0.0
            mean_w_t = float(np.mean(val_w_t)) if val_w_t else 0.0

            final_stats.update(
                {
                    "val_loss_diff": mean_loss_diff,
                    "val_loss_music": mean_loss_music,
                    "val_loss_music_normalized": mean_loss_music_norm,
                    "val_baseline": mean_baseline,
                }
            )

            improved = mean_loss_music_norm < best_val_metric
            if improved:
                best_val_metric = mean_loss_music_norm
            final_stats["best_val_loss_music_normalized"] = float(best_val_metric)

            if improved:
                if best_checkpoint_path_resolved is not None:
                    _save_checkpoint(best_checkpoint_path_resolved, "best")
                    print(f"New best validation loss ({best_val_metric:.5f}) at step {step}")
                else:
                    print(
                        f"New best validation loss ({best_val_metric:.5f}) at step {step} "
                        f"(no checkpoint path configured for best model)"
                    )

            print(
                f"[validation step {step}] diff={mean_loss_diff:.5f} "
                f"music={mean_loss_music:.5f} (norm={mean_loss_music_norm:.5f}, baseline={mean_baseline:.5f}) "
                f"wt={mean_w_t:.5f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "val/loss_diff": mean_loss_diff,
                        "val/loss_music": mean_loss_music,
                        "val/loss_music_normalized": mean_loss_music_norm,
                        "val/music_baseline": mean_baseline,
                        "val/w_t": mean_w_t,
                        "val/best_loss_music_normalized": float(best_val_metric),
                        "val/improved": int(improved),
                        **val_audio_payload,
                    },
                    step=step,
                )

            model.train()

    if checkpoint_path_resolved is not None:
        _save_checkpoint(checkpoint_path_resolved, "final")

    if wandb_run is not None:
        wandb_run.finish()

    return final_stats


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def restore_segment(
    model: STFTDiffuser,
    degraded: Tensor,
    *,
    pqmf: PQMF,
    bandplan: Sequence[BandSpec],
    scheduler_steps: int = 120,
    guidance: float = 1.5,
    trusted_blend: Optional[Tensor] = None,
) -> Tensor:
    """
    Restore a degraded waveform segment using the trained diffusion model.

    Args:
        model: Trained STFTDiffuser.
        degraded: [C, T] or [B, C, T] tensor.
        pqmf: Matching PQMF instance used during training.
        bandplan: Band specifications (for STFT configs).
        scheduler_steps: Number of DPMSolver steps.
        guidance: Classifier-free guidance weight.
        trusted_blend: Optional tensor mask [bands, freq, frames] for blending in
            trusted bins from the degraded representation (0..1).

    Returns:
        Restored waveform [B, C, T].
    """
    model.eval()
    if degraded.dim() == 2:
        degraded = degraded.unsqueeze(0)
    degraded = degraded.to(next(model.parameters()).device)
    B, C, T = degraded.shape
    bands = len(bandplan)
    stft_cfgs = build_band_stft_cfgs(bandplan)

    degraded_bands = pqmf.analysis(degraded)
    cond_reps: List[Tensor] = []
    cfg_ids: List[int] = []
    for b_idx in range(bands):
        cfg = stft_cfgs[b_idx]
        Y = stft(degraded_bands[:, b_idx], cfg)
        cond = complex_to_rep(project_consistency(Y, cfg, length=T))
        cond_reps.append(cond)
        cfg_ids.extend([b_idx] * B)
    cond_rep = torch.cat(cond_reps, dim=0)
    band_ids = torch.tensor(cfg_ids, device=degraded.device, dtype=torch.long)

    scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    scheduler.set_timesteps(scheduler_steps, device=degraded.device)

    noisy = cond_rep + 0.05 * torch.randn_like(cond_rep)
    state = noisy.clone()

    for idx, t in enumerate(scheduler.timesteps):
        # classifier-free guidance: duplicate batch
        model_input = torch.cat([state, state], dim=0)
        cond_input = torch.cat([torch.zeros_like(cond_rep), cond_rep], dim=0)
        band_ids_step = band_ids.repeat(2)
        timestep_repeat = t.repeat(model_input.shape[0])
        noise_pred = model(model_input, timestep_repeat, cond_input, band_ids_step)
        eps_uncond, eps_cond = noise_pred.chunk(2, dim=0)
        eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        out = scheduler.step(eps, t, state)
        state = out.prev_sample

        # Consistency projection per band
        recon_bands: List[Tensor] = []
        offset = 0
        for b_idx in range(bands):
            cfg = stft_cfgs[b_idx]
            band_slice = slice(offset, offset + B)
            offset += B
            X = rep_to_complex(state[band_slice])
            X = project_consistency(X, cfg, length=T)
            if trusted_blend is not None:
                blend = trusted_blend[b_idx].to(X.device)
                X = X * (1.0 - blend) + project_consistency(rep_to_complex(cond_rep[band_slice]), cfg, length=T) * blend
            state[band_slice] = complex_to_rep(X)
            recon_bands.append(istft(X, cfg, length=T))

        recon_stack = torch.stack(recon_bands, dim=1)
        full = pqmf.synthesis(recon_stack)
        full_bands = pqmf.analysis(full)
        refreshed = []
        for b_idx in range(bands):
            cfg = stft_cfgs[b_idx]
            spec = stft(full_bands[:, b_idx], cfg)
            spec = project_consistency(spec, cfg, length=T)
            refreshed.append(complex_to_rep(spec))
        state = torch.cat(refreshed, dim=0)

    # Final reconstruction
    recon_bands = []
    offset = 0
    for b_idx in range(bands):
        cfg = stft_cfgs[b_idx]
        band_slice = slice(offset, offset + B)
        offset += B
        X = rep_to_complex(state[band_slice])
        X = project_consistency(X, cfg, length=T)
        recon_bands.append(istft(X, cfg, length=T))
    recon_stack = torch.stack(recon_bands, dim=1)
    restored = pqmf.synthesis(recon_stack)
    return restored


__all__ = [
    "STFTCfg",
    "stft",
    "istft",
    "complex_to_rep",
    "rep_to_complex",
    "project_consistency",
    "PQMF",
    "TenSecondSegments",
    "BandSpec",
    "default_bandplan",
    "STFTDiffuser",
    "load_sonicmaster_split",
    "train_diffusion_restorer",
    "restore_segment",
]
