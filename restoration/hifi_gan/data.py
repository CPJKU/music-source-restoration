from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
import soundfile

from restoration.diffusion_restorer import TenSecondSegments, load_sonicmaster_split
from .effects import MusicEffectPipeline, SimpleDegradationPipeline

Tensor = torch.Tensor


class SourceFilteredRestorationDataset(Dataset[dict[str, Tensor]]):
    """10-second segments from restoration dataset filtered by source (expert training).
    
    Supports two modes:
    1. Audio mode: Loads audio files directly (use when latents_path is None)
    2. Latent mode: Loads pre-computed CoDiCodec latents as embeddings (use when latents_path is provided)
    
    Supports combining multiple dataset versions:
    - Pass single path: "/opt/datasets/.../v0.1" -> loads v0.1 only
    - Pass multiple paths: ["/opt/.../v0.1", "/opt/.../v0.2"] -> concatenates both
    - Pass list: ["v0.1", "v0.2"] -> auto-resolves under base directory
    
    Parameters
    ----------
    source_filter : str
        Source label to filter by (e.g., 'vocals', 'bass', 'drums', 'guitars', 
        'keyboards', 'synthesizers', 'orchestral', 'percussions', 'other')
    dataset_path : str | list[str]
        Path(s) to the restoration dataset on disk (audio files).
        Can be single path string, list of paths, or list of version names
    segment_seconds : float
        Duration of audio segments in seconds
    base_sample_rate : int
        Base sample rate for processing
    target_sample_rate : int
        Target sample rate for upsampling
    latents_path : str | list[str] | None
        Path(s) to pre-computed CoDiCodec latents directory (e.g., v0.1_latents/)
        If list, must match length of dataset_path list
        If provided, loads latents as embeddings instead of computing from audio
    base_dataset_dir : str | None
        Base directory for resolving version names (e.g., "/opt/datasets/.../restoration_dataset/")
        If None and dataset_path contains version names, defaults to this path
    apply_online_degradation : bool
        If True, applies SimpleDegradationPipeline to degraded_base during loading
        for additional augmentation (default: False)
    """

    def __init__(
        self,
        source_filter: str,
        dataset_path: str | list[str],
        segment_seconds: float,
        base_sample_rate: int,
        target_sample_rate: int,
        latents_path: str | list[str] | None = None,
        base_dataset_dir: str | None = None,
        apply_online_degradation: bool = False,
    ) -> None:
        self.source_filter = source_filter
        self.base_sample_rate = base_sample_rate
        self.target_sample_rate = target_sample_rate
        self.segment_seconds = segment_seconds
        
        # Initialize degradation pipeline if requested
        self.degradation = (
            SimpleDegradationPipeline(sample_rate=base_sample_rate)
            if apply_online_degradation
            else None
        )
        
        # Normalize dataset_path to list
        if isinstance(dataset_path, str):
            dataset_paths = [dataset_path]
        else:
            dataset_paths = list(dataset_path)
        
        # Resolve version names to full paths if base_dataset_dir provided
        if base_dataset_dir is None:
            base_dataset_dir = "/opt/datasets/HF_datasets/saved/restoration_dataset"
        
        resolved_paths = []
        for path in dataset_paths:
            path_obj = Path(path)
            # Check if it's a version name (e.g., "v0.1", "v0.2") or full path
            if not path_obj.is_absolute() and not path_obj.exists():
                # Assume it's a version name, resolve under base_dataset_dir
                resolved = Path(base_dataset_dir) / path
                resolved_paths.append(str(resolved))
            else:
                resolved_paths.append(path)
        
        # Normalize latents_path to list (or None) and resolve version names
        if latents_path is None:
            latents_paths = [None] * len(resolved_paths)
        elif isinstance(latents_path, str):
            # Single latents_path: resolve if it's a version name
            latent_path_obj = Path(latents_path)
            if not latent_path_obj.is_absolute() and not latent_path_obj.exists():
                # Assume it's a version name (e.g., "v0.1_latents"), resolve under base_dataset_dir
                resolved_latent = Path(base_dataset_dir) / latents_path
                latents_paths = [str(resolved_latent)] * len(resolved_paths)
            else:
                latents_paths = [latents_path] * len(resolved_paths)
        else:
            # List of latents_paths: resolve version names individually
            latents_paths = []
            for latent_path in latents_path:
                if latent_path is None:
                    latents_paths.append(None)
                else:
                    latent_path_obj = Path(latent_path)
                    if not latent_path_obj.is_absolute() and not latent_path_obj.exists():
                        # Assume it's a version name, resolve under base_dataset_dir
                        resolved_latent = Path(base_dataset_dir) / latent_path
                        latents_paths.append(str(resolved_latent))
                    else:
                        latents_paths.append(latent_path)
            
            if len(latents_paths) != len(resolved_paths):
                raise ValueError(
                    f"latents_path list length ({len(latents_paths)}) must match "
                    f"dataset_path list length ({len(resolved_paths)})"
                )
        
        # Load and concatenate datasets
        print(f"Loading restoration datasets for source '{source_filter}':")
        all_datasets = []
        for dataset_path, latent_path in zip(resolved_paths, latents_paths):
            print(f"  - {dataset_path}")
            full_dataset = load_from_disk(dataset_path)
            filtered = full_dataset.filter(lambda x: x["label"] == source_filter)
            all_datasets.append(filtered)
            print(f"    Found {len(filtered)} samples")
        
        # Concatenate all datasets
        from datasets import concatenate_datasets
        if len(all_datasets) == 1:
            self.dataset = all_datasets[0]
        else:
            self.dataset = concatenate_datasets(all_datasets)
            print(f"  Total combined: {len(self.dataset)} samples")
        
        # Handle latents: collect from all latent directories
        self.latent_files = []
        self.latents_enabled = any(lp is not None for lp in latents_paths)
        
        if self.latents_enabled:
            for latent_path in latents_paths:
                if latent_path is None:
                    continue
                
                latent_dir = Path(latent_path) / "train"
                if not latent_dir.exists():
                    raise FileNotFoundError(f"Latents directory not found: {latent_dir}")
                
                # Find all latent files for this source
                pattern = f"*_{source_filter}.pt"
                files = sorted(latent_dir.glob(pattern))
                
                if len(files) == 0:
                    print(f"    WARNING: No latent files found in {latent_dir}")
                else:
                    print(f"    Loaded {len(files)} latent files from {latent_dir}")
                
                self.latent_files.extend(files)
            
            if len(self.latent_files) == 0:
                raise ValueError(
                    f"No latent files found for source '{source_filter}' in any provided latent directories"
                )
            
            print(f"  Total latent files: {len(self.latent_files)}")
        else:
            print(f"  Audio-only mode (no latents)")

    def __len__(self) -> int:
        return len(self.latent_files) if self.latents_enabled else len(self.dataset)

    def _load_audio(self, path: str, target_sr: int) -> Tensor:
        """Load audio file and resample to target sample rate."""
        wav, sr = torchaudio.load(path, backend="soundfile")
        # load with soundfile backend to support more formats
        # Alternatively, use soundfile directly:
        # wav, sr = soundfile.read(path)
        # wav = torch.from_numpy(wav).transpose(0, 1) if wav.ndim > 1 else torch.from_numpy(wav).unsqueeze(0)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav

    def _resample_to_base(self, wav: Tensor) -> Tensor:
        """Resample from target to base sample rate."""
        if self.base_sample_rate == self.target_sample_rate:
            return wav
        return torchaudio.functional.resample(wav, self.target_sample_rate, self.base_sample_rate)

    def __getitem__(self, index: int) -> dict[str, Tensor | None]:
        # Mode 1: Load from pre-computed latents + audio
        if self.latents_enabled:
            latent_file = self.latent_files[index]
            latent_data = torch.load(latent_file, map_location="cpu")
            
            # Extract latents: shape [1, T, 8, 64] -> [T, 8, 64]
            gt_latent = latent_data["gt_latent"].squeeze(0).float()  # Clean latent
            input_latent = latent_data["input_latent"].squeeze(0).float()  # Degraded latent
            
            # Use input (degraded) latent as embedding for conditioning
            # Reshape from [T, 8, 64] to [512, T] by flattening token dimension into channels
            # CoDiCodec uses 8 tokens per frame with 64-dim features = 512 total channels
            # This gives [T, 512] then transpose to [512, T] for channel-first format
            T, num_tokens, feat_dim = input_latent.shape  # [T, 8, 64]
            embedding = input_latent.reshape(T, num_tokens * feat_dim).transpose(0, 1)  # [512, T]
            
            # Also load the actual audio from the dataset for this index
            # The metadata contains the dataset_index to match with the original dataset
            dataset_idx = latent_data["meta"]["dataset_index"]
            item = self.dataset[dataset_idx]
            
            # Load degraded (separated_stem) and clean (clean_stem) audio
            degraded_target = self._load_audio(item["separated_stem"], self.target_sample_rate)
            clean_target = self._load_audio(item["clean_stem"], self.target_sample_rate)
            
            # Resample to base rate
            degraded_base = self._resample_to_base(degraded_target)
            clean_base = self._resample_to_base(clean_target)
            
            # Ensure correct shape (channels, samples)
            if degraded_target.ndim == 1:
                degraded_target = degraded_target.unsqueeze(0)
            if clean_target.ndim == 1:
                clean_target = clean_target.unsqueeze(0)
            if degraded_base.ndim == 1:
                degraded_base = degraded_base.unsqueeze(0)
            if clean_base.ndim == 1:
                clean_base = clean_base.unsqueeze(0)
            
            # Ensure stereo (2 channels) - duplicate mono to stereo if needed
            if degraded_target.shape[0] == 1:
                degraded_target = degraded_target.repeat(2, 1)
            elif degraded_target.shape[0] > 2:
                degraded_target = degraded_target[:2]
                
            if clean_target.shape[0] == 1:
                clean_target = clean_target.repeat(2, 1)
            elif clean_target.shape[0] > 2:
                clean_target = clean_target[:2]
                
            if degraded_base.shape[0] == 1:
                degraded_base = degraded_base.repeat(2, 1)
            elif degraded_base.shape[0] > 2:
                degraded_base = degraded_base[:2]
                
            if clean_base.shape[0] == 1:
                clean_base = clean_base.repeat(2, 1)
            elif clean_base.shape[0] > 2:
                clean_base = clean_base[:2]
            
            # Apply online degradation if enabled (only to degraded_base)
            if self.degradation is not None:
                degraded_base = self.degradation(
                    degraded_base.unsqueeze(0), sample_rate=self.base_sample_rate
                ).squeeze(0)
            
            return {
                "degraded_base": degraded_base,
                "clean_base": clean_base,
                "clean_target": clean_target,
                "embedding": embedding,
                "gt_latent": gt_latent,
                "input_latent": input_latent,
            }
        
        # Mode 2: Load from audio files only (original behavior)
        item = self.dataset[index]
        
        # Load degraded (separated_stem) and clean (clean_stem) audio
        degraded_target = self._load_audio(item["separated_stem"], self.target_sample_rate)
        clean_target = self._load_audio(item["clean_stem"], self.target_sample_rate)
        
        # Resample to base rate
        degraded_base = self._resample_to_base(degraded_target)
        clean_base = self._resample_to_base(clean_target)
        
        # Ensure correct shape (channels, samples)
        if degraded_target.ndim == 1:
            degraded_target = degraded_target.unsqueeze(0)
        if clean_target.ndim == 1:
            clean_target = clean_target.unsqueeze(0)
        if degraded_base.ndim == 1:
            degraded_base = degraded_base.unsqueeze(0)
        if clean_base.ndim == 1:
            clean_base = clean_base.unsqueeze(0)
        
        # Ensure stereo (2 channels) - duplicate mono to stereo if needed
        if degraded_target.shape[0] == 1:
            degraded_target = degraded_target.repeat(2, 1)
        elif degraded_target.shape[0] > 2:
            degraded_target = degraded_target[:2]
            
        if clean_target.shape[0] == 1:
            clean_target = clean_target.repeat(2, 1)
        elif clean_target.shape[0] > 2:
            clean_target = clean_target[:2]
            
        if degraded_base.shape[0] == 1:
            degraded_base = degraded_base.repeat(2, 1)
        elif degraded_base.shape[0] > 2:
            degraded_base = degraded_base[:2]
            
        if clean_base.shape[0] == 1:
            clean_base = clean_base.repeat(2, 1)
        elif clean_base.shape[0] > 2:
            clean_base = clean_base[:2]
        
        # Apply online degradation if enabled (only to degraded_base)
        if self.degradation is not None:
            degraded_base = self.degradation(
                degraded_base.unsqueeze(0), sample_rate=self.base_sample_rate
            ).squeeze(0)
        
        return {
            "degraded_base": degraded_base,
            "clean_base": clean_base,
            "clean_target": clean_target,
            "embedding": None,
        }


class SonicMasterPairs(Dataset[dict[str, Tensor]]):
    """10-second paired segments for GAN training with online degradations."""

    def __init__(
        self,
        split: str,
        cache_dir: str | None,
        saved_dir: str | None,
        segment_seconds: float,
        base_sample_rate: int,
        target_sample_rate: int,
        apply_online_degradation: bool = True,
    ) -> None:
        dataset = load_sonicmaster_split(split, cache_dir=cache_dir, saved_dir=saved_dir)
        self.segments = TenSecondSegments(
            dataset,
            target_sr=target_sample_rate,
            segment_seconds=segment_seconds,
        )
        self.base_sample_rate = base_sample_rate
        self.target_sample_rate = target_sample_rate
        self.effects = (
            SimpleDegradationPipeline(sample_rate=base_sample_rate) if apply_online_degradation else None
        )

    def __len__(self) -> int:
        return len(self.segments)

    def _resample_to_base(self, wav: Tensor) -> Tensor:
        if self.base_sample_rate == self.target_sample_rate:
            return wav
        return torchaudio.functional.resample(wav, self.target_sample_rate, self.base_sample_rate)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.segments[index]
        degraded_target = item["degraded"]
        clean_target = item["clean"]

        degraded_base = self._resample_to_base(degraded_target)
        clean_base = self._resample_to_base(clean_target)

        if self.effects is not None:
            augmented = self.effects(degraded_base.unsqueeze(0), sample_rate=self.base_sample_rate).squeeze(0)
            if augmented.shape[-1] != degraded_base.shape[-1]:
                augmented = torch.nn.functional.interpolate(
                    augmented.unsqueeze(0),
                    size=degraded_base.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
            degraded_base = augmented

        return {
            "degraded_base": degraded_base,
            "clean_base": clean_base,
            "clean_target": clean_target,
        }


def collate_batch(batch: Iterable[dict[str, Tensor | None]]) -> dict[str, Tensor | None]:
    """Collate batch, handling both audio and latent modes."""
    # Check if we're in latent mode (has embedding)
    has_embedding = batch[0].get("embedding") is not None
    
    # Stack audio tensors (always present)
    degraded = torch.stack([item["degraded_base"] for item in batch], dim=0)
    clean_base = torch.stack([item["clean_base"] for item in batch], dim=0)
    clean_target = torch.stack([item["clean_target"] for item in batch], dim=0)
    
    result = {
        "degraded_base": degraded,
        "clean_base": clean_base,
        "clean_target": clean_target,
        "embedding": None,
    }
    
    # Add latents if in latent mode
    if has_embedding:
        embeddings = [item["embedding"] for item in batch if item["embedding"] is not None]
        gt_latents = [item["gt_latent"] for item in batch if item.get("gt_latent") is not None]
        input_latents = [item["input_latent"] for item in batch if item.get("input_latent") is not None]
        
        if embeddings:
            result["embedding"] = torch.stack(embeddings, dim=0)
        if gt_latents:
            result["gt_latent"] = torch.stack(gt_latents, dim=0)
        if input_latents:
            result["input_latent"] = torch.stack(input_latents, dim=0)
    
    return result


def create_dataloader(
    split: str,
    batch_size: int,
    segment_seconds: float,
    base_sample_rate: int,
    cache_dir: str | None = None,
    saved_dir: str | None = None,
    num_workers: int = 4,
    shuffle: bool = True,
    apply_online_degradation: bool = True,
    target_sample_rate: int | None = None,
    use_restoration_dataset: bool = False,
    restoration_dataset_path: str | list[str] | None = None,
    source_filter: str | None = None,
    latents_path: str | list[str] | None = None,
    base_dataset_dir: str | None = None,
) -> DataLoader[dict[str, Tensor | None]]:
    """Create a dataloader for either SonicMaster or restoration dataset.
    
    Parameters
    ----------
    use_restoration_dataset : bool
        If True, use SourceFilteredRestorationDataset instead of SonicMasterPairs
    restoration_dataset_path : str | list[str] | None
        Path(s) to restoration dataset (required if use_restoration_dataset=True)
        Can be single path, list of paths, or list of version names (e.g., ["v0.1", "v0.2"])
    source_filter : str | None
        Source to filter by when using restoration dataset (required if use_restoration_dataset=True)
    latents_path : str | list[str] | None
        Path(s) to pre-computed CoDiCodec latents directory (e.g., v0.1_latents/)
        Can be single path or list matching restoration_dataset_path
        If provided, loads latents as embeddings instead of audio
    base_dataset_dir : str | None
        Base directory for resolving version names
        Default: "/opt/datasets/HF_datasets/saved/restoration_dataset"
    """
    target_sr = target_sample_rate or base_sample_rate
    
    if use_restoration_dataset:
        if restoration_dataset_path is None:
            raise ValueError("restoration_dataset_path required when use_restoration_dataset=True")
        if source_filter is None:
            raise ValueError("source_filter required when use_restoration_dataset=True")
        
        dataset = SourceFilteredRestorationDataset(
            source_filter=source_filter,
            dataset_path=restoration_dataset_path,
            segment_seconds=segment_seconds,
            base_sample_rate=base_sample_rate,
            target_sample_rate=target_sr,
            latents_path=latents_path,
            base_dataset_dir=base_dataset_dir,
        )
    else:
        # Default to /opt/scratch/HF_datasets/saved/SonicMasterDataset if not specified
        if saved_dir is None:
            saved_dir = "/opt/scratch/HF_datasets/saved/SonicMasterDataset"
        
        dataset = SonicMasterPairs(
            split=split,
            cache_dir=cache_dir,
            saved_dir=saved_dir,
            segment_seconds=segment_seconds,
            base_sample_rate=base_sample_rate,
            target_sample_rate=target_sr,
            apply_online_degradation=apply_online_degradation,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )


__all__ = ["SonicMasterPairs", "SourceFilteredRestorationDataset", "create_dataloader", "collate_batch"]
