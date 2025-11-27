"""
Random segment dataset utilities for multi-source audio training.

The helpers here make it easy to sample aligned fixed-length excerpts from
datasets that store multiple audio file paths per example—ideal for source
separation or remix tasks.  A convenience loader for the saved MSS4S dataset is
included so training code can directly iterate over random 10-second chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import get_worker_info

from src.config_updates import DATA_PATH_4_SOURCES  # Fallback default

try:  # pragma: no cover - optional dependency
    import librosa
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ImportError("librosa is required for random segment loading.") from exc


class MultiAudioRandomSegmentDataset(TorchDataset):
    """
    Sample fixed-length segments from multiple audio paths stored in a HF split.

    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Sequence of dataset columns that contain audio file paths.
    duration_columns:
        Either a single column name holding a shared duration (seconds) or a list
        matching ``columns`` containing per-file durations.  ``None`` disables
        duration hints.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono.  When ``False`` the original channel layout
        is preserved.
    segment_seconds:
        Length of the sampled segment (seconds).
    seed:
        Base seed for random offset generation.
    resample_each_epoch:
        If ``True``, calling :meth:`set_epoch` will change the sampled offsets each
        epoch (useful when paired with a distributed sampler).
    """

    def __init__(
        self,
        split: str,
        columns: Sequence[str],
        duration_columns: Optional[Union[str, Sequence[str]]] = None,
        presence_columns: Optional[Dict[str, str]] = None,
        presence_suffix: Optional[str] = None,
        sr: int = 48_000,
        mono: bool = False,
        segment_seconds: float = 10.0,
        seed: int = 17,
        resample_each_epoch: bool = True,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        if len(columns) == 0:
            raise ValueError("columns must contain at least one audio column")
        self.split = split
        # Use provided root, or fall back to default from config_updates
        if root is None:
            root = Path(DATA_PATH_4_SOURCES)
        else:
            root = Path(root)
        self.ds = load_mss4s_split(split, root=root)
        self.columns = list(columns)
        self.duration_columns = duration_columns
        self.sr = int(sr)
        self.mono = bool(mono)
        self.segment_seconds = float(segment_seconds)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.presence_columns = presence_columns or {}
        self.presence_suffix = presence_suffix

        self._base_seed = int(seed)
        self._epoch = 0
        self._target_frames = int(round(self.segment_seconds * self.sr))

    # ------------------------------------------------------------------ metadata
    def set_epoch(self, epoch: int) -> None:
        """Set the logical epoch for deterministic resampling."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.ds) * 4

    # ------------------------------------------------------------------ internals
    def _rng(self, idx: int) -> np.random.Generator:
        """Worker-aware RNG to avoid duplicate segments in multi-processing."""
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        mix = self._base_seed
        mix += 1009 * worker_id
        if self.resample_each_epoch:
            mix += 7919 * self._epoch
        mix += 104729 * idx
        return np.random.default_rng(mix)

    def _durations_for_row(self, row: Dict[str, Any]) -> List[Optional[float]]:
        if self.duration_columns is None:
            return [None] * len(self.columns)
        if isinstance(self.duration_columns, str):
            value = row[self.duration_columns]
            duration = float(value) if value is not None else None
            return [duration] * len(self.columns)
        durations: List[Optional[float]] = []
        for column in self.duration_columns:
            value = row[column]
            durations.append(float(value) if value is not None else None)
        return durations

    def _choose_offset_seconds(
        self,
        row_durations: List[Optional[float]],
        rng: np.random.Generator,
    ) -> float:
        """Pick an offset that (ideally) fits inside all available durations."""
        segment_len = self.segment_seconds
        max_starts: List[float] = []
        for dur in row_durations:
            if dur is not None:
                max_starts.append(max(0.0, float(dur) - segment_len))
        if not max_starts:
            return float(rng.uniform(0.0, 5.0))
        max_start_all = max(0.0, min(max_starts))
        return float(rng.uniform(0.0, max_start_all + 1e-9))

    def _is_present(self, row: Dict[str, Any], column: str) -> bool:
        if column in self.presence_columns:
            key = self.presence_columns[column]
            return bool(row.get(key, False))
        if self.presence_suffix:
            key = f"{column}{self.presence_suffix}"
            if key in row:
                return bool(row.get(key, False))
        return bool(row.get(column))

    def _load_window(self, path: str, offset_s: float) -> torch.Tensor:
        """Load an aligned segment and pad/trim to the target length."""
        target_frames = self._target_frames
        wav, _ = librosa.load(
            path,
            sr=self.sr,
            mono=self.mono,
            offset=float(offset_s),
            duration=self.segment_seconds,
        )
        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)
        channels, frames = wav.shape
        if frames < target_frames:
            pad = np.zeros((channels, target_frames - frames), dtype=wav.dtype)
            wav = np.concatenate([wav, pad], axis=1)
        elif frames > target_frames:
            wav = wav[:, :target_frames]
        return torch.from_numpy(wav).to(torch.float32)

    # --------------------------------------------------------------------- access
    def __getitem__(self, idx: int) -> Dict[str, Any]:

        idx = idx % len(self.ds)

        row = self.ds[int(idx)]
        rng = self._rng(idx)
        row_durations = self._durations_for_row(row)
        offset_s = self._choose_offset_seconds(row_durations, rng)

        item: Dict[str, Any] = {
            "index": idx,
            "offset_seconds": offset_s,
            "segment_seconds": self.segment_seconds,
            # train: {'moisesdb': 240, 'musdb18': 100, 'dsd100': 50}  # musdb18 = musdb18hq
            # test: {'musdb18': 50, 'dsd100': 50}
            "dataset_name": row["dataset_name"]
        }
        for col in self.columns:
            present = self._is_present(row, col)
            path = row[col]
            if present and path:
                waveform = self._load_window(path, offset_s)
            else:
                # Create silent waveform with correct channel structure
                # Use same channel count as loaded audio (1 for mono, 2 for stereo)
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, self._target_frames), dtype=torch.float32)
            item[col] = {
                "waveform": waveform,
                "path": path,
                "is_present": bool(present),
            }
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "offset_seconds": [b["offset_seconds"] for b in batch],
            "segment_seconds": [b["segment_seconds"] for b in batch],
            "dataset_name": [b["dataset_name"] for b in batch],
        }
        for col in self.columns:
            waveforms = [b[col]["waveform"] for b in batch]
            paths = [b[col]["path"] for b in batch]
            present = [b[col].get("is_present", True) for b in batch]
            padded, lengths = _pad_stack(waveforms)
            out[col] = {
                "waveform": padded,
                "lengths": lengths,
                "paths": paths,
                "is_present": present,
            }
        return out

class MultiAudioRandomSegmentMixDataset(MultiAudioRandomSegmentDataset):
    """
    Variant of :class:`MultiAudioRandomSegmentDataset` that samples each source
    from a different song for mixup-style training.

    Each stem is drawn independently (with optional uniqueness constraints) so
    that the resulting batch combines stems originating from different tracks.
    The first column's offset is preserved in ``offset_seconds`` for backwards
    compatibility; per-source offsets and metadata are provided via additional
    keys (`mix_offsets`, `source_dataset_names`, `source_indices`).
    """

    def __init__(
        self,
        split: str,
        columns: Sequence[str],
        duration_columns: Optional[Union[str, Sequence[str]]] = None,
        presence_columns: Optional[Dict[str, str]] = None,
        presence_suffix: Optional[str] = None,
        sr: int = 48_000,
        mono: bool = False,
        segment_seconds: float = 10.0,
        seed: int = 17,
        resample_each_epoch: bool = True,
        ensure_unique_sources: bool = False,
        max_sampling_attempts: int = 30,
        mix_probability_start: float = 0.9,
        mix_probability_end: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        ensure_unique_sources:
            When ``True`` attempts to avoid reusing the same song for multiple
            stems within a sample. If not enough unique songs exist, duplicates
            may still occur after exhausting retries.
        max_sampling_attempts:
            Number of random draws per stem before falling back (default: 30).
        """
        super().__init__(
            split=split,
            columns=columns,
            duration_columns=duration_columns,
            presence_columns=presence_columns,
            presence_suffix=presence_suffix,
            sr=sr,
            mono=mono,
            segment_seconds=segment_seconds,
            seed=seed,
            resample_each_epoch=resample_each_epoch,
        )
        self.ensure_unique_sources = bool(ensure_unique_sources)
        self.max_sampling_attempts = max(1, int(max_sampling_attempts))
        self._mix_probability_start = float(np.clip(mix_probability_start, 0.0, 1.0))
        self._mix_probability_end = float(np.clip(mix_probability_end, 0.0, 1.0))
        self._mix_probability = self._mix_probability_start

    # ---------------------------------------------------------------- probability
    @property
    def mix_probability(self) -> float:
        return float(self._mix_probability)

    def get_mix_probability_bounds(self) -> Tuple[float, float]:
        return self._mix_probability_start, self._mix_probability_end

    def set_mix_probability(self, probability: float) -> None:
        """Update current probability of sampling mixed examples."""
        self._mix_probability = float(np.clip(probability, 0.0, 1.0))

    def reset_mix_probability(self) -> None:
        self._mix_probability = self._mix_probability_start

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = self._rng(idx)
        if rng.uniform() >= self._mix_probability:
            return super().__getitem__(idx)
        offsets_per_source: Dict[str, float] = {}
        dataset_names_per_source: Dict[str, Optional[str]] = {}
        source_indices_per_source: Dict[str, Optional[int]] = {}

        selected_indices: Set[int] = set()
        item: Dict[str, Any] = {
            "index": idx,
            "segment_seconds": self.segment_seconds,
            "dataset_name": "mix",
        }

        for col_idx, col in enumerate(self.columns):
            chosen_row: Optional[Dict[str, Any]] = None
            chosen_row_index: Optional[int] = None
            present = False
            path: Optional[str] = None

            # First attempt to sample unique rows while stems remain.
            for attempt in range(self.max_sampling_attempts):
                candidate_index = int(rng.integers(0, len(self.ds)))
                if (
                    self.ensure_unique_sources
                    and len(selected_indices) < len(self.ds)
                    and candidate_index in selected_indices
                ):
                    continue
                candidate_row = self.ds[candidate_index]
                candidate_present = self._is_present(candidate_row, col)
                candidate_path = candidate_row[col] if candidate_present else None
                if candidate_present and candidate_path:
                    chosen_row = candidate_row
                    chosen_row_index = candidate_index
                    present = True
                    path = candidate_path
                    if self.ensure_unique_sources:
                        selected_indices.add(candidate_index)
                    break

            # If uniqueness failed, allow reuse of rows.
            if chosen_row is None:
                for attempt in range(self.max_sampling_attempts):
                    candidate_index = int(rng.integers(0, len(self.ds)))
                    candidate_row = self.ds[candidate_index]
                    candidate_present = self._is_present(candidate_row, col)
                    candidate_path = candidate_row[col] if candidate_present else None
                    if candidate_present and candidate_path:
                        chosen_row = candidate_row
                        chosen_row_index = candidate_index
                        present = True
                        path = candidate_path
                        break

            if chosen_row is not None:
                durations = self._durations_for_row(chosen_row)
                duration = durations[col_idx] if col_idx < len(durations) else None
                offset_s = self._choose_offset_seconds([duration], rng)
                dataset_names_per_source[col] = chosen_row.get("dataset_name")
                source_indices_per_source[col] = chosen_row_index
            else:
                offset_s = float(rng.uniform(0.0, 5.0))
                dataset_names_per_source[col] = None
                source_indices_per_source[col] = None

            offsets_per_source[col] = offset_s

            if present and path:
                waveform = self._load_window(path, offset_s)
            else:
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, self._target_frames), dtype=torch.float32)
                path = None

            item[col] = {
                "waveform": waveform,
                "path": path,
                "is_present": bool(present),
                "source_index": source_indices_per_source[col],
                "source_dataset_name": dataset_names_per_source[col],
                "offset_seconds": offset_s,
            }

        first_col = self.columns[0] if self.columns else None
        item["offset_seconds"] = offsets_per_source.get(first_col, 0.0) if first_col else 0.0
        item["mix_offsets"] = offsets_per_source
        item["source_dataset_names"] = dataset_names_per_source
        item["source_indices"] = source_indices_per_source

        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = super().collate_fn(batch)
        out["mix_offsets"] = {
            col: [b.get("mix_offsets", {}).get(col) for b in batch]
            for col in self.columns
        }
        out["source_dataset_names"] = {
            col: [b.get("source_dataset_names", {}).get(col) for b in batch]
            for col in self.columns
        }
        out["source_indices"] = {
            col: [b.get("source_indices", {}).get(col) for b in batch]
            for col in self.columns
        }
        return out


def _pad_stack(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad waveforms with varying channel counts to a common (B, C_max, T)."""
    batch_size = len(tensors)
    channel_max = max(t.shape[0] for t in tensors)
    time = tensors[0].shape[1]
    padded = tensors[0].new_zeros((batch_size, channel_max, time))
    lengths = torch.full((batch_size,), time, dtype=torch.long)
    for idx, tensor in enumerate(tensors):
        channels = tensor.shape[0]
        padded[idx, :channels] = tensor
    return padded, lengths


# def make_collate(columns: Sequence[str]):
#     """Create a collate_fn that stacks waveforms and keeps metadata."""
#
#     def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         out: Dict[str, Any] = {
#             "index": [b["index"] for b in batch],
#             "offset_seconds": [b["offset_seconds"] for b in batch],
#             "segment_seconds": [b["segment_seconds"] for b in batch],
#         }
#         for col in columns:
#             waveforms = [b[col]["waveform"] for b in batch]
#             paths = [b[col]["path"] for b in batch]
#             present = [b[col].get("is_present", True) for b in batch]
#             padded, lengths = _pad_stack(waveforms)
#             out[col] = {
#                 "waveform": padded,
#                 "lengths": lengths,
#                 "paths": paths,
#                 "is_present": present,
#             }
#         return out
#
#     return collate_fn


def load_mss4s_split(split: str, root: Path = None) -> Dataset:
    """
    Load the specified MSS4S split and return a Hugging Face dataset.

    The dataset is stored on disk (via ``save_to_disk``).  This helper resolves the
    path and also ensures that audio columns remain as plain string paths (which is
    the format expected by :class:`MultiAudioRandomSegmentDataset`).
    """

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSS4S root not found: {root}")
    dataset_dict = DatasetDict.load_from_disk(str(root))
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not present in MSS4S dataset (available: {list(dataset_dict.keys())})")

    dataset = dataset_dict[split]
    # The saved dataset stores raw string paths.  If someone later saves it with
    # Audio features, we cast them back to plain strings to keep compatibility.
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


class MultiAudioFullSongDataset(TorchDataset):
    """
    Load full songs (no segmenting) for evaluation purposes.
    
    This dataset loads complete audio files without any segmenting, suitable for
    standard evaluation procedures like MUSDB18-HQ evaluation.
    
    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Sequence of dataset columns that contain audio file paths.
    presence_columns:
        Optional dict mapping column names to presence indicator column names.
    presence_suffix:
        Optional suffix to append to column names to find presence indicators.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono. When ``False`` the original channel layout
        is preserved.
    """
    
    def __init__(
        self,
        split: str,
        columns: Sequence[str],
        presence_columns: Optional[Dict[str, str]] = None,
        presence_suffix: Optional[str] = None,
        sr: int = 48_000,
        mono: bool = False,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        if len(columns) == 0:
            raise ValueError("columns must contain at least one audio column")
        self.split = split
        # Use provided root, or fall back to default from config_updates
        if root is None:
            root = Path(DATA_PATH_4_SOURCES)
        else:
            root = Path(root)
        self.ds = load_mss4s_split(split, root=root)
        self.columns = list(columns)
        self.sr = int(sr)
        self.mono = bool(mono)
        self.presence_columns = presence_columns or {}
        self.presence_suffix = presence_suffix
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def _is_present(self, row: Dict[str, Any], column: str) -> bool:
        if column in self.presence_columns:
            key = self.presence_columns[column]
            return bool(row.get(key, False))
        if self.presence_suffix:
            key = f"{column}{self.presence_suffix}"
            if key in row:
                return bool(row.get(key, False))
        return bool(row.get(column))
    
    def _load_full_audio(self, path: str) -> torch.Tensor:
        """Load the full audio file."""
        wav, _ = librosa.load(
            path,
            sr=self.sr,
            mono=self.mono,
        )
        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)
        return torch.from_numpy(wav).to(torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[int(idx)]
        
        item: Dict[str, Any] = {
            "index": idx,
            "dataset_name": row["dataset_name"],
        }
        for col in self.columns:
            present = self._is_present(row, col)
            path = row[col]
            if present and path:
                waveform = self._load_full_audio(path)
            else:
                # Create silent waveform with correct channel structure
                channels = 1 if self.mono else 2
                # Use a dummy length - will be padded in collate_fn
                waveform = torch.zeros((channels, 1), dtype=torch.float32)
            item[col] = {
                "waveform": waveform,
                "path": path,
                "is_present": bool(present),
            }
        return item
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that pads waveforms to the same length."""
        # Find the maximum length in the batch
        max_length = 0
        for b in batch:
            for col in self.columns:
                waveform = b[col]["waveform"]
                if waveform.shape[-1] > max_length:
                    max_length = waveform.shape[-1]
        
        # Pad all waveforms to the same length
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "dataset_name": [b["dataset_name"] for b in batch],
        }
        for col in self.columns:
            waveforms = []
            paths = []
            present = []
            lengths = []
            for b in batch:
                waveform = b[col]["waveform"]
                lengths.append(waveform.shape[-1])
                if waveform.shape[-1] < max_length:
                    # Pad with zeros
                    pad_length = max_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                waveforms.append(waveform)
                paths.append(b[col]["path"])
                present.append(b[col].get("is_present", True))
            
            out[col] = {
                "waveform": torch.stack(waveforms, dim=0),
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "paths": paths,
                "is_present": present,
            }
        return out


__all__ = [
    "MultiAudioRandomSegmentDataset",
    "MultiAudioRandomSegmentMixDataset",
    "MultiAudioFullSongDataset",
    "load_mss4s_split",
]
