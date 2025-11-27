"""
MSRBench dataset utilities for multi-source audio training.

This module provides utilities to load the MSRBench dataset which stores audio
as bytes in the HuggingFace dataset format.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from src.config_updates import DATA_PATH_MSR_BENCH  # Fallback default

try:  # pragma: no cover - optional dependency
    import librosa
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ImportError("librosa and soundfile are required for audio loading.") from exc


def load_msr_bench(split: str, root: Path = None) -> Dataset:
    """
    Load the specified MSRBench split and return a Hugging Face dataset.

    The dataset is stored on disk (via ``save_to_disk``). Audio columns are kept
    with decode=False to access the raw bytes directly.
    """
    if root is None:
        root = Path(DATA_PATH_MSR_BENCH)
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSRBench root not found: {root}")
    dataset_dict = DatasetDict.load_from_disk(str(root))
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not present in MSRBench dataset (available: {list(dataset_dict.keys())})")

    dataset = dataset_dict[split]
    # Ensure audio columns are not decoded so we can access bytes directly
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


class MultiAudioFullSongDataset(TorchDataset):
    """
    Load full songs (no segmenting) for evaluation purposes.
    
    This dataset loads complete audio files without any segmenting, suitable for
    standard evaluation procedures like MUSDB18-HQ evaluation.
    
    MSRBench dataset structure:
    - Each sample has a mixture_audio (always present)
    - Each sample has a source_audio (the single source that's present)
    - Each sample has a source_label (label indicating which source it is)
    - Multiple stem columns (vocals_audio, bass_audio, etc.) with *_is_present flags
    - Only one source is present per sample
    
    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Optional sequence of dataset columns that contain audio file paths.
        If None, uses mixture_audio and source_audio.
    presence_columns:
        Optional dict mapping column names to presence indicator column names.
    presence_suffix:
        Optional suffix to append to column names to find presence indicators.
        Defaults to '_is_present' for MSRBench.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono. When ``False`` the original channel layout
        is preserved.
    """
    
    def __init__(
        self,
        split: str,
        columns: Optional[Sequence[str]] = None,
        presence_columns: Optional[Dict[str, str]] = None,
        presence_suffix: Optional[str] = None,
        sr: int = 48_000,
        mono: bool = False,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.split = split
        # Use provided root, or fall back to default from config_updates
        if root is None:
            root = Path(DATA_PATH_MSR_BENCH)
        else:
            root = Path(root)

        if "/opt/datasets/HF_datasets/saved/" in str(root):
            self.on_musica_cluster = False
        elif "/data/<username>/" in str(root):
            self.on_musica_cluster = True
        else:
            raise ValueError(f"Dataset root path {root} not recognized for cluster settings.")

        self.ds = load_msr_bench(split, root=root)
        
        # If columns not specified, use mixture_audio and source_audio
        if columns is None:
            self.columns = ['mixture_audio', 'source_audio']
        else:
            self.columns = list(columns)
        
        self.sr = int(sr)
        self.mono = bool(mono)
        self.presence_columns = presence_columns or {}
        # Default presence suffix for MSRBench is '_is_present'
        self.presence_suffix = presence_suffix if presence_suffix is not None else '_is_present'
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def _is_present(self, row: Dict[str, Any], column: str) -> bool:
        if column in self.presence_columns:
            key = self.presence_columns[column]
            return bool(row.get(key, False))
        if self.presence_suffix:
            # For MSRBench, presence flags are like vocals_is_present (not vocals_audio_is_present)
            # So we need to remove _audio suffix if present before adding the presence suffix
            base_column = column.replace('_audio', '') if column.endswith('_audio') else column
            key = f"{base_column}{self.presence_suffix}"
            if key in row:
                return bool(row.get(key, False))
        return bool(row.get(column))
    
    def _load_full_audio(self, audio_data: Union[Dict[str, Any], str]) -> torch.Tensor:
        """
        Load the full audio file from bytes (if available) or file path.
        
        Parameters
        ----------
        audio_data:
            Either a dict with 'bytes' key containing audio bytes, or a string path
            to an audio file.
        """
        # If audio_data is a dict (from HF dataset with Audio feature), load from bytes
        if isinstance(audio_data, dict):
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                # Load from bytes
                audio_bytes = audio_data['bytes']
                wav, sr = sf.read(io.BytesIO(audio_bytes))
                # soundfile returns (samples, channels) for multi-channel, (samples,) for mono
                # Convert to (channels, samples) format
                if wav.ndim == 1:
                    wav = wav.reshape(1, -1)  # (1, samples) for mono
                else:
                    wav = wav.T  # (channels, samples)
                
                # Resample if needed
                if sr != self.sr:
                    # librosa.resample expects (channels, samples) or (samples,)
                    if wav.shape[0] == 1:
                        # Mono: resample directly
                        wav = librosa.resample(wav[0], orig_sr=sr, target_sr=self.sr)
                        wav = wav.reshape(1, -1)
                    else:
                        # Multi-channel: resample each channel
                        wav_resampled = []
                        for ch in range(wav.shape[0]):
                            wav_resampled.append(librosa.resample(wav[ch], orig_sr=sr, target_sr=self.sr))
                        wav = np.stack(wav_resampled, axis=0)
                
                # Handle mono conversion if requested
                if self.mono and wav.shape[0] > 1:
                    wav = np.mean(wav, axis=0, keepdims=True)
            elif 'path' in audio_data and audio_data['path']:
                # Fall back to path if bytes not available
                wav, _ = librosa.load(audio_data['path'], sr=self.sr, mono=self.mono)
                # librosa returns (channels, samples) or (samples,) for mono
                if wav.ndim == 1:
                    wav = np.expand_dims(wav, axis=0)
            else:
                raise ValueError("Audio data dict must contain either 'bytes' or 'path'")
        else:
            # Assume it's a file path string
            wav, _ = librosa.load(audio_data, sr=self.sr, mono=self.mono)
            # librosa returns (channels, samples) or (samples,) for mono
            if wav.ndim == 1:
                wav = np.expand_dims(wav, axis=0)
        
        # If mono=False and we got mono audio, duplicate to stereo
        if not self.mono and wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)
        
        return torch.from_numpy(wav).to(torch.float32)
    
    def _extract_stem(self, path: str) -> str:
        """
        Extract stem name from path (e.g., "Bass.zip" -> "Bass").
        
        Parameters
        ----------
        path:
            File path that may contain extension like .zip
            
        Returns
        -------
        Stem name without extension
        """
        if not path:
            raise Exception()
        # Remove any extension (e.g., .zip, .wav, etc.)
        stem = Path(path).stem
        return stem
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[int(idx)]
        
        item: Dict[str, Any] = {
            "index": idx,
            "clip_id": row.get("clip_id", ""),
            "family": row.get("family", ""),
            "source_label": row.get("source_label", ""),
            "split": row.get("split", ""),
            "duration_seconds": row.get("duration_seconds", 0.0),
            "sample_rate": row.get("sample_rate", self.sr),
        }
        
        # Load mixture_audio (always present)
        if 'mixture_audio' in row and row['mixture_audio']:
            mixture_data = row['mixture_audio']

            if self.on_musica_cluster:
                mixture_data = mixture_data.replace('/opt/datasets/music_generation/msr_bench/', '/data/<username>/msr_separation/msr_bench_files/')

            mixture_waveform = self._load_full_audio(mixture_data)
            # Extract path for metadata
            if isinstance(mixture_data, dict):
                mixture_path = mixture_data.get('path', '')
            else:
                mixture_path = mixture_data if mixture_data else ''
            item['mixture'] = {
                "waveform": mixture_waveform,
                "path": mixture_path,
                "is_present": True,
            }
        
        # Load source_audio (the single source that's present)
        if 'source_audio' in row and row['source_audio']:
            source_data = row['source_audio']

            if self.on_musica_cluster:
                source_data = source_data.replace('/opt/datasets/music_generation/msr_bench/', '/data/<username>/msr_separation/msr_bench_files/')

            source_waveform = self._load_full_audio(source_data)
            # Extract path for metadata
            if isinstance(source_data, dict):
                source_path = source_data.get('path', '')
            else:
                source_path = source_data if source_data else ''
            
            # Extract actual stem from path and validate against source_label
            source_label = row.get("source_label", "")
            
            item['source'] = {
                "waveform": source_waveform,
                "path": source_path,
                "label": source_label,
                "is_present": True,
            }
        
        # Optionally load individual stem columns if requested
        for col in self.columns:
            if col in ['mixture_audio', 'source_audio']:
                # Already handled above
                continue
            
            present = self._is_present(row, col)
            audio_data = row.get(col)

            # Extract path for metadata (could be in dict or direct string)
            if isinstance(audio_data, dict):
                path = audio_data.get('path', '')
            else:
                path = audio_data if audio_data else ''
            
            # Extract stem from path (e.g., "Bass.zip" -> "Bass")
            # CRITICAL: For MSRBench, we should use the column name as the stem identifier,
            # not the path, because the path might not match the column name.
            # The column name (e.g., 'vocals_audio') is the source of truth for which stem this is.
            # Only extract from path if path is available and we want to validate it matches.
            if path:
                path_stem = self._extract_stem(path)
                # Use column name as the stem identifier (remove _audio suffix if present)
                col_key_for_stem = col.replace('_audio', '') if col.endswith('_audio') else col
                # Normalize both for comparison (case-insensitive)
                path_stem_lower = path_stem.lower()
                col_key_lower = col_key_for_stem.lower()
                # If they don't match, warn but use column name as source of truth
                if path_stem_lower != col_key_lower and path_stem_lower not in ['', 'none', 'null']:
                    raise ValueError(f"Stem mismatch in MSRBench dataset: column '{col}' (expected stem: '{col_key_for_stem}') has path with stem '{path_stem}'")
                stem = col_key_for_stem  # Use column name as stem identifier
            else:
                # No path available, use column name
                stem = col.replace('_audio', '') if col.endswith('_audio') else col
            
            if present and audio_data:
                if self.on_musica_cluster:
                    audio_data = audio_data.replace('/opt/datasets/music_generation/msr_bench/',
                                                      '/data/<username>/msr_separation/msr_bench_files/')

                waveform = self._load_full_audio(audio_data)
            else:
                # Create silent waveform with correct channel structure
                # Use mixture length as reference if available
                if 'mixture' in item:
                    ref_length = item['mixture']['waveform'].shape[-1]
                else:
                    ref_length = 1
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, ref_length), dtype=torch.float32)
            
            # Use column name without _audio suffix for cleaner keys
            col_key = col.replace('_audio', '') if col.endswith('_audio') else col
            item[col_key] = {
                "waveform": waveform,
                "stem": stem,
                "path": path,
                "is_present": bool(present),
            }
        
        return item
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that pads waveforms to the same length."""
        # Find the maximum length in the batch
        max_length = 0
        for b in batch:
            # Check mixture and source first
            if 'mixture' in b:
                waveform = b['mixture']['waveform']
                if waveform.shape[-1] > max_length:
                    max_length = waveform.shape[-1]
            if 'source' in b:
                waveform = b['source']['waveform']
                if waveform.shape[-1] > max_length:
                    max_length = waveform.shape[-1]
            # Check other columns
            for col in self.columns:
                if col in ['mixture_audio', 'source_audio']:
                    continue
                col_key = col.replace('_audio', '') if col.endswith('_audio') else col
                if col_key in b:
                    waveform = b[col_key]["waveform"]
                    if waveform.shape[-1] > max_length:
                        max_length = waveform.shape[-1]
        
        # Pad all waveforms to the same length
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "clip_id": [b.get("clip_id", "") for b in batch],
            "family": [b.get("family", "") for b in batch],
            "source_label": [b.get("source_label", "") for b in batch],
            "split": [b.get("split", "") for b in batch],
            "duration_seconds": [b.get("duration_seconds", 0.0) for b in batch],
        }
        
        # Collate mixture
        if 'mixture' in batch[0]:
            waveforms = []
            paths = []
            lengths = []
            for b in batch:
                waveform = b['mixture']['waveform']
                lengths.append(waveform.shape[-1])
                if waveform.shape[-1] < max_length:
                    pad_length = max_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                waveforms.append(waveform)
                paths.append(b['mixture'].get('path', ''))
            out['mixture'] = {
                "waveform": torch.stack(waveforms, dim=0),
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "paths": paths,
            }
        
        # Collate source
        if 'source' in batch[0]:
            waveforms = []
            paths = []
            labels = []
            lengths = []
            for b in batch:
                waveform = b['source']['waveform']
                lengths.append(waveform.shape[-1])
                if waveform.shape[-1] < max_length:
                    pad_length = max_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                waveforms.append(waveform)
                paths.append(b['source'].get('path', ''))
                labels.append(b['source'].get('label', ''))
            out['source'] = {
                "waveform": torch.stack(waveforms, dim=0),
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "paths": paths,
                "labels": labels,
            }
        
        # Collate other columns
        for col in self.columns:
            if col in ['mixture_audio', 'source_audio']:
                continue
            col_key = col.replace('_audio', '') if col.endswith('_audio') else col
            if col_key in batch[0]:
                waveforms = []
                stems = []
                paths = []
                present = []
                lengths = []
                for b in batch:
                    waveform = b[col_key]["waveform"]
                    lengths.append(waveform.shape[-1])
                    if waveform.shape[-1] < max_length:
                        pad_length = max_length - waveform.shape[-1]
                        waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                    waveforms.append(waveform)
                    stems.append(b[col_key].get("stem", ""))
                    paths.append(b[col_key].get("path", ""))
                    present.append(b[col_key].get("is_present", False))
                out[col_key] = {
                    "waveform": torch.stack(waveforms, dim=0),
                    "lengths": torch.tensor(lengths, dtype=torch.long),
                    "stems": stems,
                    "paths": paths,
                    "is_present": present,
                }
        
        return out


__all__ = [
    "MultiAudioFullSongDataset",
    "load_msr_bench",
]
