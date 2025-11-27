"""
MSR Test Set dataset for inference-only evaluation.

This dataset loads mixture files only (no ground truth).
- If a file is longer than 10 seconds, takes the first 10 seconds
- If a file is shorter than 10 seconds, pads it with zeros to 10 seconds
- Always returns exactly 10 seconds of audio at 48kHz

Directory structure:
    root/
        Vocals/
            file1.flac
            file2.flac
        Bass/
            file1.flac
            file2.flac
        ...

Each subfolder contains mixture files, and the subfolder name indicates
which stem should be separated from the mixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

try:
    import librosa
    import soundfile as sf
except ModuleNotFoundError as exc:
    raise ImportError("librosa and soundfile are required for audio loading.") from exc


class MSRTestSetDataset(TorchDataset):
    """
    Simple dataset for inference-only evaluation with mixture files only.
    
    This dataset:
    - Loads mixture audio files from subfolders named after stems
    - Each subfolder contains mixture files
    - The subfolder name indicates which stem to separate from the mixture
    - Takes first 10 seconds if file is longer, pads to 10 seconds if shorter
    - Always returns exactly 10 seconds of audio
    - Returns only mixture waveforms (no ground truth)
    
    Directory structure:
        root/
            Vocals/
                mixture1.flac
                mixture2.flac
            Bass/
                mixture1.flac
                mixture2.flac
            ...
    
    Parameters
    ----------
    root:
        Root directory containing stem-named subfolders with mixture files.
    columns:
        Optional list of stem names to include. If None, uses all subfolders found.
        Default: None (uses all subfolders)
    sr:
        Target sampling rate for loaded audio. Default: 48000
    mono:
        Whether to load audio as mono. When False, preserves stereo. Default: False
    expected_duration:
        Expected duration in seconds. Files will be trimmed/padded to this duration.
        Default: 10.0
    tolerance:
        Not used (kept for compatibility). Default: 0.01
    """
    
    # Map folder names to column/stem names used in the project
    FOLDER_TO_COLUMN = {
        "Vocals": "vocals",
        "Bass": "bass",
        "Drums": "drums",
        "Guitars": "guitars",
        "Keyboards": "keyboards",
        "Synthesizers": "synthesizers",
        "Percussions": "percussions",
        "Orchestral Elements": "orchestral",
    }
    
    def __init__(
        self,
        root: Union[str, Path],
        columns: Optional[List[str]] = None,
        sr: int = 48_000,
        mono: bool = False,
        expected_duration: float = 10.0,
        tolerance: float = 0.01,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        self.sr = int(sr)
        self.mono = bool(mono)
        self.expected_duration = float(expected_duration)
        self.tolerance = float(tolerance)
        self.expected_samples = int(self.expected_duration * self.sr)
        
        # Discover samples from subfolders
        self.samples = self._discover_samples(columns)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.root}")
        
        print(f"MSRTestSetDataset: Found {len(self.samples)} samples (will be trimmed/padded to {self.expected_duration}s)")
    
    def _discover_samples(self, columns: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        Discover all valid samples from stem-named subfolders.
        
        Each subfolder contains mixture files, and the subfolder name indicates
        which stem should be separated. The folder name is mapped to a column/stem name
        using FOLDER_TO_COLUMN mapping.
        
        Returns a list of sample dictionaries with file paths and stem labels.
        """
        samples = []
        
        # Find all subdirectories (stem folders)
        stem_folders = [d for d in self.root.iterdir() if d.is_dir()]
        
        if not stem_folders:
            raise ValueError(f"No subfolders found in {self.root}. Expected subfolders named after stems (e.g., Vocals, Bass, Drums).")
        
        # Filter by columns if specified (filter by mapped column names)
        if columns is not None:
            columns_lower = [col.lower() for col in columns]
            # Filter folders that map to requested columns
            stem_folders = [
                folder for folder in stem_folders
                if folder.name in self.FOLDER_TO_COLUMN and 
                self.FOLDER_TO_COLUMN[folder.name].lower() in columns_lower
            ]
        
        # Process each stem folder
        for stem_folder in sorted(stem_folders):
            folder_name = stem_folder.name
            
            # Map folder name to column/stem name
            if folder_name not in self.FOLDER_TO_COLUMN:
                print(f"Warning: Folder '{folder_name}' not in FOLDER_TO_COLUMN mapping, skipping")
                continue
            
            column_name = self.FOLDER_TO_COLUMN[folder_name]
            
            # Find all audio files in this stem folder
            audio_files = sorted(stem_folder.glob("*.flac")) + sorted(stem_folder.glob("*.wav"))
            
            for audio_file in audio_files:
                samples.append({
                    "mixture_path": str(audio_file),
                    "stem_label": column_name,  # Use mapped column name, not folder name
                    "folder_name": folder_name,  # Keep folder name for reference
                    "clip_id": f"{audio_file.stem}",
                })
        
        return samples
    
    def _load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio file and return as torch tensor.
        
        - If file is longer than expected_duration, takes first expected_duration seconds
        - If file is shorter than expected_duration, pads with zeros to expected_duration
        - Always returns exactly expected_samples samples
        
        Returns tensor of shape (channels, expected_samples) at self.sr.
        """
        wav, sr = librosa.load(str(file_path), sr=self.sr, mono=self.mono, duration=self.expected_duration)
        
        # librosa returns (channels, samples) or (samples,) for mono
        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)
        
        # If mono=False and we got mono audio, duplicate to stereo
        if not self.mono and wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)
        
        # Convert to tensor
        wav_tensor = torch.from_numpy(wav).to(torch.float32)
        
        # Ensure exactly expected_samples length (trim or pad)
        current_samples = wav_tensor.shape[-1]
        if current_samples > self.expected_samples:
            # Trim to first expected_samples
            wav_tensor = wav_tensor[..., :self.expected_samples]
        elif current_samples < self.expected_samples:
            # Pad with zeros to expected_samples
            pad_length = self.expected_samples - current_samples
            wav_tensor = F.pad(wav_tensor, (0, pad_length), mode='constant', value=0.0)
        
        return wav_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # Load mixture (will be trimmed/padded to exactly expected_duration)
        mixture_waveform = self._load_audio(sample_info["mixture_path"])
        
        # Ensure waveform is exactly expected_samples (should already be from _load_audio)
        assert mixture_waveform.shape[-1] == self.expected_samples, \
            f"Expected {self.expected_samples} samples, got {mixture_waveform.shape[-1]}"
        
        # Get the mapped column/stem name
        column_name = sample_info["stem_label"]  # Already mapped from folder name
        
        item: Dict[str, Any] = {
            "index": idx,
            "clip_id": sample_info["clip_id"],
            "source_label": column_name,  # The stem to separate (mapped column name)
            # "column": column_name,  # Column/stem name
            # "stem": column_name,  # Stem name (alias)
            "mixture": {
                "waveform": mixture_waveform,
                "path": sample_info["mixture_path"],
                "is_present": True,
            },
            # "source": {
            #     "waveform": torch.zeros_like(mixture_waveform),  # No ground truth
            #     "path": "",
            #     "label": column_name,  # Mapped column/stem name
            #     "is_present": False,  # No ground truth available
            # },
            "duration_seconds": self.expected_duration,
            "sample_rate": self.sr,
        }
        
        return item
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that pads waveforms to the same length."""
        # All should be the same length (10 seconds), but pad just in case
        max_length = max(b["mixture"]["waveform"].shape[-1] for b in batch)
        
        # Collate mixture
        mixture_waveforms = []
        mixture_paths = []
        mixture_lengths = []
        
        # # Collate source (empty, but keep structure for compatibility)
        # source_waveforms = []
        # source_paths = []
        # source_labels = []
        # source_lengths = []
        
        clip_ids = []
        source_labels_list = []
        columns = []
        stems = []
        
        for b in batch:
            # Mixture
            mixture_waveform = b["mixture"]["waveform"]
            mixture_lengths.append(mixture_waveform.shape[-1])
            if mixture_waveform.shape[-1] < max_length:
                pad_length = max_length - mixture_waveform.shape[-1]
                mixture_waveform = F.pad(mixture_waveform, (0, pad_length), mode='constant', value=0.0)
            mixture_waveforms.append(mixture_waveform)
            mixture_paths.append(b["mixture"].get("path", ""))
            
            # # Source (no ground truth, but keep structure)
            # source_waveform = b["source"]["waveform"]
            # source_lengths.append(source_waveform.shape[-1])
            # if source_waveform.shape[-1] < max_length:
            #     pad_length = max_length - source_waveform.shape[-1]
            #     source_waveform = F.pad(source_waveform, (0, pad_length), mode='constant', value=0.0)
            # source_waveforms.append(source_waveform)
            # source_paths.append(b["source"].get("path", ""))
            # source_labels.append(b["source"].get("label", ""))
            
            clip_ids.append(b.get("clip_id", ""))
            source_labels_list.append(b.get("source_label", ""))
            columns.append(b.get("column", ""))
            stems.append(b.get("stem", ""))
        
        return {
            "mixture": {
                "waveform": torch.stack(mixture_waveforms, dim=0),
                "lengths": torch.tensor(mixture_lengths, dtype=torch.long),
                "paths": mixture_paths,
            },
            # "source": {
            #     "waveform": torch.stack(source_waveforms, dim=0),
            #     "lengths": torch.tensor(source_lengths, dtype=torch.long),
            #     "paths": source_paths,
            #     "labels": source_labels,
            # },
            "clip_id": clip_ids,
            "source_label": source_labels_list,
            "column": columns,
            "stem": stems,
            "duration_seconds": [b.get("duration_seconds", self.expected_duration) for b in batch],
            "sample_rate": [b.get("sample_rate", self.sr) for b in batch],
        }


__all__ = ["MSRTestSetDataset"]

