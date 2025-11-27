"""
Utility wrapper around torch_audiomentations to build a music-focused effects chain.

The pipeline currently applies a handful of lightweight effects followed by an
optional impulse-response convolution stage.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations import (
    AddColoredNoise,
    ApplyImpulseResponse,    
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PolarityInversion,
    ShuffleChannels,
)
import torchaudio



logger = logging.getLogger(__name__)

IrPaths = Optional[Union[Sequence[str], str]]


class Compressor(BaseWaveformTransform):
    """
    Apply dynamic range compression using dasp_pytorch.

    This reduces the dynamic range by attenuating signals above a threshold,
    making quiet parts relatively louder and loud parts relatively quieter.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int = 48_000,
        threshold_db: tuple[float, float] = (-24.0, -12.0),
        ratio: tuple[float, float] = (2.0, 6.0),
        attack_ms: tuple[float, float] = (5.0, 20.0),
        release_ms: tuple[float, float] = (50.0, 200.0),
        knee_db: tuple[float, float] = (0.0, 6.0),
        makeup_gain_db: tuple[float, float] = (0.0, 6.0),
        p: float = 0.1,
    ) -> None:
        """
        Initialize the compressor transform.

        Parameters
        ----------
        sample_rate:
            Sample rate for processing.
        threshold_db:
            Range for threshold in dB (min, max). Signals above this are compressed.
        ratio:
            Range for compression ratio (min, max). Higher values = more compression.
        attack_ms:
            Range for attack time in milliseconds (min, max). How quickly compression engages.
        release_ms:
            Range for release time in milliseconds (min, max). How quickly compression releases.
        knee_db:
            Range for knee width in dB (min, max). Smoother transition at threshold.
        makeup_gain_db:
            Range for makeup gain in dB (min, max). Compensates for level reduction.
        p:
            Probability of applying this transform.
        """
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee_db = knee_db
        self.makeup_gain_db = makeup_gain_db
        self._compressor = None

    def randomize_parameters(self, samples: torch.Tensor, sample_rate: int, targets=None, target_rate: Optional[int] = None):
        """Randomize compressor parameters for each batch."""
        batch_size = samples.shape[0]
        
        # Import here to avoid hard dependency
        try:
            import dasp_pytorch
        except ImportError:
            raise RuntimeError("dasp_pytorch is required for Compressor transform but is not installed")
        
        if self._compressor is None or self._compressor.sample_rate != sample_rate:
            self._compressor = dasp_pytorch.Compressor(sample_rate=sample_rate)
        
        # Generate random normalized parameters [0, 1] for each sample in batch
        self.transform_parameters["compressor_params"] = torch.rand(
            batch_size, self._compressor.num_params, device=samples.device
        )

    def apply_transform(
        self, samples: torch.Tensor, sample_rate: Optional[int] = None, targets=None, target_rate: Optional[int] = None
    ):
        """Apply compression with randomized parameters."""
        from torch_audiomentations.utils.object_dict import ObjectDict
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Get compressor instance
        try:
            import dasp_pytorch
        except ImportError:
            # If dasp_pytorch not available, return unchanged
            return ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
        
        if self._compressor is None or self._compressor.sample_rate != sample_rate:
            self._compressor = dasp_pytorch.Compressor(sample_rate=sample_rate)
        
        # Apply compression with normalized parameters
        params = self.transform_parameters.get("compressor_params")
        if params is None:
            # Fallback: use conservative default parameters
            params = torch.ones(samples.shape[0], self._compressor.num_params, device=samples.device) * 0.3
        
        processed = self._compressor.process_normalized(samples, params)
        
        return ObjectDict(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class NoiseShapedReverb(BaseWaveformTransform):
    """
    Apply artificial reverberation using dasp_pytorch.

    Adds spatial depth and ambience to the audio signal while
    maintaining clarity through noise shaping.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int = 48_000,
        room_size: tuple[float, float] = (0.3, 0.7),
        damping: tuple[float, float] = (0.3, 0.7),
        wet_level: tuple[float, float] = (0.1, 0.4),
        dry_level: tuple[float, float] = (0.7, 0.9),
        width: tuple[float, float] = (0.5, 1.0),
        p: float = 0.1,
    ) -> None:
        """
        Initialize the reverb transform.

        Parameters
        ----------
        sample_rate:
            Sample rate for processing.
        room_size:
            Range for room size (min, max). Larger = longer reverb tail.
        damping:
            Range for high-frequency damping (min, max). Higher = darker reverb.
        wet_level:
            Range for reverb amount (min, max). How much reverb to mix in.
        dry_level:
            Range for dry signal level (min, max). Original signal level.
        width:
            Range for stereo width (min, max). Spatial spread of reverb.
        p:
            Probability of applying this transform.
        """
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.width = width
        self._reverb = None

    def randomize_parameters(self, samples: torch.Tensor, sample_rate: int, targets=None, target_rate: Optional[int] = None):
        """Randomize reverb parameters for each batch."""
        batch_size = samples.shape[0]
        
        # Import here to avoid hard dependency
        try:
            import dasp_pytorch
        except ImportError:
            raise RuntimeError("dasp_pytorch is required for NoiseShapedReverb transform but is not installed")
        
        if self._reverb is None or self._reverb.sample_rate != sample_rate:
            self._reverb = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)
        
        # Generate random normalized parameters [0, 1] for each sample in batch
        self.transform_parameters["reverb_params"] = torch.rand(
            batch_size, self._reverb.num_params, device=samples.device
        )

    def apply_transform(
        self, samples: torch.Tensor, sample_rate: Optional[int] = None, targets=None, target_rate: Optional[int] = None
    ):
        """Apply reverb with randomized parameters."""
        from torch_audiomentations.utils.object_dict import ObjectDict
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Get reverb instance
        try:
            import dasp_pytorch
        except ImportError:
            # If dasp_pytorch not available, return unchanged
            return ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
        
        if self._reverb is None or self._reverb.sample_rate != sample_rate:
            self._reverb = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)
        
        # Apply reverb with normalized parameters
        params = self.transform_parameters.get("reverb_params")
        if params is None:
            # Fallback: use conservative default parameters (subtle reverb)
            params = torch.ones(samples.shape[0], self._reverb.num_params, device=samples.device) * 0.25
        
        processed = self._reverb.process_normalized(samples, params)
        
        return ObjectDict(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class DownUpSample(BaseWaveformTransform):
    """
    Temporarily resample audio to a lower rate before restoring the original sample rate.

    This can introduce subtle artifacts emulating bandwidth-limited processing stages.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(self, source_sample_rate: int = 48_000, target_sample_rate: int = 16_000, p: float = 0.1) -> None:
        if Compose is None:  # pragma: no cover - runtime guard
            raise RuntimeError("torch_audiomentations is required for DownUpSample transform")
        if torchaudio is None:  # pragma: no cover - runtime guard
            raise RuntimeError("torchaudio is required for DownUpSample transform")
        super().__init__(p=p)
        self.target_sample_rate = int(target_sample_rate)
        self.source_sample_rate = int(source_sample_rate)

    def apply_transform(self, samples: torch.Tensor, sample_rate: Optional[int] = None, targets=None, target_rate: Optional[int] = None):
        """Apply downsampling and upsampling. Must return ObjectDict for torch_audiomentations compatibility."""
        from torch_audiomentations.utils.object_dict import ObjectDict
        
        if sample_rate is None:
            sample_rate = self.source_sample_rate
        if sample_rate == self.target_sample_rate:
            return ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )

        batch, channels, frames = samples.shape
        flattened = samples.reshape(batch * channels, frames).contiguous()
        dtype = samples.dtype
        device = samples.device

        resample_down = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to("cpu")
        resample_up = torchaudio.transforms.Resample(orig_freq=self.target_sample_rate, new_freq=sample_rate).to("cpu")

        float_data = flattened.to(device="cpu", dtype=torch.float32)
        processed = resample_up(resample_down(float_data))
        processed = processed.reshape(batch, channels, -1)

        if processed.size(-1) != frames:
            if processed.size(-1) > frames:
                processed = processed[..., :frames]
            else:
                pad = frames - processed.size(-1)
                processed = F.pad(processed, (0, pad))

        processed = processed.to(device=device, dtype=dtype)
        
        return ObjectDict(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class MusicEffectPipeline:
    """Prepare a reusable audio augmentation pipeline for music signals."""

    def __init__(
        self,
        sample_rate: int,
        ir_paths: IrPaths = None,
        *,
        gain_db: tuple[float, float] = (-4.0, 2.0),
        noise_snr_db: tuple[float, float] = (24.0, 48.0),
        high_pass_hz: tuple[float, float] = (100.0, 150.0),
        low_pass_hz: tuple[float, float] = (12_000.0, 20_000.0),
        shuffle_p: float = 0.25,
        polarity_p: float = 0.1,
        ir_p: float = 0.1,
    ) -> None:
        if Compose is None:  # pragma: no cover - runtime guard
            raise RuntimeError("torch_audiomentations is required for MusicEffectPipeline")

        self.sample_rate = int(sample_rate)
        self._ir_paths = self._process_ir_paths(ir_paths)
        self._ir_probability = float(ir_p)
        # signal flow principle: fix problems → control dynamics → shape tone → add space → final adjustments
        transforms = [
            # Input - stage effects
            DownUpSample(target_sample_rate=16_000, p=0.1),
            HighPassFilter(min_cutoff_freq=float(high_pass_hz[0]), max_cutoff_freq=float(high_pass_hz[1]), p=0.1),
            # LowPassFilter(min_cutoff_freq=float(low_pass_hz[0]), max_cutoff_freq=float(low_pass_hz[1]), p=0.1),
            
            # Dynamic range compression can be too aggressive for music
            Compressor(sample_rate=self.sample_rate, p=0.1),

            # Tone shaping and noise
            AddColoredNoise(min_snr_in_db=float(noise_snr_db[0]), max_snr_in_db=float(noise_snr_db[1]), p=0.1),

            # Spatial effects
            # NoiseShapedReverb(sample_rate=self.sample_rate, p=0.1),            
        ]

        if self._ir_paths:
            transforms.append(ApplyImpulseResponse(self._ir_paths, p=self._ir_probability, compensate_for_propagation_delay=True))
        else:
            logger.debug("MusicEffectPipeline initialized without impulse responses; skipping ApplyImpulseResponse.")

        transforms.extend([
            # Final - output stage effects
            Gain(min_gain_in_db=float(gain_db[0]), max_gain_in_db=float(gain_db[1]), p=0.1),  # HERE: changed gain p=0.5 to p=0.1
            PolarityInversion(p=float(polarity_p)),
            ShuffleChannels(p=float(shuffle_p), sample_rate=self.sample_rate),
        ])

        self.effect_chain = Compose(transforms=transforms, output_type="tensor")

    def __call__(self, samples, sample_rate: Optional[int] = None, **kwargs):  # type: ignore[override]
        """
        Apply the composed augmentation chain.

        Parameters
        ----------
        samples:
            Tensor shaped (batch, channels, frames) with values in -1..1 range.
        sample_rate:
            Optionally override the stored sample rate at call time.
        kwargs:
            Forwarded to the underlying Compose callable.
        """
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate

        # original_device = samples.device
        # original_dtype = samples.dtype
        #
        # processed = samples.detach().to(device="cpu", dtype=torch.float32)
        # augmented = self.effect_chain(processed, sample_rate=sr, **kwargs)
        #
        # return augmented.to(device=original_device, dtype=original_dtype)
        return self.effect_chain(samples, sample_rate=sr, **kwargs)

    def update_ir_paths(self, ir_paths: IrPaths) -> None:
        """Update impulse-response sources and rebuild the transform if needed."""
        processed = self._process_ir_paths(ir_paths)
        if processed == self._ir_paths:
            return
        self._ir_paths = processed

        transforms = list(self.effect_chain.transforms)
        transforms = [t for t in transforms if not isinstance(t, ApplyImpulseResponse)]
        if self._ir_paths:
            transforms.append(ApplyImpulseResponse(self._ir_paths, p=self._ir_probability, compensate_for_propagation_delay=True, output_type="tensor"))
        else:
            logger.debug("Removed ApplyImpulseResponse from pipeline due to missing IR files.")

        self.effect_chain = Compose(transforms=transforms, output_type="tensor")

    def _process_ir_paths(self, ir_paths: IrPaths) -> List[str]:
        """Process IR paths input to ensure it resolves to a list of audio files."""
        if ir_paths is None:
            return []
        if isinstance(ir_paths, str):
            path = ir_paths
            if path.endswith((".wav", ".flac")):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"IR file not found: {path}")
                return [path]
            if os.path.isdir(path):
                files: list[str] = []
                for root, _, filenames in os.walk(path):
                    for filename in filenames:
                        if filename.endswith((".wav", ".flac")):
                            files.append(os.path.join(root, filename))
                if not files:
                    logger.warning("No .wav or .flac files found in directory: %s", path)
                return files
            logger.warning("Provided ir_paths string is neither a file nor a directory: %s", path)
            return []
        if isinstance(ir_paths, Sequence):
            files = [
                os.fspath(path)
                for path in ir_paths
                if isinstance(path, str)
                and path.endswith((".wav", ".flac"))
                and os.path.isfile(path)
            ]
            removed = len(ir_paths) - len(files)
            if removed:
                logger.warning("Discarded %d invalid IR paths from the provided list.", removed)
            return files

        logger.warning("Invalid ir_paths format. Expected None, str, or Sequence[str].")
        return []


__all__ = ["MusicEffectPipeline", "Compressor", "NoiseShapedReverb", "DownUpSample"]
