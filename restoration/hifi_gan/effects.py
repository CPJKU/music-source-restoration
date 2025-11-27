from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
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
    from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
    from torch_audiomentations.utils.object_dict import ObjectDict

    HAS_TA = True
except ImportError:  # pragma: no cover - optional dependency
    Compose = None
    ApplyImpulseResponse = None  # type: ignore[assignment]
    BaseWaveformTransform = object  # type: ignore[assignment]
    ObjectDict = dict  # type: ignore[assignment]
    HAS_TA = False

try:  # pragma: no cover - optional dependency
    import torchaudio

    HAS_TORCHAUDIO = True
except ImportError:  # pragma: no cover - optional dependency
    torchaudio = None  # type: ignore[assignment]
    HAS_TORCHAUDIO = False

logger = logging.getLogger(__name__)

IrPaths = Optional[Union[Sequence[str], str]]


if HAS_TA:

    class Compressor(BaseWaveformTransform):
        """Dynamic range compression via dasp_pytorch."""

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
            super().__init__(p=p)
            self.sample_rate = int(sample_rate)
            self.threshold_db = threshold_db
            self.ratio = ratio
            self.attack_ms = attack_ms
            self.release_ms = release_ms
            self.knee_db = knee_db
            self.makeup_gain_db = makeup_gain_db
            self._compressor = None

        def randomize_parameters(
            self,
            samples: torch.Tensor,
            sample_rate: int,
            targets=None,
            target_rate: Optional[int] = None,
        ) -> None:
            batch_size = samples.shape[0]
            try:
                import dasp_pytorch  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("dasp_pytorch is required for Compressor transform but is not installed") from exc

            if self._compressor is None or self._compressor.sample_rate != sample_rate:
                self._compressor = dasp_pytorch.Compressor(sample_rate=sample_rate)

            self.transform_parameters["compressor_params"] = torch.rand(
                batch_size,
                self._compressor.num_params,
                device=samples.device,
            )

        def apply_transform(
            self,
            samples: torch.Tensor,
            sample_rate: Optional[int] = None,
            targets=None,
            target_rate: Optional[int] = None,
        ):
            try:
                import dasp_pytorch  # type: ignore
            except ImportError:  # pragma: no cover - optional dependency
                return ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)

            sample_rate = int(sample_rate or self.sample_rate)
            if self._compressor is None or self._compressor.sample_rate != sample_rate:
                self._compressor = dasp_pytorch.Compressor(sample_rate=sample_rate)

            params = self.transform_parameters.get("compressor_params")
            if params is None:
                params = torch.full(
                    (samples.shape[0], self._compressor.num_params),
                    0.3,
                    device=samples.device,
                )
            processed = self._compressor.process_normalized(samples, params)
            return ObjectDict(samples=processed, sample_rate=sample_rate, targets=targets, target_rate=target_rate)


    class NoiseShapedReverb(BaseWaveformTransform):
        """Adds noise-shaped artificial reverb via dasp_pytorch."""

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
            super().__init__(p=p)
            self.sample_rate = int(sample_rate)
            self.room_size = room_size
            self.damping = damping
            self.wet_level = wet_level
            self.dry_level = dry_level
            self.width = width
            self._reverb = None

        def randomize_parameters(
            self,
            samples: torch.Tensor,
            sample_rate: int,
            targets=None,
            target_rate: Optional[int] = None,
        ) -> None:
            batch_size = samples.shape[0]
            try:
                import dasp_pytorch  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("dasp_pytorch is required for NoiseShapedReverb transform but is not installed") from exc

            if self._reverb is None or self._reverb.sample_rate != sample_rate:
                self._reverb = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)

            self.transform_parameters["reverb_params"] = torch.rand(
                batch_size,
                self._reverb.num_params,
                device=samples.device,
            )

        def apply_transform(
            self,
            samples: torch.Tensor,
            sample_rate: Optional[int] = None,
            targets=None,
            target_rate: Optional[int] = None,
        ):
            try:
                import dasp_pytorch  # type: ignore
            except ImportError:  # pragma: no cover - optional dependency
                return ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)

            sample_rate = int(sample_rate or self.sample_rate)
            if self._reverb is None or self._reverb.sample_rate != sample_rate:
                self._reverb = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)

            params = self.transform_parameters.get("reverb_params")
            if params is None:
                params = torch.full(
                    (samples.shape[0], self._reverb.num_params),
                    0.25,
                    device=samples.device,
                )
            processed = self._reverb.process_normalized(samples, params)
            return ObjectDict(samples=processed, sample_rate=sample_rate, targets=targets, target_rate=target_rate)


else:  # pragma: no cover - executed when torch_audiomentations missing

    class Compressor:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch_audiomentations is required for Compressor transform.")


    class NoiseShapedReverb:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("torch_audiomentations is required for NoiseShapedReverb transform.")


class DownUpSample(BaseWaveformTransform):
    """Downsample then upsample to introduce codec-style artefacts."""

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(self, source_sample_rate: int = 48_000, target_sample_rate: int = 16_000, p: float = 0.1) -> None:
        if not HAS_TA:
            raise RuntimeError("torch_audiomentations is required for DownUpSample transform.")
        if not HAS_TORCHAUDIO:
            raise RuntimeError("torchaudio is required for DownUpSample transform.")
        super().__init__(p=p)
        self.target_sample_rate = int(target_sample_rate)
        self.source_sample_rate = int(source_sample_rate)

    def apply_transform(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
        targets=None,
        target_rate: Optional[int] = None,
    ):
        sample_rate = int(sample_rate or self.source_sample_rate)
        if sample_rate == self.target_sample_rate:
            return ObjectDict(samples=samples, sample_rate=sample_rate, targets=targets, target_rate=target_rate)

        batch, channels, frames = samples.shape
        flattened = samples.reshape(batch * channels, frames).contiguous()
        dtype = samples.dtype

        resample_down = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to("cpu")  # type: ignore[arg-type]
        resample_up = torchaudio.transforms.Resample(orig_freq=self.target_sample_rate, new_freq=sample_rate).to("cpu")  # type: ignore[arg-type]

        float_data = flattened.to(device="cpu", dtype=torch.float32)
        processed = resample_up(resample_down(float_data))
        processed = processed.reshape(batch, channels, -1)

        if processed.size(-1) != frames:
            if processed.size(-1) > frames:
                processed = processed[..., :frames]
            else:
                pad = frames - processed.size(-1)
                processed = F.pad(processed, (0, pad))

        processed = processed.to(device=samples.device, dtype=dtype)
        return ObjectDict(samples=processed, sample_rate=sample_rate, targets=targets, target_rate=target_rate)


class CustomAddBackgroundNoise(BaseWaveformTransform):
    """Custom background noise addition that works with newer torchaudio versions.
    
    This avoids the torchaudio.info() compatibility issue in torch_audiomentations.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        background_paths: list[str],
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 20.0,
        p: float = 0.5,
    ) -> None:
        if not HAS_TORCHAUDIO:
            raise RuntimeError("torchaudio is required for CustomAddBackgroundNoise.")
        super().__init__(p=p)
        self.background_paths = background_paths
        self.min_snr_db = min_snr_in_db
        self.max_snr_db = max_snr_in_db
        
        # Pre-load all background noise files to avoid runtime I/O and compatibility issues
        self.background_samples = []
        logger.info(f"Pre-loading {len(background_paths)} background noise files...")
        for path in background_paths:
            try:
                audio, sr = torchaudio.load(path)
                # Convert to mono by averaging channels if needed
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                self.background_samples.append((audio, sr))
            except Exception as e:
                logger.warning(f"Failed to load background noise {path}: {e}")
        
        if not self.background_samples:
            raise RuntimeError("No valid background noise files could be loaded")
        
        logger.info(f"Successfully loaded {len(self.background_samples)} background noise files")

    def apply_transform(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
        targets=None,
        target_rate: Optional[int] = None,
    ):
        """Add background noise at random SNR."""
        batch_size, num_channels, num_samples = samples.shape
        
        # Process each sample in batch
        output = samples.clone()
        for b in range(batch_size):
            # Randomly select a background noise
            bg_audio, bg_sr = self.background_samples[torch.randint(len(self.background_samples), (1,)).item()]
            
            # Resample if needed
            if bg_sr != sample_rate:
                bg_audio = torchaudio.functional.resample(bg_audio, bg_sr, sample_rate)
            
            # Get random segment of background noise
            if bg_audio.shape[-1] >= num_samples:
                start_idx = torch.randint(0, bg_audio.shape[-1] - num_samples + 1, (1,)).item()
                bg_segment = bg_audio[:, start_idx:start_idx + num_samples]
            else:
                # Loop background if too short
                repeats = (num_samples // bg_audio.shape[-1]) + 1
                bg_audio_repeated = bg_audio.repeat(1, repeats)
                bg_segment = bg_audio_repeated[:, :num_samples]
            
            # Match channels (broadcast mono to stereo if needed)
            if bg_segment.shape[0] == 1 and num_channels > 1:
                bg_segment = bg_segment.repeat(num_channels, 1)
            elif bg_segment.shape[0] > num_channels:
                bg_segment = bg_segment[:num_channels]
            
            # Move to same device as input
            bg_segment = bg_segment.to(samples.device)
            
            # Calculate SNR and mix
            snr_db = torch.rand(1).item() * (self.max_snr_db - self.min_snr_db) + self.min_snr_db
            
            # Compute signal and noise power
            signal = samples[b]
            signal_power = torch.mean(signal ** 2)
            noise_power = torch.mean(bg_segment ** 2)
            
            # Calculate noise scaling factor
            snr_linear = 10 ** (snr_db / 10)
            if noise_power > 0:
                noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            else:
                noise_scale = 0.0
            
            # Mix signal and noise
            output[b] = signal + noise_scale * bg_segment
        
        return ObjectDict(samples=output, sample_rate=sample_rate, targets=targets, target_rate=target_rate)


class SimpleDegradationPipeline:
    """Simple degradation pipeline with downsampling and background noise.
    
    Applies codec-style degradation via downsampling/upsampling and adds
    background noise from a directory of audio files.
    """

    def __init__(
        self,
        sample_rate: int,
        background_noise_dir: str = "/opt/datasets/music_generation/background_sounds/gramophone/",
        *,
        downsample_rate: int = 16_000,
        noise_snr_db: tuple[float, float] = (0.0, 5.0),
        downsample_p: float = 0.1,
        noise_p: float = 0.1,
    ) -> None:
        """Initialize simple degradation pipeline.
        
        Parameters
        ----------
        sample_rate : int
            Target sample rate for processing
        background_noise_dir : str
            Path to directory containing background noise audio files
        downsample_rate : int
            Intermediate sample rate for downsample/upsample degradation
        noise_snr_db : tuple[float, float]
            Min and max SNR in dB for background noise
        downsample_p : float
            Probability of applying downsampling degradation
        noise_p : float
            Probability of applying background noise
        """
        self.sample_rate = int(sample_rate)
        self.background_noise_dir = background_noise_dir
        self.debug_message_shown = True

        if not HAS_TA or Compose is None:
            logger.warning("torch_audiomentations not available. SimpleDegradationPipeline disabled.")
            self.effect_chain = None
            return
        if not HAS_TORCHAUDIO:
            logger.warning("torchaudio not available. SimpleDegradationPipeline disabled.")
            self.effect_chain = None
            return

        # Collect background noise files
        noise_files = self._collect_audio_files(background_noise_dir)
        if not noise_files:
            logger.warning(f"No audio files found in {background_noise_dir}. Background noise disabled.")
        else:
            logger.info(f"Found {len(noise_files)} background noise files from {background_noise_dir}.")
        
        transforms = []
        
        # Add downsampling degradation
        transforms.append(
            DownUpSample(
                source_sample_rate=sample_rate,
                target_sample_rate=int(downsample_rate),
                p=float(downsample_p),
            )
        )
        
        # Add background noise if files found - use custom implementation
        if noise_files:
            try:
                transforms.append(
                    CustomAddBackgroundNoise(
                        background_paths=noise_files,
                        min_snr_in_db=float(noise_snr_db[0]),
                        max_snr_in_db=float(noise_snr_db[1]),
                        p=float(noise_p),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to initialize background noise: {e}")

        self.effect_chain = Compose(transforms=transforms, output_type="tensor")

    def __call__(self, samples: torch.Tensor, sample_rate: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Apply degradation pipeline to audio samples.
        
        Parameters
        ----------
        samples : torch.Tensor
            Audio samples with shape [batch, channels, samples]
        sample_rate : int, optional
            Sample rate of the audio
        
        Returns
        -------
        torch.Tensor
            Degraded audio samples
        """
        if self.effect_chain is None:
            return samples
        elif self.debug_message_shown:
            print("Applying SimpleDegradationPipeline")
            self.debug_message_shown = False
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        return self.effect_chain(samples, sample_rate=sr, **kwargs)

    def _collect_audio_files(self, directory: str) -> List[str]:
        """Collect all audio files from directory.
        
        Parameters
        ----------
        directory : str
            Path to directory containing audio files
        
        Returns
        -------
        List[str]
            List of absolute paths to audio files
        """
        if not os.path.isdir(directory):
            logger.warning(f"Background noise directory not found: {directory}")
            return []
        
        audio_files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith((".wav", ".flac", ".mp3", ".ogg")):
                    audio_files.append(os.path.join(root, filename))
        
        if audio_files:
            logger.info(f"Found {len(audio_files)} background noise files in {directory}")
        
        return audio_files


class MusicEffectPipeline:
    """Reusable augmentation chain combining tone, dynamics, and spatial effects."""

    def __init__(
        self,
        sample_rate: int,
        ir_paths: IrPaths = None,
        *,
        gain_db: tuple[float, float] = (-5.0, 2.0),
        noise_snr_db: tuple[float, float] = (24.0, 48.0),
        high_pass_hz: tuple[float, float] = (100.0, 150.0),
        low_pass_hz: tuple[float, float] = (12_000.0, 20_000.0),
        shuffle_p: float = 0.25,
        polarity_p: float = 0.1,
        ir_p: float = 0.1,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._ir_probability = float(ir_p)
        self._ir_paths = self._process_ir_paths(ir_paths)

        if not HAS_TA or Compose is None:
            logger.warning("torch_audiomentations not available. MusicEffectPipeline disabled.")
            self.effect_chain = None
            return
        if not HAS_TORCHAUDIO:
            logger.warning("torchaudio not available. MusicEffectPipeline disabled.")
            self.effect_chain = None
            return

        transforms = [
            DownUpSample(target_sample_rate=16_000, p=0.1),
            HighPassFilter(
                min_cutoff_freq=float(high_pass_hz[0]),
                max_cutoff_freq=min(float(high_pass_hz[1]), self.sample_rate / 2.5),
                p=0.1,
            ),
            LowPassFilter(
                min_cutoff_freq=float(low_pass_hz[0]),
                max_cutoff_freq=min(float(low_pass_hz[1]), self.sample_rate / 2.5),
                p=0.1,
            ),
            AddColoredNoise(
                min_snr_in_db=float(noise_snr_db[0]),
                max_snr_in_db=float(noise_snr_db[1]),
                p=0.1,
            ),
        ]

        if self._ir_paths and ApplyImpulseResponse is not None:
            transforms.append(
                ApplyImpulseResponse(
                    self._ir_paths,
                    p=self._ir_probability,
                    compensate_for_propagation_delay=True,
                )
            )
        elif not self._ir_paths:
            logger.debug("MusicEffectPipeline initialized without impulse responses.")

        transforms.extend(
            [
                Gain(min_gain_in_db=float(gain_db[0]), max_gain_in_db=float(gain_db[1]), p=0.1),
                PolarityInversion(p=float(polarity_p)),
                ShuffleChannels(p=float(shuffle_p), sample_rate=self.sample_rate),
            ]
        )

        self.effect_chain = Compose(transforms=transforms, output_type="tensor")

    def __call__(self, samples: torch.Tensor, sample_rate: Optional[int] = None, **kwargs) -> torch.Tensor:
        if self.effect_chain is None:
            return samples
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        return self.effect_chain(samples, sample_rate=sr, **kwargs)

    def update_ir_paths(self, ir_paths: IrPaths) -> None:
        processed = self._process_ir_paths(ir_paths)
        if processed == self._ir_paths:
            return
        self._ir_paths = processed
        if self.effect_chain is None:
            return

        transforms = list(self.effect_chain.transforms)
        transforms = [t for t in transforms if ApplyImpulseResponse is None or not isinstance(t, ApplyImpulseResponse)]
        if self._ir_paths and ApplyImpulseResponse is not None:
            transforms.append(
                ApplyImpulseResponse(
                    self._ir_paths,
                    p=self._ir_probability,
                    compensate_for_propagation_delay=True,
                )
            )
        self.effect_chain = Compose(transforms=transforms, output_type="tensor")

    def _process_ir_paths(self, ir_paths: IrPaths) -> List[str]:
        if ir_paths is None:
            return []
        if isinstance(ir_paths, str):
            path = os.fspath(ir_paths)
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


__all__ = ["MusicEffectPipeline", "SimpleDegradationPipeline", "CustomAddBackgroundNoise", "Compressor", "NoiseShapedReverb", "DownUpSample"]

