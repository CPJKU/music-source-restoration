from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

Tensor = torch.Tensor


@dataclass
class DiscriminatorOutput:
    score: Tensor
    features: List[Tensor]


class SpectralSubDiscriminator(nn.Module):
    """2D ConvNet operating on complex STFT (real/imag channels) with dilated convs.
    
    Memory optimization: Stores 3 strategic feature maps with aggressive compression:
    - Input projection (early features): 2x2 spatial pooling + 4x channel projection
    - Middle dilation layer (mid-level features): 2x2 spatial pooling + 4x channel projection
    - Final score (high-level features): 2x2 spatial pooling + 4x channel projection
    
    Uses random projection (Johnson-Lindenstrauss) to reduce channels while preserving
    approximate distances for feature matching. Total memory reduction: ~94% vs full features.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        dilation_factors: Sequence[int] = (1, 2, 4, 8, 16),
        feature_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.dilated_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                )
                for d in dilation_factors
            ]
        )
        self.logits = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.num_dilations = len(dilation_factors)
        
        # Random projection matrices for channel reduction (fixed, no gradients)
        # Johnson-Lindenstrauss lemma: random projection preserves pairwise distances
        reduced_channels = base_channels // feature_reduction
        self.feat_proj = nn.Parameter(
            torch.randn(base_channels, reduced_channels) / (base_channels ** 0.5),
            requires_grad=False
        )
        self.score_proj = nn.Parameter(
            torch.randn(1, max(1, reduced_channels // 4)) / (1 ** 0.5),
            requires_grad=False
        )

    def forward(self, spec: Tensor) -> DiscriminatorOutput:
        features: List[Tensor] = []
        x = self.input(spec)
        
        # Store compressed input projection (early features)
        # Step 1: Spatial pooling (4x reduction)
        x_pooled = F.avg_pool2d(x, 2)  # [B, C, H/2, W/2]
        # Step 2: Random channel projection (4x reduction)
        B, C, H, W = x_pooled.shape
        x_flat = x_pooled.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        x_proj = x_flat @ self.feat_proj  # [B*H*W, C/4]
        x_proj = x_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C/4, H, W]
        features.append(x_proj.detach())
        
        # Process dilated layers, storing compressed middle layer
        mid_idx = self.num_dilations // 2  # Index 2 for 5 layers (dilation=4)
        for idx, conv in enumerate(self.dilated_layers):
            x = F.leaky_relu(x, 0.2)
            x = conv(x)
            if idx == mid_idx:
                # Apply same compression: spatial pooling + random projection
                x_pooled = F.avg_pool2d(x, 2)
                B, C, H, W = x_pooled.shape
                x_flat = x_pooled.permute(0, 2, 3, 1).reshape(B * H * W, C)
                x_proj = x_flat @ self.feat_proj
                x_proj = x_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2)
                features.append(x_proj.detach())
        
        x = F.leaky_relu(x, 0.2)
        score = self.logits(x)
        
        # Store compressed final score
        score_pooled = F.avg_pool2d(score, 2)  # [B, 1, H/2, W/2]
        B, C, H, W = score_pooled.shape
        score_flat = score_pooled.permute(0, 2, 3, 1).reshape(B * H * W, C)
        score_proj = score_flat @ self.score_proj
        score_proj = score_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append(score_proj.detach())
        
        return DiscriminatorOutput(score=score.mean(dim=[2, 3]), features=features)


class MultiScaleSTFTDiscriminator(nn.Module):
    """MS-STFT discriminator across multiple window sizes."""

    def __init__(
        self,
        fft_sizes: Sequence[int],
        hop_sizes: Sequence[int],
        win_lengths: Sequence[int],
        sample_rate: int,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("FFTs, hops, and window lengths must align.")
        self.configs = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.sample_rate = sample_rate
        self.discriminators = nn.ModuleList([SpectralSubDiscriminator() for _ in self.configs])

    def _stft_features(self, wav: Tensor, fft: int, hop: int, win: int) -> Tensor:
        window = torch.hann_window(win, device=wav.device)
        spec = torch.stft(
            wav,
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
            center=True,
        )
        real = spec.real.unsqueeze(1)
        imag = spec.imag.unsqueeze(1)
        feats = torch.cat([real, imag], dim=1)
        return feats

    def forward(self, waveform: Tensor) -> List[DiscriminatorOutput]:
        """
        Args:
            waveform: [B, C, T] stereo audio
        """
        batch, channels, samples = waveform.shape
        wav = waveform.reshape(batch * channels, samples)
        outputs: List[DiscriminatorOutput] = []
        for disc, (fft, hop, win) in zip(self.discriminators, self.configs):
            spec = self._stft_features(wav, fft, hop, win)
            outputs.append(disc(spec))
        return outputs


class PeriodDiscriminator(nn.Module):
    """Period discriminator from HiFi-GAN."""

    def __init__(self, period: int, channels: Sequence[int] = (64, 128, 256, 512, 512)) -> None:
        super().__init__()
        self.period = period
        convs: List[nn.Module] = []
        in_channels = 2  # stereo
        for idx, out_channels in enumerate(channels):
            kernel_size = 5 if idx < len(channels) - 1 else 3
            stride = 3 if idx < len(channels) - 1 else 1
            convs.append(
                nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
            )
            in_channels = out_channels
        self.convs = nn.ModuleList(convs)
        self.final = nn.utils.weight_norm(nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, waveform: Tensor) -> DiscriminatorOutput:
        batch, channels, samples = waveform.shape
        if samples % self.period != 0:
            pad = self.period - (samples % self.period)
            waveform = F.pad(waveform, (0, pad), mode="reflect")
            samples = waveform.shape[-1]
        x = waveform.view(batch, channels, samples // self.period, self.period)
        features: List[Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        score = self.final(x)
        features.append(score)
        score = score.flatten(1, -1)
        return DiscriminatorOutput(score=score.mean(dim=1, keepdim=True), features=features)


class MultiPeriodDiscriminator(nn.Module):
    """Collection of period discriminators over diverse periods."""

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, waveform: Tensor) -> List[DiscriminatorOutput]:
        return [disc(waveform) for disc in self.discriminators]


class SubBandDiscriminator(nn.Module):
    """1D discriminator operating on pre-defined sub-band views."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = base_channels
        layers.append(nn.Conv1d(in_channels, channels, kernel_size=15, padding=7))
        for _ in range(1, num_layers):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2))
            channels *= 2
        self.layers = nn.ModuleList(layers)
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> DiscriminatorOutput:
        features: List[Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                features.append(h)
        score = self.final(h)
        features.append(score)
        return DiscriminatorOutput(score=score.mean(dim=2, keepdim=True), features=features)


class MultiBandDiscriminator(nn.Module):
    """Approximate multi-band discriminator using strided views."""

    def __init__(self, band_strides: Sequence[int] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.band_strides = band_strides
        self.discriminators = nn.ModuleList([SubBandDiscriminator(in_channels=2) for _ in band_strides])

    def _band_views(self, waveform: Tensor, stride: int) -> Tensor:
        if stride == 1:
            return waveform
        return F.avg_pool1d(waveform, kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, waveform: Tensor) -> List[DiscriminatorOutput]:
        outputs: List[DiscriminatorOutput] = []
        for stride, disc in zip(self.band_strides, self.discriminators):
            view = self._band_views(waveform, stride)
            outputs.append(disc(view))
        return outputs


class MelSubDiscriminator(nn.Module):
    """1D ConvNet operating on mel-spectrogram features.
    
    This is a simpler, more stable alternative to STFT-based discriminators.
    Uses mel-scale frequency representation which is perceptually motivated.
    
    Memory optimization: Only stores 3 strategic feature maps (detached):
    - Input projection (early features)
    - Middle layer (mid-level features)
    - Final score (high-level features)
    """

    def __init__(self, n_mels: int = 128, base_channels: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = base_channels
        # Initial projection from mel bins to channels
        layers.append(nn.Conv1d(n_mels, channels, kernel_size=7, padding=3))
        
        # Downsampling layers
        for _ in range(1, num_layers):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2))
            channels *= 2
        
        self.layers = nn.ModuleList(layers)
        self.num_conv_layers = sum(1 for layer in layers if isinstance(layer, nn.Conv1d))
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, mel: Tensor) -> DiscriminatorOutput:
        """Forward pass on mel-spectrogram.
        
        Parameters
        ----------
        mel : Tensor
            Mel-spectrogram [B, n_mels, T]
            
        Returns
        -------
        DiscriminatorOutput
            Score and intermediate features (detached)
        """
        features: List[Tensor] = []
        x = mel
        conv_idx = 0
        mid_idx = self.num_conv_layers // 2
        
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                # Store only input projection, middle layer, detached
                if conv_idx == 0:  # Input projection
                    features.append(x.detach())
                elif conv_idx == mid_idx:  # Middle layer
                    features.append(x.detach())
                conv_idx += 1
        
        score = self.final(x)
        # Store final score (detached)
        features.append(score.detach())
        return DiscriminatorOutput(score=score.mean(dim=2, keepdim=True), features=features)


class MultiScaleMelDiscriminator(nn.Module):
    """Multi-scale mel-spectrogram discriminator.
    
    This discriminator operates on mel-spectrograms at multiple resolutions.
    It's simpler and more stable than STFT-based discriminators, making it
    a good fallback option if training becomes unstable.
    
    Use this as an alternative to MultiScaleSTFTDiscriminator for:
    - More stable training (no phase information to model)
    - Lower memory footprint
    - Faster training (simpler features)
    
    Trade-offs:
    - Less detailed frequency discrimination
    - May miss high-frequency artifacts
    - Better for perceptual quality than technical accuracy
    """

    def __init__(
        self,
        sample_rate: int,
        mel_configs: Sequence[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize multi-scale mel discriminator.
        
        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz
        mel_configs : Sequence[Tuple[int, int, int]], optional
            List of (n_fft, hop_length, n_mels) configurations.
            Default: [(2048, 512, 128), (1024, 256, 64), (512, 128, 32)]
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        if mel_configs is None:
            mel_configs = [
                (2048, 512, 128),  # High resolution
                (1024, 256, 64),   # Medium resolution
                (512, 128, 32),    # Low resolution
            ]
        
        self.configs = mel_configs
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0.0,
                f_max=sample_rate // 2,
                power=2.0,
            )
            for n_fft, hop_length, n_mels in mel_configs
        ])
        
        self.discriminators = nn.ModuleList([
            MelSubDiscriminator(n_mels=n_mels)
            for _, _, n_mels in mel_configs
        ])

    def forward(self, waveform: Tensor) -> List[DiscriminatorOutput]:
        """Forward pass on audio waveform.
        
        Parameters
        ----------
        waveform : Tensor
            Audio waveform [B, C, T] (stereo)
            
        Returns
        -------
        List[DiscriminatorOutput]
            One output per mel configuration
        """
        batch, channels, samples = waveform.shape
        # Flatten stereo channels for processing
        wav = waveform.reshape(batch * channels, samples)
        
        outputs: List[DiscriminatorOutput] = []
        for mel_transform, disc in zip(self.mel_transforms, self.discriminators):
            # Compute mel spectrogram
            mel = mel_transform(wav)  # [B*C, n_mels, T]
            # Log scale (perceptual)
            mel = torch.log10(mel + 1e-6)
            # Discriminate
            outputs.append(disc(mel))
        
        return outputs


class FinallyDiscriminatorBundle(nn.Module):
    """Flexible discriminator supporting either MS-STFT or MS-Mel discriminators.
    
    Supports two discriminator types:
    - 'stft': Multi-Scale STFT discriminators (default, more detailed frequency discrimination)
    - 'mel': Multi-Scale Mel discriminators (lighter memory, more stable training)
    
    Uses discriminators at both base (24kHz) and target (48kHz) sample rates.
    This provides multi-resolution spectral discrimination which is ideal for music quality.
    
    Removed components (for stability and simplicity):
    - MultiPeriodDiscriminator (MPD): Time-domain patterns, prone to mode collapse
    - MultiBandDiscriminator (MBD): Frequency bands, adds complexity without clear benefit
    
    The dict-based output format is preserved for compatibility with existing training code.
    Keys: 'ms_disc_base', 'ms_disc_target' (where disc is either stft or mel)
    """

    def __init__(
        self,
        base_sample_rate: int,
        target_sample_rate: int,
        base_fft_sizes: Sequence[int] = (2048, 1024, 512),
        target_fft_sizes: Sequence[int] = (4096, 2048, 1024),
        discriminator_type: str = "mel",
    ) -> None:
        """Initialize discriminator bundle.
        
        Parameters
        ----------
        base_sample_rate : int
            Base sample rate (e.g., 24000 Hz)
        target_sample_rate : int
            Target sample rate (e.g., 48000 Hz)
        base_fft_sizes : Sequence[int]
            FFT sizes for base sample rate
        target_fft_sizes : Sequence[int]
            FFT sizes for target sample rate
        discriminator_type : str
            Type of discriminator: 'stft' or 'mel'
        """
        super().__init__()
        
        if discriminator_type not in ("stft", "mel"):
            raise ValueError(f"discriminator_type must be 'stft' or 'mel', got '{discriminator_type}'")
        
        self.discriminator_type = discriminator_type
        
        if discriminator_type == "stft":
            # Base sample rate (24kHz) STFT discriminator
            hop_sizes = [fft // 4 for fft in base_fft_sizes]
            win_lengths = base_fft_sizes
            self.ms_disc_base = MultiScaleSTFTDiscriminator(
                base_fft_sizes, hop_sizes, win_lengths, base_sample_rate
            )
            
            # Target sample rate (48kHz) STFT discriminator
            hop_sizes_target = [fft // 4 for fft in target_fft_sizes]
            win_lengths_target = target_fft_sizes
            self.ms_disc_target = MultiScaleSTFTDiscriminator(
                target_fft_sizes, hop_sizes_target, win_lengths_target, target_sample_rate
            )
        else:  # mel
            # Base sample rate (24kHz) Mel discriminator
            # Mel configs: (n_fft, hop_length, n_mels)
            base_mel_configs = [(fft, fft // 4, 128 // (2 ** i)) for i, fft in enumerate(base_fft_sizes)]
            self.ms_disc_base = MultiScaleMelDiscriminator(
                sample_rate=base_sample_rate,
                mel_configs=base_mel_configs,
            )
            
            # Target sample rate (48kHz) Mel discriminator
            target_mel_configs = [(fft, fft // 4, 128 // (2 ** i)) for i, fft in enumerate(target_fft_sizes)]
            self.ms_disc_target = MultiScaleMelDiscriminator(
                sample_rate=target_sample_rate,
                mel_configs=target_mel_configs,
            )

    def forward(
        self,
        base_waveform: Tensor,
        target_waveform: Tensor | None = None,
    ) -> dict[str, List[DiscriminatorOutput]]:
        """Return discriminator outputs for adversarial and feature-matching losses.
        
        The dict format groups discriminator outputs by type, which is then flattened
        by the trainer using flatten_outputs() for loss computation.
        
        Parameters
        ----------
        base_waveform : Tensor
            Audio at base sample rate (24kHz) [B, C, T]
        target_waveform : Tensor, optional
            Audio at target sample rate (48kHz) [B, C, T]
            If None, uses base_waveform for target discriminator
            
        Returns
        -------
        dict[str, List[DiscriminatorOutput]]
            Dict with keys 'ms_disc_base' and 'ms_disc_target'
            Each value is a list of DiscriminatorOutput (one per scale)
        """
        outputs: dict[str, List[DiscriminatorOutput]] = {
            "ms_disc_base": self.ms_disc_base(base_waveform),
        }
        
        # Use target waveform if provided, otherwise use base waveform
        waveform_for_target = target_waveform if target_waveform is not None else base_waveform
        outputs["ms_disc_target"] = self.ms_disc_target(waveform_for_target)
        
        return outputs


__all__ = [
    "DiscriminatorOutput",
    "MultiScaleSTFTDiscriminator",
    "MultiScaleMelDiscriminator",
    "MultiPeriodDiscriminator",
    "MultiBandDiscriminator",
    "FinallyDiscriminatorBundle",
]
