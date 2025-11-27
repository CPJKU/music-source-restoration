from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _leaky_relu(x: Tensor, negative_slope: float = 0.2) -> Tensor:
    return F.leaky_relu(x, negative_slope=negative_slope)


class ResidualBlock1d(nn.Module):
    """1D residual block with weight-normalised conv for waveform paths."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + _leaky_relu(self.conv(x))


class ResidualBlock2d(nn.Module):
    """2D residual block for spectrogram processing."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + _leaky_relu(self.conv(x))


class Downsample1d(nn.Module):
    """Conv + residual stack for 1D downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_res_blocks: int = 2,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.pre = nn.Sequential(*[ResidualBlock1d(in_channels) for _ in range(num_res_blocks)])
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pre(x))


class Upsample1d(nn.Module):
    """Transposed conv + skip merge for 1D upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_res_blocks: int = 2,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.post = nn.Sequential(*[ResidualBlock1d(out_channels) for _ in range(num_res_blocks)])
        self.merge = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.deconv(x)
        x = self.post(x)
        # Pad to match skip length if needed
        if x.shape[-1] < skip.shape[-1]:
            pad = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, pad))
        elif x.shape[-1] > skip.shape[-1]:
            x = x[..., : skip.shape[-1]]
        return self.merge(torch.cat([x, skip], dim=1))


class Downsample2d(nn.Module):
    """2D strided conv downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        num_res_blocks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.pre = nn.Sequential(*[ResidualBlock2d(in_channels) for _ in range(num_res_blocks)])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pre(x))


class Upsample2d(nn.Module):
    """2D transposed conv upsampling with skip merge."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        num_res_blocks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.post = nn.Sequential(*[ResidualBlock2d(out_channels) for _ in range(num_res_blocks)])
        self.merge = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.deconv(x)
        x = self.post(x)
        diff_y = skip.shape[-2] - x.shape[-2]
        diff_x = skip.shape[-1] - x.shape[-1]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                (
                    math.floor(diff_x / 2),
                    math.ceil(diff_x / 2),
                    math.floor(diff_y / 2),
                    math.ceil(diff_y / 2),
                ),
            )
        return self.merge(torch.cat([x, skip], dim=1))


# -----------------------------------------------------------------------------
# Spectral UNet
# -----------------------------------------------------------------------------


class SpectralUNet(nn.Module):
    """2D UNet over complex spectrogram representations."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_scales: int = 4,
        stride: int = 2,
        num_res_blocks: int = 2,
        out_channels: int = 256,
        add_positional_encoding: bool = True,
        freq_bins: int = 512,
        max_frames: int = 512,
    ) -> None:
        super().__init__()
        self.add_positional_encoding = add_positional_encoding
        if add_positional_encoding:
            # Use register_buffer instead of Parameter to make it non-trainable
            # This prevents NaN issues during adversarial training
            self.register_buffer("positional", torch.randn(1, in_channels, freq_bins, max_frames))
        channels: List[int] = [base_channels * (2**i) for i in range(num_scales)]
        self.input = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        downs = []
        for idx in range(num_scales - 1):
            downs.append(Downsample2d(channels[idx], channels[idx + 1], stride=stride, num_res_blocks=num_res_blocks))
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.Sequential(
            ResidualBlock2d(channels[-1]),
            ResidualBlock2d(channels[-1]),
        )

        ups = []
        for idx in reversed(range(num_scales - 1)):
            ups.append(Upsample2d(channels[idx + 1], channels[idx], stride=stride, num_res_blocks=num_res_blocks))
        self.ups = nn.ModuleList(ups)
        self.output = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: [B, C_in, F, T]
        Returns:
            Tensor of shape [B, C_out, F, T]
        """
        if self.add_positional_encoding:
            pos = self.positional[..., : spec.shape[-1]]
            spec = spec + pos

        skips: List[Tensor] = []
        x = self.input(spec)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for skip, up in zip(reversed(skips[:-1]), self.ups):
            x = up(x, skip)
        return self.output(x)


# -----------------------------------------------------------------------------
# HiFi-GAN style upsampler blocks
# -----------------------------------------------------------------------------


class HiFiResBlock(nn.Module):
    """Residual block from HiFi-GAN with harmonic dilations."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Sequence[int] = (1, 3, 9)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for d in dilations:
            layers.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(channels, channels, kernel_size, padding=d * (kernel_size // 2), dilation=d),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = out + layer(out)
        return out


class HiFiUpsampleBlock(nn.Module):
    """Upsample block with transposed convolution and residual stack."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        kernel_size = stride * 2
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=stride // 2 + stride % 2,
            output_padding=stride % 2,
        )
        self.resblocks = nn.ModuleList(
            [
                HiFiResBlock(out_channels, kernel_size=3, dilations=(1, 3, 5)),
                HiFiResBlock(out_channels, kernel_size=3, dilations=(1, 6, 9)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = _leaky_relu(self.deconv(x))
        res = 0.0
        for block in self.resblocks:
            res = res + block(x)
        return res / len(self.resblocks)


class HiFiUpsampler(nn.Module):
    """Stack of HiFi-GAN style upsampling blocks with feature concatenation."""

    def __init__(
        self,
        in_channels: int,
        upsample_scales: Sequence[int],
        embedding_channels: int,
        out_channels: int,
        fuse_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels + embedding_channels, in_channels, kernel_size=1)
        channels = in_channels
        blocks: List[nn.Module] = []
        for scale in upsample_scales:
            blocks.append(HiFiUpsampleBlock(channels, channels // 2, stride=scale))
            channels = channels // 2
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, out_channels, kernel_size=7, padding=3),
        )
        self.fuse_mode = fuse_mode

    def forward(self, features: Tensor, embedding_map: Tensor) -> Tensor:
        """
        Args:
            features: [B, C, T]
            embedding_map: [B, E, T] (already resampled to match time axis)
        """
        if self.fuse_mode == "add":
            x = torch.cat([features, embedding_map], dim=1)
            x = self.proj(x)
        else:
            x = torch.cat([features, embedding_map], dim=1)
            x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


# -----------------------------------------------------------------------------
# Wave UNet (1D)
# -----------------------------------------------------------------------------


class WaveUNet(nn.Module):
    """1D UNet operating on stereo waveforms."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        stride: int = 2,
        num_scales: int = 4,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        channels: List[int] = [base_channels * (2**i) for i in range(num_scales)]
        self.input = nn.Conv1d(in_channels, channels[0], kernel_size=5, padding=2)
        downs = []
        for idx in range(num_scales - 1):
            downs.append(
                Downsample1d(
                    channels[idx],
                    channels[idx + 1],
                    stride=stride,
                    num_res_blocks=num_res_blocks,
                    kernel_size=4,
                )
            )
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.Sequential(*[ResidualBlock1d(channels[-1]) for _ in range(4)])
        ups = []
        for idx in reversed(range(num_scales - 1)):
            ups.append(
                Upsample1d(
                    channels[idx + 1],
                    channels[idx],
                    stride=stride,
                    num_res_blocks=num_res_blocks,
                    kernel_size=4,
                )
            )
        self.ups = nn.ModuleList(ups)
        self.output = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        skips: List[Tensor] = []
        x = self.input(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for skip, up in zip(reversed(skips[:-1]), self.ups):
            x = up(x, skip)
        return self.output(x)


# -----------------------------------------------------------------------------
# Spectral mask refinement
# -----------------------------------------------------------------------------


@dataclass
class STFTConfig:
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None
    center: bool = True

    def window(self, device: torch.device) -> Tensor:
        length = self.win_length or self.n_fft
        return torch.hann_window(length, device=device)


class SpectralMaskNet(nn.Module):
    """Predicts multiplicative masks for magnitude refinement."""

    def __init__(self, stft_cfg: STFTConfig | None = None) -> None:
        super().__init__()
        self.stft_cfg = stft_cfg or STFTConfig()
        self.encoder = SpectralUNet(
            in_channels=2,
            base_channels=16,
            num_scales=3,
            out_channels=2,
            add_positional_encoding=False,
            freq_bins=self.stft_cfg.n_fft // 2 + 1,
        )

    def _stft(self, wav: Tensor) -> Tensor:
        cfg = self.stft_cfg
        window = cfg.window(wav.device)
        spec = torch.stft(
            wav,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length or cfg.n_fft,
            center=cfg.center,
            window=window,
            return_complex=True,
        )
        return spec

    def _istft(self, spec: Tensor, length: int) -> Tensor:
        cfg = self.stft_cfg
        window = cfg.window(spec.device)
        wav = torch.istft(
            spec,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length or cfg.n_fft,
            center=cfg.center,
            window=window,
            length=length,
        )
        return wav

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: [B, C, T]
        Returns:
            Refined waveform [B, C, T]
        """
        batch, channels, samples = waveform.shape
        wav = waveform.reshape(batch * channels, samples)
        spec = self._stft(wav)  # [BC, F, K]
        mag = spec.abs()
        phase = torch.angle(spec)
        mag_norm = torch.log1p(mag)
        phase_real = torch.cos(phase)
        features = torch.stack([mag_norm, phase_real], dim=1)  # [BC, 2, F, K]
        mask = self.encoder(features).sigmoid()  # [BC, 2, F, K]
        # Use the first channel as magnitude mask
        mag_mask = mask[:, 0]  # [BC, F, K]
        refined_mag = mag * mag_mask
        complex_spec = torch.polar(refined_mag, phase)
        wav_hat = self._istft(complex_spec, length=samples)
        wav_hat = wav_hat.view(batch, channels, samples)
        return wav_hat


# -----------------------------------------------------------------------------
# Upsample head
# -----------------------------------------------------------------------------


class UpsampleWaveUNetHead(nn.Module):
    """WaveUNet head with explicit resampling to target sample rate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor
        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
        )
        self.wave_unet = WaveUNet(base_channels, base_channels, base_channels=base_channels)
        self.post = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, T] audio at base sample rate.
        Returns:
            Audio upsampled by upsample_factor.
        """
        batch, channels, samples = x.shape
        target_len = int(round(samples * self.upsample_factor))
        flat = x.reshape(batch * channels, samples)
        x = torchaudio.functional.resample(
            flat,
            orig_freq=24000,                 # ratio only; actual rate cancels
            new_freq=24000 * self.upsample_factor,
            resampling_method="sinc_interp_hann",
            lowpass_filter_width=32,
            rolloff=0.9475937,
            beta=14.769656459379492,
        ).view(batch, channels, -1)
        # Guard against off‑by‑one length from resample
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = self.pre(x)
        x = self.wave_unet(x)
        x = self.post(x)
        return x


__all__ = [
    "ResidualBlock1d",
    "ResidualBlock2d",
    "SpectralUNet",
    "HiFiUpsampler",
    "WaveUNet",
    "SpectralMaskNet",
    "UpsampleWaveUNetHead",
    "STFTConfig",
]

