from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import MusicSSLFeatureExtractor
from .modules import (
    HiFiUpsampler,
    SpectralMaskNet,
    SpectralUNet,
    STFTConfig,
    UpsampleWaveUNetHead,
    WaveUNet,
)

Tensor = torch.Tensor


class FinallyGenerator(nn.Module):
    """FINALLY-style generator for music restoration."""

    def __init__(
        self,
        base_sample_rate: int = 24_000,
        target_sample_rate: int = 48_000,
        stft_cfg: STFTConfig | None = None,
        spectral_channels: int = 256,
        upsample_scales: tuple[int, ...] = (8, 8, 8),
        embedding_backbone: str = "codicodec",
        pretrained_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.base_sample_rate = base_sample_rate
        self.target_sample_rate = target_sample_rate
        self.stft_cfg = stft_cfg or STFTConfig(n_fft=2048, hop_length=512)
        self.spectral_channels = spectral_channels

        self.embedding_extractor = MusicSSLFeatureExtractor(
            target_sample_rate=base_sample_rate,
            backbone=embedding_backbone,
        )
        self.embedding_extractor.requires_grad_(False)
        embedding_dim = self.embedding_extractor.ssl_dim
        
        # Embedding projection for transfer learning
        if pretrained_embedding_dim is not None and pretrained_embedding_dim != embedding_dim:
            self.embedding_proj = nn.Linear(embedding_dim, pretrained_embedding_dim)
            print(
                f"[FinallyGenerator] Adding embedding projection: {embedding_dim} -> {pretrained_embedding_dim} "
                f"for transfer learning (backbone={embedding_backbone})"
            )
            effective_embedding_dim = pretrained_embedding_dim
        else:
            self.embedding_proj = None
            effective_embedding_dim = embedding_dim
        
        self.ssl_dim = embedding_dim  # Original dimension from extractor
        self.effective_ssl_dim = effective_embedding_dim  # Dimension after projection

        self.spectral_unet = SpectralUNet(
            in_channels=4,
            base_channels=32,
            num_scales=4,
            out_channels=spectral_channels,
            freq_bins=self.stft_cfg.n_fft // 2 + 1,
            max_frames=1024,
        )
        self.proj_spec = nn.Conv1d(spectral_channels, spectral_channels, kernel_size=1)
        self.hifi = HiFiUpsampler(
            in_channels=spectral_channels,
            upsample_scales=upsample_scales,
            embedding_channels=effective_embedding_dim,
            out_channels=2,
        )
        self.wave_unet = WaveUNet(
            in_channels=4,
            out_channels=2,
            base_channels=64,
            num_scales=4,
        )
        self.spectral_mask = SpectralMaskNet(self.stft_cfg)
        upsample_factor = target_sample_rate // base_sample_rate
        self.upsample_head = UpsampleWaveUNetHead(in_channels=2, out_channels=2, upsample_factor=upsample_factor)

    def _stft(self, waveform: Tensor) -> Tensor:
        cfg = self.stft_cfg
        window = cfg.window(waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length or cfg.n_fft,
            window=window,
            return_complex=True,
            center=cfg.center,
        )
        return spec

    def _spectral_features(self, waveform: Tensor) -> Tensor:
        batch, channels, samples = waveform.shape
        wav = waveform.reshape(batch * channels, samples)
        spec = self._stft(wav)  # [BC, F, T]
        mag = spec.abs()
        real = spec.real
        imag = spec.imag
        phase = torch.atan2(imag, real + 1e-6)
        feats = torch.stack([mag, real, imag, phase], dim=1)
        return feats

    def forward(
        self,
        degraded: Tensor,
        stage: Literal["stage1", "stage2", "stage3"] = "stage3",
        sample_rate: int | None = None,
        precomputed_embeddings: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            degraded: [B, 2, T] stereo audio at base sample rate.
            stage: Training stage flag controlling upsample head usage.
            sample_rate: Optional override if degraded is not at base sample rate.
        """
        if sample_rate is None:
            sample_rate = self.base_sample_rate
        if sample_rate != self.base_sample_rate:
            degraded = F.interpolate(
                degraded,
                scale_factor=self.base_sample_rate / sample_rate,
                mode="linear",
                align_corners=False,
            )

        spec_feats = self._spectral_features(degraded)
        spec_latent = self.spectral_unet(spec_feats)
        spec_tokens = spec_latent.mean(dim=2)  # [B*2, C, T]
        # Restore batch dimension
        batch = degraded.shape[0]
        tokens = spec_tokens.view(batch, degraded.shape[1], spec_tokens.shape[1], spec_tokens.shape[-1])
        tokens = tokens.mean(dim=1)  # merge stereo features
        tokens = self.proj_spec(tokens)

        with torch.no_grad():
            if precomputed_embeddings is not None:
                embedding_map = precomputed_embeddings.to(degraded.device)
            else:
                embedding_map = self.embedding_extractor(degraded, self.base_sample_rate)
        
        # Apply embedding projection if transfer learning
        if self.embedding_proj is not None:
            # embedding_map is [B, ssl_dim, T]
            embedding_map = embedding_map.transpose(1, 2)  # [B, T, ssl_dim]
            embedding_map = self.embedding_proj(embedding_map)  # [B, T, pretrained_dim]
            embedding_map = embedding_map.transpose(1, 2)  # [B, pretrained_dim, T]

        embedding_map = F.interpolate(embedding_map, size=tokens.shape[-1], mode="linear", align_corners=False)
        hifi_out = self.hifi(tokens, embedding_map)
        base_wave = torch.tanh(hifi_out)

        # Align lengths
        if base_wave.shape[-1] != degraded.shape[-1]:
            base_wave = F.interpolate(base_wave, size=degraded.shape[-1], mode="linear", align_corners=False)

        wave_input = torch.cat([base_wave, degraded], dim=1)
        refined = self.wave_unet(wave_input)
        refined = torch.tanh(refined)
        refined = refined + degraded
        refined = torch.tanh(refined)
        base_out = self.spectral_mask(refined)

        if stage == "stage3":
            target_out = self.upsample_head(base_out)
            return {"base": base_out, "target": target_out}
        return {"base": base_out}


__all__ = ["FinallyGenerator"]
