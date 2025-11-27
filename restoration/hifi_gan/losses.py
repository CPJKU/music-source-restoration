from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .discriminators import DiscriminatorOutput
from .modules import STFTConfig
from .erb_filterbank import make_erb_filterbank_for_stft

# Try to use fast PyTorch-native SoftDTW (no compilation)
# Falls back to Numba version if not available
try:
    from .soft_dtw_torch import SoftDTWTorch as SoftDTW
    _USING_TORCH_SOFTDTW = True
except ImportError:
    print("Warning: PyTorch-native SoftDTW not available, falling back to Numba version.")
    from .soft_dtw import SoftDTW
    _USING_TORCH_SOFTDTW = False

Tensor = torch.Tensor


@dataclass
class LMOSConfig:
    alpha: float = 1.0
    stft_sizes: Sequence[int] = (1024, 2048, 4096)
    hop_lengths: Sequence[int] = (256, 512, 1024)


class LMOSLoss(nn.Module):
    """Latent Music Objective Score."""

    def __init__(self, embedding_extractor: nn.Module, cfg: LMOSConfig | None = None) -> None:
        super().__init__()
        self.embedding = embedding_extractor
        self.cfg = cfg or LMOSConfig()

    def _stft_mag(self, wav: Tensor, n_fft: int, hop_length: int) -> Tensor:
        window = torch.hann_window(n_fft, device=wav.device)
        spec = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        return spec.abs()

    def forward(self, reference: Tensor, estimate: Tensor, sample_rate: int) -> Tensor:
        """
        Args:
            reference: [B, C, T]
            estimate: [B, C, T]
        """
        emb_ref = self.embedding(reference, sample_rate)
        emb_est = self.embedding(estimate, sample_rate)
        emb_loss = F.mse_loss(emb_est, emb_ref)
        ref_flat = reference.reshape(reference.shape[0] * reference.shape[1], reference.shape[-1])
        est_flat = estimate.reshape(estimate.shape[0] * estimate.shape[1], estimate.shape[-1])

        spec_losses: List[Tensor] = []
        for n_fft, hop in zip(self.cfg.stft_sizes, self.cfg.hop_lengths):
            ref_mag = self._stft_mag(ref_flat, n_fft, hop)
            est_mag = self._stft_mag(est_flat, n_fft, hop)
            spec_losses.append(torch.mean(torch.abs(ref_mag - est_mag)))
        spec_loss = torch.stack(spec_losses).mean()
        return self.cfg.alpha * emb_loss + spec_loss


class LeastSquaresGANLoss:
    """Least-squares GAN objectives with optional label smoothing."""

    def generator(self, fake_scores: Iterable[Tensor], real_target: float = 1.0, **_: object) -> Tensor:
        losses = [torch.mean((score - real_target) ** 2) for score in fake_scores]
        return torch.stack(losses).mean()

    def discriminator(
        self,
        real_scores: Iterable[Tensor],
        fake_scores: Iterable[Tensor],
        real_target: float = 1.0,
        fake_target: float = 0.0,
        **_: object,
    ) -> Tensor:
        losses_real = [torch.mean((score - real_target) ** 2) for score in real_scores]
        losses_fake = [torch.mean((score - fake_target) ** 2) for score in fake_scores]
        return torch.stack(losses_real + losses_fake).mean()


class HingeGANLoss:
    """Hinge loss GAN objective."""

    def generator(self, fake_scores: Iterable[Tensor], **_: object) -> Tensor:
        losses = [torch.mean(-score) for score in fake_scores]
        return torch.stack(losses).mean()

    def discriminator(
        self,
        real_scores: Iterable[Tensor],
        fake_scores: Iterable[Tensor],
        **_: object,
    ) -> Tensor:
        losses_real = [torch.mean(F.relu(1.0 - score)) for score in real_scores]
        losses_fake = [torch.mean(F.relu(1.0 + score)) for score in fake_scores]
        return torch.stack(losses_real + losses_fake).mean()


def feature_matching_loss(
    real_outputs: Iterable[DiscriminatorOutput],
    fake_outputs: Iterable[DiscriminatorOutput],
) -> Tensor:
    real_list = list(real_outputs)
    fake_list = list(fake_outputs)
    if not real_list or not fake_list:
        return torch.tensor(0.0)
    total: List[Tensor] = []
    for real, fake in zip(real_list, fake_list):
        for real_feat, fake_feat in zip(real.features, fake.features):
            total.append(F.l1_loss(fake_feat, real_feat))
    if total:
        return torch.stack(total).mean()
    device = fake_list[0].features[0].device
    return torch.tensor(0.0, device=device)


def r1_gradient_penalty(
    discriminator: nn.Module,
    real_base: Tensor,
    real_target: Tensor | None = None,
) -> Tensor:
    """Compute R1 gradient penalty for discriminator regularization.
    
    R1 penalty: E[||∇_x D(x)||^2] where x ~ real data
    Encourages discriminator to have smooth gradients, preventing mode collapse.
    
    Parameters
    ----------
    discriminator : nn.Module
        Discriminator network
    real_base : Tensor
        Real audio samples at base sample rate [B, C, T]
    real_target : Tensor, optional
        Real audio samples at target sample rate [B, C, T]
        
    Returns
    -------
    Tensor
        R1 gradient penalty (scalar)
    """
    inputs: List[Tensor] = []
    real_base = real_base.requires_grad_(True)
    inputs.append(real_base)
    if real_target is not None:
        real_target = real_target.requires_grad_(True)
        inputs.append(real_target)
    
    # Forward pass through discriminator
    disc_outputs = discriminator(real_base, real_target)
    
    # Flatten all discriminator scores
    all_scores = []
    for key in sorted(disc_outputs.keys()):
        for out in disc_outputs[key]:
            all_scores.append(out.score)
    
    # Compute gradients of discriminator output w.r.t. input
    # Sum all scores to get scalar for backward
    score_sum = sum(s.sum() for s in all_scores)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=score_sum,
        inputs=tuple(inputs),
        create_graph=True,  # Allow gradients of gradients
        retain_graph=True,
        only_inputs=True,
    )
    
    penalty = 0.0
    batch_terms: List[Tensor] = []
    for grad in gradients:
        reshaped = grad.reshape(grad.shape[0], -1)
        batch_terms.append(reshaped.pow(2).sum(dim=1))
    if batch_terms:
        penalty = torch.stack(batch_terms, dim=0).sum(dim=0).mean()
    else:
        penalty = torch.tensor(0.0, device=real_base.device)
    
    return penalty


class SpectralConvergence(nn.Module):
    """Spectral convergence metric."""

    def __init__(self, stft_cfg: STFTConfig) -> None:
        super().__init__()
        self.stft_cfg = stft_cfg

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        cfg = self.stft_cfg
        window = cfg.window(reference.device)
        ref = torch.stft(
            reference,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length or cfg.n_fft,
            window=window,
            return_complex=True,
            center=cfg.center,
        )
        est = torch.stft(
            estimate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length or cfg.n_fft,
            window=window,
            return_complex=True,
            center=cfg.center,
        )
        numerator = torch.norm(ref - est, p="fro")
        denominator = torch.norm(ref, p="fro")
        return numerator / (denominator + 1e-6)


class OnsetWeightedLoss(nn.Module):
    """Onset-aware term using high-pass energy derivatives."""

    def __init__(self, frame_length: int = 1024, hop_length: int = 256) -> None:
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        def onset_envelope(wav: Tensor) -> Tensor:
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=48_000,
                n_fft=2048,
                hop_length=self.hop_length,
                n_mels=32,
            )(wav)
            mel = torch.log1p(mel)
            diff = mel[:, :, 1:] - mel[:, :, :-1]
            diff = diff.clamp_min(0.0)
            return diff.mean(dim=1)

        ref_env = onset_envelope(reference)
        est_env = onset_envelope(estimate)
        return F.l1_loss(est_env, ref_env)


class LPAPSLoss(nn.Module):
    """Lightweight perceptual audio similarity surrogate (LPAPS-like)."""

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            center=True,
        )

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        ref_mel = torch.log1p(self.mel_transform(reference))
        est_mel = torch.log1p(self.mel_transform(estimate))
        return F.l1_loss(est_mel, ref_mel)


class HumanFeedbackSurrogateLoss(nn.Module):
    """Combined differentiable surrogate for perceptual ranking.
    
    DEPRECATED: This loss is speech-oriented. Use MusicPerceptualLoss for music restoration.
    """

    def __init__(
        self,
        lpaps_weight: float = 1.0,
        spec_conv_weight: float = 1.0,
        onset_weight: float = 1.0,
        sample_rate: int = 48_000,
    ) -> None:
        super().__init__()
        self.lpaps = LPAPSLoss(sample_rate)
        self.spec_conv = SpectralConvergence(STFTConfig(n_fft=4096, hop_length=1024))
        self.onset = OnsetWeightedLoss(hop_length=256)
        self.w1 = lpaps_weight
        self.w2 = spec_conv_weight
        self.w3 = onset_weight

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        lpaps_term = self.lpaps(reference, estimate)
        spec_term = self.spec_conv(reference, estimate)
        onset_term = self.onset(reference, estimate)
        return -(self.w1 * lpaps_term + self.w2 * spec_term + self.w3 * onset_term)


class AWeightedSTFTLoss(nn.Module):
    """Multi-resolution STFT loss with A-weighting for music.
    
    Adapted from the paper's speech perceptual loss to music domain by:
    - Using A-weighting to flatten frequency response (not speech-shaped)
    - Multi-resolution STFT at [2048, 4096, 8192] for 48kHz audio
    - Spectral convergence + log-magnitude L1
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_ffts: Sequence[int] = (2048, 4096, 8192),
        hop_lengths: Sequence[int] | None = None,
        sc_weight: float = 0.5,
        logmag_weight: float = 0.5,
        log_eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.sr = sample_rate
        self.n_ffts = list(n_ffts)
        self.hops = list(hop_lengths) if hop_lengths else [n // 4 for n in self.n_ffts]
        self.sc_w = sc_weight
        self.lm_w = logmag_weight
        self.log_eps = log_eps

        # Register windows and A-weighting vectors
        for i, n_fft in enumerate(self.n_ffts):
            self.register_buffer(f"window_{i}", torch.hann_window(n_fft))
            F = n_fft // 2 + 1
            freqs = torch.linspace(0.0, self.sr / 2.0, F)
            aw = self._a_weight_linear(freqs)
            self.register_buffer(f"aweight_{i}", aw)

    @staticmethod
    def _a_weight_db(f_hz: Tensor) -> Tensor:
        """ITU/IEC A-weighting curve in dB."""
        f2 = f_hz * f_hz + 1e-20
        ra_num = (12194.0**2) * (f2**2)
        ra_den = (f2 + 20.6**2) * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
        ra = ra_num / ra_den
        return 20.0 * torch.log10(ra + 1e-20) + 2.0

    def _a_weight_linear(self, freqs: Tensor) -> Tensor:
        """Convert A-weighting from dB to linear gain."""
        adb = self._a_weight_db(freqs)
        return (10.0 ** (adb / 20.0)).clamp_min(1e-3)

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """
        Args:
            reference: [B, C, T]
            estimate: [B, C, T]
        """
        # Flatten channels to process all separately
        ref_flat = reference.reshape(reference.shape[0] * reference.shape[1], reference.shape[-1])
        est_flat = estimate.reshape(estimate.shape[0] * estimate.shape[1], estimate.shape[-1])

        total = estimate.new_zeros(())
        for i, (n_fft, hop) in enumerate(zip(self.n_ffts, self.hops)):
            window = getattr(self, f"window_{i}").to(estimate.device)
            aweight = getattr(self, f"aweight_{i}").to(estimate.device)

            # Compute STFT
            Se = torch.stft(est_flat, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
            St = torch.stft(ref_flat, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)

            # Apply A-weighting to magnitudes
            Me = Se.abs() * aweight.view(-1, 1)
            Mt = St.abs() * aweight.view(-1, 1)

            # Spectral convergence
            sc = (Me - Mt).norm(p="fro") / (Mt.norm(p="fro") + 1e-12)

            # Log-magnitude L1
            le = torch.log(Me + self.log_eps)
            lt = torch.log(Mt + self.log_eps)
            lm = F.l1_loss(le, lt)

            total = total + (self.sc_w * sc + self.lm_w * lm)

        return total / len(self.n_ffts)


class InstantaneousFrequencyLoss(nn.Module):
    """Instantaneous frequency loss for phase coherence.
    
    Measures the L1 distance between phase time-derivatives (IF),
    which captures transient sharpness and phase alignment.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """
        Args:
            reference: [B, C, T]
            estimate: [B, C, T]
        """
        ref_flat = reference.reshape(reference.shape[0] * reference.shape[1], reference.shape[-1])
        est_flat = estimate.reshape(estimate.shape[0] * estimate.shape[1], estimate.shape[-1])

        window = self.window.to(reference.device)
        if not isinstance(window, Tensor):
            window = torch.hann_window(self.n_fft, device=reference.device)
        Se = torch.stft(est_flat, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)
        St = torch.stft(ref_flat, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)

        # Phase unwrapping approximation: phase difference over time
        phase_e = torch.angle(Se)
        phase_t = torch.angle(St)

        # Instantaneous frequency: d(phase)/dt
        if_e = phase_e[:, :, 1:] - phase_e[:, :, :-1]
        if_t = phase_t[:, :, 1:] - phase_t[:, :, :-1]

        # Wrap to [-π, π]
        if_e = torch.remainder(if_e + torch.pi, 2 * torch.pi) - torch.pi
        if_t = torch.remainder(if_t + torch.pi, 2 * torch.pi) - torch.pi

        return F.l1_loss(if_e, if_t)


class GainConsistencyLoss(nn.Module):
    """Frame-wise RMS consistency for dynamic range preservation.
    
    Penalizes deviations in the loudness envelope to prevent over-compression
    or dynamic range loss.
    """

    def __init__(self, frame_len: int = 2048, hop: int = 512) -> None:
        super().__init__()
        self.frame_len = frame_len
        self.hop = hop

    def _rms_envelope(self, wav: Tensor) -> Tensor:
        """Compute RMS envelope using sliding window."""
        # wav: [B, C, T]
        B, C, T = wav.shape
        wav = wav.reshape(B * C, T)

        # Unfold to frames
        frames = F.unfold(
            wav.unsqueeze(1).unsqueeze(1),  # [B*C, 1, 1, T]
            kernel_size=(1, self.frame_len),
            stride=(1, self.hop),
        )  # [B*C, frame_len, n_frames]

        rms = torch.sqrt(torch.mean(frames**2, dim=1) + 1e-8)  # [B*C, n_frames]
        return rms.reshape(B, C, -1)

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """
        Args:
            reference: [B, C, T]
            estimate: [B, C, T]
        """
        ref_rms = self._rms_envelope(reference)
        est_rms = self._rms_envelope(estimate)
        return F.l1_loss(torch.log1p(est_rms), torch.log1p(ref_rms))


class StereoImagingLoss(nn.Module):
    """Stereo imaging loss for spatial coherence.
    
    Preserves mid-side balance to maintain stereo width and localization.
    Critical for music production quality.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """
        Args:
            reference: [B, 2, T] stereo
            estimate: [B, 2, T] stereo
        """
        if reference.shape[1] != 2 or estimate.shape[1] != 2:
            return torch.tensor(0.0, device=reference.device)

        # Mid-side decomposition
        ref_mid = (reference[:, 0] + reference[:, 1]) / 2
        ref_side = (reference[:, 0] - reference[:, 1]) / 2
        est_mid = (estimate[:, 0] + estimate[:, 1]) / 2
        est_side = (estimate[:, 0] - estimate[:, 1]) / 2

        # L2 loss on mid and side separately
        loss_mid = F.mse_loss(est_mid, ref_mid)
        loss_side = F.mse_loss(est_side, ref_side)

        return loss_mid + loss_side


class MultiMelSNRLoss(nn.Module):
    """Multi-Mel-SNR loss using scale-invariant projection and multiple mel configurations.
    
    Differentiable version of the Multi-Mel-SNR metric for use as a training loss.
    Uses three mel-spectrogram configurations to capture different frequency/time resolutions.
    
    Returns a positive loss value in range [0, max_loss] where:
    - Mel-SNR >= 40 dB (excellent) → loss ≈ 0
    - Mel-SNR ≈ 20 dB (good) → loss ≈ 1.0
    - Mel-SNR ≈ 0 dB (poor) → loss ≈ 2.5
    
    Note: Mel-SNR rarely goes below 0 dB even for very noisy waveforms due to
    scale-invariant projection and mel-filtering. The loss is calibrated based
    on observed mel-SNR ranges, not raw waveform SNR.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        eps: float = 1e-8,
        target_snr: float = 40.0,  # Target mel-SNR in dB (loss = 0 at this point)
        max_loss: float = 5.0,  # Maximum loss value at very low SNR
        steepness: float = 0.0137,  # DEPRECATED: kept for backward compatibility, not used
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.target_snr = target_snr
        self.max_loss = max_loss
        self.steepness = steepness
        
        # Three mel configurations: (n_fft, hop_length, n_mels)
        self.configs = [
            (512, 256, 80),
            (1024, 512, 128),
            (2048, 1024, 192)
        ]
        
        # Pre-compute mel filterbanks and register as buffers
        for i, (n_fft, hop_length, n_mels) in enumerate(self.configs):
            # Create mel filterbank
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=float(sample_rate // 2),
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm="slaney",
                mel_scale="htk"
            )
            self.register_buffer(f"mel_fb_{i}", mel_fb)
            
            # Register window as buffer
            window = torch.hann_window(n_fft)
            self.register_buffer(f"window_{i}", window)

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """Compute positive-range multi-mel-SNR loss.
        
        Parameters
        ----------
        reference : Tensor
            Ground truth audio, shape [B, C, T]
        estimate : Tensor
            Predicted audio, shape [B, C, T]
            
        Returns
        -------
        Tensor
            Positive loss value in range [0, max_loss] where:
            - Higher SNR (better quality) → lower loss
            - SNR at target_snr → loss ≈ 0
            - Very low SNR → loss approaches max_loss
        """
        # Flatten channels to process all separately
        ref_flat = reference.reshape(reference.shape[0] * reference.shape[1], reference.shape[-1])
        est_flat = estimate.reshape(estimate.shape[0] * estimate.shape[1], estimate.shape[-1])
        
        # Scale-invariant normalization per sample
        alpha = (ref_flat * est_flat).sum(dim=1, keepdim=True) / (
            (est_flat * est_flat).sum(dim=1, keepdim=True) + self.eps
        )
        est_scaled = alpha * est_flat
        
        snrs = []
        for i, (n_fft, hop_length, n_mels) in enumerate(self.configs):
            # Get device-aware buffers (explicit .to() for safety)
            mel_fb = getattr(self, f"mel_fb_{i}").to(reference.device)
            window = getattr(self, f"window_{i}").to(reference.device)
            
            # Compute STFT
            spec_ref = torch.stft(
                ref_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            spec_est = torch.stft(
                est_scaled,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            
            # Power spectrogram
            power_ref = spec_ref.abs().pow(2)
            power_est = spec_est.abs().pow(2)
            
            # Apply mel filterbank: [B, F, T] @ [F, M] -> [B, M, T]
            M_ref = torch.matmul(power_ref.transpose(1, 2), mel_fb).transpose(1, 2)
            M_est = torch.matmul(power_est.transpose(1, 2), mel_fb).transpose(1, 2)
            
            # SNR in dB per sample
            signal_power = M_ref.pow(2).sum(dim=[1, 2])
            noise_power = (M_ref - M_est).pow(2).sum(dim=[1, 2])
            snr = 10.0 * torch.log10(signal_power / (noise_power + self.eps) + self.eps)
            snrs.append(snr)
        
        # Average across configurations and samples
        multi_snr = torch.stack(snrs).mean(dim=0).mean()
        
        # Convert SNR to positive loss with precise calibration
        # Since mel-SNR rarely goes below 0 dB (even for very noisy signals),
        # we calibrate the loss based on observed mel-SNR values:
        # - Mel-SNR >= 40 dB (excellent quality) → loss = 0
        # - Mel-SNR = 20 dB (good quality) → loss ≈ 1.0
        # - Mel-SNR = 0 dB (very poor quality) → loss = 2.5
        #
        # Use piecewise exponential for better dynamic range:
        # For gap in [0, 20]: loss = 5.0 * (1 - exp(-0.035 * gap))
        # This gives: gap=0→loss=0, gap=20→loss=1.0, gap=40→loss=2.5
        
        snr_gap = torch.clamp(self.target_snr - multi_snr, min=0.0)  # Positive when below target
        
        # Exponential with k chosen for: at gap=20 → loss=1.0, at gap=40 → loss=2.5
        # 1.0 = 5.0 * (1 - exp(-k * 20))
        # 0.2 = 1 - exp(-k * 20)
        # exp(-k * 20) = 0.8
        # k * 20 = -ln(0.8) = 0.223
        # k = 0.0115
        # Check: at gap=40: 5.0 * (1 - exp(-0.0115 * 40)) = 5.0 * (1 - exp(-0.46)) = 5.0 * 0.369 = 1.85
        # Need higher k for loss=2.5 at gap=40:
        # 2.5 = 5.0 * (1 - exp(-k * 40))
        # 0.5 = 1 - exp(-k * 40)
        # exp(-k * 40) = 0.5
        # k = 0.693 / 40 = 0.0173 (our current value)
        k = 0.0173
        loss = self.max_loss * (1.0 - torch.exp(-k * snr_gap))
        
        return loss
    
    def get_snr_and_loss(self, reference: Tensor, estimate: Tensor) -> tuple[Tensor, Tensor]:
        """Compute both SNR value and loss for debugging/analysis.
        
        Parameters
        ----------
        reference : Tensor
            Ground truth audio, shape [B, C, T]
        estimate : Tensor
            Predicted audio, shape [B, C, T]
            
        Returns
        -------
        tuple[Tensor, Tensor]
            (multi_mel_snr in dB, loss value)
        """
        ref_flat = reference.reshape(reference.shape[0] * reference.shape[1], reference.shape[-1])
        est_flat = estimate.reshape(estimate.shape[0] * estimate.shape[1], estimate.shape[-1])
        
        # Scale-invariant normalization
        alpha = (ref_flat * est_flat).sum(dim=1, keepdim=True) / (
            (est_flat * est_flat).sum(dim=1, keepdim=True) + self.eps
        )
        est_scaled = alpha * est_flat
        
        snrs = []
        for i, (n_fft, hop_length, n_mels) in enumerate(self.configs):
            # Get device-aware buffers (explicit .to() for safety)
            mel_fb = getattr(self, f"mel_fb_{i}").to(reference.device)
            window = getattr(self, f"window_{i}").to(reference.device)
            
            # Compute STFT
            spec_ref = torch.stft(
                ref_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            spec_est = torch.stft(
                est_scaled,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            
            # Power spectrogram
            power_ref = spec_ref.abs().pow(2)
            power_est = spec_est.abs().pow(2)
            
            # Apply mel filterbank
            M_ref = torch.matmul(power_ref.transpose(1, 2), mel_fb).transpose(1, 2)
            M_est = torch.matmul(power_est.transpose(1, 2), mel_fb).transpose(1, 2)
            
            signal_power = M_ref.pow(2).sum(dim=[1, 2])
            noise_power = (M_ref - M_est).pow(2).sum(dim=[1, 2])
            snr = 10.0 * torch.log10(signal_power / (noise_power + self.eps) + self.eps)
            snrs.append(snr)
        
        multi_snr = torch.stack(snrs).mean(dim=0).mean()
        snr_gap = torch.clamp(self.target_snr - multi_snr, min=0.0)
        
        k = 0.0173
        loss = self.max_loss * (1.0 - torch.exp(-k * snr_gap))
        
        return multi_snr, loss


class MusicPerceptualLoss(nn.Module):
    """Music-aware perceptual loss combining multiple music-specific metrics.
    
    This replaces HumanFeedbackSurrogateLoss (speech-oriented) with losses
    tailored for music restoration:
    - A-weighted multi-res STFT (flat frequency response, not speech-shaped)
    - Instantaneous frequency (phase coherence, transient sharpness)
    - Gain consistency (dynamic range preservation)
    - Stereo imaging (spatial coherence)
    - Multi-Mel-SNR (scale-invariant spectral quality)
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        w_stft: float = 1.0,
        w_if: float = 0.75,
        w_gain: float = 0.75,
        w_stereo: float = 1.0,
        w_mel_snr: float = 1.5,        
    ) -> None:
        super().__init__()
        self.stft_loss = AWeightedSTFTLoss(sample_rate=sample_rate)
        self.if_loss = InstantaneousFrequencyLoss(n_fft=2048, hop_length=512)
        self.gain_loss = GainConsistencyLoss(frame_len=2048, hop=512)
        self.stereo_loss = StereoImagingLoss()
        self.mel_snr_loss = MultiMelSNRLoss(
            sample_rate=sample_rate,                   
        )

        self.w_stft = w_stft
        self.w_if = w_if
        self.w_gain = w_gain
        self.w_stereo = w_stereo
        self.w_mel_snr = w_mel_snr

    def forward(self, reference: Tensor, estimate: Tensor) -> Tensor:
        """
        Args:
            reference: [B, 2, T] stereo audio at 48kHz
            estimate: [B, 2, T] stereo audio at 48kHz
            
        Returns:
            Weighted sum of music perceptual metrics
        """
        stft_term = self.stft_loss(reference, estimate)
        if_term = self.if_loss(reference, estimate)
        gain_term = self.gain_loss(reference, estimate)
        stereo_term = self.stereo_loss(reference, estimate)
        mel_snr_term = self.mel_snr_loss(reference, estimate)

        total = (
            self.w_stft * stft_term
            + self.w_if * if_term
            + self.w_gain * gain_term
            + self.w_stereo * stereo_term
            + self.w_mel_snr * mel_snr_term
        )
        return total
    

class ERBSoftDTWMetric(torch.nn.Module):
    """
    Zimtohrli-like differentiable metric:
      - ERB front-end (128 filters on ERB scale)
      - log-loudness + partial loudness normalization
      - SoftDTW over time on ERB features

    Returns:
      distance: SoftDTW distance (smaller = better)
      similarity: exp(-alpha * distance) in (0, 1]
    """
    def __init__(
        self,
        sample_rate: int = 48_000,
        n_filters: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        low_lim: float = 50.0,
        high_lim: float | None = None,
        softdtw_gamma: float = 3.0,
        alpha: float = 1e-5,
        frame_subsample: int = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.frame_subsample = frame_subsample

        # Precompute ERB filterbank (constant, no grad)
        fbanks_np = make_erb_filterbank_for_stft(
            sr=sample_rate,
            n_fft=n_fft,
            n_filters=n_filters,
            low_lim=low_lim,
            high_lim=high_lim,
        )  # (C, F)
        fbanks = torch.from_numpy(fbanks_np)  # (C, F)
        self.register_buffer("erb_fbank", fbanks)  # no grad

        # SoftDTW over time
        self.softdtw = SoftDTW(gamma=softdtw_gamma, normalize=False)

    def _erb_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, T) or (B, C, T) for stereo
        returns: log-ERB spectrogram (B, C_erb, T_frames)
        
        If stereo input (B, 2, T), processes each channel and averages ERB features.
        """
        # Handle stereo by averaging channels
        if audio.dim() == 3:
            # Stereo: (B, 2, T) -> average to (B, T)
            audio = audio.mean(dim=1)
        
        B, T = audio.shape

        # STFT: (B, F, T_frames, 2) complex
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=audio.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # (B, F, T_frames)

        # Power spectrogram
        power = (stft.real**2 + stft.imag**2)  # (B, F, T_frames)
        # F should match erb_fbank.shape[1]
        assert power.shape[1] == self.erb_fbank.shape[1], \
            f"STFT freq bins ({power.shape[1]}) != ERB filt size ({self.erb_fbank.shape[1]})"

        # ERB filtering: (C, F) @ (B, F, T) -> (B, C, T)
        # use einsum for clarity: c f, b f t -> b c t
        erb_fbank = self.erb_fbank.to(audio.device)
        erb = torch.einsum("cf, bft -> bct", erb_fbank, power)

        # optional temporal subsampling
        if self.frame_subsample > 1:
            erb = erb[:, :, ::self.frame_subsample]

        # log-loudness
        eps = 1e-8
        log_erb = torch.log10(erb + eps)  # (B, C, T)

        return log_erb

    def _partial_loudness_normalization(
        self,
        ref_log: torch.Tensor,
        gen_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bring maxima 82% closer, like in the paper (approx).
        Operates per-sample in batch.
        """
        # (B, 1, 1)
        max_ref = ref_log.amax(dim=(1, 2), keepdim=True)
        max_gen = gen_log.amax(dim=(1, 2), keepdim=True)
        delta = max_ref - max_gen  # (B,1,1)

        ref_adj = ref_log - 0.41 * delta
        gen_adj = gen_log + 0.41 * delta
        return ref_adj, gen_adj

    def forward(self, ref_audio: torch.Tensor, gen_audio: torch.Tensor):
        """
        ref_audio, gen_audio: (B, T), (B, C, T) for stereo, or (T,)
        Assumes both at self.sample_rate (no resampling inside).
        
        Returns:
          distance: (B,) SoftDTW distance
          similarity: (B,) exp(-alpha * distance) in (0,1]
        """
        # Ensure batch dimension
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if gen_audio.dim() == 1:
            gen_audio = gen_audio.unsqueeze(0)

        assert ref_audio.shape == gen_audio.shape, "ref & gen must have same shape"

        # ERB "neurogram"
        ref_log = self._erb_spectrogram(ref_audio)  # (B, C, T)
        gen_log = self._erb_spectrogram(gen_audio)  # (B, C, T')

        # partial loudness normalization
        ref_norm, gen_norm = self._partial_loudness_normalization(ref_log, gen_log)

        # SoftDTW expects (B, T, D): time=frames, D=features
        x = ref_norm.transpose(1, 2)  # (B, T, C)
        y = gen_norm.transpose(1, 2)  # (B, T', C)

        # SoftDTW distance per batch element
        dist = self.softdtw(x, y)  # (B,)
        # similarity in (0,1], differentiable
        sim = torch.exp(-self.alpha * dist)

        return dist, sim


class ERBSoftDTWLoss(torch.nn.Module):
    """
    Gamified perceptual loss using ERBSoftDTWMetric.
    
    Instead of minimizing large raw distances, this uses similarity [0,1] to scale
    a base loss, making the metric more suitable as a regularization term.
    
    Key idea: When perceptual similarity is high, reduce the base loss penalty.
    
    Usage:
        base_loss = F.l1_loss(estimate, reference)  # Main reconstruction loss
        perceptual_loss = erb_loss(reference, estimate, base_loss)
        total_loss = base_loss + lambda_perceptual * perceptual_loss
    """
    def __init__(
        self,
        sample_rate: int = 48_000,
        n_filters: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        low_lim: float = 50.0,
        high_lim: float | None = None,
        softdtw_gamma: float = 1.0,
        alpha: float = 0.001,  # Smaller alpha for more gradual similarity decay
        frame_subsample: int = 2,  # Subsample for efficiency
        lambda_gamify: float = 1.0,  # Scaling factor for gamification
        use_distance: bool = False,  # If True, return raw distance; if False, use gamified version
    ):
        super().__init__()
        self.metric = ERBSoftDTWMetric(
            sample_rate=sample_rate,
            n_filters=n_filters,
            n_fft=n_fft,
            hop_length=hop_length,
            low_lim=low_lim,
            high_lim=high_lim,
            softdtw_gamma=softdtw_gamma,
            alpha=alpha,
            frame_subsample=frame_subsample,
        )
        self.lambda_gamify = lambda_gamify
        self.use_distance = use_distance
    
    def forward(
        self, 
        reference: torch.Tensor, 
        estimate: torch.Tensor,
        base_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            reference: (B, C, T) or (B, T) clean audio
            estimate: (B, C, T) or (B, T) generated audio
            base_loss: Optional (B,) or scalar base loss to scale by similarity
        
        Returns:
            loss: If use_distance=True, returns normalized distance.
                  If use_distance=False and base_loss provided, returns gamified loss.
                  Otherwise returns (1 - similarity) as penalty.
        """
        dist, sim = self.metric(reference, estimate)  # (B,), (B,)
        
        if self.use_distance:
            # Return normalized distance (divide by typical scale ~2000 to get reasonable range)
            return (dist / 100.0).mean()  # Heuristic: scale to ~0-50 range
        
        if base_loss is not None:
            # Gamified: scale base loss by perceptual dissimilarity
            # When sim=1 (perfect), scale=1.0 (no penalty)
            # When sim=0 (terrible), scale=1.0 + lambda_gamify (increased penalty)
            scale = 1.0 + self.lambda_gamify * (1.0 - sim)  # (B,)
            
            # Handle both scalar and per-sample base_loss
            if base_loss.dim() == 0:
                # Scalar base_loss: broadcast to (B,)
                return (base_loss * scale).mean()
            else:
                # Per-sample base_loss: (B,)
                return (base_loss * scale).mean()
        else:
            # No base loss: return dissimilarity as penalty
            return (1.0 - sim).mean()


__all__ = [
    "LMOSLoss",
    "LMOSConfig",
    "LeastSquaresGANLoss",
    "HingeGANLoss",
    "feature_matching_loss",
    "HumanFeedbackSurrogateLoss",  # Deprecated, use MusicPerceptualLoss
    "MusicPerceptualLoss",
    "MultiMelSNRLoss",
    "AWeightedSTFTLoss",
    "InstantaneousFrequencyLoss",
    "GainConsistencyLoss",
    "StereoImagingLoss",
    "ERBSoftDTWMetric",
    "ERBSoftDTWLoss",
]
