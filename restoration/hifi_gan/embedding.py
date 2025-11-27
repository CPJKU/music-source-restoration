from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torchaudio

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel, AutoProcessor  # type: ignore

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None  # type: ignore
    AutoProcessor = None  # type: ignore
    HAS_TRANSFORMERS = False

try:  # pragma: no cover - optional dependency
    import codicodec  # type: ignore

    HAS_CODICODEC = True
except ImportError:  # pragma: no cover - optional dependency
    codicodec = None  # type: ignore
    HAS_CODICODEC = False

Tensor = torch.Tensor


def _has_torchaudio_bundle(name: str) -> bool:
    try:
        getattr(torchaudio.pipelines, name)
        return True
    except AttributeError:
        return False


class MusicSSLFeatureExtractor(nn.Module):
    """
    Flexible frontend that can produce embeddings from:
      * HuggingFace MERT (default when available),
      * torchaudio pipeline bundles,
      * CoDiCodec latents (preferred when `prefer_codicodec=True`),
      * or a log-mel fallback.

    Passing precomputed embeddings or codec latents skips expensive recomputation.
    """

    BACKBONE_PRESETS = {
        "mert26m": {"hf_model_name": "m-a-p/MERT-v1-26M"},
        "mert95m": {"hf_model_name": "m-a-p/MERT-v1-95M"},
        "mert330m": {"hf_model_name": "m-a-p/MERT-v1-330M"},
        "codicodec": {"prefer_codicodec": True, "codec_as_channels": True},
        "mel": {},
    }

    def __init__(
        self,
        target_sample_rate: int = 48_000,
        bundle_name: str = "MERT_V1",
        mel_bins: int = 128,
        mel_hop: int = 320,
        hf_model_name: str = "m-a-p/MERT-v1-26M",
        backbone: str = "codicodec",
    ) -> None:
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.bundle_name = bundle_name
        self.mel_bins = mel_bins
        self.mel_hop = mel_hop
        self.mel_bins = mel_bins
        self.prefer_codicodec = False
        self.codec_as_channels = False

        preset_key = backbone.lower().replace(" ", "") if backbone else ""
        preset = self.BACKBONE_PRESETS.get(preset_key, {})
        if "hf_model_name" in preset:
            self.hf_model_name = preset["hf_model_name"]
        if preset.get("prefer_codicodec"):
            self.prefer_codicodec = True
        if preset.get("codec_as_channels"):
            self.codec_as_channels = True
        self.backbone = preset_key or "mel"

        self.register_buffer("_buffer", torch.zeros(1), persistent=False)

        # Initialise backend containers
        self.hf_processor = None
        self.hf_model = None
        self.hf_sample_rate = target_sample_rate
        self.hf_dim = mel_bins

        self.bundle_model: Optional[nn.Module] = None
        self.bundle_sample_rate = target_sample_rate
        self.bundle_dim = mel_bins

        self.codec = None
        self.codec_sample_rate = target_sample_rate
        self.codec_dim = mel_bins

        self.active_backend = "mel"
        self.ssl_sample_rate = target_sample_rate
        self.ssl_dim = mel_bins
        self.mel_transform = None

        self._init_hf_backend()
        self._init_bundle_backend()
        self._init_codec_backend()

        if self.prefer_codicodec and self.codec is not None:
            self._set_active_backend("codicodec")
        elif self.hf_model is not None:
            self._set_active_backend("hf")
        elif self.bundle_model is not None:
            self._set_active_backend("bundle")
        else:
            self._set_active_backend("mel")

    # --------------------------------------------------------------------- #
    # Backend initialisation helpers
    # --------------------------------------------------------------------- #

    def _init_hf_backend(self) -> None:
        if not HAS_TRANSFORMERS:
            return
        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.hf_model_name, trust_remote_code=True)  # type: ignore[arg-type]
            self.hf_model = AutoModel.from_pretrained(self.hf_model_name, trust_remote_code=True)  # type: ignore[arg-type]
            self.hf_model.eval()
            self.hf_model.requires_grad_(False)
            self.hf_sample_rate = int(getattr(self.hf_processor, "sampling_rate", self.target_sample_rate))
            if hasattr(self.hf_model.config, "hidden_size"):
                self.hf_dim = int(self.hf_model.config.hidden_size)
            else:
                self.hf_dim = int(getattr(self.hf_model.config, "projection_dim", self.mel_bins))
        except Exception:  # pragma: no cover - optional dependency failure
            self.hf_processor = None
            self.hf_model = None
            self.hf_sample_rate = self.target_sample_rate
            self.hf_dim = self.mel_bins

    def _init_bundle_backend(self) -> None:
        if not _has_torchaudio_bundle(self.bundle_name):
            return
        try:
            bundle = getattr(torchaudio.pipelines, self.bundle_name)
            model = bundle.get_model()
            model.eval()
            model.requires_grad_(False)
            self.bundle_model = model
            self.bundle_sample_rate = bundle.sample_rate
            self.bundle_dim = bundle.embedding_dim
        except Exception:  # pragma: no cover - optional dependency failure
            self.bundle_model = None
            self.bundle_sample_rate = self.target_sample_rate
            self.bundle_dim = self.mel_bins

    def _init_codec_backend(self) -> None:
        if not HAS_CODICODEC:
            return
        try:
            if hasattr(codicodec.EncoderDecoder, "load_pretrained"):
                self.codec = codicodec.EncoderDecoder.load_pretrained()
            else:
                self.codec = codicodec.EncoderDecoder()
            self.codec_sample_rate = int(getattr(self.codec, "sample_rate", self.target_sample_rate))
            self.codec_dim = self._infer_codec_dim()
        except Exception:  # pragma: no cover - optional dependency failure
            self.codec = None
            self.codec_sample_rate = self.target_sample_rate
            self.codec_dim = self.mel_bins

    def _set_active_backend(self, name: str) -> None:
        if name == "codicodec" and self.codec is not None:
            self.active_backend = "codicodec"
            self.ssl_sample_rate = self.codec_sample_rate
            self.ssl_dim = self.codec_dim
        elif name == "hf" and self.hf_model is not None:
            self.active_backend = "hf"
            self.ssl_sample_rate = self.hf_sample_rate
            self.ssl_dim = self.hf_dim
        elif name == "bundle" and self.bundle_model is not None:
            self.active_backend = "bundle"
            self.ssl_sample_rate = self.bundle_sample_rate
            self.ssl_dim = self.bundle_dim
        else:
            self.active_backend = "mel"
            self.ssl_sample_rate = self.target_sample_rate
            self.ssl_dim = self.mel_bins

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ssl_sample_rate,
            n_fft=1024,
            hop_length=self.mel_hop,
            n_mels=self.mel_bins,
            f_min=20.0,
            f_max=self.ssl_sample_rate / 2.0,
            center=True,
            power=2.0,
        )

    # --------------------------------------------------------------------- #
    # Forward path
    # --------------------------------------------------------------------- #

    def _resample(self, wav: Tensor, orig_sr: int, target_sr: int) -> Tensor:
        if orig_sr == target_sr:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, target_sr)

    @torch.inference_mode()
    def forward(
        self,
        waveform: Tensor,
        sample_rate: int,
        embeddings: Optional[Tensor] = None,
        codec_latents: Optional[Tensor | Sequence[Tensor]] = None,
    ) -> Tensor:
        """
        Args:
            waveform: [B, C, T] audio.
            sample_rate: sampling rate of the audio.
            embeddings: precomputed embeddings to reuse.
            codec_latents: optional CoDiCodec latents (list or tensor).
        Returns:
            Tensor shaped [B, D, T_feat].
        """
        if embeddings is not None:
            return embeddings

        device = waveform.device
        batch, channels, _ = waveform.shape

        # If codec latents are supplied or backend prefers codec, handle first.
        if codec_latents is not None:
            formatted = self._format_codec_latent_batch(codec_latents, device=device)
            self._update_dim_if_needed(formatted.shape[1])
            return formatted

        if self.active_backend == "codicodec" and self.codec is not None:
            try:
                mono = waveform.mean(dim=1) if channels > 1 else waveform.squeeze(1)
                codec_audio = self._resample(mono, sample_rate, self.codec_sample_rate)
                features = self._compute_codec_features(codec_audio, device=device)
                self._update_dim_if_needed(features.shape[1])
                return features
            except Exception:
                # Fall back gracefully
                if self.hf_model is not None:
                    self._set_active_backend("hf")
                elif self.bundle_model is not None:
                    self._set_active_backend("bundle")
                else:
                    self._set_active_backend("mel")

        # HuggingFace MERT
        if self.hf_model is not None and self.active_backend == "hf":
            mono = waveform.mean(dim=1) if channels > 1 else waveform.squeeze(1)
            wav_resampled = self._resample(mono, sample_rate, self.hf_sample_rate)
            audio_list = [wav_resampled[i].detach().cpu().to(torch.float32).numpy() for i in range(batch)]
            processor_inputs = self.hf_processor(
                audio_list,
                sampling_rate=self.hf_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            if hasattr(processor_inputs, "to"):
                processor_inputs = processor_inputs.to(device)  # type: ignore[union-attr]
            else:
                processor_inputs = {k: v.to(device) for k, v in processor_inputs.items()}  # type: ignore[union-attr]
            outputs = self.hf_model(**processor_inputs, output_hidden_states=False)
            hidden = outputs.last_hidden_state  # [B, frames, hidden]
            feats = hidden.permute(0, 2, 1).contiguous()
            self._update_dim_if_needed(feats.shape[1])
            return feats

        # torchaudio Spectral UNet bundle fallback
        if self.bundle_model is not None and self.active_backend == "bundle":
            mono = waveform.mean(dim=1) if channels > 1 else waveform.squeeze(1)
            wav_resampled = self._resample(mono, sample_rate, self.bundle_sample_rate)
            bundle = self.bundle_model.to(device)
            feats = bundle.extract_features(wav_resampled)[0]  # type: ignore[attr-defined]
            feats = feats.permute(0, 2, 1).contiguous()
            self._update_dim_if_needed(feats.shape[1])
            return feats

        # Final fallback: log-mel features
        mono = waveform.mean(dim=1, keepdim=True) if channels > 1 else waveform
        wav_resampled = self._resample(mono, sample_rate, self.ssl_sample_rate)
        mel = self.mel_transform(wav_resampled.squeeze(1).to(device))
        mel = torch.log10(mel + 1e-6)
        self._update_dim_if_needed(mel.shape[1])
        return mel

    # --------------------------------------------------------------------- #
    # CoDiCodec helpers
    # --------------------------------------------------------------------- #

    def _format_codec_latent_batch(
        self,
        latents: Tensor | Sequence[Tensor],
        device: torch.device,
    ) -> Tensor:
        tensors: list[Tensor] = []
        if isinstance(latents, torch.Tensor):
            tensors = self._split_codec_tensor(latents)
        else:
            for item in latents:
                tensor = torch.as_tensor(item)
                tensors.append(self._reshape_codec_latent(tensor))

        max_len = max(t.shape[0] for t in tensors)
        dim = tensors[0].shape[1]
        padded = []
        for tensor in tensors:
            if tensor.shape[0] < max_len:
                pad = torch.zeros(max_len - tensor.shape[0], dim, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        stacked = torch.stack(padded, dim=0).to(device=device, dtype=torch.float32)  # [B, seq, dim]
        return stacked.permute(0, 2, 1).contiguous()

    def _split_codec_tensor(self, tensor: Tensor) -> list[Tensor]:
        """
        Normalise codec latent tensor into a list of [seq, dim] tensors.
        Accepts shapes like:
            [B, F, K, D], [B, T, D], [F, K, D], [T, D]
        """
        if tensor.ndim == 4:
            return [
                self._reshape_codec_latent(tensor[idx])
                for idx in range(tensor.shape[0])
            ]
        if tensor.ndim == 3:
            return [
                self._reshape_codec_latent(tensor[idx])
                for idx in range(tensor.shape[0])
            ]
        if tensor.ndim == 2:
            return [self._reshape_codec_latent(tensor)]
        raise ValueError("Unsupported codec latent tensor shape.")

    def _reshape_codec_latent(self, latent: Tensor) -> Tensor:
        latent = latent.detach().cpu().to(torch.float32)
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim == 3:
            latent = latent.reshape(-1, latent.shape[-1])
        elif latent.ndim == 2:
            pass
        elif latent.ndim == 1:
            latent = latent.unsqueeze(0)
        else:
            latent = latent.view(-1, latent.shape[-1])
        return latent

    def _compute_codec_features(self, wav: Tensor, device: torch.device) -> Tensor:
        if self.codec is None:
            raise RuntimeError("CoDiCodec backend is not initialised.")
        latents: list[Tensor] = []
        for sample in wav:
            latents.append(self._encode_codec_single(sample))
        formatted = self._format_codec_latent_batch(latents, device=device)
        return formatted

    def _encode_codec_single(self, sample: Tensor) -> Tensor:
        assert self.codec is not None
        audio_tensor = sample.detach().cpu().unsqueeze(0).to(torch.float32)
        encode_fn = None
        for name in ("encode_latents", "encode", "forward"):
            if hasattr(self.codec, name):
                encode_fn = getattr(self.codec, name)
                break
        if encode_fn is None:
            raise AttributeError("CoDiCodec encoder does not expose an encode method.")

        latent = None
        for attempt in range(3):
            try:
                if attempt == 0:
                    latent = encode_fn(audio_tensor, sample_rate=self.codec_sample_rate)
                elif attempt == 1:
                    latent = encode_fn(audio_tensor)
                else:
                    latent = encode_fn(audio_tensor.numpy(), sample_rate=self.codec_sample_rate)
                break
            except TypeError:
                continue

        if latent is None:
            raise RuntimeError("Unable to encode audio with CoDiCodec.")

        if isinstance(latent, (list, tuple)):
            latent = latent[0]
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        return self._reshape_codec_latent(latent_tensor)

    def _infer_codec_dim(self) -> int:
        if self.codec is None:
            return self.mel_bins
        try:
            dummy = torch.zeros(self.codec_sample_rate, dtype=torch.float32)
            features = self._encode_codec_single(dummy)
            # If codec_as_channels=True, return tokens×features (e.g., 8×64=512)
            # Otherwise return just feature dim (64)
            if self.codec_as_channels and features.ndim == 2:
                # features is [T, D], but we want to know the original shape before reshape
                # For CoDiCodec: 8 tokens × 64 dims = 512
                return 512  # 8 * 64 for CoDiCodec
            return features.shape[1]
        except Exception:
            return self.mel_bins

    def _update_dim_if_needed(self, dim: int) -> None:
        if dim != self.ssl_dim:
            self.ssl_dim = dim


__all__ = ["MusicSSLFeatureExtractor"]
