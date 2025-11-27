#!/usr/bin/env python3
"""Inference script for Finally GAN expert models.

Loads a trained expert model and runs inference on restoration dataset samples,
upsampling from base sample rate (24kHz) to target sample rate (48kHz).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm

from .configs import ModelConfig
from .generator import FinallyGenerator


def load_expert_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> FinallyGenerator:
    """Load a trained expert model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the checkpoint file (e.g., best_model.pt)
    device : torch.device | None
        Device to load model on. If None, uses CUDA if available.
        
    Returns
    -------
    FinallyGenerator
        Loaded generator model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = Path(checkpoint_path)
    
    # If checkpoint_path is a directory, find the latest checkpoint
    if checkpoint_path.is_dir():
        # Look for common checkpoint patterns
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        
        # Prefer best_model.pt, otherwise take the latest by name
        best_model = checkpoint_path / "best_model.pt"
        if best_model.exists():
            checkpoint_path = best_model
        else:
            # Sort by step number if present, otherwise by name
            checkpoint_files.sort(key=lambda p: p.stem)
            checkpoint_path = checkpoint_files[-1]
        
        print(f"Found checkpoint: {checkpoint_path.name}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        # Use default config if not saved - need to infer embedding backbone from weights
        print("Warning: model_config not found in checkpoint, inferring from weights")
        
        # Check if transfer learning was used (projection layer exists)
        has_projection = "embedding_proj.weight" in checkpoint["generator"]
        
        # Check if MERT/HF model weights exist
        has_hf_model = any("embedding_extractor.hf_model" in k for k in checkpoint["generator"].keys())
        
        # Infer backbone based on what weights exist
        if has_hf_model:
            # MERT model weights present
            hf_dim = checkpoint["generator"]["embedding_extractor.hf_model.encoder.layers.0.attention.k_proj.weight"].shape[0]
            if hf_dim == 768:
                embedding_backbone = "mert95m"
            elif hf_dim == 1024:
                embedding_backbone = "mert330m"
            else:
                embedding_backbone = "mert26m"
            print(f"Detected HuggingFace model with hidden_dim={hf_dim}, using backbone='{embedding_backbone}'")
        else:
            # No HF model, check hifi.proj to infer embedding dim
            proj_shape = checkpoint["generator"]["hifi.proj.weight"].shape
            total_channels = proj_shape[1]  # [out, in, k]
            spectral_channels = 256  # assumed
            embedding_dim = total_channels - spectral_channels
            
            # Map embedding dims to backbones
            backbone_map = {
                768: "mert95m",
                128: "encodec",
                512: "codicodec",
                80: "mel",
            }
            embedding_backbone = backbone_map.get(embedding_dim, "mel")
            print(f"Inferred embedding_dim={embedding_dim}, using backbone='{embedding_backbone}'")
        
        # Check pretrained_embedding_dim for transfer learning
        pretrained_embedding_dim = None
        if has_projection:
            proj_out_dim = checkpoint["generator"]["embedding_proj.weight"].shape[0]
            pretrained_embedding_dim = proj_out_dim
            print(f"Detected projection layer: transfer learning with pretrained_embedding_dim={pretrained_embedding_dim}")
        
        model_cfg = ModelConfig(
            base_sample_rate=24000,
            target_sample_rate=48000,
            embedding_backbone=embedding_backbone,
            pretrained_embedding_dim=pretrained_embedding_dim,
        )

    print("Using {} backbone".format(model_cfg.embedding_backbone))
    
    # For inference with saved HF model weights, we use 'mel' backend during init
    # to avoid downloading the HF model, then load the saved weights
    has_hf_weights = any("embedding_extractor.hf_model" in k for k in checkpoint["generator"].keys())
    
    if has_hf_weights:
        print("[inference] Checkpoint contains HF model weights - using mel backend for initialization")
        # Initialize with mel to avoid HF model download, we'll load the weights next
        model = FinallyGenerator(
            base_sample_rate=model_cfg.base_sample_rate,
            target_sample_rate=model_cfg.target_sample_rate,
            embedding_backbone="mel",  # Use mel to skip HF download
            pretrained_embedding_dim=model_cfg.pretrained_embedding_dim,
        )
    else:
        # No HF weights, use the detected backbone normally
        model = FinallyGenerator(
            base_sample_rate=model_cfg.base_sample_rate,
            target_sample_rate=model_cfg.target_sample_rate,
            embedding_backbone=model_cfg.embedding_backbone,
            pretrained_embedding_dim=model_cfg.pretrained_embedding_dim,
        )
    
    # Load state dict (use strict=False to allow for missing/extra keys in embedding extractor)
    if "generator" in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["generator"], strict=False)
        
        # After loading, update the active backend if HF weights were loaded
        if has_hf_weights:
            # Manually set the backend to match what was trained
            model.embedding_extractor.active_backend = "hf"
            model.embedding_extractor.ssl_dim = checkpoint["generator"]["embedding_extractor.hf_model.encoder.layers.0.attention.k_proj.weight"].shape[0]
            model.embedding_extractor.ssl_sample_rate = 24000  # MERT uses 24kHz
            print(f"[inference] Loaded HF model weights - backend set to 'hf' with dim={model.embedding_extractor.ssl_dim}")
            
            # Filter out expected "unexpected" keys (HF model weights that we loaded into attributes)
            unexpected_keys = [k for k in unexpected_keys if not k.startswith("embedding_extractor.hf_model.")]
            if not unexpected_keys:
                print("[inference] All checkpoint weights loaded successfully")
        
        if missing_keys:
            print(f"[inference] WARNING: Missing keys (using random init): {missing_keys}")
        if unexpected_keys:
            print(f"[inference] WARNING: Unexpected keys (not loaded): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    elif "generator_state" in checkpoint:
        model.load_state_dict(checkpoint["generator_state"], strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "ema" in checkpoint:
        # Try EMA weights if available
        model.load_state_dict(checkpoint["ema"], strict=False)
    else:
        raise KeyError(f"Could not find model state in checkpoint. Available keys: {list(checkpoint.keys())}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


@torch.no_grad()
def run_inference(
    model: FinallyGenerator,
    audio_path: str | Path,
    device: torch.device | None = None,
    segment_seconds: float | None = None,
) -> torch.Tensor:
    """Run inference on an audio file.
    
    Parameters
    ----------
    model : FinallyGenerator
        Loaded generator model
    audio_path : str | Path
        Path to input audio file
    device : torch.device | None
        Device to run inference on
    segment_seconds : float | None
        If provided, processes audio in segments of this duration.
        Useful for long files to avoid OOM.
        
    Returns
    -------
    torch.Tensor
        Enhanced audio at target sample rate (48kHz), shape [channels, samples]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    print(f"Loaded audio: {audio.shape} at {sr}Hz")
    
    # Resample to base sample rate if needed
    if sr != model.base_sample_rate:
        print(f"Resampling from {sr}Hz to {model.base_sample_rate}Hz")
        audio = torchaudio.functional.resample(audio, sr, model.base_sample_rate)
    
    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    
    # Process
    if segment_seconds is None:
        # Process entire file at once
        audio_batch = audio.unsqueeze(0).to(device)  # [1, 2, T]
        output_dict = model(audio_batch, stage="stage3")
        enhanced = output_dict["target"].squeeze(0)  # [2, T']
    else:
        # Process in segments (overlap-add)
        segment_samples = int(segment_seconds * model.base_sample_rate)
        hop_samples = segment_samples // 2  # 50% overlap
        
        segments = []
        num_segments = (audio.shape[-1] + hop_samples - 1) // hop_samples
        
        print(f"Processing {num_segments} segments...")
        for i in tqdm(range(0, audio.shape[-1], hop_samples)):
            segment = audio[:, i:i + segment_samples]
            
            # Pad if last segment is shorter
            if segment.shape[-1] < segment_samples:
                pad_length = segment_samples - segment.shape[-1]
                segment = torch.nn.functional.pad(segment, (0, pad_length))
            
            segment_batch = segment.unsqueeze(0).to(device)
            output_dict = model(segment_batch, stage="stage3")
            enhanced_segment = output_dict["target"].squeeze(0).cpu()
            segments.append(enhanced_segment)
        
        # Overlap-add reconstruction
        upsample_factor = model.target_sample_rate // model.base_sample_rate
        output_hop = hop_samples * upsample_factor
        output_length = audio.shape[-1] * upsample_factor
        
        enhanced = torch.zeros(2, output_length)
        weight = torch.zeros(2, output_length)
        
        for i, segment in enumerate(segments):
            start = i * output_hop
            end = start + segment.shape[-1]
            if end > output_length:
                segment = segment[:, :output_length - start]
                end = output_length
            
            # Hann window for smooth blending
            window = torch.hann_window(segment.shape[-1])
            windowed = segment * window.unsqueeze(0)
            
            enhanced[:, start:end] += windowed
            weight[:, start:end] += window.unsqueeze(0)
        
        # Normalize
        enhanced = enhanced / (weight + 1e-8)
    
    return enhanced.cpu()


def main():
    parser = argparse.ArgumentParser(description="Run inference with Finally GAN expert model")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source type (vocals, bass, drums, guitars, etc.)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="restoration/sonicmaster_finally_gan/checkpoints/finally_gan_experts",
        help="Base directory for expert checkpoints",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input audio file path (if not using dataset)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to restoration dataset (if using dataset)",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Index of sample to process from dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./enhanced_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Process audio in segments (useful for long files)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.source}_perceptual_24k_binorm"
    model = load_expert_model(checkpoint_path, device)
    
    # Get input file
    if args.input:
        input_path = args.input
    elif args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}")
        ds = load_from_disk(args.dataset_path)
        
        # Filter by source if needed
        if "label" in ds[0]:
            ds_filtered = ds.filter(lambda x: x["label"] == args.source)
            print(f"Filtered to {len(ds_filtered)} {args.source} samples")
            if len(ds_filtered) == 0:
                print(f"No samples found for source '{args.source}'")
                return
            sample = ds_filtered[args.dataset_index]
        else:
            sample = ds[args.dataset_index]
        
        input_path = sample["separated_stem"]
        print(f"Processing sample {args.dataset_index}: {input_path}")
    else:
        raise ValueError("Must provide either --input or --dataset-path")
    
    # Run inference
    print("Running inference...")
    enhanced = run_inference(
        model,
        input_path,
        device=device,
        segment_seconds=args.segment_seconds,
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving enhanced audio to {output_path}")
    torchaudio.save(
        output_path,
        enhanced,
        model.target_sample_rate,
    )
    
    print(f"Done! Enhanced audio saved at {model.target_sample_rate}Hz")
    print(f"Output shape: {enhanced.shape}")


if __name__ == "__main__":
    main()
