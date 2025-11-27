#!/usr/bin/env python3
"""Expert training script for FinallyGAN.

Usage:
    # Stage 1: Generalist warmup on SonicMaster
    python train_expert.py --config configs/experts/stage1_generalist_warmup.json
    
    # Stage 2: Expert fine-tuning per source (20k steps)
    python train_expert.py --expert vocals --stage finetune
    python train_expert.py --expert bass --stage finetune
    # ... for all sources
    
    # Stage 3: Expert feature matching (perceptual + STFT feature matching, 20k steps)
    # Uses frozen discriminator for feature matching loss without adversarial training
    python train_expert.py --expert vocals --stage feature_matching
    python train_expert.py --expert bass --stage feature_matching
    # ... for all sources
    
    # Stage 4 (optional): Expert adversarial training per source (20k steps)
    python train_expert.py --expert vocals --stage adversarial
    python train_expert.py --expert bass --stage adversarial
    # ... for all sources
    
    # Stage 5 (optional): Expert audio perceptual training per source (20k steps)
    python train_expert.py --expert vocals --stage audio_perceptual
    python train_expert.py --expert bass --stage audio_perceptual
    # ... for all sources
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from restoration.sonicmaster_finally_gan.configs import DataConfig, ModelConfig, StageConfig, TrainConfig
from restoration.sonicmaster_finally_gan.trainer import FinallyGanTrainer


SOURCES = [
    "vocals",
    "bass",
    "drums",
    "guitars",
    "keyboards",
    "synthesizers",
    "orchestral",
    "percussions",
]


def load_config(config_path: str) -> tuple[ModelConfig, TrainConfig, DataConfig]:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        cfg = json.load(f)
    
    model_cfg = ModelConfig(**cfg["model"])
    
    # Convert stage dicts to StageConfig objects
    stages = [StageConfig(**stage) for stage in cfg["train"]["stages"]]
    train_cfg_dict = cfg["train"].copy()
    train_cfg_dict["stages"] = stages
    train_cfg = TrainConfig(**train_cfg_dict)
    
    data_cfg = DataConfig(**cfg["data"])
    
    return model_cfg, train_cfg, data_cfg


def get_expert_config_path(expert: str, stage: str) -> str:
    """Get config path for expert training."""
    if stage == "finetune":
        return f"configs/experts/expert_{expert}_finetune.json"
    elif stage == "adversarial":
        return f"configs/experts/expert_{expert}_adversarial.json"
    elif stage == "audio_perceptual":
        return f"configs/experts/expert_{expert}_audio_perceptual.json"
    elif stage == "feature_matching":
        return f"configs/experts/expert_{expert}_feature_matching.json"
    elif stage == "feature_matching_safe":
        return f"configs/experts/expert_{expert}_feature_matching_safe.json"
    elif stage == "perceptual":
        return f"configs/experts/expert_{expert}_perceptual.json"
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 'finetune', 'adversarial', 'audio_perceptual', 'feature_matching', 'feature_matching_safe', or 'perceptual'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FinallyGAN expert models")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (for generalist training)",
    )
    parser.add_argument(
        "--expert",
        type=str,
        choices=SOURCES,
        help="Expert source to train (vocals, bass, drums, etc.)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["finetune", "adversarial", "audio_perceptual", "feature_matching", "feature_matching_safe", "perceptual"],
        help="Training stage (finetune, adversarial, audio_perceptual, feature_matching, feature_matching_safe, or perceptual)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    elif args.expert and args.stage:
        config_path = get_expert_config_path(args.expert, args.stage)
    else:
        parser.error("Must provide either --config or both --expert and --stage")
    
    print(f"Loading configuration from {config_path}")
    model_cfg, train_cfg, data_cfg = load_config(config_path)
    
    # Override checkpoint path if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Note: Trainer handles checkpoint loading internally
    
    # Log configuration
    print("\n" + "="*80)
    print("Configuration:")
    print("="*80)
    print(f"Model: {model_cfg}")
    print(f"Training: {train_cfg}")
    print(f"Data: {data_cfg}")
    print("="*80 + "\n")
    
    # Create trainer and start training
    trainer = FinallyGanTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        resume_from=args.resume,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
