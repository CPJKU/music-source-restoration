from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class ModelConfig:
    base_sample_rate: int = 24_000
    target_sample_rate: int = 48_000
    stft_n_fft: int = 2048
    stft_hop: int = 512
    spectral_channels: int = 256
    upsample_scales: Sequence[int] = (8, 8, 8)
    embedding_backbone: str = "mert95m"
    pretrained_embedding_dim: int | None = None  # For transfer learning: original embedding dimension


@dataclass
class StageConfig:
    name: str
    max_steps: int
    batch_size: int
    lr: float
    lambda_lmos: float
    lambda_gan: float
    lambda_fm: float
    lambda_music_perc: float  # Renamed from lambda_hf
    lambda_audio_perc: float = 0.0  # Audio perceptual loss (ERBSoftDTW)
    use_adversarial: bool = False
    use_feature_matching: bool = False  # Use feature matching loss without adversarial training
    use_perceptual: bool = False
    use_upsample_head: bool = False
    ema: float = 0.999
    gradient_clip: float = 1.0
    warmup_steps: int = 2000
    accumulate_steps: int = 1  # Gradient accumulation steps
    gan_objective: str = "ls"  # Options: "ls", "hinge"
    adv_ramp_steps: int = 0
    disc_input_noise_std: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    """Training configuration optimized for L40 GPU (48GB VRAM)."""
    stages: List[StageConfig] = field(
        default_factory=lambda: [
            StageConfig(
                name="stage1",
                max_steps=40_000,
                batch_size=12,  # Increased from 8 for L40
                lr=2e-4,
                lambda_lmos=1.0,
                lambda_gan=0.0,
                lambda_fm=0.0,
                lambda_music_perc=0.0,
                use_adversarial=False,
                use_perceptual=False,
                use_upsample_head=False,
                accumulate_steps=2,  # Effective batch size: 24
            ),
            StageConfig(
                name="stage2",
                max_steps=60_000,
                batch_size=8,  # Increased from 6 for L40
                lr=2e-4,
                lambda_lmos=20.0,
                lambda_gan=0.4,
                lambda_fm=20.0,
                lambda_music_perc=0.0,
                use_adversarial=True,
                use_perceptual=False,
                use_upsample_head=False,
                accumulate_steps=2,  # Effective batch size: 16
            ),
            StageConfig(
                name="stage3",
                max_steps=80_000,
                batch_size=6,  # Increased from 4 for L40
                lr=1.5e-4,
                lambda_lmos=0.5,
                lambda_gan=5.0,
                lambda_fm=15.0,
                lambda_music_perc=5.0,  # Music-specific perceptual loss
                use_adversarial=True,
                use_perceptual=True,
                use_upsample_head=True,
                accumulate_steps=2,  # Effective batch size: 12
            ),
        ]
    )
    betas: tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 1e-2
    mixed_precision: str = "bf16"
    save_every: int = 5_000
    log_every: int = 100
    eval_every: int = 2_000
    checkpoint_dir: str = "checkpoints/finally_gan"
    
    # W&B logging
    use_wandb: bool = False
    wandb_project: str = "finally-gan-music"
    wandb_entity: str | None = "something_with_audio"
    wandb_group: str | None = "finally_gan"
    wandb_job_type: str | None = None
    wandb_run_name: str | None = "finally_gan_experiment"
    wandb_dir: str | None = None
    
    # Validation settings
    val_audio_dir: str = "validation_audio"
    val_save_samples: int = 4  # Number of audio samples to save per validation
    
    # Top-k checkpoint saving
    keep_top_k_checkpoints: int = 2  # Keep only best K checkpoints by validation loss
    
    # torch.compile optimization (PyTorch 2.0+)
    use_torch_compile: bool = True
    torch_compile_mode: str = "default"  # Options: "default", "reduce-overhead", "max-autotune"
    
    # Transfer learning settings
    freeze_hifi_for_embedding_switch: bool = False  # Freeze HiFi-GAN layers when switching embeddings
    projection_lr_multiplier: float = 10.0  # Learning rate multiplier for embedding projection layer


@dataclass
class DataConfig:
    segment_seconds: float = 10.0
    cache_dir: str | None = None
    saved_dir: str | None = "/opt/scratch/HF_datasets/saved/SonicMasterDataset"
    apply_online_degradation: bool = False  # Dataset already contains degraded samples
    num_workers: int = 6
    val_split: str | None = "validation"
    val_batches: int = 4
    
    # Expert training parameters
    use_restoration_dataset: bool = False
    restoration_dataset_path: str = "/opt/datasets/HF_datasets/saved/restoration_dataset/v0.1"
    source_filter: str | None = None  # e.g., 'vocals', 'bass', 'drums', etc.
    latents_path: str | None = None  # Path to pre-computed CoDiCodec latents (e.g., v0.1_latents/)
    
    # Audio preprocessing
    normalize_degraded: bool = False  # Normalize degraded audio to [-1, 1] before model input
    normalize_target: bool = False  # Normalize target audio to [-1, 1] before computing losses


__all__ = ["ModelConfig", "TrainConfig", "DataConfig", "StageConfig"]
