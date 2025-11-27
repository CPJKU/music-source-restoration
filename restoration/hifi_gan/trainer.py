from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
HAS_ACCELERATE = True


try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .configs import DataConfig, ModelConfig, StageConfig, TrainConfig
from .data import create_dataloader
from .discriminators import DiscriminatorOutput, FinallyDiscriminatorBundle
from .generator import FinallyGenerator
from .losses import (
    LMOSLoss,
    LeastSquaresGANLoss,
    MusicPerceptualLoss,
    feature_matching_loss,
    HingeGANLoss,
    ERBSoftDTWLoss,
)
from .modules import STFTConfig
from .evaluation_utils import save_validation_samples

Tensor = torch.Tensor


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Add new parameters to shadow (e.g., from transfer learning)
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            # Ensure shadow is on same device as param
            if self.shadow[name].device != param.device:
                self.shadow[name] = self.shadow[name].to(param.device)
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Only apply EMA if shadow exists (parameter may have been frozen during training)
            if name not in self.shadow:
                continue
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, Tensor]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


def flatten_outputs(outputs: Dict[str, List[DiscriminatorOutput]]) -> List[DiscriminatorOutput]:
    flat: List[DiscriminatorOutput] = []
    for key in sorted(outputs.keys()):
        flat.extend(outputs[key])
    return flat


class FinallyGanTrainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
        device: torch.device | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.resume_from = Path(resume_from) if resume_from else None
        self.data_cfg = data_cfg
        
        # Initialize Accelerator for distributed training and automatic mixed precision
        if HAS_ACCELERATE:
            self.accelerator = Accelerator(  # type: ignore
                mixed_precision=train_cfg.mixed_precision,
                gradient_accumulation_steps=1,  # We handle this manually per stage
                log_with="wandb" if train_cfg.use_wandb and HAS_WANDB else None,
            )
            self.device = self.accelerator.device
            print(f"[accelerator] Initialized with {self.accelerator.num_processes} process(es)")
            print(f"[accelerator] Mixed precision: {self.accelerator.mixed_precision}")
        else:
            self.accelerator = None
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[accelerator] Accelerate not available, using manual device management")

        stft_cfg = STFTConfig(n_fft=model_cfg.stft_n_fft, hop_length=model_cfg.stft_hop)
        self.generator = FinallyGenerator(
            base_sample_rate=model_cfg.base_sample_rate,
            target_sample_rate=model_cfg.target_sample_rate,
            stft_cfg=stft_cfg,
            spectral_channels=model_cfg.spectral_channels,
            upsample_scales=tuple(model_cfg.upsample_scales),
            embedding_backbone=model_cfg.embedding_backbone,
            pretrained_embedding_dim=model_cfg.pretrained_embedding_dim,
        )        
        self.discriminator = FinallyDiscriminatorBundle(
            base_sample_rate=model_cfg.base_sample_rate,
            target_sample_rate=model_cfg.target_sample_rate,
        )

        self.lmos_loss = LMOSLoss(self.generator.embedding_extractor)
        self.gan_losses = {
            "ls": LeastSquaresGANLoss(),
            "hinge": HingeGANLoss(),
        }
        self.default_gan_objective = "ls"
        self.music_perc_loss = MusicPerceptualLoss(sample_rate=model_cfg.target_sample_rate)
        self.audio_perc_loss = ERBSoftDTWLoss(sample_rate=model_cfg.target_sample_rate, alpha=1e-5)
        
        # Move to device if not using Accelerator (Accelerator will handle this in prepare())
        if not HAS_ACCELERATE:
            self.generator = self.generator.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
            self.lmos_loss = self.lmos_loss.to(self.device)
            self.music_perc_loss = self.music_perc_loss.to(self.device)
            self.audio_perc_loss = self.audio_perc_loss.to(self.device)
        
        # Apply torch.compile for optimization (PyTorch 2.0+)
        if train_cfg.use_torch_compile and hasattr(torch, 'compile'):
            print("[trainer] Compiling models with torch.compile...")
            self.generator = torch.compile(self.generator, mode=train_cfg.torch_compile_mode)
            self.discriminator = torch.compile(self.discriminator, mode=train_cfg.torch_compile_mode)
            print(f"[trainer] Models compiled with mode='{train_cfg.torch_compile_mode}'")
        elif train_cfg.use_torch_compile:
            print("[trainer] torch.compile requested but not available (requires PyTorch 2.0+)")

        self.ema = EMA(self.generator, decay=train_cfg.stages[-1].ema if train_cfg.stages else 0.999)

        self.checkpoint_dir = Path(train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B setup
        self.wandb_run = None
        if train_cfg.use_wandb and HAS_WANDB and not HAS_ACCELERATE:
            # Only initialize wandb manually if not using Accelerator
            self._setup_wandb()
        
        # Track best checkpoints
        self.best_checkpoints: List[Dict[str, object]] = []
    
    def _load_checkpoint_after_prepare(
        self,
        checkpoint_path: str | Path,
        optim_g: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler._LRScheduler,
        optim_d: torch.optim.Optimizer | None = None,
        scheduler_d: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """Load model, optimizer, and scheduler states from checkpoint after accelerator.prepare().
        
        This must be called AFTER accelerator.prepare() to ensure the wrapped models
        are loaded correctly.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file
        optim_g : torch.optim.Optimizer
            Generator optimizer (already prepared)
        scheduler_g : torch.optim.lr_scheduler._LRScheduler
            Generator scheduler (already prepared)
        optim_d : torch.optim.Optimizer, optional
            Discriminator optimizer (already prepared), if using adversarial training
        scheduler_d : torch.optim.lr_scheduler._LRScheduler, optional
            Discriminator scheduler (already prepared), if using adversarial training
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[trainer] Loading checkpoint from {checkpoint_path}")
        
        # Load on CPU first to avoid OOM, then let Accelerate handle device placement
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Unwrap models if using Accelerator (they're wrapped by prepare())
        generator_to_load = self.accelerator.unwrap_model(self.generator) if HAS_ACCELERATE and self.accelerator else self.generator
        discriminator_to_load = self.accelerator.unwrap_model(self.discriminator) if HAS_ACCELERATE and self.accelerator else self.discriminator
        
        # Load model states
        # For transfer learning, skip HiFi-GAN input projection layer (dimension mismatch expected)
        checkpoint_state = checkpoint["generator"].copy()
        
        # Check if we're using transfer learning (projection layer exists)
        has_projection = hasattr(generator_to_load, 'embedding_proj') and generator_to_load.embedding_proj is not None
        
        # Check for dimension mismatch in hifi.proj layer
        skip_hifi_proj = False
        if 'hifi.proj.weight' in checkpoint_state:
            checkpoint_shape = checkpoint_state['hifi.proj.weight'].shape
            current_shape = None
            for name, param in generator_to_load.named_parameters():
                if name == 'hifi.proj.weight':
                    current_shape = param.shape
                    break
            
            if current_shape is not None and checkpoint_shape != current_shape:
                skip_hifi_proj = True
                print(f"[transfer_learning] HiFi-GAN input dimension mismatch detected:")
                print(f"  Checkpoint: {checkpoint_shape} (expecting {checkpoint_shape[1] - 256}-dim embeddings)")
                print(f"  Current model: {current_shape} (expecting {current_shape[1] - 256}-dim embeddings)")


        if skip_hifi_proj:
            # Skip loading hifi.proj weights (dimension mismatch expected due to different embedding dims)
            keys_to_skip = [k for k in checkpoint_state.keys() if 'hifi.proj' in k]
            for key in keys_to_skip:
                print(f"[transfer_learning] Skipping checkpoint key: {key}")
                del checkpoint_state[key]
            
            if has_projection:
                print(f"[transfer_learning] Using projection layer - HiFi-GAN input layer will be retrained")
            else:
                print(f"[transfer_learning] HiFi-GAN input layer will be randomly initialized due to dimension change")
        
        # Use strict=False to allow for architectural changes (e.g., Parameter -> Buffer, missing projection layer)
        missing_keys, unexpected_keys = generator_to_load.load_state_dict(checkpoint_state, strict=False)
        
        # Filter out expected missing keys for transfer learning
        if has_projection or skip_hifi_proj:
            expected_missing = ['embedding_proj.weight', 'embedding_proj.bias', 'hifi.proj.weight', 'hifi.proj.bias']
            missing_keys = [k for k in missing_keys if not any(expected in k for expected in expected_missing)]
        
        if missing_keys:
            print(f"[trainer] Generator missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[trainer] Generator unexpected keys: {unexpected_keys}")
        print("[trainer] Loaded generator state")
        
        # Only load discriminator if it was actually trained in previous stage
        # If stage1 didn't use adversarial training, discriminator is just random init
        prev_stage_config = checkpoint.get("stage", {})
        prev_stage_name = prev_stage_config.get("name", "") if isinstance(prev_stage_config, dict) else ""
        prev_used_adv = prev_stage_config.get("use_adversarial", False) if isinstance(prev_stage_config, dict) else False
        
        # CRITICAL: Don't load discriminator when transitioning from finetune to adversarial
        # The finetune stage may have saved discriminator weights, but they're untrained/random
        # Loading them can cause numerical instability. Always start fresh for adversarial.
        is_finetune_to_adversarial = "finetune" in prev_stage_name and any("adversarial" in s.name for s in self.train_cfg.stages)
        
        if prev_used_adv and not is_finetune_to_adversarial:
            missing_keys, unexpected_keys = discriminator_to_load.load_state_dict(checkpoint["discriminator"], strict=False)
            if missing_keys:
                print(f"[trainer] Discriminator missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[trainer] Discriminator unexpected keys: {unexpected_keys}")
            print("[trainer] Loaded discriminator state from adversarial-trained checkpoint")
        else:
            print("[trainer] Discriminator will use fresh random initialization")
            if is_finetune_to_adversarial:
                print("[trainer] Reason: Transitioning from finetune to adversarial stage (disc not trained)")
            else:
                print("[trainer] Reason: Previous stage didn't use adversarial training")
        
        # Load EMA state
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
            print("[trainer] Loaded EMA state")
        
        print(f"[trainer] Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        print(f"[trainer] Resuming from stage: {checkpoint.get('stage', 'unknown')}")
        print("[trainer] NOTE: Optimizer and scheduler states NOT loaded - using fresh optimizers for new stage")
        
        # Verify loaded weights don't have NaN
        self._check_model_for_nan("after checkpoint load")
    
    def _check_model_for_nan(self, context: str = "") -> bool:
        """Check if model weights contain NaN values.
        
        Parameters
        ----------
        context : str
            Context string for logging (e.g., "after checkpoint load")
            
        Returns
        -------
        bool
            True if NaN found, False otherwise
        """
        has_nan = False
        
        # Check generator
        gen_model = self.accelerator.unwrap_model(self.generator) if HAS_ACCELERATE and self.accelerator else self.generator
        for name, param in gen_model.named_parameters():
            if torch.isnan(param).any():
                print(f"[NaN CHECK] {context}: Generator parameter '{name}' contains NaN!")
                has_nan = True
                break
        
        # Check discriminator
        disc_model = self.accelerator.unwrap_model(self.discriminator) if HAS_ACCELERATE and self.accelerator else self.discriminator
        for name, param in disc_model.named_parameters():
            if torch.isnan(param).any():
                print(f"[NaN CHECK] {context}: Discriminator parameter '{name}' contains NaN!")
                has_nan = True
                break
        
        if not has_nan:
            print(f"[NaN CHECK] {context}: ✓ No NaN values detected in model parameters")
        
        return has_nan

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not HAS_WANDB:
            print("[wandb] wandb not available; disabling logging")
            self.train_cfg.use_wandb = False
            return
        
        init_kwargs = {
            "project": self.train_cfg.wandb_project,
            "config": {
                "model": asdict(self.model_cfg),
                "train": asdict(self.train_cfg),
                "data": asdict(self.data_cfg),
            },
        }
        
        if self.train_cfg.wandb_entity:
            init_kwargs["entity"] = self.train_cfg.wandb_entity
        if self.train_cfg.wandb_group:
            init_kwargs["group"] = self.train_cfg.wandb_group
        if self.train_cfg.wandb_run_name:
            init_kwargs["name"] = self.train_cfg.wandb_run_name
        if self.train_cfg.wandb_dir:
            init_kwargs["dir"] = self.train_cfg.wandb_dir
        
        # Set job_type to source for expert training
        if self.data_cfg.source_filter:
            init_kwargs["job_type"] = self.data_cfg.source_filter
        
        self.wandb_run = wandb.init(**init_kwargs)
        print(f"[wandb] Initialized run: {self.wandb_run.name}")

    def _prepare_batch(self, batch: dict[str, Tensor | None]) -> dict[str, Tensor | None]:
        prepared: dict[str, Tensor | None] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
    
    def _normalize_audio(self, audio: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        """Normalize audio to [-1, 1] range and return normalization factor for denormalization.
        
        Parameters
        ----------
        audio : Tensor
            Input audio tensor
        eps : float
            Small epsilon for numerical stability
            
        Returns
        -------
        tuple[Tensor, Tensor]
            (normalized_audio, scale_factor)
        """
        max_val = audio.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=eps)  # Avoid division by zero
        normalized = audio / max_val
        return normalized, max_val

    def _build_optimizer(self, params: Iterable[nn.Parameter], stage: StageConfig) -> torch.optim.Optimizer:
        """Build optimizer with support for parameter groups and different learning rates.
        
        If transfer learning is enabled (pretrained_embedding_dim is set), creates
        parameter groups with different learning rates for the embedding projection layer.
        """
        # Check if transfer learning is active
        has_projection = hasattr(self.generator, 'embedding_proj') and self.generator.embedding_proj is not None
        
        if has_projection and self.train_cfg.projection_lr_multiplier != 1.0:
            # Create parameter groups with different learning rates
            projection_params = []
            other_params = []
            
            for name, param in self.generator.named_parameters():
                if not param.requires_grad:
                    continue
                if 'embedding_proj' in name:
                    projection_params.append(param)
                else:
                    other_params.append(param)
            
            param_groups = [
                {'params': other_params, 'lr': stage.lr},
                {'params': projection_params, 'lr': stage.lr * self.train_cfg.projection_lr_multiplier},
            ]
            
            print(f"[optimizer] Using parameter groups:")
            print(f"  - Standard params: lr={stage.lr:.2e}")
            print(f"  - Projection layer: lr={stage.lr * self.train_cfg.projection_lr_multiplier:.2e}")
            
            return torch.optim.AdamW(
                param_groups,
                betas=self.train_cfg.betas,
                weight_decay=self.train_cfg.weight_decay,
            )
        else:
            # Standard single-group optimizer
            return torch.optim.AdamW(
                params,
                lr=stage.lr,
                betas=self.train_cfg.betas,
                weight_decay=self.train_cfg.weight_decay,
            )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, stage: StageConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """Build LR scheduler with optional warmup.
        
        If warmup_steps > 0, uses linear warmup followed by cosine annealing.
        Otherwise, uses cosine annealing from the start.
        """
        if stage.warmup_steps > 0:
            # Linear warmup followed by cosine annealing
            def lr_lambda(current_step: int) -> float:
                if current_step < stage.warmup_steps:
                    # Linear warmup from 0 to 1
                    return float(current_step) / float(max(1, stage.warmup_steps))
                else:
                    # Cosine annealing from 1 to eta_min/lr
                    progress = float(current_step - stage.warmup_steps) / float(
                        max(1, stage.max_steps - stage.warmup_steps)
                    )
                    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # No warmup, just cosine annealing
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage.max_steps,
                eta_min=stage.lr * 0.001,
            )

    def _save_checkpoint(
        self,
        stage: StageConfig,
        step: int,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer | None,
        scheduler_g: torch.optim.lr_scheduler._LRScheduler,
        scheduler_d: torch.optim.lr_scheduler._LRScheduler | None,
        val_loss: float | None = None,
    ) -> None:
        """Save checkpoint and manage top-k best checkpoints."""
        if val_loss is not None:
            ckpt_path = self.checkpoint_dir / f"{stage.name}_step{step:07d}_val{val_loss:.6f}.pt"
        else:
            ckpt_path = self.checkpoint_dir / f"{stage.name}_step{step:07d}.pt"
        
        payload = {
            "stage": asdict(stage),
            "step": step,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer_g": optim_g.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
        }
        if optim_d is not None:
            payload["optimizer_d"] = optim_d.state_dict()
        if scheduler_d is not None:
            payload["scheduler_d"] = scheduler_d.state_dict()
        if val_loss is not None:
            payload["val_loss"] = val_loss

        torch.save(payload, ckpt_path)
        
        # Manage top-k checkpoints if validation loss provided
        if val_loss is not None:
            self.best_checkpoints.append({"path": ckpt_path, "val_loss": val_loss})
            self.best_checkpoints.sort(key=lambda item: item["val_loss"])
            
            # Remove checkpoints beyond top-k
            while len(self.best_checkpoints) > self.train_cfg.keep_top_k_checkpoints:
                removed = self.best_checkpoints.pop(-1)
                try:
                    Path(removed["path"]).unlink()
                    print(f"[checkpoint] Removed: {Path(removed['path']).name} (val_loss={removed['val_loss']:.6f})")
                except FileNotFoundError:
                    pass

    def train(self) -> None:
        for stage in self.train_cfg.stages:
            print(f"[trainer] Starting stage {stage.name}")
            train_loader = create_dataloader(
                split="train",
                batch_size=stage.batch_size,
                segment_seconds=self.data_cfg.segment_seconds,
                base_sample_rate=self.model_cfg.base_sample_rate,
                target_sample_rate=self.model_cfg.target_sample_rate,
                cache_dir=self.data_cfg.cache_dir,
                saved_dir=self.data_cfg.saved_dir,
                num_workers=self.data_cfg.num_workers,
                shuffle=True,
                apply_online_degradation=self.data_cfg.apply_online_degradation,
                use_restoration_dataset=self.data_cfg.use_restoration_dataset,
                restoration_dataset_path=self.data_cfg.restoration_dataset_path,
                source_filter=self.data_cfg.source_filter,
                latents_path=self.data_cfg.latents_path,
            )

            val_loader: DataLoader[dict[str, Tensor]] | None = None
            if self.data_cfg.val_split and not self.data_cfg.use_restoration_dataset:
                # Skip validation for restoration dataset (no splits)
                val_loader = create_dataloader(
                    split=self.data_cfg.val_split,
                    batch_size=stage.batch_size,
                    segment_seconds=self.data_cfg.segment_seconds,
                    base_sample_rate=self.model_cfg.base_sample_rate,
                    target_sample_rate=self.model_cfg.target_sample_rate,
                    cache_dir=self.data_cfg.cache_dir,
                    saved_dir=self.data_cfg.saved_dir,
                    num_workers=self.data_cfg.num_workers,
                    shuffle=False,
                    apply_online_degradation=False,
                )
            
            # Apply transfer learning freezing if enabled
            if self.train_cfg.freeze_hifi_for_embedding_switch:
                frozen_count = 0
                trainable_count = 0
                for name, param in self.generator.named_parameters():
                    if "hifi" in name and "hifi.proj" not in name and "embedding_proj" not in name:
                        param.requires_grad = False
                        frozen_count += 1
                    elif param.requires_grad:
                        trainable_count += 1
                
                print(f"[transfer_learning] Frozen {frozen_count} HiFi-GAN parameters")
                print(f"[transfer_learning] Training {trainable_count} parameters")
                print(f"[transfer_learning] Trainable components: embedding extractor projection, spectral UNet, wave UNet, upsample head")
            else:
                # Explicitly unfreeze all parameters for this stage
                unfrozen_count = 0
                for name, param in self.generator.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1
                
                if unfrozen_count > 0:
                    print(f"[transfer_learning] Unfrozen {unfrozen_count} parameters for this stage")


            optim_g = self._build_optimizer(self.generator.parameters(), stage)
            scheduler_g = self._build_scheduler(optim_g, stage)
            optim_d: torch.optim.Optimizer | None = None
            scheduler_d: torch.optim.lr_scheduler._LRScheduler | None = None
            if stage.use_adversarial:
                # Use 2x HIGHER LR for discriminator to help it learn quickly
                # Discriminator needs to learn real vs fake fast to provide useful gradients
                # Generator is pre-trained from finetune, so disc needs to catch up
                disc_lr = stage.lr * 0.5
                optim_d = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=disc_lr,
                    betas=self.train_cfg.betas,
                    weight_decay=self.train_cfg.weight_decay,
                )
                print(f"[trainer] Discriminator LR: {disc_lr} (2x generator LR: {stage.lr})")
                scheduler_d = self._build_scheduler(optim_d, stage)
            
            # Prepare models, optimizers, and dataloaders with Accelerator
            if HAS_ACCELERATE and self.accelerator is not None:
                self.generator, self.discriminator, optim_g, scheduler_g, train_loader = self.accelerator.prepare(
                    self.generator, self.discriminator, optim_g, scheduler_g, train_loader
                )
                if optim_d is not None and scheduler_d is not None:
                    optim_d, scheduler_d = self.accelerator.prepare(optim_d, scheduler_d)
                if val_loader is not None:
                    val_loader = self.accelerator.prepare(val_loader)
                
                # Load checkpoint after prepare if specified (only on first stage)
                if self.resume_from and stage == self.train_cfg.stages[0]:
                    self._load_checkpoint_after_prepare(
                        checkpoint_path=self.resume_from,
                        optim_g=optim_g,
                        scheduler_g=scheduler_g,
                        optim_d=optim_d,
                        scheduler_d=scheduler_d,
                    )
                
                # Initialize W&B tracking if enabled
                if self.train_cfg.use_wandb and HAS_WANDB:
                    # Build wandb init kwargs with job_type for expert training
                    wandb_kwargs = {}
                    if self.train_cfg.wandb_entity:
                        wandb_kwargs["entity"] = self.train_cfg.wandb_entity
                    if self.train_cfg.wandb_group:
                        wandb_kwargs["group"] = self.train_cfg.wandb_group
                    if self.train_cfg.wandb_run_name:
                        wandb_kwargs["name"] = self.train_cfg.wandb_run_name
                    if self.data_cfg.source_filter:
                        wandb_kwargs["job_type"] = self.data_cfg.source_filter
                    
                    self.accelerator.init_trackers(
                        project_name=self.train_cfg.wandb_project,
                        config={
                            "model": asdict(self.model_cfg),
                            "train": asdict(self.train_cfg),
                            "data": asdict(self.data_cfg),
                        },
                        init_kwargs={"wandb": wandb_kwargs} if wandb_kwargs else {}
                    )

                    # When using Accelerate, trackers initialize wandb globally (wandb.run).
                    # Mirror that into self.wandb_run on the main process so manual
                    # `wandb.log` guards (which check self.wandb_run) work correctly.
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main:
                        try:
                            # wandb.run is set by the tracker; assign to trainer attribute
                            self.wandb_run = wandb.run
                            if self.wandb_run is not None:
                                print(f"[wandb] Initialized run (via Accelerate): {self.wandb_run.name}")
                        except Exception:
                            # Non-fatal: continue without setting self.wandb_run
                            pass

            self.generator.train()
            self.discriminator.train()
            
            gan_objective = (stage.gan_objective or self.default_gan_objective).lower()
            gan_loss_fn = self.gan_losses.get(gan_objective, self.gan_losses[self.default_gan_objective])
            label_smoothing = max(0.0, min(stage.label_smoothing, 0.49))
            disc_noise_std = max(0.0, stage.disc_input_noise_std)
            total_adv_steps = max(stage.adv_ramp_steps, 0)
            disc_pretrain_steps = min(500, total_adv_steps) if stage.use_adversarial else 0
            ramp_start = disc_pretrain_steps
            ramp_end = max(total_adv_steps, ramp_start)
            if stage.use_adversarial and ramp_end == ramp_start:
                ramp_end = ramp_start + 1
            real_target = 1.0 - label_smoothing
            fake_target = label_smoothing
            
            def _maybe_add_disc_noise(tensor: Tensor | None) -> Tensor | None:
                if tensor is None or disc_noise_std <= 0:
                    return tensor
                noise = torch.randn_like(tensor)
                return tensor + noise * disc_noise_std
            
            # Check for NaN before starting training
            self._check_model_for_nan(f"before {stage.name} training starts")

            iterator = iter(train_loader)
            running_losses = defaultdict(float)
                                  

            for step in tqdm(range(1, stage.max_steps + 1), desc=f"{stage.name}"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch = self._prepare_batch(batch)

                degraded_base = batch["degraded_base"]
                clean_base = batch["clean_base"]
                clean_target = batch["clean_target"]
                embedding_batch = batch.get("embedding")
                
                # Normalize degraded audio if enabled
                degraded_scale = None
                if self.data_cfg.normalize_degraded:
                    degraded_base, degraded_scale = self._normalize_audio(degraded_base)
                if clean_target is not None and self.data_cfg.normalize_target:
                    clean_target, _ = self._normalize_audio(clean_target)
                if clean_base is not None and self.data_cfg.normalize_target:
                    clean_base, _ = self._normalize_audio(clean_base)

                # Gradient accumulation logic                 
                accumulate = stage.accumulate_steps
                start_cycle = (accumulate == 1) or ((step - 1) % accumulate == 0)
                
                # Zero gradients at the start of accumulation cycle
                if start_cycle:
                    optim_g.zero_grad(set_to_none=True)
                    if stage.use_adversarial and optim_d is not None:
                        optim_d.zero_grad(set_to_none=True)
                elif stage.use_adversarial and optim_d is not None and step <= disc_pretrain_steps:
                    # When discriminator pretrains every step, clear grads each iteration.
                    optim_d.zero_grad(set_to_none=True)

                # DEBUG: Log input ranges for first few steps
                if step <= 10:
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main:
                        print(f"\n[DEBUG STEP {step}] Input ranges:")
                        print(f"  degraded_base: [{degraded_base.min():.4f}, {degraded_base.max():.4f}], mean={degraded_base.mean():.4f}")
                        print(f"  clean_base: [{clean_base.min():.4f}, {clean_base.max():.4f}], mean={clean_base.mean():.4f}")
                        if torch.isnan(degraded_base).any() or torch.isnan(clean_base).any():
                            print(f"  ❌ NaN detected in input data!")

                # Accelerator handles mixed precision automatically
                outputs = self.generator(
                    degraded_base,
                    stage=stage.name,
                    precomputed_embeddings=embedding_batch if torch.is_tensor(embedding_batch) else None,
                )
                base_out = outputs["base"]
                target_out = outputs.get("target")
                
                # DEBUG: Log output ranges for first few steps
                if step <= 10:
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main:
                        print(f"  base_out: [{base_out.min():.4f}, {base_out.max():.4f}], mean={base_out.mean():.4f}")
                        if torch.isnan(base_out).any():
                            print(f"  ❌ NaN detected in generator output!")
                            self._check_model_for_nan(f"step {step} - NaN in generator output")

                if target_out is None:
                    lmos_ref = clean_base
                    lmos_estimate = base_out
                    lmos_sr = self.model_cfg.base_sample_rate
                else:
                    lmos_ref = clean_target
                    lmos_estimate = target_out
                    lmos_sr = self.model_cfg.target_sample_rate

                loss_lmos = self.lmos_loss(lmos_ref, lmos_estimate, lmos_sr)
                loss_adv = torch.tensor(0.0, device=self.device)
                loss_fm = torch.tensor(0.0, device=self.device)
                loss_music_perc = torch.tensor(0.0, device=self.device)
                loss_audio_perc = torch.tensor(0.0, device=self.device)
                
                # Determine if we need discriminator forward pass (for adversarial or feature matching)
                need_discriminator = stage.use_adversarial or stage.use_feature_matching
                
                # Generator adversarial contribution ramps after discriminator pre-training
                if stage.use_adversarial:
                    if step < disc_pretrain_steps:
                        gen_adv_scale = 0.0
                    elif step < ramp_end:
                        ramp_span = max(1, ramp_end - ramp_start)
                        gen_adv_scale = float(step - ramp_start) / float(ramp_span)
                    else:
                        gen_adv_scale = 1.0
                else:
                    gen_adv_scale = 0.0

                # Compute discriminator features for adversarial or feature matching
                if need_discriminator:
                    
                    # CRITICAL: Disable discriminator gradients during generator loss computation
                    # For feature matching only: we only need features, discriminator stays frozen
                    # For adversarial: we compute discriminator loss separately below
                    for param in self.discriminator.parameters():
                        param.requires_grad = False
                    
                    # Pass generator outputs WITHOUT detach for generator update (need gradients)
                    base_for_disc = _maybe_add_disc_noise(base_out)
                    target_for_disc = _maybe_add_disc_noise(target_out)
                    real_base_for_disc = _maybe_add_disc_noise(clean_base)
                    real_target_for_disc = _maybe_add_disc_noise(clean_target if target_out is not None else None)
                    disc_fake = self.discriminator(base_for_disc, target_for_disc)
                    with torch.no_grad():
                        disc_real = self.discriminator(real_base_for_disc, real_target_for_disc)
                    
                    # Clip discriminator scores to prevent extreme values that cause NaN gradients
                    fake_scores = [torch.clamp(out.score, min=-10.0, max=10.0) for out in flatten_outputs(disc_fake)]
                    real_scores = [torch.clamp(out.score, min=-10.0, max=10.0) for out in flatten_outputs(disc_real)]
                    
                    # DEBUG: Check disc scores for NaN/Inf
                    if step <= 15:
                        is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                        if is_main:
                            fake_scores_stats = [f"mean={s.mean().item():.4f}, range=[{s.min().item():.4f}, {s.max().item():.4f}]" for s in fake_scores]
                            print(f"  fake_scores: {fake_scores_stats}")
                            for i, s in enumerate(fake_scores):
                                if torch.isnan(s).any() or torch.isinf(s).any():
                                    print(f"  ❌ NaN/Inf in fake_scores[{i}]!")
                    
                    # Compute adversarial loss only if use_adversarial is True
                    if stage.use_adversarial:
                        loss_adv = gan_loss_fn.generator(fake_scores, real_target=real_target) * gen_adv_scale
                    
                    # Compute feature matching loss if enabled (can be used with or without adversarial)
                    # Feature matching: match intermediate discriminator features between real and fake
                    # This provides perceptual guidance without requiring discriminator training
                    if stage.use_feature_matching or stage.use_adversarial:
                        fm_scale = gen_adv_scale if stage.use_adversarial else 1.0
                        loss_fm = feature_matching_loss(
                            flatten_outputs(disc_real),
                            flatten_outputs(disc_fake),
                        ) * fm_scale
                    
                    # DEBUG: Log disc losses for first few steps
                    if step <= 15:
                        is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                        if is_main:
                            print(f"  loss_adv: {loss_adv.item():.6f}, loss_fm: {loss_fm.item():.6f}")
                            if torch.isnan(loss_adv) or torch.isnan(loss_fm):
                                print(f"  ❌ NaN in discriminator losses!")

                if stage.use_perceptual and target_out is not None and stage.lambda_music_perc > 0:
                    loss_music_perc = self.music_perc_loss(clean_target, target_out)
                if stage.lambda_audio_perc > 0 and target_out is not None:
                    loss_audio_perc = self.audio_perc_loss(clean_target, target_out)

                total_loss = (
                    stage.lambda_lmos * loss_lmos
                    + stage.lambda_gan * loss_adv
                    + stage.lambda_fm * loss_fm
                    + stage.lambda_music_perc * loss_music_perc
                    + stage.lambda_audio_perc * loss_audio_perc
                ) / accumulate
                
                # Check for NaN in outputs before computing loss
                has_nan = False
                if torch.isnan(base_out).any() or torch.isinf(base_out).any():
                    has_nan = True
                    print(f"\n[NaN DETECTED IN OUTPUT] Step {step}:")
                    print(f"  base_out: NaN={torch.isnan(base_out).sum().item()}, Inf={torch.isinf(base_out).sum().item()}")
                    print(f"  base_out range: [{base_out.min().item()}, {base_out.max().item()}]")
                
                if target_out is not None and (torch.isnan(target_out).any() or torch.isinf(target_out).any()):
                    has_nan = True
                    print(f"  target_out: NaN={torch.isnan(target_out).sum().item()}, Inf={torch.isinf(target_out).sum().item()}")
                
                # Check for NaN in losses
                if torch.isnan(total_loss) or torch.isnan(loss_lmos):
                    has_nan = True
                    print(f"\n[NaN DETECTED IN LOSS] Step {step}:")
                    print(f"  total_loss: {total_loss.item()}")
                    print(f"  loss_lmos: {loss_lmos.item()}")
                    print(f"  loss_adv: {loss_adv.item()}")
                    print(f"  loss_fm: {loss_fm.item()}")
                    print(f"  loss_music_perc: {loss_music_perc.item()}")
                
                # If NaN detected, skip this batch and continue
                if has_nan:
                    print(f"\n⚠️  SKIPPING BATCH at step {step} due to NaN/Inf")
                    print(f"   Input stats: degraded_base [{degraded_base.min().item():.6f}, {degraded_base.max().item():.6f}]")
                    print(f"   Input stats: clean_base [{clean_base.min().item():.6f}, {clean_base.max().item():.6f}]")
                    print(f"   Batch size: {degraded_base.shape[0]}")
                    
                    # Check if model weights are corrupted
                    self._check_model_for_nan(f"at step {step} after NaN detection")
                    
                    # Zero gradients and continue to next batch
                    optim_g.zero_grad()
                    if optim_d is not None:
                        optim_d.zero_grad()
                    
                    # Log skipped sample count
                    if HAS_ACCELERATE and self.accelerator is not None:
                        self.accelerator.log({"train/nan_skipped_samples": 1}, step=step)
                    
                    continue  # Skip backward pass and optimizer step

                # Use accelerator for backward pass
                if HAS_ACCELERATE and self.accelerator is not None:
                    self.accelerator.backward(total_loss)
                else:
                    total_loss.backward()

                # Discriminator update (with gradient accumulation matching generator)
                loss_r1 = torch.tensor(0.0, device=self.device)  # Define outside if-block for logging
                if stage.use_adversarial and optim_d is not None:
                    # Re-enable discriminator gradients for discriminator training
                    for param in self.discriminator.parameters():
                        param.requires_grad = True
                    
                    # Detach generator outputs to prevent gradients flowing back to generator
                    base_detached = _maybe_add_disc_noise(base_out.detach())
                    target_detached = _maybe_add_disc_noise(target_out.detach() if target_out is not None else None)
                    
                    # Compute discriminator scores on detached outputs
                    disc_fake_detached = self.discriminator(base_detached, target_detached)
                    real_base_for_disc = _maybe_add_disc_noise(clean_base)
                    real_target_for_disc = _maybe_add_disc_noise(clean_target if target_out is not None else None)
                    disc_real_for_disc = self.discriminator(real_base_for_disc, real_target_for_disc)
                    
                    # Clip discriminator scores to prevent extreme gradients
                    fake_scores_d = [torch.clamp(out.score, min=-10.0, max=10.0) for out in flatten_outputs(disc_fake_detached)]
                    real_scores_d = [torch.clamp(out.score, min=-10.0, max=10.0) for out in flatten_outputs(disc_real_for_disc)]
                    
                    # Train discriminator at FULL strength (no warmup scaling)
                    # Discriminator needs to learn quickly to provide useful signal to generator
                    # CRITICAL: Divide by accumulate to match generator's gradient scale
                    loss_d = gan_loss_fn.discriminator(
                        real_scores_d,
                        fake_scores_d,
                        real_target=real_target,
                        fake_target=fake_target,
                    ) / accumulate
                    
                    # Add R1 gradient penalty every 8 steps for stability
                    # R1 penalty: E[||∇_x D(x)||^2] encourages smooth discriminator gradients
                    # Compute less frequently (every 8 steps) to save computation
                    if step % 8 == 0 and clean_base is not None:
                        from .losses import r1_gradient_penalty
                        clean_target_for_penalty = clean_target if target_out is not None else None
                        loss_r1 = r1_gradient_penalty(
                            self.discriminator,
                            clean_base,
                            clean_target_for_penalty,
                        ) * 10.0  # lambda_r1 = 10.0
                        loss_d = loss_d + loss_r1 / accumulate
                    
                    # DEBUG: Log discriminator loss and scores for first few steps
                    if step <= 15:
                        is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                        if is_main:
                            real_scores_stats = [f"mean={s.mean().item():.4f}, range=[{s.min().item():.4f}, {s.max().item():.4f}]" for s in real_scores_d]
                            fake_scores_d_stats = [f"mean={s.mean().item():.4f}, range=[{s.min().item():.4f}, {s.max().item():.4f}]" for s in fake_scores_d]
                            print(f"  [DISC] real_scores: {real_scores_stats}")
                            print(f"  [DISC] fake_scores: {fake_scores_d_stats}")
                            print(f"  [DISC] loss_d: {loss_d.item():.6f}, gen_adv_scale: {gen_adv_scale:.3f}")

                    # Use accelerator for backward pass
                    if HAS_ACCELERATE and self.accelerator is not None:
                        self.accelerator.backward(loss_d)
                    else:
                        loss_d.backward()
                
                is_accum_boundary = (step % accumulate == 0)
                
                # Update generator on accumulation boundary
                if is_accum_boundary:
                    grad_norm_g = None
                    if stage.gradient_clip > 0:
                        if HAS_ACCELERATE and self.accelerator is not None:
                            grad_norm_g = self.accelerator.clip_grad_norm_(self.generator.parameters(), stage.gradient_clip)
                        else:
                            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), stage.gradient_clip)
                    
                    # DEBUG: Log gradient norm before optimizer step
                    if step <= 10:
                        is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                        if is_main:
                            print(f"[DEBUG STEP {step}] Gradient norm before optim step: {grad_norm_g}")
                    
                    optim_g.step()
                    scheduler_g.step()
                    self.ema.update(self.generator)
                    
                    # DEBUG: Check weights after optimizer step
                    if step <= 10:
                        is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                        if is_main:
                            has_nan = False
                            for name, param in self.generator.named_parameters():
                                if torch.isnan(param).any():
                                    print(f"  ❌ NaN in {name} after optimizer step!")
                                    has_nan = True
                                    break
                            if not has_nan:
                                print(f"  ✓ All weights OK after optimizer step")
                
                # Update discriminator on every accumulation boundary and during pre-training
                if stage.use_adversarial and optim_d is not None:
                    should_update_disc = is_accum_boundary or step <= disc_pretrain_steps
                    if should_update_disc:
                        # DEBUG: Check discriminator gradients before clipping
                        if step <= 15:
                            is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                            if is_main:
                                grad_norm_d = 0.0
                                for p in self.discriminator.parameters():
                                    if p.grad is not None:
                                        grad_norm_d += p.grad.norm().item() ** 2
                                grad_norm_d = grad_norm_d ** 0.5
                                print(f"  [DISC] grad_norm before clip: {grad_norm_d:.6f}")
                        
                        if stage.gradient_clip > 0:
                            if HAS_ACCELERATE and self.accelerator is not None:
                                grad_norm_d_clipped = self.accelerator.clip_grad_norm_(self.discriminator.parameters(), stage.gradient_clip)
                            else:
                                grad_norm_d_clipped = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), stage.gradient_clip)
                            
                            if step <= 15:
                                is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                                if is_main:
                                    print(f"  [DISC] grad_norm after clip: {grad_norm_d_clipped}")
                                    if step % 16 == 0 and loss_r1.item() > 0:
                                        print(f"  [DISC] R1 penalty: {loss_r1.item():.6f}")
                        
                        optim_d.step()
                        if scheduler_d is not None:
                            scheduler_d.step()

                running_losses["lmos"] += loss_lmos.item()
                running_losses["adv"] += loss_adv.item()
                running_losses["fm"] += loss_fm.item()
                running_losses["music_perc"] += loss_music_perc.item()
                running_losses["audio_perc"] += loss_audio_perc.item()

                if step % self.train_cfg.log_every == 0:
                    mean_losses = {k: v / self.train_cfg.log_every for k, v in running_losses.items()}
                    
                    # Only log on main process
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main:
                        status = " | ".join(f"{k}:{v:.4f}" for k, v in mean_losses.items())
                        print(f"[{stage.name}] step {step}: {status}")
                        
                        # Log to W&B
                        if self.wandb_run is not None and HAS_WANDB:
                            current_lr = optim_g.param_groups[0]["lr"]
                            wandb_metrics = {
                                "train/lmos": mean_losses["lmos"],
                                "train/adv": mean_losses["adv"],
                                "train/fm": mean_losses["fm"],
                                "train/music_perc": mean_losses["music_perc"],
                                "train/audio_perc": mean_losses["audio_perc"],
                                "train/total_loss": sum(mean_losses.values()),
                                "train/lr": current_lr,
                                "train/stage": stage.name,
                            }
                            wandb.log(wandb_metrics, step=step)
                    
                    running_losses = defaultdict(float)

                if step % self.train_cfg.save_every == 0:
                    # Only save on main process
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main:
                        self._save_checkpoint(stage, step, optim_g, optim_d, scheduler_g, scheduler_d)

                if (
                    self.train_cfg.eval_every > 0
                    and val_loader is not None
                    and step % self.train_cfg.eval_every == 0
                ):
                    # Synchronize before validation
                    if HAS_ACCELERATE and self.accelerator is not None:
                        self.accelerator.wait_for_everyone()
                    
                    val_metrics = self.evaluate(stage, val_loader, step)
                    
                    # Save checkpoint with validation loss (only on main process)
                    is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
                    if is_main and val_metrics and "val/lmos" in val_metrics:
                        val_loss = val_metrics["val/lmos"]
                        self._save_checkpoint(
                            stage, step, optim_g, optim_d, 
                            scheduler_g, scheduler_d, val_loss=val_loss
                        )
                    
                    # Log to W&B (only on main process)
                    if is_main and self.wandb_run is not None and HAS_WANDB:
                        wandb.log(val_metrics, step=step)

            # Stage end checkpoint (only on main process)
            is_main = not HAS_ACCELERATE or self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                self._save_checkpoint(stage, stage.max_steps, optim_g, optim_d, scheduler_g, scheduler_d)
                print(f"[trainer] Finished stage {stage.name}")
                
                # Run final evaluation on 2 samples if no validation set was used
                if val_loader is None:
                    print(f"[trainer] Running final evaluation (no validation set used during training)")
                    self._run_final_evaluation(stage, train_loader)
        
        # Close W&B run
        if self.wandb_run is not None and HAS_WANDB:
            self.wandb_run.finish()

    @torch.no_grad()
    def _run_final_evaluation(
        self,
        stage: StageConfig,
        train_loader: DataLoader[dict[str, Tensor | None]],
        step=None
    ) -> None:
        """Run quick evaluation on 2 samples from training data and log to W&B.
        
        Parameters
        ----------
        stage : StageConfig
            Current training stage
        train_loader : DataLoader
            Training data loader
        """
        self.generator.eval()
        self.ema.apply(self.generator)
        
        # Get 2 samples from training data
        iterator = iter(train_loader)
        batch = next(iterator)
        batch = self._prepare_batch(batch)
        
        degraded_base = batch["degraded_base"][:2]  # Take only 2 samples
        clean_base = batch["clean_base"][:2]
        clean_target = batch["clean_target"][:2]
        embedding_batch = batch.get("embedding")
        if torch.is_tensor(embedding_batch):
            embedding_batch = embedding_batch[:2]
        
        # Normalize degraded audio if enabled
        if self.data_cfg.normalize_degraded:
            degraded_base, _ = self._normalize_audio(degraded_base)
        
        outputs = self.generator(
            degraded_base,
            stage=stage.name,
            precomputed_embeddings=embedding_batch if torch.is_tensor(embedding_batch) else None,
        )
        
        target_out = outputs.get("target")
        
        if target_out is None:
            restored_audio = outputs["base"]
            clean_audio = clean_base
            degraded_audio = degraded_base
            sample_rate = self.model_cfg.base_sample_rate
        else:
            restored_audio = target_out
            clean_audio = clean_target
            degraded_audio = torch.nn.functional.interpolate(
                degraded_base.unsqueeze(1) if degraded_base.ndim == 2 else degraded_base,
                size=clean_target.shape[-1],
                mode='linear',
                align_corners=False
            )
            if degraded_audio.ndim == 3 and degraded_audio.shape[1] == 1:
                degraded_audio = degraded_audio.squeeze(1)
            sample_rate = self.model_cfg.target_sample_rate
        
        # Move to CPU for evaluation
        clean_audio_cpu = clean_audio.detach().cpu()
        restored_audio_cpu = restored_audio.detach().cpu()
        degraded_audio_cpu = degraded_audio.detach().cpu()
        
        # Save samples and compute metrics
        audio_dir = Path(self.train_cfg.checkpoint_dir) / self.train_cfg.val_audio_dir
        audio_metrics, wandb_audio = save_validation_samples(
            step=stage.max_steps if step is None else step,
            output_dir=audio_dir,
            clean_audio=clean_audio_cpu,
            restored_audio=restored_audio_cpu,
            degraded_audio=degraded_audio_cpu,
            sample_rate=sample_rate,
            num_samples=2,
            use_wandb=self.train_cfg.use_wandb,
        )
        
        # Log metrics
        print(f"[final_eval:{stage.name}] Audio quality metrics:")
        for key, value in audio_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Log to W&B
        if self.wandb_run is not None and HAS_WANDB:
            final_metrics = {f"final/{k}": v for k, v in audio_metrics.items()}
            final_metrics.update(wandb_audio)
            wandb.log(final_metrics, step=stage.max_steps)
        
        self.ema.restore(self.generator)
        self.generator.train()
    
    @torch.no_grad()
    def evaluate(
        self, 
        stage: StageConfig, 
        val_loader: DataLoader[dict[str, Tensor | None]], 
        step: int | None = None
    ) -> Dict[str, float]:
        """Run validation and return metrics including audio quality scores.
        
        Parameters
        ----------
        stage : StageConfig
            Current training stage configuration
        val_loader : DataLoader
            Validation data loader
        step : int, optional
            Current training step (for audio saving)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.ema.apply(self.generator)
        
        total_lmos = 0.0
        batches = 0
        
        # Store first batch for audio quality metrics
        first_batch_clean = None
        first_batch_restored = None
        first_batch_degraded = None
        
        for batch_idx, batch in enumerate(val_loader):
            if self.data_cfg.val_batches and batch_idx >= self.data_cfg.val_batches:
                break
            
            batch = self._prepare_batch(batch)
            degraded_base = batch["degraded_base"]
            clean_base = batch["clean_base"]
            clean_target = batch["clean_target"]
            embedding_batch = batch.get("embedding")
            
            # Normalize degraded audio if enabled
            degraded_scale = None
            if self.data_cfg.normalize_degraded:
                degraded_base, degraded_scale = self._normalize_audio(degraded_base)
            
            outputs = self.generator(
                degraded_base,
                stage=stage.name,
                precomputed_embeddings=embedding_batch if torch.is_tensor(embedding_batch) else None,
            )
            
            target_out = outputs.get("target")
            
            # Compute LMOS loss
            if target_out is None:
                lmos = self.lmos_loss(clean_base, outputs["base"], self.model_cfg.base_sample_rate)
                restored_audio = outputs["base"]
                clean_audio = clean_base
                degraded_audio = degraded_base
            else:
                lmos = self.lmos_loss(clean_target, target_out, self.model_cfg.target_sample_rate)
                restored_audio = target_out
                clean_audio = clean_target
                # Upsample degraded audio to match target sample rate for fair comparison
                degraded_audio = torch.nn.functional.interpolate(
                    degraded_base.unsqueeze(1) if degraded_base.ndim == 2 else degraded_base,
                    size=clean_target.shape[-1],
                    mode='linear',
                    align_corners=False
                )
                if degraded_audio.ndim == 3 and degraded_audio.shape[1] == 1:
                    degraded_audio = degraded_audio.squeeze(1)
            
            total_lmos += lmos.item()
            batches += 1
            
            # Store first batch for detailed metrics
            if batch_idx == 0:
                first_batch_clean = clean_audio.detach().cpu()
                first_batch_restored = restored_audio.detach().cpu()
                first_batch_degraded = degraded_audio.detach().cpu()
        
        mean_lmos = total_lmos / max(batches, 1)
        metrics = {"val/lmos": mean_lmos}
        
        # Compute audio quality metrics and save samples if step provided
        if step is not None and first_batch_clean is not None:
            sample_rate = (
                self.model_cfg.target_sample_rate 
                if stage.use_upsample_head 
                else self.model_cfg.base_sample_rate
            )
            
            audio_dir = Path(self.train_cfg.checkpoint_dir) / self.train_cfg.val_audio_dir
            audio_metrics, wandb_audio = save_validation_samples(
                step=step,
                output_dir=audio_dir,
                clean_audio=first_batch_clean,
                restored_audio=first_batch_restored,
                degraded_audio=first_batch_degraded,
                sample_rate=sample_rate,
                num_samples=self.train_cfg.val_save_samples,
                use_wandb=self.train_cfg.use_wandb,
            )
            
            # Add audio metrics with val/ prefix
            for key, value in audio_metrics.items():
                metrics[f"val/{key}"] = value
            
            # Add W&B audio objects
            if wandb_audio and self.wandb_run is not None and HAS_WANDB:
                for key, value in wandb_audio.items():
                    metrics[key] = value
        
        # Print metrics
        print(f"[eval:{stage.name}] LMOS={mean_lmos:.4f}")
        if "val/si_snr_restored" in metrics:
            print(f"[eval:{stage.name}] SI-SNR: input={metrics.get('val/si_snr_input', 0):.2f} dB, "
                  f"restored={metrics['val/si_snr_restored']:.2f} dB")
        if "val/sdr_restored" in metrics:
            print(f"[eval:{stage.name}] SDR: input={metrics.get('val/sdr_input', 0):.2f} dB, "
                  f"restored={metrics['val/sdr_restored']:.2f} dB")
        
        self.ema.restore(self.generator)
        self.generator.train()
        
        return metrics


__all__ = ["FinallyGanTrainer"]
