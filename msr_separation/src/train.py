import shutil
import tempfile
import random
import json
import re
import uuid

from filelock import FileLock
from munch import DefaultMunch
from torch.utils.data import Subset

from src.utils import (
    logging_setup,
    parse_yaml,
    initialize_config,
    ignore_warnings,
    download_checkpoint_from_wandb,
    inject_dataset_paths_into_config,
)

ignore_warnings()
import os

import sys
import torch
import pathlib
import numpy as np
import soundfile as sf
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sacred import Experiment
from sacred import SETTINGS
from datasets import Dataset, concatenate_datasets
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from tqdm import tqdm

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from src.config_updates import add_configs, LOG_PATH, CKPT_PATH, MUSICA_SCRATCH_4_STEM, MUSICA_SCRATCH_8_STEM
import src.config_updates as config_updates_module
from src.training.lightningmodule.helper import load_ckpt


def restore_columns_from_checkpoint(lightning_module, full_ckpt, context_name="checkpoint"):
    """
    Restore columns from checkpoint to ensure correct indexing.
    
    The model output order matches the training column order (from encode_target_sources),
    so we MUST use the same column order that was used during training.
    When a checkpoint is loaded from a previous training stage, the columns from that stage
    are what the model was trained with, and we must use the same order for indexing.
    
    Args:
        lightning_module: The lightning module to restore columns for
        full_ckpt: Full checkpoint dictionary from load_ckpt()
        context_name: Context name for logging (e.g., "MSRBench evaluation", "validation")
    
    Returns:
        bool: True if columns were restored from checkpoint, False if using config columns
    """
    try:
        # PyTorch Lightning stores hyper_parameters in the checkpoint
        if 'hyper_parameters' in full_ckpt:
            hparams = full_ckpt['hyper_parameters']
            if 'columns' in hparams:
                ckpt_columns = hparams['columns']
                if ckpt_columns is not None:
                    # Convert to list if needed (might be stored as tuple or other type)
                    ckpt_columns_list = list(ckpt_columns) if not isinstance(ckpt_columns, list) else ckpt_columns
                    if len(ckpt_columns_list) > 0:
                        print(f"[{context_name}] ✓ Restoring columns from checkpoint: {ckpt_columns_list}")
                        lightning_module.columns = ckpt_columns_list
                        return True
                    else:
                        print(f"[{context_name}] ⚠ Warning: Checkpoint has empty columns, using config: {lightning_module.columns}")
                else:
                    print(f"[{context_name}] ⚠ Warning: Checkpoint hyper_parameters has None columns, using config: {lightning_module.columns}")
            else:
                print(f"[{context_name}] ⚠ Warning: Columns not found in checkpoint hyper_parameters")
                print(f"  Using config columns: {lightning_module.columns}")
                print("  ⚠ CRITICAL: This may cause incorrect indexing if config columns differ from training order!")
        else:
            print(f"[{context_name}] ⚠ Warning: No hyper_parameters in checkpoint")
            print(f"  Using config columns: {lightning_module.columns}")
            print("  ⚠ CRITICAL: This may cause incorrect indexing if config columns differ from training order!")
    except Exception as e:
        print(f"[{context_name}] ⚠ Warning: Error restoring columns from checkpoint: {e}")
        print(f"  Using config columns: {lightning_module.columns}")
        print("  ⚠ CRITICAL: This may cause incorrect indexing if config columns differ from training order!")
    
    return False

ex = Experiment('music_source_separation')


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    cmd_short = cmd.split('\\')[-1]

    config_yaml_file = None

    batch_size = -1  # default batch size
    tqdm_rate = 1  # 100

    # if checkpoint is in the folder where the checkpoints are stored, set these variables:
    resume_wandb_id = None
    resume_ckpt_name = None

    # otherwise, set this variable:
    resume_ckpt_path = None

    # Fine-tuning: load checkpoint from a previous wandb run but create a new wandb run
    # Uses last.ckpt from CKPT_PATH/finetune_config_filename/finetune_wandb_id/last.ckpt
    finetune_wandb_id = None
    # Config filename/path for fine-tuning checkpoint (default: 'roformer_4s-dataset_step_lr_scheduler')
    finetune_config_filename = None

    # Optional MSRBench evaluation settings (override via YAML configs)
    msr_bench_evaluation = None

    # Optional inference dataset generation settings (override via YAML configs)
    generate_inference_dataset = None

    # Optional MSR test set inference settings (override via YAML configs)
    msr_testset_inference = None

    # Validate and upload only: load checkpoint, run validation, create new wandb run, and upload samples
    validate_and_upload_only = False

    lightning_module = dict(
        args=dict(
            lr_scheduler=dict(
                schedule_mode=None,
                num_warmup_steps=50_000,
                lr_end=2e-7
            )
        )
    )

    locals().update(parse_yaml(config_yaml_file))


add_configs(ex)


def upload_checkpoint_and_samples_to_wandb(
        logger,
        checkpoint_path: str,
        model: torch.nn.Module,
        lightning_module,
        data_module,
        device,
        num_samples: int = 30,
        config=None,
        only_upload_samples=False
):
    """
    Upload checkpoint as artifact and generate audio samples for wandb.

    Args:
        logger: WandbLogger instance
        checkpoint_path: Path to checkpoint file
        model: Model for inference
        lightning_module: Lightning module (for columns and augmentation)
        data_module: Data module (to access validation dataset)
        device: Device to run inference on
        num_samples: Number of samples to generate (default: 6)
    """
    import wandb

    if wandb.run is None:
        print("WARNING: wandb.run is None, skipping upload")
        return

    print(f"\n{'=' * 80}")
    print("Uploading checkpoint and audio samples to wandb...")
    print(f"{'=' * 80}")

    try:
        if not only_upload_samples:
            # 1. Upload checkpoint as artifact
            print(f"\nUploading checkpoint: {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint_artifact = wandb.Artifact(
                    name=f"checkpoint-{wandb.run.id}",
                    type="model",
                    description=f"Final checkpoint after training"
                )
                checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(checkpoint_artifact)
                print("✓ Checkpoint uploaded as artifact")
            else:
                print(f"WARNING: Checkpoint not found: {checkpoint_path}")

        # 2. Generate and upload audio samples
        print(f"\nGenerating {num_samples} audio samples...")

        # Get validation dataset
        if data_module.val_dataset is None:
            print("WARNING: Validation dataset not available, skipping audio sample upload")
            return

        val_dataset = data_module.val_dataset

        columns = lightning_module.columns if hasattr(lightning_module, 'columns') else None
        if columns is None:
            print("WARNING: Columns not found in lightning module, skipping audio sample upload")
            return

        # Create mapping from column name to index based on lightning_module.columns order
        # This ensures we index into separated_sources correctly, matching the model's output order
        print("Lightning Columns: ", lightning_module.columns)
        column_to_idx = {col: idx for idx, col in enumerate(lightning_module.columns)}

        sampling_rate = lightning_module.sampling_rate if hasattr(lightning_module, 'sampling_rate') else 48000
        use_augmentation = lightning_module.augment if hasattr(lightning_module, 'augment') else False

        # Select samples
        dataset_size = len(val_dataset)
        if dataset_size == 0:
            print("WARNING: Validation dataset is empty, skipping audio sample upload")
            return

        num_samples = min(num_samples, dataset_size)

        # For MSRBench we want deterministic sample selection so that runs are comparable.
        if getattr(lightning_module, 'msr_bench_validation', False):
            msr_seed = getattr(lightning_module, 'msr_bench_seed', 42)
            rng = random.Random(msr_seed)
            available_indices = list(range(dataset_size))
            rng.shuffle(available_indices)
            sample_indices = sorted(available_indices[:num_samples])
            print(f"[MSRBench] Using deterministic sample indices for wandb upload (seed={msr_seed}): {sample_indices}")
        else:
            sample_indices = random.sample(range(dataset_size), num_samples)

        model.eval()
        device_obj = torch.device(device) if isinstance(device, str) else device
        model.to(device_obj)

        # Determine precision setting from trainer or config (to match training precision)
        # Get precision from trainer if available, otherwise try config, then default to full precision
        use_autocast = False
        autocast_dtype = None
        precision = "16-mixed"

        # # Try to get precision from trainer first (most reliable in training context)
        # if hasattr(lightning_module, 'trainer') and lightning_module.trainer is not None:
        #     trainer = lightning_module.trainer
        #     precision = getattr(trainer, 'precision', None)

        # Fallback to config if trainer precision not available
        if precision is None and config is not None:
            trainer_config = config.get('train', {}).get('trainer', {}).get('args', {})
            precision = trainer_config.get('precision', None)
            if precision:
                print(f"Using precision from config: {precision}")

        # Apply precision setting
        if precision == "16-mixed":
            use_autocast = True
            autocast_dtype = torch.float16
            print("Using FP16 mixed precision for inference (matching training precision)")
        elif precision == "bf16-mixed":
            use_autocast = True
            autocast_dtype = torch.bfloat16
            print("Using BF16 mixed precision for inference (matching training precision)")
        elif precision in ["32-true", "32"]:
            use_autocast = False
            print("Using FP32 precision for inference (matching training precision)")
        else:
            # Unknown precision or not found, default to no autocast
            if precision:
                print(f"Unknown precision setting '{precision}', using FP32 for inference")
            else:
                print("No precision setting found, using FP32 precision for inference")

        # Apply EMA weights if enabled (for inference, we want to use EMA model)
        if hasattr(lightning_module, 'use_ema') and lightning_module.use_ema and lightning_module.ema is not None:
            print("Applying EMA weights for sample generation...")
            lightning_module.ema.apply()
            print("EMA weights applied successfully")

        # Create temporary directory for audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_count = 0

            for sample_idx, idx in enumerate(sample_indices):
                try:
                    # Load sample from dataset
                    sample = {}
                    sample = val_dataset[idx]

                    # Detect if this is MSRBench dataset (has 'mixture' and 'source' keys)
                    is_msr_bench = 'mixture' in sample and 'source' in sample

                    if is_msr_bench:
                        # Handle MSRBench dataset structure
                        if not sample['mixture'].get('is_present', False) or not sample['source'].get('is_present',
                                                                                                      False):
                            continue

                        # Get mixture and source waveforms
                        mixture, source_gt, source_label = None, None, None
                        mixture = sample['mixture']['waveform']  # [2, T]
                        source_gt = sample['source']['waveform']  # [2, T]
                        source_label = sample['source'].get('label', 'unknown')

                        # Trim to minimum length
                        min_length = min(mixture.shape[-1], source_gt.shape[-1])
                        mixture = mixture[:, :min_length]
                        source_gt = source_gt[:, :min_length]

                        # Save mixture
                        mixture_path = os.path.join(temp_dir, f"sample_{sample_idx}_mixture.wav")
                        sf.write(mixture_path, mixture.T.cpu().numpy(), sampling_rate)

                        # Log mixture
                        wandb_dict = {}
                        wandb_dict = {
                            f"audio_samples/sample_{sample_idx}/mixture": wandb.Audio(mixture_path,
                                                                                      sample_rate=sampling_rate),
                        }

                        # Run inference with correct precision
                        with torch.no_grad():
                            mixture_batch = {}
                            mixture_batch = mixture.unsqueeze(0).to(device_obj)  # [1, 2, T]
                            input_dict = {'mixture': mixture_batch}

                            # Add label_vector for FiLM conditioning if enabled
                            if hasattr(lightning_module,
                                       'use_film_conditioning') and lightning_module.use_film_conditioning:
                                label_vector = lightning_module.label_vectors.flatten().unsqueeze(0).to(device_obj)
                                input_dict['label_vector'] = label_vector

                            # Use autocast if mixed precision was used during training
                            # Note: autocast automatically handles mixed precision - it uses FP16 for operations
                            # that benefit from it (matmuls, convolutions) and FP32 for operations that need
                            # higher precision (STFT, reductions). Inner autocast(enabled=False) blocks in the
                            # model will automatically force FP32 for those specific operations, matching training.
                            output_dict = {}
                            if use_autocast and autocast_dtype is not None:
                                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                                    output_dict = model(input_dict)
                            else:
                                output_dict = model(input_dict)
                            separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, 2, T]
                            separated_sources = separated_sources.cpu()

                        # Find the index of the source label in columns using the column_to_idx mapping
                        separated_stem = None
                        if source_label in columns:
                            source_idx = column_to_idx[source_label]
                            separated_stem = separated_sources[source_idx]  # [2, T]

                            # Trim separated_stem to match actual length
                            if separated_stem.shape[-1] > min_length:
                                separated_stem = separated_stem[:, :min_length]
                            elif separated_stem.shape[-1] < min_length:
                                pad_length = min_length - separated_stem.shape[-1]
                                separated_stem = torch.nn.functional.pad(separated_stem, (0, pad_length),
                                                                         mode='constant', value=0.0)

                            # Save audio files
                            base_name, clean_path, separated_path = "", "", ""
                            base_name = f"sample_{sample_idx}_source_{source_label}"
                            clean_path = os.path.join(temp_dir, f"{base_name}_clean.wav")
                            separated_path = os.path.join(temp_dir, f"{base_name}_separated.wav")

                            sf.write(clean_path, source_gt.T.cpu().numpy(), sampling_rate)
                            sf.write(separated_path, separated_stem.T.cpu().numpy(), sampling_rate)

                            # Add audio to wandb dict
                            wandb_dict[f"audio_samples/sample_{sample_idx}/{source_label}/clean"] = wandb.Audio(
                                clean_path, sample_rate=sampling_rate)
                            wandb_dict[f"audio_samples/sample_{sample_idx}/{source_label}/separated"] = wandb.Audio(
                                separated_path, sample_rate=sampling_rate)

                            uploaded_count += 2  # 2 files (clean, separated)
                        else:
                            print(f"Warning: Source label '{source_label}' not found in columns {columns}")

                        # Log all audio files for this sample
                        wandb.log(wandb_dict)
                        uploaded_count += 1  # +1 for mixture
                    else:
                        # Handle standard multi-column dataset structure
                        # Prepare batch data with augmentation (like in training)
                        batch_data = {}
                        waveform_length = None
                        original_presence = {}

                        # Get waveform length from first available column
                        for column in columns:
                            if column in sample and sample[column]['waveform'] is not None:
                                waveform_length = sample[column]['waveform'].shape[-1]
                                break

                        if waveform_length is None:
                            continue

                        # Prepare batch
                        for column in columns:
                            if column in sample:
                                sample_column = sample[column]
                                column_is_present = bool(sample_column.get('is_present', True))
                            else:
                                sample_column = {}
                                column_is_present = False
                            original_presence[column] = column_is_present

                            if column in sample and column_is_present:
                                batch_data[column] = {
                                    "waveform": sample[column]['waveform'].unsqueeze(0),
                                    "lengths": torch.tensor([sample[column]['waveform'].shape[-1]]),
                                    "paths": [sample[column].get('path', '')],
                                    "is_present": [sample[column].get('is_present', True)]
                                }
                            else:
                                empty_waveform = torch.zeros((1, 2, waveform_length))
                                batch_data[column] = {
                                    "waveform": empty_waveform,
                                    "lengths": torch.tensor([waveform_length]),
                                    "paths": [""],
                                    "is_present": [False]
                                }

                        # Apply augmentation and create mixture
                        batch_data = lightning_module.create_mixtures(batch_data)
                        mixture = batch_data['mixture'].squeeze(0)  # [2, T]

                        # Save mixture once per sample (not per stem)
                        mixture_path = os.path.join(temp_dir, f"sample_{sample_idx}_mixture.wav")
                        sf.write(mixture_path, mixture.T.cpu().numpy(), sampling_rate)

                        # Log mixture once per sample
                        wandb_dict = {
                            f"audio_samples/sample_{sample_idx}/mixture": wandb.Audio(mixture_path,
                                                                                      sample_rate=sampling_rate),
                        }

                        # Run inference with correct precision
                        with torch.no_grad():
                            mixture_batch = mixture.unsqueeze(0).to(device_obj)  # [1, 2, T]
                            input_dict = {'mixture': mixture_batch}

                            # Use autocast if mixed precision was used during training
                            # Note: autocast automatically handles mixed precision - it uses FP16 for operations
                            # that benefit from it (matmuls, convolutions) and FP32 for operations that need
                            # higher precision (STFT, reductions). Inner autocast(enabled=False) blocks in the
                            # model will automatically force FP32 for those specific operations, matching training.
                            if use_autocast and autocast_dtype is not None:
                                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                                    output_dict = model(input_dict)
                            else:
                                output_dict = model(input_dict)
                            separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, 2, T]
                            separated_sources = separated_sources.cpu()

                        # Save audio files for each stem that is present
                        for column in columns:
                            is_present_batch = batch_data[column]['is_present'][0]
                            is_present_original = original_presence.get(column, False)
                            if not (is_present_batch and is_present_original):
                                continue

                            # Get the correct index for this column based on model's output order
                            source_idx = column_to_idx[column]

                            # Get waveforms
                            clean_stem = batch_data[column]["waveform"].squeeze(0)  # [2, T]
                            if use_augmentation and 'aug_waveform' in batch_data[column]:
                                aug_stem = batch_data[column]["aug_waveform"].squeeze(0)  # [2, T]
                            else:
                                aug_stem = clean_stem  # Fallback to clean if no augmentation

                            separated_stem = separated_sources[source_idx]  # [2, T]

                            # Save audio files (transpose from [2, T] to [T, 2] for soundfile)
                            base_name = f"sample_{sample_idx}_stem_{column}"
                            clean_path = os.path.join(temp_dir, f"{base_name}_clean.wav")
                            aug_path = os.path.join(temp_dir, f"{base_name}_augmented.wav")
                            separated_path = os.path.join(temp_dir, f"{base_name}_separated.wav")

                            sf.write(clean_path, clean_stem.T.cpu().numpy(), sampling_rate)
                            sf.write(aug_path, aug_stem.T.cpu().numpy(), sampling_rate)
                            sf.write(separated_path, separated_stem.T.cpu().numpy(), sampling_rate)

                            # Add stem-specific audio to wandb dict (grouped under sample section)
                            wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/clean"] = wandb.Audio(clean_path,
                                                                                                          sample_rate=sampling_rate)
                            wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/augmented"] = wandb.Audio(aug_path,
                                                                                                              sample_rate=sampling_rate)
                            wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/separated"] = wandb.Audio(
                                separated_path, sample_rate=sampling_rate)

                            uploaded_count += 3  # 3 files per stem (clean, augmented, separated)

                        # Log all audio files for this sample at once
                        wandb.log(wandb_dict)
                        uploaded_count += 1  # +1 for mixture

                except Exception as e:
                    print(f"Error generating sample {sample_idx} (idx {idx}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"✓ Generated and uploaded {len(sample_indices)} audio samples ({uploaded_count} audio files)")

    except Exception as e:
        print(f"ERROR: Failed to upload checkpoint/samples to wandb: {e}")
        import traceback
        traceback.print_exc()


def _collect_hf_batch_dirs(output_dir: str, include_top_level: bool = False):
    """
    Collect Hugging Face dataset directories under output_dir that can be concatenated.
    """
    batch_dirs = []
    seen = set()

    temp_batches_dir = os.path.join(output_dir, "_temp_batches")
    if os.path.exists(temp_batches_dir):
        for item in sorted(os.listdir(temp_batches_dir)):
            if not item.startswith("rank_") or "_batch_" not in item:
                continue
            candidate = os.path.join(temp_batches_dir, item)
            if os.path.isdir(candidate) and candidate not in seen:
                batch_dirs.append(candidate)
                seen.add(candidate)

    if include_top_level and os.path.exists(output_dir):
        excluded = {"_temp_batches", "audio_files", "final_dataset"}
        dataset_markers = ("dataset_info.json", "dataset_dict.json", "state.json")
        for item in sorted(os.listdir(output_dir)):
            if item in excluded:
                continue
            candidate = os.path.join(output_dir, item)
            if not os.path.isdir(candidate):
                continue
            if candidate in seen:
                continue
            if any(os.path.exists(os.path.join(candidate, marker)) for marker in dataset_markers):
                batch_dirs.append(candidate)
                seen.add(candidate)

    return batch_dirs


def _print_label_distribution(label_counts, total_samples, final_dataset_path=None):
    """
    Pretty-print the label distribution and total sample count.
    """
    if not label_counts:
        print("No label distribution available.")
        return

    print("\nSample distribution by label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"Total samples: {total_samples}")
    if final_dataset_path:
        print(f"Final dataset saved to: {final_dataset_path}")


def _print_rank_statistics_table(all_rank_stats):
    """
    Pretty-print rank-level statistics summary.
    """
    if not all_rank_stats:
        print("Warning: No rank statistics found!")
        return

    print("\n" + "=" * 80)
    print("RANK STATISTICS SUMMARY")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Samples':<10} {'Batches':<10} {'Avg SI-SNR (dB)':<20} {'Std SI-SNR (dB)':<20}")
    print("-" * 80)

    total_samples = 0
    all_avg_si_snr = []
    for stats in sorted(all_rank_stats, key=lambda x: x['rank']):
        rank_num = stats['rank']
        samples = stats['samples_collected']
        batches = stats['num_batches']
        avg_snr = stats['avg_si_snr']
        std_snr = stats['std_si_snr']

        total_samples += samples
        if avg_snr is not None:
            all_avg_si_snr.append(avg_snr)

        avg_str = f"{avg_snr:.4f}" if avg_snr is not None else "N/A"
        std_str = f"{std_snr:.4f}" if std_snr is not None else "N/A"
        print(f"{rank_num:<6} {samples:<10} {batches:<10} {avg_str:<20} {std_str:<20}")

    print("-" * 80)
    print(f"{'TOTAL':<6} {total_samples:<10} {'':<10} {'':<20} {'':<20}")

    if all_avg_si_snr:
        overall_avg = np.mean(all_avg_si_snr)
        overall_std = np.std(all_avg_si_snr)
        print(f"\nOverall Average SI-SNR across all ranks: {overall_avg:.4f} ± {overall_std:.4f} dB")
    print("=" * 80 + "\n")


def _broadcast_object_to_ranks(obj, rank, callback=None, include_rank0=False):
    """
    Broadcast a Python object from rank 0 to all other ranks and optionally execute a callback.
    """
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if distributed:
        obj_list = [obj]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        shared_obj = obj_list[0]
        if callback is not None and shared_obj is not None:
            if include_rank0 or rank != 0:
                callback(shared_obj)
        return shared_obj
    else:
        if callback is not None and obj is not None and (include_rank0 or rank != 0):
            callback(obj)
        return obj


def _share_rank_statistics_with_other_ranks(all_rank_stats, rank):
    """
    Share rank statistics summary with other ranks for consistent logging.
    """
    _broadcast_object_to_ranks(
        all_rank_stats if rank == 0 else None,
        rank,
        callback=_print_rank_statistics_table
    )


def _share_label_distribution_with_other_ranks(dataset_info, rank):
    """
    Share label distribution summary so every rank sees it.
    """
    def _callback(info):
        if info is None:
            return
        _print_label_distribution(
            info.get('label_counts', {}),
            info.get('num_samples', 0),
            info.get('path')
        )

    _broadcast_object_to_ranks(
        dataset_info if rank == 0 else None,
        rank,
        callback=_callback
    )


def _wait_for_all_stats_files(output_dir, world_size, max_retries=30, retry_delay=0.5):
    """
    Wait for all rank stats files to be available with retries.
    
    Args:
        output_dir: Directory containing _temp_batches folder
        world_size: Total number of ranks
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        List of stats dictionaries, one per rank
    """
    import time
    all_rank_stats = []
    missing_ranks = []
    
    for attempt in range(max_retries):
        all_rank_stats = []
        missing_ranks = []
        
        for r in range(world_size):
            stats_file = os.path.join(output_dir, "_temp_batches", f"rank_{r}_stats.json")
            if os.path.exists(stats_file):
                try:
                    # Try to read the file to ensure it's complete
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        # Verify the stats contain expected fields
                        if 'rank' in stats and stats['rank'] == r:
                            all_rank_stats.append(stats)
                        else:
                            missing_ranks.append(r)
                except (json.JSONDecodeError, IOError, OSError) as e:
                    # File exists but not readable yet or incomplete
                    missing_ranks.append(r)
            else:
                missing_ranks.append(r)
        
        if len(all_rank_stats) == world_size:
            # All stats files found and readable
            return all_rank_stats
        
        if attempt < max_retries - 1:
            if missing_ranks:
                print(f"[Rank 0] Waiting for stats files from ranks {missing_ranks} (attempt {attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)
    
    # Final attempt - return what we have
    if missing_ranks:
        print(f"[Rank 0] WARNING: Could not find stats files for ranks {missing_ranks} after {max_retries} attempts")
    return all_rank_stats


def _print_msrbench_results(results, results_path):
    """
    Pretty-print MSRBench evaluation results (same format used on rank 0).
    """
    print("\n" + "=" * 100)
    print("MSRBench Evaluation Results (Final)")
    print("=" * 100)
    print(f"\nFiles evaluated per stem:")
    for label in sorted(results['files_evaluated'].keys()):
        count = results['files_evaluated'][label]
        print(f"  {label:15s}: {count:5d} files")

    print(f"\nPer-source SI-SNR scores:")
    for label, metrics in sorted(results['per_source'].items()):
        if 'mean_si_snr' in metrics:
            print(f"  {label:15s}: {metrics['mean_si_snr']:7.4f} ± {metrics['std_si_snr']:7.4f} dB "
                  f"({metrics['num_windows']} windows)")

    if results['overall'] and 'mean_si_snr' in results['overall']:
        print(f"\nOverall SI-SNR: {results['overall']['mean_si_snr']:.4f} ± "
              f"{results['overall']['std_si_snr']:.4f} dB "
              f"({results['overall']['num_windows']} windows)")

    print(f"\nPer-source Multi-Mel-SNR scores:")
    for label, metrics in sorted(results['per_source'].items()):
        if 'mean_multi_mel_snr' in metrics:
            print(f"  {label:15s}: {metrics['mean_multi_mel_snr']:7.4f} ± {metrics['std_multi_mel_snr']:7.4f} dB")

    if results['overall'] and 'mean_multi_mel_snr' in results['overall']:
        print(f"\nOverall Multi-Mel-SNR: {results['overall']['mean_multi_mel_snr']:.4f} ± "
              f"{results['overall']['std_multi_mel_snr']:.4f} dB")

    if results.get('per_dt_group'):
        print(f"\n{'=' * 100}")
        print("SI-SNR Performance by Degradation Type Group (Average across all stems)")
        print("=" * 100)

        print(f"\n{'DT Group':<20} {'Name':<25} {'SI-SNR (dB)':<20} {'Windows':<10}")
        print("-" * 100)

        if results['overall']:
            print(f"{'Overall':<20} {'All Groups':<25} "
                  f"{results['overall']['mean_si_snr']:6.4f} ± {results['overall']['std_si_snr']:6.4f}    "
                  f"{results['overall']['num_windows']:<10}")

        dt_group_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
        for dt_group in dt_group_order:
            if dt_group in results['per_dt_group']:
                dt_data = results['per_dt_group'][dt_group]
                print(f"{dt_group:<20} {dt_data['name']:<25} "
                      f"{dt_data['mean_si_snr']:6.4f} ± {dt_data['std_si_snr']:6.4f}    "
                      f"{dt_data['num_windows']:<10}")

        print("-" * 100)
        print("=" * 100)

        print(f"\n{'=' * 100}")
        print("SI-SNR Performance by Degradation Type Group AND Stem")
        print("=" * 100)

        columns_meta = results.get('metadata', {}).get('columns', sorted(results['per_source'].keys()))
        dt_group_names_meta = results.get('metadata', {}).get('dt_group_names', {
            'DT0': 'Reference',
            'DT1-DT4': 'Analog/Acoustic',
            'DT5-DT8': 'Traditional Codecs',
            'DT9-DT12': 'Neural Codecs'
        })

        stem_labels_sorted = sorted(columns_meta)
        header = f"{'DT Group':<20} {'Name':<25}"
        for stem in stem_labels_sorted:
            header += f" {stem:<15}"
        header += f" {'Avg':<15}"
        print(f"\n{header}")
        print("-" * 100)

        for dt_group in dt_group_order:
            if dt_group in results['per_dt_group']:
                row = f"{dt_group:<20} {dt_group_names_meta[dt_group]:<25}"
                dt_group_stem_values = []

                for stem in stem_labels_sorted:
                    dt_stem_data = results.get('per_dt_group_and_stem', {}).get(dt_group, {}).get(stem)
                    if dt_stem_data:
                        stem_avg = dt_stem_data['mean_si_snr']
                        dt_group_stem_values.append(stem_avg)
                        row += f" {stem_avg:11.4f}" + (" " * 5)
                    else:
                        row += (" " * 7) + f" {'---':<8}"

                if len(dt_group_stem_values) > 0:
                    dt_group_avg = np.mean(dt_group_stem_values)
                    row += f" {dt_group_avg:11.4f}"
                elif dt_group in results['per_dt_group']:
                    dt_data = results['per_dt_group'][dt_group]
                    row += f" {dt_data['mean_si_snr']:11.4f}"
                else:
                    row += f" {'---':<12}"

                print(row)

        print("-" * 100)
        avg_row = f"{'Avg per stem':<20} {'Across all DT':<25}"
        for stem in stem_labels_sorted:
            if stem in results['per_source']:
                stem_avg = results['per_source'][stem]['mean_si_snr']
                avg_row += f" {stem_avg:11.4f}"
            else:
                avg_row += f" {'---':<12}"
        if results['overall']:
            avg_row += f" {results['overall']['mean_si_snr']:11.4f}"
        else:
            avg_row += f" {'---':<12}"
        print(avg_row)
        print("-" * 100)
        print("=" * 100)

    if 'fad_clap' in results:
        print(f"\nPer-source FAD-CLAP scores:")
        for label, score in sorted(results['fad_clap']['per_source'].items()):
            print(f"  {label:15s}: {score:7.4f}")

        if results['fad_clap']['overall'] is not None:
            print(f"\nOverall FAD-CLAP: {results['fad_clap']['overall']:.4f}")

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


def _share_msrbench_results_with_other_ranks(results_payload, rank):
    """
    Share MSRBench evaluation results so that every rank prints the summary.
    """
    def _callback(payload):
        if payload is None:
            return
        _print_msrbench_results(payload.get('results'), payload.get('results_path'))

    _broadcast_object_to_ranks(
        results_payload if rank == 0 else None,
        rank,
        callback=_callback
    )


def _combine_and_save_hf_datasets(output_dir: str, include_top_level: bool = False):
    """
    Load, concatenate, and save Hugging Face datasets present in output_dir.
    """
    all_batch_dirs = _collect_hf_batch_dirs(output_dir, include_top_level=include_top_level)

    if not all_batch_dirs:
        print("No batches found to combine!")
        return None

    print(f"Found {len(all_batch_dirs)} batches to combine")

    batch_datasets = []
    for batch_dir in tqdm(all_batch_dirs, desc="Loading batches"):
        try:
            batch_dataset = Dataset.load_from_disk(batch_dir)
            batch_datasets.append(batch_dataset)
        except Exception as e:
            print(f"Error loading batch {batch_dir}: {e}")

    if not batch_datasets:
        print("No valid batches to combine!")
        return None

    print("Concatenating datasets...")
    final_dataset = concatenate_datasets(batch_datasets)

    label_counts = {}
    for sample in tqdm(final_dataset, desc="Computing statistics"):
        label = sample.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1

    final_dataset_path = os.path.join(output_dir, "final_dataset")
    print(f"Saving final dataset with {len(final_dataset)} samples to {final_dataset_path}...")
    final_dataset.save_to_disk(final_dataset_path)

    dataset_info = {
        'path': final_dataset_path,
        'label_counts': label_counts,
        'num_samples': len(final_dataset)
    }

    _print_label_distribution(label_counts, len(final_dataset), final_dataset_path)

    return dataset_info


def run_msrbench_evaluation(
        configs,
        msr_eval_cfg,
        dataset_path_msr_bench,
        logger,
        rank
):
    """
    Run MSRBench evaluation using the existing evaluation utilities.
    Supports multi-GPU evaluation with proper sharding across ranks.
    """
    if not msr_eval_cfg or not msr_eval_cfg.get('enabled', False):
        return

    # Get world_size for multi-GPU evaluation
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        print(f"[Rank {rank}] Starting MSRBench evaluation with {world_size} GPUs")
    else:
        # Try to get from environment
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
        if world_size > 1:
            print(f"[Rank {rank}] Detected {world_size} GPUs from environment")
            # print(f"[Rank {rank}] Detected {world_size} GPUs from environment, but distributed not initialized")
            # print(f"[Rank {rank}] Running evaluation on single GPU (rank {rank})")
            # world_size = 1
        else:
            print(f"[Rank {rank}] Starting MSRBench evaluation on single GPU")

    print("\n" + "=" * 80)
    print("Starting MSRBench evaluation...")
    print("=" * 80)

    try:
        from src import evaluate as evaluate_module
        from src.data.msr_bench import MultiAudioFullSongDataset
    except SystemExit as exc:
        raise RuntimeError(
            "Failed to import MSRBench evaluation helpers. "
            "Please ensure the 'transformers' package is installed."
        ) from exc

    eval_mode = (msr_eval_cfg.get('mode') or 'full').lower()
    subset_modes = {'validation_subset', 'validation', 'val', 'val500'}
    use_subset = eval_mode in subset_modes

    eval_ckpt_path = msr_eval_cfg.get('checkpoint_path')
    eval_wandb_id = msr_eval_cfg.get('wandb_id')
    eval_config_name = msr_eval_cfg.get('config_name')

    if eval_ckpt_path is None:
        if eval_wandb_id:
            eval_ckpt_path = os.path.join(CKPT_PATH, eval_config_name, eval_wandb_id, "last.ckpt")
        else:
            raise Exception()

    if eval_wandb_id and not os.path.exists(eval_ckpt_path):
        print(f"Checkpoint {eval_ckpt_path} not found locally. Downloading from wandb...")
        wandb_project = msr_eval_cfg.get('wandb_project', 'MSR_Separation')
        wandb_entity = msr_eval_cfg.get('wandb_entity', 'something_with_audio')
        eval_ckpt_path = download_checkpoint_from_wandb(
            wandb_id=eval_wandb_id,
            checkpoint_path=eval_ckpt_path,
            project=wandb_project,
            entity=wandb_entity,
            artifact_name=f"checkpoint-{eval_wandb_id}",
        )

    if not os.path.exists(eval_ckpt_path):
        raise FileNotFoundError(f"MSRBench evaluation checkpoint not found: {eval_ckpt_path}")

    print(f"Using checkpoint for MSRBench evaluation: {eval_ckpt_path}")

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_module = initialize_config(configs['lightning_module'])
    eval_module = eval_module.to(eval_device)

    init_new_stems_from_other = False  # configs.get('init_new_stems_from_other', False)
    add_noise_to_new_stems = False  # configs.get('add_noise_to_new_stems', False)

    full_ckpt = load_ckpt(
        eval_ckpt_path,
        eval_module.model,
        False,
        init_new_stems_from_other=init_new_stems_from_other,
        add_noise_to_new_stems=add_noise_to_new_stems,
    )

    # CRITICAL: Restore columns from checkpoint to ensure correct indexing
    restore_columns_from_checkpoint(eval_module, full_ckpt, "MSRBench evaluation")

    if hasattr(eval_module, 'use_ema') and eval_module.use_ema and getattr(eval_module, 'ema', None) is not None:
        if 'ema_state_dict' in full_ckpt:
            eval_module.ema.load_state_dict(full_ckpt['ema_state_dict'])
            print("Loaded EMA state from checkpoint for MSRBench evaluation.")
        print("Applying EMA weights for MSRBench evaluation...")
        eval_module.ema.apply()

    eval_model = eval_module.model
    eval_model.eval()

    full_split = msr_eval_cfg.get('full_split', 'train')
    validation_split = msr_eval_cfg.get('validation_split', full_split)
    dataset_split = validation_split if use_subset else full_split
    dataset_sr = getattr(eval_module, 'sampling_rate', 48000)

    print(f"Loading MSRBench split '{dataset_split}' from {dataset_path_msr_bench}")
    msr_dataset = MultiAudioFullSongDataset(
        split=dataset_split,
        columns=None,
        sr=dataset_sr,
        mono=False,
        root=dataset_path_msr_bench,
    )

    total_files = len(msr_dataset)
    shuffle_samples = msr_eval_cfg.get('shuffle_full_samples', True)
    subset_info = ""

    if use_subset:
        subset_limit = msr_eval_cfg.get('validation_sample_limit', 500)
        if subset_limit is None or subset_limit <= 0:
            subset_limit = 500
        subset_limit = min(subset_limit, total_files)
        subset_seed = msr_eval_cfg.get('validation_seed')
        if subset_seed is None:
            subset_seed = getattr(eval_module, 'msr_bench_seed', 42)
        rng = random.Random(subset_seed)
        all_indices = list(range(total_files))
        rng.shuffle(all_indices)
        selected_indices = sorted(all_indices[:subset_limit])
        msr_dataset = Subset(msr_dataset, selected_indices)
        subset_info = f" (subset of {subset_limit} samples, seed={subset_seed})"
        shuffle_samples = False

    print(f"[Rank {rank}] MSRBench dataset ready: {len(msr_dataset)} samples{subset_info}")

    # Get precision from config
    precision = configs.get('train', {}).get('trainer', {}).get('args', {}).get('precision', '16-mixed')

    try:
        results = evaluate_module.evaluate_msr_bench(
            model=eval_model,
            lightning_module=eval_module,
            dataset=msr_dataset,
            device=eval_device,
            window_duration=msr_eval_cfg.get('window_duration', 10.0),
            sr=dataset_sr,
            max_samples=None,
            shuffle_samples=shuffle_samples,
            calculate_fad_clap=msr_eval_cfg.get('calculate_fad_clap', False),
            fad_clap_batch_size=msr_eval_cfg.get('fad_clap_batch_size', 16),
            precision=precision,
            rank=rank,
            world_size=world_size,
        )
    except Exception as e:
        print(f"[Rank {rank}] ERROR: MSRBench evaluation failed: {e}")
        raise

    # Only rank 0 should save results to disk
    if rank == 0:
        output_dir = msr_eval_cfg.get('output_dir')
        if output_dir is None:
            mode_suffix = 'validation_subset' if use_subset else 'full'
            output_dir = os.path.join(LOG_PATH, "msr_evaluation", f"msr_bench_eval_{mode_suffix}_{eval_wandb_id}")
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "msr_bench_results.json")
        with open(results_path, 'w') as fp:
            json.dump(results, fp, indent=2)
    else:
        results_path = None

    results_payload = {
        'results': results,
        'results_path': results_path
    } if rank == 0 else None

    # Print final results (same format as evaluate.py)
    if rank == 0:
        _print_msrbench_results(results, results_path)

        # Log to wandb (only rank 0)
        if logger is not None and hasattr(logger, 'experiment'):
            overall_metrics = results.get('overall', {})
            log_payload = {}
            if overall_metrics.get('mean_si_snr') is not None:
                log_payload['msr_bench/overall_si_snr'] = overall_metrics['mean_si_snr']
            if overall_metrics.get('mean_multi_mel_snr') is not None:
                log_payload['msr_bench/overall_multi_mel_snr'] = overall_metrics['mean_multi_mel_snr']
            fad_overall = results.get('fad_clap', {}).get('overall') if 'fad_clap' in results else None
            if fad_overall is not None:
                log_payload['msr_bench/overall_fad_clap'] = fad_overall
            if log_payload:
                logger.experiment.log(log_payload)

    _share_msrbench_results_with_other_ranks(results_payload, rank)

    # Wait for all ranks to finish before proceeding
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_msr_testset_inference(
        trainer,
        pl_model,
        configs,
        testset_cfg,
        rank=0
):
    """
    Run inference on MSR test set and save predictions stem-wise.
    
    Args:
        trainer: PyTorch Lightning trainer
        pl_model: Lightning module with loaded checkpoint
        configs: Configuration object
        testset_cfg: Configuration dict with keys:
            - testset_root: Root directory of test set
            - output_dir: Output directory for predictions (optional, derived from testset_root if not provided)
            - checkpoint_path: Path to checkpoint (if None, uses trainer's checkpoint)
            - batch_size: Batch size for inference
            - precision: 'fp32', 'fp16', or 'bf16' (default: 'fp32')
        rank: Current rank (for distributed inference)
    """
    from src.data.msr_testset import MSRTestSetDataset
    from pathlib import Path
    
    # Get configuration
    testset_root = testset_cfg.get('testset_root')
    output_dir = testset_cfg.get('output_dir', None)  # Optional, will be derived if not provided
    checkpoint_path = testset_cfg.get('checkpoint_path', None)
    batch_size = testset_cfg.get('batch_size', 2)
    precision = testset_cfg.get('precision', '16-mixed')
    
    if testset_root is None:
        raise ValueError("testset_root must be specified in testset_cfg")
    
    # Derive output_dir from testset_root if not provided
    if output_dir is None:
        testset_path = Path(testset_root)
        parent_dir = testset_path.parent
        
        # Find existing separated_v* folders and get the highest version number
        existing_versions = []
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"{testset_path.stem}_separated_v"):
                    # Extract version number from "separated_v{N}"
                    try:
                        version_num = int(item.name.split(f"{testset_path.stem}_separated_v")[1])
                        existing_versions.append(version_num)
                    except (ValueError, IndexError):
                        # Skip if format doesn't match
                        continue
        
        # Determine next version number
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
        
        output_dir = str(parent_dir / f"{testset_path.stem}_separated_v{next_version}")
        
        if rank == 0:
            print(f"Derived output_dir from testset_root: {output_dir}")
    
    # Load checkpoint if specified
    if checkpoint_path is not None:
        if rank == 0:
            print(f"Loading checkpoint from: {checkpoint_path}")
        full_ckpt = load_ckpt(checkpoint_path, pl_model)
        
        # CRITICAL: Restore columns from checkpoint to ensure correct indexing
        restore_columns_from_checkpoint(pl_model, full_ckpt, "Test set inference")
        
        # Load EMA state if present
        if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
            if 'ema_state_dict' in full_ckpt:
                pl_model.ema.load_state_dict(full_ckpt['ema_state_dict'])
                if rank == 0:
                    print("Loaded EMA state from checkpoint")
            else:
                if rank == 0:
                    print("Warning: EMA enabled but no EMA state found in checkpoint.")
        # Apply EMA weights if enabled
        if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
            if rank == 0:
                print("Applying EMA weights for inference...")
            pl_model.ema.apply()
    
    # Set model to eval mode
    pl_model.eval()
    model = pl_model.model
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if rank == 0:
            print(f"Moving model to GPU: {device}")
        pl_model = pl_model.to(device)
        model = model.to(device)
    else:
        device = torch.device("cpu")
        if rank == 0:
            print("WARNING: CUDA not available, using CPU for inference")
    
    if rank == 0:
        print(f"Using device: {device} for inference")
        print(f"Precision: {precision}")
    
    # Setup precision
    use_autocast = False
    autocast_dtype = None
    if precision == '16-mixed':
        use_autocast = True
        autocast_dtype = torch.float16
    elif precision == 'bf16':
        use_autocast = True
        autocast_dtype = torch.bfloat16
    elif precision == 'fp32':
        use_autocast = False
    else:
        if rank == 0:
            print(f"Warning: Unknown precision '{precision}', using FP32")
    
    # Load dataset
    if rank == 0:
        print(f"Loading MSRTestSetDataset from: {testset_root}")
    dataset = MSRTestSetDataset(root=testset_root)
    collate_fn = dataset.collate_fn
    
    # Get columns from model
    columns = pl_model.columns if hasattr(pl_model, 'columns') else None
    if columns is None:
        if rank == 0:
            print("Warning: Could not get columns from model, using default")
        columns = ['vocals', 'bass', 'drums', 'other']
    
    if rank == 0:
        print(f"Model columns: {columns}")
    
    # Create reverse mapping from column name to index
    column_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid issues with distributed
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create output directory structure (stem-wise folders)
    # Map folder names back from column names
    COLUMN_TO_FOLDER = {v: k for k, v in MSRTestSetDataset.FOLDER_TO_COLUMN.items()}
    
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create stem folders
    stem_folders = {}
    for column in columns:
        if column == 'other':
            pass
        else:
            # Get folder name from column name
            folder_name = COLUMN_TO_FOLDER.get(column, column.capitalize())
            stem_folder = output_base / folder_name
            stem_folder.mkdir(exist_ok=True)
            stem_folders[column] = stem_folder
            if rank == 0:
                print(f"Created output folder for {column}: {stem_folder}")
    
    # Run inference
    if rank == 0:
        print(f"\nStarting inference on {len(dataset)} samples...")
    
    num_processed = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"[Rank {rank}] Inference", disable=(rank != 0))):
            # Get mixture
            mixture = batch_data['mixture']['waveform'].to(device)  # [batch_size, 2, samples]
            clip_ids = batch_data['clip_id']
            source_labels = batch_data['source_label']
            
            # Run inference with correct precision
            input_dict = {'mixture': mixture}
            if use_autocast and autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    output_dict = model(input_dict)
            else:
                output_dict = model(input_dict)
            
            separated_sources = output_dict['waveform']  # [batch_size, n_sources, 2, samples]
            
            # Save predictions for each sample in batch
            batch_size_actual = mixture.shape[0]
            for i in range(batch_size_actual):
                clip_id = clip_ids[i]
                source_label = source_labels[i]  # The stem we want to separate
                
                # Find the index of the source label in columns
                if source_label not in column_to_idx:
                    if rank == 0:
                        print(f"Warning: Source label '{source_label}' not found in columns {columns}, skipping")
                    continue
                
                source_idx = column_to_idx[source_label]
                
                # Get the separated source for this stem
                separated_stem = separated_sources[i, source_idx]  # [2, samples]
                
                # Convert to numpy and ensure float32 precision
                separated_audio = separated_stem.cpu().numpy().astype(np.float32)  # [2, samples]
                
                # Save to the appropriate stem folder
                output_folder = stem_folders[source_label]
                output_path = output_folder / f"{clip_id}.wav"
                
                # Get sample rate
                sample_rate = dataset.sr
                
                # Save as WAV file (transpose from [2, T] to [T, 2] for soundfile)
                sf.write(str(output_path), separated_audio.T, sample_rate)
                
                num_processed += 1
    
    if rank == 0:
        print(f"\n✓ Inference complete! Processed {num_processed} samples")
        print(f"Predictions saved to: {output_dir}")
        print(f"Stem-wise folders:")
        for column, folder in stem_folders.items():
            num_files = len(list(folder.glob("*.wav")))
            print(f"  {column}: {folder} ({num_files} files)")
        
        # Set permissions so others can access the files
        print(f"\nSetting permissions on output directory: {output_dir}")
        import subprocess
        
        try:
            # Set directory permissions: a+rwx (read, write, execute for all)
            result = subprocess.run(
                ['find', output_dir, '-type', 'd', '-print0'],
                capture_output=True,
                check=True
            )
            if result.stdout:
                subprocess.run(
                    ['xargs', '-0', '-P', '8', '-n', '500', 'chmod', 'a+rwx'],
                    input=result.stdout,
                    check=True
                )
            
            # Set file permissions: a+rw (read, write for all)
            result = subprocess.run(
                ['find', output_dir, '-type', 'f', '-print0'],
                capture_output=True,
                check=True
            )
            if result.stdout:
                subprocess.run(
                    ['xargs', '-0', '-P', '8', '-n', '1000', 'chmod', 'a+rw'],
                    input=result.stdout,
                    check=True
                )
            
            print("✓ Permissions set successfully")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to set permissions: {e}")
        except Exception as e:
            print(f"Warning: Error setting permissions: {e}")
    
    # Wait for all ranks to finish
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return output_dir


def generate_inference_dataset_with_trainer(
        trainer,
        pl_model,
        data_module,
        configs,
        output_dir,
        num_samples=10000,
        num_epochs=5,
        save_interval=1000,
        split='train',
        seed=42,
        rank=0,
        world_size=1,
        reuse_existing_batches_only=False
):
    """
    Generate a HuggingFace dataset from model predictions on the training set.

    This function:
    1. Uses the trainer and correct precision (matching training)
    2. Works on multiple GPUs simultaneously without processing samples twice
    3. Calculates SI-SNR to verify precision handling
    4. Stores samples (mixture, clean source, prediction, etc.) in files
    5. Creates a HuggingFace dataset containing only paths to samples
    6. Saves periodically to avoid memory issues

    Args:
        trainer: PyTorch Lightning trainer
        pl_model: Lightning module (with model and augmentation)
        data_module: Data module containing the dataset
        configs: Configuration dict
        output_dir: Directory to save the dataset and audio files
        num_samples: Target number of samples to generate
        num_epochs: Number of epochs to iterate through dataset
        save_interval: Save dataset every N samples
        split: Dataset split to use ('train' or 'val')
        rank: Current process rank (for distributed training)
        world_size: Total number of processes (for distributed training)
        reuse_existing_batches_only: If True, skip generation and only concatenate
            pre-existing Hugging Face datasets found in output_dir

    Returns:
        Path to the final saved HuggingFace dataset
    """
    # Only rank 0 should create output directory structure
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        audio_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        base_temp_dir = os.path.join(output_dir, "_temp_batches")
        os.makedirs(base_temp_dir, exist_ok=True)

    # Wait for rank 0 to create directories
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    reuse_existing_batches_only = bool(reuse_existing_batches_only)
    if reuse_existing_batches_only:
        if rank == 0:
            print("\nReuse-existing-batches flag enabled. Skipping sample generation and concatenating existing HF datasets.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        dataset_info = None
        if rank == 0:
            dataset_info = _combine_and_save_hf_datasets(
                output_dir,
                include_top_level=True
            )

        _share_label_distribution_with_other_ranks(dataset_info, rank)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if dataset_info and rank == 0:
            return dataset_info.get('path')
        return None

    # Get dataset based on split
    if split == 'train':
        dataset = data_module.train_dataset
    elif split == 'val':
        dataset = data_module.val_dataset
    else:
        raise ValueError(f"Unknown split: {split}")

    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' is None")

    # Get columns and other config
    columns = pl_model.columns if hasattr(pl_model, 'columns') else None
    if columns is None:
        raise ValueError("pl_model.columns must be set")

    # Create mapping from column name to index based on pl_model.columns order
    # This ensures we index into separated_sources correctly, matching the model's output order
    column_to_idx = {col: idx for idx, col in enumerate(pl_model.columns)}

    sampling_rate = getattr(pl_model, 'sampling_rate', 48000)
    use_augmentation = getattr(pl_model, 'augment', False)

    if not use_augmentation:
        print("WARNING: Augmentation is disabled. Setting augment=True for dataset generation.")
        pl_model.augment = True
        use_augmentation = True

    # Get precision setting from trainer
    use_autocast = False
    autocast_dtype = None
    precision = "16-mixed"
    if hasattr(trainer, 'precision'):
        precision = trainer.precision
        if precision == "16-mixed":
            use_autocast = True
            autocast_dtype = torch.float16
            print(f"[Rank {rank}] Using FP16 mixed precision for inference (matching training precision)")
        elif precision == "bf16-mixed":
            use_autocast = True
            autocast_dtype = torch.bfloat16
            print(f"[Rank {rank}] Using BF16 mixed precision for inference (matching training precision)")
        elif precision in ["32-true", "32"]:
            use_autocast = False
            print(f"[Rank {rank}] Using FP32 precision for inference (matching training precision)")

    # Get model and set to eval mode
    model = pl_model.model
    
    # Unwrap DDP if present (for inference, we don't need DDP wrapper)
    if hasattr(model, 'module'):
        # Model is wrapped in DDP or DataParallel
        print(f"[Rank {rank}] Unwrapping model from DDP/DataParallel wrapper...")
        model = model.module
    
    model.eval()

    # Apply EMA if enabled
    if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
        print(f"[Rank {rank}] Applying EMA weights for inference...")
        pl_model.ema.apply()
        print(f"[Rank {rank}] EMA weights applied successfully")

    # Get device - prioritize model's current device, then trainer, then rank-based
    if torch.cuda.is_available():
        # First, check what device the model is currently on
        model_device = next(model.parameters()).device
        
        # Try to get device from trainer strategy (most reliable for DDP)
        if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'root_device'):
            device = trainer.strategy.root_device
        elif hasattr(trainer, 'devices') and trainer.devices:
            if isinstance(trainer.devices, list):
                # For multi-GPU, use the device corresponding to this rank
                device_idx = rank % len(trainer.devices) if len(trainer.devices) > 1 else trainer.devices[0]
                device = torch.device(f"cuda:{device_idx}")
            else:
                device = torch.device(f"cuda:{trainer.devices}")
        elif model_device.type == 'cuda':
            # Model is already on a GPU, use that device
            device = model_device
        else:
            # Fallback: use rank-based device assignment
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        
        # Move model to device if not already there
        if model_device != device:
            print(f"[Rank {rank}] Moving model from {model_device} to {device} for inference...")
            model = model.to(device)
            pl_model.model = model  # Update reference in pl_model
        else:
            print(f"[Rank {rank}] Model already on device {device}")
    else:
        device = torch.device("cpu")
        if next(model.parameters()).device != device:
            print(f"[Rank {rank}] Moving model to CPU for inference...")
            model = model.to(device)
            pl_model.model = model
        print(f"[Rank {rank}] WARNING: CUDA not available, using CPU for inference")
    
    print(f"[Rank {rank}] Using device: {device} for inference")

    # Get dataset length and create indices for this rank
    dataset_length = len(dataset)
    print(f"[Rank {rank}] Dataset length: {dataset_length}")

    # Calculate indices for this rank (distribute samples across GPUs)
    all_indices = list(range(dataset_length))
    # Shuffle deterministically based on rank
    rng = random.Random(seed + rank)  # Different seed per rank
    rng.shuffle(all_indices)
    # Distribute indices across ranks
    rank_indices = [all_indices[i] for i in range(rank, len(all_indices), world_size)]
    print(f"[Rank {rank}] Processing {len(rank_indices)} samples (out of {dataset_length} total)")

    # Collect samples
    samples_collected = 0
    current_batch = []
    batch_num = 0
    temp_dirs = []
    samples_saved = 0  # Tracks how many samples this rank has persisted
    all_si_snr_values = []
    run_id = uuid.uuid4().hex  # Ensures uniqueness across reruns

    def sanitize_label(label):
        """Return a filesystem-safe label string."""
        if label is None:
            return "unknown"
        label_str = str(label)
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", label_str)
        return sanitized or "unknown"

    def flush_current_batch(batch, batch_idx, saved_count):
        """Save current batch to disk and return updated counters."""
        if not batch:
            return batch_idx, saved_count

        batch_to_process = batch

        # Create batch directory with rank prefix to avoid conflicts
        batch_dir = os.path.join(
            output_dir, "_temp_batches", f"rank_{rank}_batch_{batch_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)
        audio_batch_dir = os.path.join(batch_dir, "audio_files")
        os.makedirs(audio_batch_dir, exist_ok=True)

        # Process samples: save audio files and create sample dicts
        processed_samples = []
        # Capture save_interval from outer scope
        for idx, sample_data_within_loop in enumerate(batch_to_process):
            label = sample_data_within_loop.get('label', 'unknown')
            safe_label = sanitize_label(label)
            dataset_index = sample_data_within_loop.get('dataset_index')
            if isinstance(dataset_index, int):
                dataset_index_str = f"{dataset_index:09d}"
            else:
                dataset_index_str = sanitize_label(dataset_index) if dataset_index is not None else "na"
            epoch_value = sample_data_within_loop.get('epoch', 0)
            sequence_num = saved_count + idx
            unique_id = (
                f"{run_id}_rank{rank}_epoch{epoch_value}_idx{dataset_index_str}_seq{sequence_num:010d}"
            )
            file_prefix = f"sample_{unique_id}"

            # Save audio files
            clean_path = os.path.join(audio_batch_dir, f"{file_prefix}_clean_{safe_label}.wav")
            aug_path = os.path.join(audio_batch_dir, f"{file_prefix}_aug_{safe_label}.wav")
            separated_path = os.path.join(audio_batch_dir, f"{file_prefix}_separated_{safe_label}.wav")
            mixture_path = os.path.join(audio_batch_dir, f"{file_prefix}_mixture.wav")

            # Save audio files (transpose from [2, T] to [T, 2] for soundfile)
            # Convert to float32 if needed (soundfile doesn't support float16)
            clean_stem = sample_data_within_loop['clean_stem'].astype(np.float32)
            aug_stem = sample_data_within_loop['augmented_stem'].astype(np.float32)
            separated_stem = sample_data_within_loop['separated_stem'].astype(np.float32)
            mixture = sample_data_within_loop['mixture'].astype(np.float32)
            
            sf.write(clean_path, clean_stem.T, sampling_rate)
            sf.write(aug_path, aug_stem.T, sampling_rate)
            sf.write(separated_path, separated_stem.T, sampling_rate)
            sf.write(mixture_path, mixture.T, sampling_rate)

            # Create sample dict with file paths (relative to batch_dir)
            processed_sample = {
                'clean_stem': clean_path,  # Absolute path
                'augmented_stem': aug_path,
                'separated_stem': separated_path,
                'mixture': mixture_path,
                'label': sample_data_within_loop['label'],
                'original_path': sample_data_within_loop.get('original_path', ''),
                'dataset_index': sample_data_within_loop['dataset_index'],
                'epoch': sample_data_within_loop['epoch'],
                'si_snr': sample_data_within_loop.get('si_snr', None),
                'offset_seconds': sample_data_within_loop.get('offset_seconds', 0.0),
                'segment_seconds': sample_data_within_loop.get('segment_seconds', 10.0),
                'dataset_name': sample_data_within_loop.get('dataset_name', 'unknown')
            }
            processed_samples.append(processed_sample)

        # Create and save dataset for this batch
        batch_dataset = Dataset.from_list(processed_samples)
        batch_dataset.save_to_disk(batch_dir)
        temp_dirs.append(batch_dir)

        print(f"[Rank {rank}] Saved batch {batch_idx} with {len(batch_to_process)} samples to {batch_dir}")
        batch_idx += 1
        saved_count += len(batch_to_process)

        return batch_idx, saved_count

    # ------------------------------------ function end -------------------------------------------

    # Process samples across epochs
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # Shuffle indices for this epoch (different per rank due to different seed)
        epoch_indices = rank_indices.copy()
        rng.shuffle(epoch_indices)

        for idx in tqdm(epoch_indices, desc=f"[Rank {rank}] Processing epoch {epoch + 1}", disable=(rank != 0)):
            try:
                # Set epoch for dataset to get different random offsets
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)

                # Load sample from dataset
                sample = dataset[idx]

                # Prepare batch data with augmentation
                waveform_length = None
                for column in columns:
                    if column in sample and sample[column].get('waveform') is not None:
                        waveform_length = sample[column]['waveform'].shape[-1]
                        break

                if waveform_length is None:
                    continue

                batch_data = {}
                original_presence = {}
                for column in columns:
                    if column in sample and sample[column].get('is_present', False):
                        batch_data[column] = {
                            "waveform": sample[column]['waveform'].unsqueeze(0),
                            "lengths": torch.tensor([sample[column]['waveform'].shape[-1]]),
                            "paths": [sample[column].get('path', '')],
                            "is_present": [sample[column].get('is_present', True)]
                        }
                        original_presence[column] = True
                    else:
                        empty_waveform = torch.zeros((1, 2, waveform_length))
                        batch_data[column] = {
                            "waveform": empty_waveform,
                            "lengths": torch.tensor([waveform_length]),
                            "paths": [""],
                            "is_present": [False]
                        }
                        original_presence[column] = False

                # Apply augmentation and create mixture
                batch_data = pl_model.create_mixtures(batch_data)

                # Extract mixture (keep on device for inference, will move to CPU later)
                mixture = batch_data['mixture'].squeeze(0)  # [2, T]

                # Run inference with correct precision
                with torch.no_grad():
                    mixture_batch = mixture.unsqueeze(0).to(device)  # [1, 2, T]
                    input_dict = {'mixture': mixture_batch}

                    # Add label_vector for FiLM conditioning if enabled
                    if hasattr(pl_model, 'use_film_conditioning') and pl_model.use_film_conditioning:
                        label_vector = pl_model.label_vectors.flatten().unsqueeze(0).to(device)
                        input_dict['label_vector'] = label_vector

                    # Use autocast if mixed precision was used during training
                    if use_autocast and autocast_dtype is not None:
                        with torch.cuda.amp.autocast(dtype=autocast_dtype):
                            output_dict = model(input_dict)
                    else:
                        output_dict = model(input_dict)
                    separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, 2, T]
                    separated_sources = separated_sources.cpu()

                # Create samples for each present source
                for column in columns:
                    is_present_batch = batch_data[column]['is_present'][0]
                    is_present_original = original_presence.get(column, False)

                    # Only include samples where the source is actually present
                    if not (is_present_batch and is_present_original):
                        continue

                    # Get the correct index for this column based on model's output order
                    source_idx = column_to_idx[column]

                    # Extract clean stem (ground truth, before augmentation)
                    clean_stem = batch_data[column]["waveform"].squeeze(0).cpu()  # [2, T]

                    # Extract augmented stem
                    if 'aug_waveform' in batch_data[column]:
                        aug_stem = batch_data[column]["aug_waveform"].squeeze(0).cpu()  # [2, T]
                    else:
                        aug_stem = clean_stem

                    # Extract separated stem (prediction) - already on CPU from earlier
                    separated_stem = separated_sources[source_idx]  # [2, T]

                    # Calculate SI-SNR to verify precision handling
                    clean_stem_tensor = clean_stem  # [2, T]
                    separated_stem_tensor = separated_stem  # [2, T]
                    si_snr_val = si_snr(separated_stem_tensor, clean_stem_tensor).mean().item()
                    all_si_snr_values.append(si_snr_val)

                    # Get original path
                    original_path = sample[column].get('path', '') if column in sample else ""

                    if column not in original_path:
                        print(f"[Rank {rank}] Warning: Possible mismatch between column '{column}' and original path '{original_path}'")

                    # Store sample data (convert to numpy, ensuring CPU)
                    mixture_cpu = mixture.cpu()
                    sample_data = {
                        'clean_stem': clean_stem.numpy(),  # Convert to numpy for saving
                        'augmented_stem': aug_stem.numpy(),
                        'separated_stem': separated_stem.numpy(),
                        'mixture': mixture_cpu.numpy(),
                        'label': column,
                        'original_path': original_path,
                        'dataset_index': idx,
                        'epoch': epoch,
                        # 'si_snr': si_snr_val,
                        'offset_seconds': sample.get('offset_seconds', 0.0),
                        'segment_seconds': sample.get('segment_seconds', 10.0),
                        'dataset_name': sample.get('dataset_name', 'unknown')
                    }
                    current_batch.append(sample_data)
                    samples_collected += 1

                    # Print average SI-SNR every 500 samples
                    if samples_collected % 500 == 0 and all_si_snr_values:
                        avg_si_snr = np.mean(all_si_snr_values)
                        std_si_snr = np.std(all_si_snr_values)
                        print(f"[Rank {rank}] After {samples_collected} samples: Average SI-SNR = {avg_si_snr:.4f} ± {std_si_snr:.4f} dB")

                    # Save batch if interval reached
                    if len(current_batch) >= save_interval:
                        batch_num, samples_saved = flush_current_batch(
                            current_batch, batch_num, samples_saved
                        )
                        current_batch = []

                    # Check if we've collected enough samples
                    if samples_collected >= num_samples:
                        break

                if samples_collected >= num_samples:
                    break

            except Exception as e:
                print(f"[Rank {rank}] Error processing sample {idx} in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if samples_collected >= num_samples:
            break

    # Save any remaining samples
    if current_batch:
        batch_num, samples_saved = flush_current_batch(
            current_batch, batch_num, samples_saved
        )
        current_batch = []

    print(f"[Rank {rank}] Collected {samples_collected} samples total in {len(temp_dirs)} batches")
    avg_si_snr = None
    std_si_snr = None
    if all_si_snr_values:
        avg_si_snr = np.mean(all_si_snr_values)
        std_si_snr = np.std(all_si_snr_values)
        print(f"[Rank {rank}] Average SI-SNR: {avg_si_snr:.4f} ± {std_si_snr:.4f} dB")
    
    # Save rank statistics to file for aggregation by rank 0
    stats_file = os.path.join(output_dir, "_temp_batches", f"rank_{rank}_stats.json")
    stats_data = {
        'rank': rank,
        'samples_collected': samples_collected,
        'num_batches': len(temp_dirs),
        'avg_si_snr': float(avg_si_snr) if avg_si_snr is not None else None,
        'std_si_snr': float(std_si_snr) if std_si_snr is not None else None,
        'num_si_snr_values': len(all_si_snr_values)
    }
    # Write stats file and ensure it's flushed to disk
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk
    
    print(f"[Rank {rank}] Stats file written: {stats_file}")

    # Wait for all ranks to finish
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only rank 0 combines all batches into final dataset
    if rank == 0:
        # Wait for all stats files with retries
        all_rank_stats = _wait_for_all_stats_files(output_dir, world_size)
        
        _print_rank_statistics_table(all_rank_stats)
        _share_rank_statistics_with_other_ranks(all_rank_stats, rank)
        
        print("\nCombining all batches from all ranks into final dataset...")
        dataset_info = _combine_and_save_hf_datasets(
            output_dir,
            include_top_level=reuse_existing_batches_only
        )

        _share_label_distribution_with_other_ranks(dataset_info, rank)

        if dataset_info:
            return dataset_info.get('path')
        return None
    else:
        _share_rank_statistics_with_other_ranks(None, rank)
        _share_label_distribution_with_other_ranks(None, rank)
        return None


def generate_inference_dataset_msrbench_with_trainer(
        trainer,
        pl_model,
        data_module,
        configs,
        output_dir,
        save_interval=1000,
        split='val',
        rank=0,
        world_size=1,
        reuse_existing_batches_only=False
):
    """
    Generate a HuggingFace dataset from model predictions on the MSRBench dataset.

    This function:
    1. Uses the trainer and correct precision (matching training)
    2. Works on multiple GPUs simultaneously without processing samples twice
    3. Calculates SI-SNR to verify precision handling
    4. Stores samples (mixture, clean source, prediction, etc.) in files
    5. Creates a HuggingFace dataset containing only paths to samples
    6. Saves periodically to avoid memory issues
    7. Processes all samples in the dataset (typically ~26000 samples, all 10 seconds long)

    Args:
        trainer: PyTorch Lightning trainer
        pl_model: Lightning module (with model and augmentation)
        data_module: Data module containing the dataset
        configs: Configuration dict
        output_dir: Directory to save the dataset and audio files
        save_interval: Save dataset every N samples
        split: Dataset split to use ('train' or 'val')
        rank: Current process rank (for distributed training)
        world_size: Total number of processes (for distributed training)
        reuse_existing_batches_only: If True, skip sample generation and only
            concatenate any Hugging Face datasets already present in output_dir

    Returns:
        Path to the final saved HuggingFace dataset
    """
    # Only rank 0 should create output directory structure
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        audio_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        base_temp_dir = os.path.join(output_dir, "_temp_batches")
        os.makedirs(base_temp_dir, exist_ok=True)

    # Wait for rank 0 to create directories
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    reuse_existing_batches_only = bool(reuse_existing_batches_only)
    if reuse_existing_batches_only:
        if rank == 0:
            print("\nReuse-existing-batches flag enabled for MSRBench generation. Concatenating existing HF datasets.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        dataset_info = None
        if rank == 0:
            dataset_info = _combine_and_save_hf_datasets(
                output_dir,
                include_top_level=True
            )

        _share_label_distribution_with_other_ranks(dataset_info, rank)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if dataset_info and rank == 0:
            return dataset_info.get('path')
        return None

    # Get dataset based on split
    if split == 'train':
        dataset = data_module.train_dataset
    elif split == 'val':
        dataset = data_module.val_dataset
    else:
        raise ValueError(f"Unknown split: {split}")

    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' is None")

    # Get columns and other config
    columns = pl_model.columns if hasattr(pl_model, 'columns') else None
    if columns is None:
        raise ValueError("pl_model.columns must be set")

    # Create mapping from column name to index based on pl_model.columns order
    # This ensures we index into separated_sources correctly, matching the model's output order
    column_to_idx = {col: idx for idx, col in enumerate(pl_model.columns)}

    sampling_rate = getattr(pl_model, 'sampling_rate', 48000)

    # Get precision setting from trainer
    use_autocast = False
    autocast_dtype = None
    precision = "16-mixed"
    if hasattr(trainer, 'precision'):
        precision = trainer.precision
        if precision == "16-mixed":
            use_autocast = True
            autocast_dtype = torch.float16
            print(f"[Rank {rank}] Using FP16 mixed precision for inference (matching training precision)")
        elif precision == "bf16-mixed":
            use_autocast = True
            autocast_dtype = torch.bfloat16
            print(f"[Rank {rank}] Using BF16 mixed precision for inference (matching training precision)")
        elif precision in ["32-true", "32"]:
            use_autocast = False
            print(f"[Rank {rank}] Using FP32 precision for inference (matching training precision)")

    # Get model and set to eval mode
    model = pl_model.model
    
    # Unwrap DDP if present (for inference, we don't need DDP wrapper)
    if hasattr(model, 'module'):
        # Model is wrapped in DDP or DataParallel
        print(f"[Rank {rank}] Unwrapping model from DDP/DataParallel wrapper...")
        model = model.module
    
    model.eval()

    # Apply EMA if enabled
    if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
        print(f"[Rank {rank}] Applying EMA weights for inference...")
        pl_model.ema.apply()
        print(f"[Rank {rank}] EMA weights applied successfully")

    # Get device - prioritize model's current device, then trainer, then rank-based
    if torch.cuda.is_available():
        # First, check what device the model is currently on
        model_device = next(model.parameters()).device
        
        # Try to get device from trainer strategy (most reliable for DDP)
        if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'root_device'):
            device = trainer.strategy.root_device
        elif hasattr(trainer, 'devices') and trainer.devices:
            if isinstance(trainer.devices, list):
                # For multi-GPU, use the device corresponding to this rank
                device_idx = rank % len(trainer.devices) if len(trainer.devices) > 1 else trainer.devices[0]
                device = torch.device(f"cuda:{device_idx}")
            else:
                device = torch.device(f"cuda:{trainer.devices}")
        elif model_device.type == 'cuda':
            # Model is already on a GPU, use that device
            device = model_device
        else:
            # Fallback: use rank-based device assignment
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        
        # Move model to device if not already there
        if model_device != device:
            print(f"[Rank {rank}] Moving model from {model_device} to {device} for inference...")
            model = model.to(device)
            pl_model.model = model  # Update reference in pl_model
        else:
            print(f"[Rank {rank}] Model already on device {device}")
    else:
        device = torch.device("cpu")
        if next(model.parameters()).device != device:
            print(f"[Rank {rank}] Moving model to CPU for inference...")
            model = model.to(device)
            pl_model.model = model
        print(f"[Rank {rank}] WARNING: CUDA not available, using CPU for inference")
    
    print(f"[Rank {rank}] Using device: {device} for inference")

    # Get dataset length and create indices for this rank
    dataset_length = len(dataset)
    print(f"[Rank {rank}] Dataset length: {dataset_length}")

    # Calculate indices for this rank (distribute samples across GPUs)
    # Simply distribute indices across ranks without shuffling
    all_indices = list(range(dataset_length))
    rank_indices = [all_indices[i] for i in range(rank, len(all_indices), world_size)]
    print(f"[Rank {rank}] Processing {len(rank_indices)} samples (out of {dataset_length} total)")

    # Collect samples
    samples_collected = 0
    current_batch = []
    batch_num = 0
    temp_dirs = []
    samples_saved = 0  # Tracks how many samples have been persisted by this rank
    all_si_snr_values = []
    run_id = uuid.uuid4().hex  # Ensures unique naming across reruns

    def sanitize_label(label):
        """Return a filesystem-safe label string."""
        if label is None:
            return "unknown"
        label_str = str(label)
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", label_str)
        return sanitized or "unknown"

    def flush_current_batch(batch, batch_idx, saved_count):
        """Save current batch to disk and return updated counters."""
        if not batch:
            return batch_idx, saved_count

        batch_to_process = batch

        # Create batch directory with rank prefix to avoid conflicts
        batch_dir = os.path.join(output_dir, "_temp_batches", f"rank_{rank}_batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        audio_batch_dir = os.path.join(batch_dir, "audio_files")
        os.makedirs(audio_batch_dir, exist_ok=True)

        # Process samples: save audio files and create sample dicts
        processed_samples = []
        for idx, sample_data_within_loop in enumerate(batch_to_process):
            label = sample_data_within_loop.get('label', 'unknown')
            safe_label = sanitize_label(label)
            dataset_index = sample_data_within_loop.get('dataset_index')
            if isinstance(dataset_index, int):
                dataset_index_str = f"{dataset_index:09d}"
            else:
                dataset_index_str = sanitize_label(dataset_index) if dataset_index is not None else "na"
            clip_id_value = sample_data_within_loop.get('clip_id', '')
            clip_id_str = sanitize_label(clip_id_value) if clip_id_value else "noclip"
            sequence_num = saved_count + idx
            unique_id = f"{run_id}_rank{rank}_idx{dataset_index_str}_clip{clip_id_str}_seq{sequence_num:010d}"
            file_prefix = f"sample_{unique_id}"

            # Save audio files
            clean_path = os.path.join(audio_batch_dir, f"{file_prefix}_clean_{safe_label}.wav")
            separated_path = os.path.join(audio_batch_dir, f"{file_prefix}_separated_{safe_label}.wav")
            mixture_path = os.path.join(audio_batch_dir, f"{file_prefix}_mixture.wav")

            # Save audio files (transpose from [C, T] to [T, C] for soundfile)
            clean_stem = sample_data_within_loop['clean_stem'].astype(np.float32)
            separated_stem = sample_data_within_loop['separated_stem'].astype(np.float32)
            mixture = sample_data_within_loop['mixture'].astype(np.float32)
            
            sf.write(clean_path, clean_stem.T, sampling_rate)
            sf.write(separated_path, separated_stem.T, sampling_rate)
            sf.write(mixture_path, mixture.T, sampling_rate)

            # Create sample dict with file paths (absolute paths)
            processed_sample = {
                'clean_stem': clean_path,  # Absolute path
                'separated_stem': separated_path,
                'mixture': mixture_path,
                'label': sample_data_within_loop['label'],
                'mixture_path': sample_data_within_loop.get('mixture_path', ''),
                'source_path': sample_data_within_loop.get('source_path', ''),
                'dataset_index': sample_data_within_loop['dataset_index'],
                'si_snr': sample_data_within_loop.get('si_snr', None),
                'clip_id': sample_data_within_loop.get('clip_id', ''),
                'family': sample_data_within_loop.get('family', ''),
                'duration_seconds': sample_data_within_loop.get('duration_seconds', 0.0),
                'dataset_name': sample_data_within_loop.get('dataset_name', 'msr_bench')
            }
            processed_samples.append(processed_sample)

        # Create and save dataset for this batch
        batch_dataset = Dataset.from_list(processed_samples)
        batch_dataset.save_to_disk(batch_dir)
        temp_dirs.append(batch_dir)

        print(f"[Rank {rank}] Saved batch {batch_idx} with {len(batch_to_process)} samples to {batch_dir}")
        batch_idx += 1
        saved_count += len(batch_to_process)

        return batch_idx, saved_count

    # Process all samples (no epochs needed - process each sample once)
    for idx in tqdm(rank_indices, desc=f"[Rank {rank}] Processing samples", disable=(rank != 0)):
        try:
            # Load sample from dataset
            sample = dataset[idx]

            # MSRBench structure:
            # - sample['mixture']: dict with 'waveform' [C, T], 'path', 'is_present'
            # - sample['source']: dict with 'waveform' [C, T], 'path', 'label', 'is_present'

            # Check if mixture and source are present
            if 'mixture' not in sample or not sample['mixture'].get('is_present', False):
                continue
            
            if 'source' not in sample or not sample['source'].get('is_present', False):
                continue
            
            # Get source label
            source_label = sample['source'].get('label', '')
            if not source_label or source_label not in columns:
                continue
            
            # Get the index of this source in the model output using the column_to_idx mapping
            # This ensures we use the correct index matching the model's output order
            source_idx = column_to_idx[source_label]
            
            # Extract mixture and source waveforms
            # All samples are exactly 10 seconds long, no trimming needed
            mixture = sample['mixture']['waveform']  # [C, T]
            source_gt = sample['source']['waveform']  # [C, T]

            # Run inference with correct precision
            with torch.no_grad():
                mixture_batch = mixture.unsqueeze(0).to(device)  # [1, C, T]
                input_dict = {'mixture': mixture_batch}

                # Add label_vector for FiLM conditioning if enabled
                if hasattr(pl_model, 'use_film_conditioning') and pl_model.use_film_conditioning:
                    label_vector = pl_model.label_vectors.flatten().unsqueeze(0).to(device)
                    input_dict['label_vector'] = label_vector

                # Use autocast if mixed precision was used during training
                if use_autocast and autocast_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        output_dict = model(input_dict)
                else:
                    output_dict = model(input_dict)
                separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, C, T]
                separated_sources = separated_sources.cpu()

            # Extract prediction for this source
            separated_stem = separated_sources[source_idx]  # [C, T]

            # Trim separated_stem to match source length (in case of padding)
            source_length = source_gt.shape[-1]
            if separated_stem.shape[-1] > source_length:
                separated_stem = separated_stem[:, :source_length]
            elif separated_stem.shape[-1] < source_length:
                # Pad if shorter (shouldn't happen, but handle it)
                pad_length = source_length - separated_stem.shape[-1]
                separated_stem = torch.nn.functional.pad(separated_stem, (0, pad_length), mode='constant', value=0.0)

            # Calculate SI-SNR to verify precision handling
            clean_stem_tensor = source_gt  # [C, T]
            separated_stem_tensor = separated_stem  # [C, T]
            si_snr_val = si_snr(separated_stem_tensor, clean_stem_tensor).mean().item()
            all_si_snr_values.append(si_snr_val)

            # Get original paths
            mixture_path = sample['mixture'].get('path', '')
            source_path = sample['source'].get('path', '')

            # Store sample data (convert to numpy, ensuring CPU)
            mixture_cpu = mixture.cpu()
            sample_data = {
                'clean_stem': source_gt.cpu().numpy(),  # Convert to numpy for saving
                'separated_stem': separated_stem.numpy(),
                'mixture': mixture_cpu.numpy(),
                'label': source_label,
                'mixture_path': mixture_path,
                'source_path': source_path,
                'dataset_index': idx,
                # 'si_snr': si_snr_val,
                'clip_id': sample.get('clip_id', ''),
                'family': sample.get('family', ''),
                'duration_seconds': sample.get('duration_seconds', 0.0),
                'dataset_name': 'msr_bench'
            }
            current_batch.append(sample_data)
            samples_collected += 1

            # Print average SI-SNR every 500 samples
            if samples_collected % 500 == 0 and all_si_snr_values:
                avg_si_snr = np.mean(all_si_snr_values)
                std_si_snr = np.std(all_si_snr_values)
                print(f"[Rank {rank}] After {samples_collected} samples: Average SI-SNR = {avg_si_snr:.4f} ± {std_si_snr:.4f} dB")

            # Save batch if interval reached
            if len(current_batch) >= save_interval:
                batch_num, samples_saved = flush_current_batch(current_batch, batch_num, samples_saved)
                current_batch = []

        except Exception as e:
                print(f"[Rank {rank}] Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save any remaining samples
    if current_batch:
        batch_num, samples_saved = flush_current_batch(current_batch, batch_num, samples_saved)
        current_batch = []

    print(f"[Rank {rank}] Collected {samples_collected} samples total in {len(temp_dirs)} batches")
    avg_si_snr = None
    std_si_snr = None
    if all_si_snr_values:
        avg_si_snr = np.mean(all_si_snr_values)
        std_si_snr = np.std(all_si_snr_values)
        print(f"[Rank {rank}] Average SI-SNR: {avg_si_snr:.4f} ± {std_si_snr:.4f} dB")
    
    # Save rank statistics to file for aggregation by rank 0
    stats_file = os.path.join(output_dir, "_temp_batches", f"rank_{rank}_stats.json")
    stats_data = {
        'rank': rank,
        'samples_collected': samples_collected,
        'num_batches': len(temp_dirs),
        'avg_si_snr': float(avg_si_snr) if avg_si_snr is not None else None,
        'std_si_snr': float(std_si_snr) if std_si_snr is not None else None,
        'num_si_snr_values': len(all_si_snr_values)
    }
    # Write stats file and ensure it's flushed to disk
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk
    
    print(f"[Rank {rank}] Stats file written: {stats_file}")

    # Wait for all ranks to finish
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only rank 0 combines all batches into final dataset
    if rank == 0:
        # Wait for all stats files with retries
        all_rank_stats = _wait_for_all_stats_files(output_dir, world_size)
        
        _print_rank_statistics_table(all_rank_stats)
        _share_rank_statistics_with_other_ranks(all_rank_stats, rank)
        
        print("\nCombining all batches from all ranks into final dataset...")
        dataset_info = _combine_and_save_hf_datasets(
            output_dir,
            include_top_level=reuse_existing_batches_only
        )
        _share_label_distribution_with_other_ranks(dataset_info, rank)
        if dataset_info:
            return dataset_info.get('path')
        return None
    else:
        _share_rank_statistics_with_other_ranks(None, rank)
        _share_label_distribution_with_other_ranks(None, rank)
        return None


@ex.command(prefix="wandb")
def get_wandb_logger(config, name=None, project="MSR_Separation", rank0_only=True, tags=[]):
    if project is None:
        project = "MSR_Separation"

    resume_wandb_id = config.get('resume_wandb_id', None)

    config_filename = pathlib.Path(config.get('config_yaml_file', None)).stem
    log_dir = os.path.join(LOG_PATH, config_filename)
    os.makedirs(log_dir, exist_ok=True)

    # Configure wandb settings to prevent hanging on finish
    # Use thread start method to avoid blocking
    import wandb
    settings = wandb.Settings(
        _disable_stats=True,  # Disable stats collection that might cause delays
        start_method="thread",  # Use thread-based start method
        # Note: console="off" removed to enable logging to wandb site
    )

    wandb_logger = WandbLogger(
        entity="something_with_audio",
        project=project,
        tags=tags,
        config=config,
        name=name,
        id=resume_wandb_id,
        dir=log_dir,
        resume="must" if resume_wandb_id else None,
        settings=settings
    )

    return wandb_logger


@ex.command
def main(
        _run,
        _config,
        _log,
        _rnd,
        _seed,
        rank=0
):
    rank = rank_zero_only.rank
    print("rank:", rank)

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    print("local_rank:", local_rank)

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    print("num_nodes:", num_nodes)
    if num_nodes > 1:
        print("Updating number of nodes:", num_nodes)
        _config['train']['trainer']['args']['num_nodes'] = num_nodes

    import socket
    hostname = socket.gethostname()
    print("hostname:", hostname)

    configs = DefaultMunch.fromDict(_config)

    # Get dataset paths from config (required, no defaults)
    if 'dataset_path_4_sources' not in _config:
        raise ValueError("dataset_path_4_sources must be specified in config file")
    if 'dataset_path_8_sources' not in _config:
        raise ValueError("dataset_path_8_sources must be specified in config file")
    if 'dataset_path_msr_bench' not in _config:
        raise ValueError("dataset_path_msr_bench must be specified in config file")

    dataset_path_4_sources = _config['dataset_path_4_sources']
    dataset_path_8_sources = _config['dataset_path_8_sources']
    dataset_path_msr_bench = _config['dataset_path_msr_bench']

    if hostname.startswith('n'):
        print("MUSICA cluster detected")

        # Update paths for MUSICA cluster (like config_updates.py does)
        MUSICA_DATA_PATH = "/data/<username>/"  # replace with username
        dataset_path_4_sources = os.path.join(MUSICA_DATA_PATH, "msr_separation/mss4s_musica")
        dataset_path_8_sources = os.path.join(MUSICA_DATA_PATH, "msr_separation/mss8s_musica_v2_other_is_present")
        dataset_path_msr_bench = os.path.join(MUSICA_DATA_PATH, "msr_separation/msr_bench")
        # dataset_path_msr_bench stays as is (no MUSICA-specific path for MSRBench)

        print(f"MUSICA cluster: Updated dataset_path_4_sources={dataset_path_4_sources}")
        print(f"MUSICA cluster: Updated dataset_path_8_sources={dataset_path_8_sources}")
        print(f"MUSICA cluster: dataset_path_msr_bench={dataset_path_msr_bench}")

        # Check if 4-stem or 8-stem dataset is used
        using_4_stem_dataset = len(configs['datamodule']['args']['train_dataloader']['dataset']['args']['columns']) == 4

        # Only rank 0 should perform data copying to avoid redundant operations
        if rank == 0:
            if using_4_stem_dataset:
                if not os.path.exists(MUSICA_SCRATCH_4_STEM):
                    lock = FileLock(os.path.join("/tmp/", f'copy_data_{hostname}.lock'))

                    with lock:
                        print("Copying 4-stem dataset to scratch...")
                        shutil.copytree(dataset_path_4_sources, MUSICA_SCRATCH_4_STEM, copy_function=shutil.copy)
                        print("Copy completed.")
                else:
                    print(f"MUSICA_SCRATCH_4_STEM path already exists: {MUSICA_SCRATCH_4_STEM}")

                # Use scratch path if it exists
                if os.path.exists(MUSICA_SCRATCH_4_STEM):
                    dataset_path_4_sources = MUSICA_SCRATCH_4_STEM
                    print(f"Using scratch path for 4-stem dataset: {dataset_path_4_sources}")

            else:
                if not os.path.exists(MUSICA_SCRATCH_8_STEM):
                    lock = FileLock(os.path.join("/tmp/", f'copy_data_{hostname}.lock'))

                    with lock:
                        print("Copying 8-stem dataset to scratch...")
                        shutil.copytree(dataset_path_8_sources, MUSICA_SCRATCH_8_STEM, copy_function=shutil.copy)
                        print("Copy completed.")
                else:
                    print(f"MUSICA_SCRATCH_8_STEM path already exists: {MUSICA_SCRATCH_8_STEM}")

                # Use scratch path if it exists
                if os.path.exists(MUSICA_SCRATCH_8_STEM):
                    dataset_path_8_sources = MUSICA_SCRATCH_8_STEM
                    print(f"Using scratch path for 8-stem dataset: {dataset_path_8_sources}")

        # Wait for rank 0 to finish copying before other ranks proceed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Non-rank-0 processes should also use scratch paths if they exist
        if rank != 0:
            if using_4_stem_dataset and os.path.exists(MUSICA_SCRATCH_4_STEM):
                dataset_path_4_sources = MUSICA_SCRATCH_4_STEM
            elif not using_4_stem_dataset and os.path.exists(MUSICA_SCRATCH_8_STEM):
                dataset_path_8_sources = MUSICA_SCRATCH_8_STEM

        # additionally set wandb cache and log paths to scratch
        os.environ["WANDB_CACHE_DIR"] = "/scratch/<username>/msr_separation/wandb/cache"
        os.environ["WANDB_DIR"] = "/scratch/<username>/msr_separation/wandb/runs"

    # Print final dataset paths
    print(f"Final dataset paths:")
    print(f"  dataset_path_4_sources = {dataset_path_4_sources}")
    print(f"  dataset_path_8_sources = {dataset_path_8_sources}")
    print(f"  dataset_path_msr_bench = {dataset_path_msr_bench}")

    # Broadcast dataset paths from rank 0 to all ranks before initializing datasets
    # This ensures all ranks use the same paths even if they're on different nodes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        print("Broadcasting dataset paths from rank 0 to all ranks...")
        # Broadcast dataset paths from rank 0
        data_paths_list = [dataset_path_4_sources, dataset_path_8_sources, dataset_path_msr_bench]
        torch.distributed.broadcast_object_list(data_paths_list, src=0)
        dataset_path_4_sources = data_paths_list[0]
        dataset_path_8_sources = data_paths_list[1]
        dataset_path_msr_bench = data_paths_list[2]
        if rank != 0:
            print(f"Rank {rank}: Received dataset paths from rank 0:")
            print(f"  dataset_path_4_sources={dataset_path_4_sources}")
            print(f"  dataset_path_8_sources={dataset_path_8_sources}")
            print(f"  dataset_path_msr_bench={dataset_path_msr_bench}")

    # Update config with the final dataset paths (after MUSICA detection and broadcasting)
    _config['dataset_path_4_sources'] = dataset_path_4_sources
    _config['dataset_path_8_sources'] = dataset_path_8_sources
    _config['dataset_path_msr_bench'] = dataset_path_msr_bench

    logger = None
    if rank == 0:
        logger = get_wandb_logger(_config)

    print("Config main: ")
    print(_config)

    # deterministic
    if configs['deterministic']:
        torch.use_deterministic_algorithms(True, warn_only=True)
        pl.seed_everything(configs['manual_seed'], workers=True)
        configs['train']['trainer']['args']['deterministic'] = True

    config_filename = pathlib.Path(configs['config_yaml_file']).stem

    ckpt_save_path = os.path.join(CKPT_PATH, config_filename)
    os.makedirs(ckpt_save_path, exist_ok=True)
    run_id = len([f for f in os.listdir(ckpt_save_path) if f.startswith('run_')])

    wandb_id = None

    if rank == 0:
        if logger is not None:
            wandb_id = logger.experiment.id
            if wandb_id is not None:
                os.makedirs(os.path.join(ckpt_save_path, f"{wandb_id}"), exist_ok=True)

    # Broadcast wandb_id from rank 0 to all ranks for multi-GPU training
    # This ensures all processes know the checkpoint directory path
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Convert wandb_id to a list for broadcasting (handles None case)
        wandb_id_list = [wandb_id] if wandb_id is not None else [None]
        torch.distributed.broadcast_object_list(wandb_id_list, src=0)
        wandb_id = wandb_id_list[0]
        if rank != 0 and wandb_id is not None:
            print(f"Rank {rank}: Received wandb_id from rank 0: {wandb_id}")

    # data module
    if configs['batch_size'] > 0:
        print(f"Use batchsize of {configs['batch_size']}")
        configs['datamodule']['args']['train_dataloader']['batch_size'] = configs['batch_size']
        # configs['datamodule']['args']['train_dataloader']['num_workers'] = configs['batch_size']  # why is num_workers = batchsize?
        if 'val_dataloader' in configs['datamodule']['args']:
            configs['datamodule']['args']['val_dataloader']['batch_size'] = configs['batch_size']
        # configs['datamodule']['args']['val_dataloader']['num_workers'] = configs['batch_size']

    # Inject dataset paths into dataset configs
    inject_dataset_paths_into_config(
        _config,
        dataset_path_4_sources=dataset_path_4_sources,
        dataset_path_8_sources=dataset_path_8_sources,
        dataset_path_msr_bench=dataset_path_msr_bench
    )
    configs = DefaultMunch.fromDict(_config)  # Recreate configs with injected paths

    print('Initialize data module')
    data_module = initialize_config(configs['datamodule'])

    # model
    print('Initialize lightning module')
    pl_model = initialize_config(configs['lightning_module'])

    # callbacks
    print('Initialize callbacks')
    callbacks_configs = configs['train']['callbacks']
    callbacks = []
    for callback_config in callbacks_configs:
        if callback_config['name'] == 'checkpoint': callback_config['args']['dirpath'] = os.path.join(ckpt_save_path,
                                                                                                      f"{wandb_id}") if wandb_id else os.path.join(
            ckpt_save_path, f"{run_id}")
        if callback_config['name'] == 'tqdm' and configs['tqdm_rate'] > 0:
            callback_config['args']['refresh_rate'] = configs['tqdm_rate']
        callback = initialize_config(callback_config)
        callbacks.append(callback)

    # trainer
    configs['train']['trainer']['args']['callbacks'] = callbacks
    configs['train']['trainer']['args']['logger'] = logger
    trainer = initialize_config(configs['train']['trainer'])

    print("Run ID: ", run_id)

    if rank == 0:
        logger.experiment.config.update({'run_id': run_id})

    # checkpoints path
    finetune_ckpt_path = None
    if configs['resume_wandb_id'] is not None:  # if resume_wandb_id is set
        resume_ckpt_name = configs['resume_ckpt_name'] if configs['resume_ckpt_name'] else 'last'
        resume_ckpt_name = resume_ckpt_name + '.ckpt' if resume_ckpt_name[:-5] != '.ckpt' else resume_ckpt_name
        ckpt_resume_path = os.path.join(CKPT_PATH, config_filename, configs['resume_wandb_id'], resume_ckpt_name)

        print(f'Resuming run with wandb_id {configs["resume_wandb_id"]}')
        print(f'Resume checkpoint: {ckpt_resume_path}')

        # Try to download from wandb if not found locally (only rank 0)
        if rank == 0:
            if not os.path.exists(ckpt_resume_path):
                print(f'Checkpoint not found locally, attempting to download from wandb...')
                try:
                    ckpt_resume_path = download_checkpoint_from_wandb(
                        wandb_id=configs['resume_wandb_id'],
                        checkpoint_path=ckpt_resume_path,
                        project="MSR_Separation",
                        entity="something_with_audio",
                        artifact_name=f"checkpoint-{configs['resume_wandb_id']}"
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_resume_path}. Error: {e}")

        # Wait for rank 0 to finish downloading before other ranks proceed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    elif configs['resume_ckpt_path']:  # if resume_ckpt_path is set but resume_wandb_id is not set
        ckpt_resume_path = configs['resume_ckpt_path']
        print(f'Resuming run.')
        print(f'Resume checkpoint: {ckpt_resume_path}')
    elif configs['finetune_wandb_id'] is not None:  # if finetune_wandb_id is set
        # Get finetune config filename from config, fallback to default if not set
        finetune_config_filename = configs.get('finetune_config_filename', 'roformer_4s-dataset_step_lr_scheduler')
        finetune_ckpt_path = os.path.join(CKPT_PATH, finetune_config_filename, configs['finetune_wandb_id'],
                                          'last.ckpt')
        print(f'Fine-tuning from checkpoint with wandb_id {configs["finetune_wandb_id"]}')
        print(f'Fine-tuning config filename: {finetune_config_filename}')
        print(f'Fine-tuning checkpoint: {finetune_ckpt_path}')

        # Try to download from wandb if not found locally (only rank 0)
        if rank == 0:
            if not os.path.exists(finetune_ckpt_path):
                print(f'Checkpoint not found locally, attempting to download from wandb...')
                try:
                    finetune_ckpt_path = download_checkpoint_from_wandb(
                        wandb_id=configs['finetune_wandb_id'],
                        checkpoint_path=finetune_ckpt_path,
                        project="MSR_Separation",
                        entity="something_with_audio"
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Fine-tuning checkpoint not found: {finetune_ckpt_path}. Error: {e}")

        # Wait for rank 0 to finish downloading before other ranks proceed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # For fine-tuning, we only load model weights, not optimizer states or epoch info
        # So we don't set ckpt_resume_path - it will be loaded manually below
        ckpt_resume_path = None
    else:
        ckpt_resume_path = None

    if configs['deterministic']: torch.use_deterministic_algorithms(True, warn_only=True)

    # ─── Enable Torch Flags ─────────────────────────────────────────────────────────
    if configs['train']['trainer'].get('torch_flags', False):
        print("⚡ Mixed precision enabled — enabling all attention backends (Flash, Mem-Efficient, Math)")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Load model weights for fine-tuning (without optimizer states or epoch info)
    # This must be done before compilation and channels_last conversion
    if finetune_ckpt_path is not None:
        print(f'Loading model weights from fine-tuning checkpoint (without optimizer states)...')
        init_new_stems_from_other = configs.get('init_new_stems_from_other', False)
        add_noise_to_new_stems = configs.get('add_noise_to_new_stems', False)
        full_ckpt = load_ckpt(
            finetune_ckpt_path,
            pl_model.model,
            configs['map_4stem_to_9stem'],
            init_new_stems_from_other=init_new_stems_from_other,
            add_noise_to_new_stems=add_noise_to_new_stems
        )
        
        # CRITICAL: Restore columns from checkpoint to ensure correct indexing
        # Note: For fine-tuning, we might want to keep config columns if they differ,
        # but we should at least warn if there's a mismatch
        restore_columns_from_checkpoint(pl_model, full_ckpt, "Fine-tuning")
        
        # Load EMA state if present (EMA is part of LightningModule, not the model)
        if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
            if 'ema_state_dict' in full_ckpt:
                pl_model.ema.load_state_dict(full_ckpt['ema_state_dict'])
                print("Loaded EMA state from checkpoint")
            else:
                print("Warning: EMA enabled but no EMA state found in checkpoint. EMA will use current model weights.")
        print('Model weights loaded successfully. Starting fresh training with new optimizer and epoch counter.')

    # ─── Channels Last ──────────────────────────────────────────────────────────────
    if configs['train']['trainer'].get('channels_last', False):
        pl_model = pl_model.to(memory_format=torch.channels_last)

    # ─── Precision ──────────────────────────────────────────────────────────────────
    # Already handled above via `precision_setting` and its effects

    # ─── Compile ────────────────────────────────────────────────────────────────────
    if configs['train']['trainer'].get('compile', False):
        try:
            print("🔧 Compiling model with torch.compile...")
            pl_model = torch.compile(pl_model, mode="default", fullgraph=False)
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.cuda.empty_cache()

    # Check evaluation/test-only modes
    msr_eval_cfg = configs.get('msr_bench_evaluation') or {}
    msr_eval_only = bool(msr_eval_cfg.get('only', False))
    msr_eval_enabled = bool(msr_eval_cfg.get('enabled', False))

    gen_inf_cfg = configs.get('generate_inference_dataset') or {}
    gen_inf_only = bool(gen_inf_cfg.get('only', False))
    gen_inf_enabled = bool(gen_inf_cfg.get('enabled', False))

    testset_cfg = configs.get('msr_testset_inference') or {}
    testset_inference_enabled = bool(testset_cfg.get('enabled', False))
    testset_inference_only = bool(testset_cfg.get('only', False))

    validate_and_upload_only = configs.get('validate_and_upload_only', False)

    # Handle validate_and_upload_only mode
    if validate_and_upload_only:
        print("\n" + "=" * 80)
        print("Validate and upload-only mode enabled.")
        print("=" * 80)

        # Get checkpoint info from msr_bench_evaluation config
        eval_wandb_id = msr_eval_cfg.get('wandb_id')
        eval_config_name = msr_eval_cfg.get('config_name', config_filename)

        if eval_wandb_id is None:
            raise ValueError("validate_and_upload_only mode requires msr_bench_evaluation.wandb_id to be set")

        # Construct checkpoint path
        validate_ckpt_path = os.path.join(CKPT_PATH, eval_config_name, eval_wandb_id, "last.ckpt")

        # Download checkpoint if not found locally (only rank 0)
        if rank == 0:
            if not os.path.exists(validate_ckpt_path):
                print(f"Checkpoint not found locally, attempting to download from wandb...")
                wandb_project = "MSR_Separation"
                wandb_entity = "something_with_audio"
                try:
                    validate_ckpt_path = download_checkpoint_from_wandb(
                        wandb_id=eval_wandb_id,
                        checkpoint_path=validate_ckpt_path,
                        project=wandb_project,
                        entity=wandb_entity,
                        artifact_name=f"checkpoint-{eval_wandb_id}",
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Validation checkpoint not found: {validate_ckpt_path}. Error: {e}")

        # Wait for rank 0 to finish downloading before other ranks proceed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not os.path.exists(validate_ckpt_path):
            raise FileNotFoundError(f"Validation checkpoint not found: {validate_ckpt_path}")

        print(f"Loading checkpoint for validation: {validate_ckpt_path}")

        # Load checkpoint manually (model weights and EMA)
        init_new_stems_from_other = False  # configs.get('init_new_stems_from_other', False)
        add_noise_to_new_stems = False  # configs.get('add_noise_to_new_stems', False)

        full_ckpt = load_ckpt(
            validate_ckpt_path,
            pl_model.model,
            False,
            init_new_stems_from_other=init_new_stems_from_other,
            add_noise_to_new_stems=add_noise_to_new_stems,
        )

        # CRITICAL: Restore columns from checkpoint to ensure correct indexing
        restore_columns_from_checkpoint(pl_model, full_ckpt, "Validation")

        # Load EMA state if present
        if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
            if 'ema_state_dict' in full_ckpt:
                pl_model.ema.load_state_dict(full_ckpt['ema_state_dict'])
                print("Loaded EMA state from checkpoint for validation.")
            print("Applying EMA weights for validation...")
            pl_model.ema.apply()

        print("Checkpoint loaded successfully.")

        # # Close existing logger if any and create a new one (not resuming)
        # if rank == 0 and logger is not None:
        #     try:
        #         logger.experiment.finish()
        #     except:
        #         pass

        # # Create a new wandb logger (not resuming)
        # if rank == 0:
        #     # Temporarily clear resume_wandb_id to create new run
        #     old_resume_wandb_id = _config.get('resume_wandb_id')
        #     _config['resume_wandb_id'] = None
        #     logger = get_wandb_logger(_config)
        #     wandb_id = logger.experiment.id if logger is not None else None
        #     # Restore original resume_wandb_id (though we won't use it)
        #     _config['resume_wandb_id'] = old_resume_wandb_id
        #
        #     if wandb_id is not None:
        #         print(f"Created new wandb run with ID: {wandb_id}")
        #         # Update trainer with new logger
        #         trainer.logger = logger
        # else:
        #     wandb_id = None

        # Broadcast wandb_id from rank 0 to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            wandb_id_list = [wandb_id] if wandb_id is not None else [None]
            torch.distributed.broadcast_object_list(wandb_id_list, src=0)
            wandb_id = wandb_id_list[0]
            if rank != 0 and wandb_id is not None:
                print(f"Rank {rank}: Received wandb_id from rank 0: {wandb_id}")

        # Run validation only
        print("\n" + "=" * 80)
        print("Running validation...")
        print("=" * 80)
        # trainer.validate(
        #     model=pl_model,
        #     datamodule=data_module,
        #     ckpt_path=None  # Don't load checkpoint again, we already loaded it manually
        # )

        # Upload checkpoint and samples to wandb
        if rank == 0 and logger is not None and wandb_id is not None:
            print("\n" + "=" * 80)
            print("Uploading checkpoint and samples to wandb...")
            print("=" * 80)
            device = next(pl_model.model.parameters()).device
            upload_checkpoint_and_samples_to_wandb(
                logger=logger,
                checkpoint_path=validate_ckpt_path,
                model=pl_model.model,
                lightning_module=pl_model,
                data_module=data_module,
                device=device,
                num_samples=30,
                config=_config,
                only_upload_samples=True
            )
        elif rank == 0 and logger is not None and wandb_id is None:
            print("WARNING: wandb_id is None, skipping checkpoint and audio sample upload")

        print("\n✓ Validate and upload completed successfully!")

        # Skip MSRBench evaluation and other post-training operations
        # We'll just finalize wandb and exit
        if msr_eval_enabled:
            print("Skipping MSRBench evaluation (validate_and_upload_only mode).")
        if gen_inf_enabled:
            print("Skipping inference dataset generation (validate_and_upload_only mode).")

    elif msr_eval_only:
        print("MSRBench evaluation-only mode enabled. Skipping training/testing.")
    elif gen_inf_only:
        print("Inference dataset generation-only mode enabled. Skipping training/testing.")
    elif testset_inference_only:
        print("MSR Test Set inference-only mode enabled. Skipping training/testing.")
    else:
        # Fit, evaluate, and save checkpoints.
        trainer.fit(
            model=pl_model,
            train_dataloaders=None,
            val_dataloaders=None,
            datamodule=data_module,
            ckpt_path=ckpt_resume_path
        )

        # # Run test after training completes
        # print("Training completed. Running test evaluation...")

        # Ensure wandb_id is broadcast to all ranks (distributed might be initialized by now)
        # This is a safety check in case distributed wasn't initialized earlier when we first tried to broadcast
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # All ranks participate in broadcast - rank 0 sends, others receive
            wandb_id_list = [wandb_id] if wandb_id is not None else [None]
            torch.distributed.broadcast_object_list(wandb_id_list, src=0)
            if wandb_id_list[0] is not None:
                old_wandb_id = wandb_id
                wandb_id = wandb_id_list[0]
                if rank != 0 and old_wandb_id != wandb_id:
                    print(f"Rank {rank}: Received wandb_id from rank 0 (late broadcast): {wandb_id}")

        # Upload checkpoint and audio samples to wandb (only on rank 0)
        if rank == 0 and logger is not None and wandb_id is not None:
            # Get device from model
            device = next(pl_model.model.parameters()).device
            upload_checkpoint_and_samples_to_wandb(
                logger=logger,
                checkpoint_path=os.path.join(ckpt_save_path, wandb_id, "last.ckpt"),
                model=pl_model.model,
                lightning_module=pl_model,
                data_module=data_module,
                device=device,
                num_samples=30,
                config=_config,
                only_upload_samples=False
            )
        elif rank == 0 and logger is not None and wandb_id is None:
            print("WARNING: wandb_id is None, skipping checkpoint and audio sample upload")

    if msr_eval_enabled and not validate_and_upload_only:
        run_msrbench_evaluation(
            configs=configs,
            msr_eval_cfg=msr_eval_cfg,
            dataset_path_msr_bench=dataset_path_msr_bench,
            logger=logger,
            rank=rank
        )

    # ─── Optional Inference Dataset Generation ──────────────────────────────────────
    if gen_inf_enabled and not validate_and_upload_only:
        if rank == 0:
            print("\n" + "=" * 80)
            print("Starting inference dataset generation...")
            print("=" * 80)

        # Get world_size from trainer or environment
        world_size = 1
        if hasattr(trainer, 'world_size'):
            world_size = trainer.world_size
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            # Try to get from environment
            world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))

        output_dir = gen_inf_cfg.get('output_dir')

        msr_bench = gen_inf_cfg.get('msr_bench', False)
        reuse_existing_batches_only = bool(gen_inf_cfg.get('reuse_existing_batches_only', False))
        if msr_bench:
            if output_dir is None:
                output_dir = os.path.join(LOG_PATH, config_filename, "msrbench_inference_dataset")

            dataset_path = generate_inference_dataset_msrbench_with_trainer(
                trainer=trainer,
                pl_model=pl_model,
                data_module=data_module,
                configs=configs,
                output_dir=output_dir,
                save_interval=gen_inf_cfg.get('save_interval', 1000),
                split=gen_inf_cfg.get('split', 'val'),
                rank=rank,
                world_size=world_size,
                reuse_existing_batches_only=reuse_existing_batches_only
            )

            if rank == 0 and dataset_path:
                print(f"\n✓ MSRBench Inference dataset generation complete!")
                print(f"  Dataset saved to: {dataset_path}")
                if logger is not None:
                    logger.experiment.log({"msrbench_inference_dataset/path": dataset_path})
        else:
            if output_dir is None:
                output_dir = os.path.join(LOG_PATH, config_filename, "inference_dataset")

            dataset_path = generate_inference_dataset_with_trainer(
                trainer=trainer,
                pl_model=pl_model,
                data_module=data_module,
                configs=configs,
                output_dir=output_dir,
                num_samples=gen_inf_cfg.get('num_samples', 10000),
                num_epochs=gen_inf_cfg.get('num_epochs', 1),
                save_interval=gen_inf_cfg.get('save_interval', 1000),
                split=gen_inf_cfg.get('split', 'train'),
                seed=gen_inf_cfg.get('seed', 42),
                rank=rank,
                world_size=world_size,
                reuse_existing_batches_only=reuse_existing_batches_only
            )

            if rank == 0 and dataset_path:
                print(f"\n✓ Inference dataset generation complete!")
                print(f"  Dataset saved to: {dataset_path}")
                if logger is not None:
                    logger.experiment.log({"inference_dataset/path": dataset_path})

    # ─── MSR Test Set Inference ───────────────────────────────────────────────────────
    if testset_inference_enabled and not validate_and_upload_only:
        if rank == 0:
            print("\n" + "=" * 80)
            print("Starting MSR test set inference...")
            print("=" * 80)

        testset_wandb_id = testset_cfg.get('wandb_id')
        
        # Get checkpoint path if specified
        checkpoint_path = os.path.join(CKPT_PATH, config_filename, testset_wandb_id, "last.ckpt")
        
        if checkpoint_path and not os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"Warning: Checkpoint not found at {checkpoint_path}, attempting to download from wandb...")

            if testset_wandb_id:
                wandb_project = "MSR_Separation"
                wandb_entity = "something_with_audio"
                try:
                    checkpoint_path = download_checkpoint_from_wandb(
                        wandb_id=testset_wandb_id,
                        checkpoint_path=checkpoint_path,
                        project=wandb_project,
                        entity=wandb_entity,
                        artifact_name=f"checkpoint-{testset_wandb_id}",
                    )
                    if rank == 0:
                        print(f"Downloaded checkpoint to: {checkpoint_path}")
                except Exception as e:
                    if rank == 0:
                        print(f"Failed to download checkpoint: {e}")
                    checkpoint_path = None
        
        # Update testset_cfg with checkpoint path
        if checkpoint_path:
            testset_cfg['checkpoint_path'] = checkpoint_path
        
        output_dir = run_msr_testset_inference(
            trainer=trainer,
            pl_model=pl_model,
            configs=configs,
            testset_cfg=testset_cfg,
            rank=rank
        )
        
        if rank == 0 and output_dir:
            print(f"\n✓ MSR test set inference complete!")
            print(f"  Predictions saved to: {output_dir}")
            if logger is not None:
                logger.experiment.log({"msr_testset_inference/output_dir": output_dir})
        
        if testset_inference_only:
            return

        return

    # finalize wandb run with timeout and error handling
    # Wait for uploads to complete before finishing
    if rank == 0 and logger is not None:
        try:
            import wandb
            import threading
            import time

            print("\n" + "=" * 80)
            print("Finalizing wandb run and waiting for uploads to complete...")
            print("=" * 80)

            # Wait for wandb to finish uploading artifacts and media
            # Artifacts and audio files are uploaded asynchronously, so we need to wait
            print("\nWaiting for wandb uploads to complete (checkpoint + audio samples)...")
            max_upload_wait = 300  # Maximum 5 minutes for uploads (adjust if needed)
            upload_check_interval = 5  # Check every 5 seconds
            elapsed_time = 0

            # Initial wait for uploads to start and make progress
            print(f"  Waiting {max_upload_wait} seconds for uploads to complete...")
            print(f"  (This ensures checkpoint artifact and audio files are uploaded)")

            while elapsed_time < max_upload_wait:
                try:
                    if wandb.run is not None:
                        time.sleep(upload_check_interval)
                        elapsed_time += upload_check_interval

                        # Print progress every 30 seconds
                        if elapsed_time % 30 == 0:
                            remaining = max_upload_wait - elapsed_time
                            print(f"  Waiting... {elapsed_time}/{max_upload_wait}s elapsed ({remaining}s remaining)")
                    else:
                        print("  wandb.run is None, proceeding...")
                        break
                except Exception as e:
                    print(f"  Note: Error during upload wait: {e}")
                    break

            if elapsed_time >= max_upload_wait:
                print(f"✓ Maximum upload wait time ({max_upload_wait}s) reached. Proceeding with finish...")
            else:
                print(f"✓ Upload wait completed ({elapsed_time} seconds)")

            # Additional short wait for final sync
            print("  Giving wandb additional 10 seconds for final sync...")
            time.sleep(10)

            # Now finish wandb run
            finished = threading.Event()
            finish_error = None
            finish_success = False

            def finish_wandb():
                """Finish wandb in a separate thread with timeout."""
                nonlocal finish_error, finish_success
                try:
                    # Try finishing through the logger first
                    logger.experiment.finish(exit_code=0)
                    finish_success = True
                except Exception as e:
                    finish_error = e
                finally:
                    finished.set()

            # Start finishing in a separate thread
            finish_thread = threading.Thread(target=finish_wandb, daemon=False)
            finish_thread.start()

            # Wait for finish with timeout (30 seconds - increased for uploads)
            if finished.wait(timeout=30):
                if finish_error is not None:
                    print(f"WARNING: Error during wandb.finish(): {finish_error}")
                    print("Training completed successfully, but wandb sync may be incomplete")
                elif finish_success:
                    print("✓ wandb run finalized successfully")
            else:
                print("WARNING: wandb.finish() timed out after 30 seconds")
                print("Force closing wandb connections...")

            # Force close wandb connections to prevent hanging
            # This allows the process to exit even if sync is still in progress
            try:
                if wandb.run is not None:
                    # Try to gracefully close the backend
                    try:
                        if hasattr(wandb.run, '_backend'):
                            wandb.run._backend._shutdown()
                    except:
                        pass
                    # Force finish without waiting for sync
                    try:
                        wandb.finish(exit_code=0)
                    except:
                        pass
            except Exception as e:
                print(f"Note: Error closing wandb backend: {e}")

            # Join the thread with a short timeout
            if finish_thread.is_alive():
                finish_thread.join(timeout=5)

            print("\nTraining completed. Exiting...")

        except Exception as e:
            print(f"ERROR: Failed to finalize wandb run: {e}")
            print("The training completed successfully, but wandb sync may be incomplete")
            # Try to force finish wandb even if there was an error
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=0)
            except:
                pass
        finally:
            # Final wait before exit to ensure everything is synced
            print("Waiting 5 seconds for final sync...")
            time.sleep(5)

            # Force exit using os._exit to bypass any cleanup that might hang
            # This is necessary on SLURM clusters where wandb might hang
            # os._exit(0) immediately terminates the process, bypassing cleanup
            print("Exiting process...")
            os._exit(0)


@ex.automain
def default_command():
    return main()
