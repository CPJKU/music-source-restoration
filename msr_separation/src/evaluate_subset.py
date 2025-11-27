"""
Evaluation script for source separation models.

This script evaluates trained models on test sets using SI-SNR and FAD-CLAP metrics.
It can be used to evaluate models on the test set after training is complete.
"""

import os
import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from scipy.linalg import sqrtm
import wandb
import random

warnings.filterwarnings("ignore")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    print("Error: The 'transformers' library is not installed.")
    print("Please install it to run FAD-CLAP calculations:")
    print("pip install transformers")
    exit(1)

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from src.utils import initialize_config, download_checkpoint_from_wandb
from src.training.lightningmodule.helper import load_ckpt
from src.data.msr_bench import MultiAudioFullSongDataset
from src.training.metrics.multi_mel_snr import multi_mel_snr, MultiMelSNRMetric


def init_wandb_run(args, default_name: str):
    """Initialize a Weights & Biases run for evaluation logging."""
    if isinstance(default_name, Path):
        default_name = default_name.stem
    run_name = args.wandb_run_name or f"{default_name}-{args.wandb_id or 'manual'}"
    config_dict = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("wandb_")
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=config_dict,
        mode=args.wandb_mode,
        settings=wandb.Settings(save_code=False),
    )


def log_wandb_metrics(metrics: dict, prefix: str = ""):
    """Log nested metric dictionary to wandb by flattening numeric entries."""
    if wandb.run is None:
        return

    flat_metrics = {}

    def recurse(obj, parent_key=""):
        if isinstance(obj, (int, float)):
            flat_metrics[parent_key] = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}/{key}" if parent_key else key
                recurse(value, new_key)

    recurse(metrics, prefix)
    if flat_metrics:
        wandb.log(flat_metrics)


def load_audio(file_path, sr=48000):
    """Load audio file and return as torch tensor."""
    try:
        wav, samplerate = sf.read(file_path)
        if samplerate != sr:
            # Resample if needed (simple approach - in practice you might want proper resampling)
            pass
        if wav.ndim > 1:
            wav = wav.T  # Convert to [channels, samples]
        else:
            wav = wav[np.newaxis, :]
        return torch.from_numpy(wav).float()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_clap_embeddings(file_paths, model, processor, device, batch_size=16):
    """Get CLAP embeddings for a list of audio files."""
    model.to(device)
    all_embeddings = []
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating embeddings", ncols=100, leave=False):
        batch_paths = file_paths[i:i+batch_size]
        audio_batch = []
        for path in batch_paths:
            try:
                wav, sr = sf.read(path)
                if wav.ndim == 2 and wav.shape[1] == 2:
                    audio_batch.append(wav[:, 0])  # Left channel
                    audio_batch.append(wav[:, 1])  # Right channel
                elif wav.ndim == 1:
                    audio_batch.append(wav)
                else:
                    continue
            except Exception:
                continue

        if not audio_batch:
            continue

        try:
            inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)
            
            all_embeddings.append(audio_features.cpu().numpy())
        except Exception:
            continue
            
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)


def get_clap_embeddings_from_arrays(audio_arrays, model, processor, device, batch_size=16):
    """Get CLAP embeddings from a list of numpy audio arrays."""
    model.to(device)
    all_embeddings = []
    
    for i in range(0, len(audio_arrays), batch_size):
        batch_arrays = audio_arrays[i:i+batch_size]
        audio_batch = []
        
        for audio in batch_arrays:
            # audio shape: [channels, samples] or [samples]
            if audio.ndim == 2 and audio.shape[0] == 2:  # Stereo [2, n_samples]
                # Use left channel for CLAP
                audio_batch.append(audio[0])  # [n_samples]
            elif audio.ndim == 1:  # Mono [n_samples]
                audio_batch.append(audio)
            else:
                # Handle other cases - flatten if needed
                audio_batch.append(audio.flatten())
        
        if not audio_batch:
            continue
        
        try:
            inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)
            
            all_embeddings.append(audio_features.cpu().numpy())
        except Exception as e:
            continue
    
    if not all_embeddings:
        return np.array([])
    
    return np.concatenate(all_embeddings, axis=0)


def calculate_frechet_distance(embeddings1, embeddings2):
    """Calculate Fréchet Distance between two sets of embeddings."""
    if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
        return None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return None

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fad_score


def select_msr_bench_subset(total_files: int, subset_size: int, seed: int) -> list[int]:
    """
    Select a deterministic subset of MSRBench samples, matching the training-time logic.

    Args:
        total_files: Total number of files in the dataset.
        subset_size: Number of files to select.
        seed: RNG seed for deterministic selection.

    Returns:
        Sorted list of selected indices.
    """
    if subset_size is None:
        raise ValueError("subset_size must be provided for subset selection")

    if total_files == 0:
        return []

    subset_size = min(int(subset_size), total_files)
    rng = random.Random(int(seed))
    all_indices = list(range(total_files))
    rng.shuffle(all_indices)
    selected = sorted(all_indices[:subset_size])
    return selected


def evaluate_model_on_dataset(model, dataloader, device, output_dir, n_sources=4, precision=None):
    """
    Evaluate model on a dataset and save separated audio files.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        output_dir: Directory to save separated audio files
        n_sources: Number of sources to separate
        precision: Precision setting from config (e.g., "16-mixed", "bf16-mixed", "32-true")
    
    Returns:
        List of (target_path, output_path) tuples for metric calculation
    """
    model.eval()
    model.to(device)

    # Determine precision setting from config (to match training precision)
    use_autocast = False
    autocast_dtype = None
    if precision:
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
            # Unknown precision, default to no autocast
            print(f"Unknown precision setting '{precision}', using FP32 for inference")
    else:
        print("No precision setting provided, using FP32 precision for inference")
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_pairs = []
    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    
    print(f"Evaluating model on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Prepare input
            mixture = batch_data['mixture'].to(device)  # [batch_size, 2, n_samples]
            target_sources = batch_data['target_sources']  # [batch_size, n_sources, 2, n_samples]
            
            # Run inference
            input_dict = {'mixture': mixture}
            # output_dict = model(input_dict)

            # Use autocast if mixed precision was used during training
            if use_autocast and autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    output_dict = model(input_dict)
            else:
                output_dict = model(input_dict)


            separated_sources = output_dict['waveform']  # [batch_size, n_sources, 2, n_samples]
            
            # Save separated audio files and calculate SI-SNR and Multi-Mel-SNR
            batch_size = mixture.shape[0]
            for i in range(batch_size):
                # Get file names (assuming they exist in batch_data)
                if 'file_paths' in batch_data:
                    base_name = Path(batch_data['file_paths'][i]).stem
                else:
                    base_name = f"sample_{batch_idx}_{i}"
                
                # Save each separated source
                for source_idx in range(n_sources):
                    # Get target and separated audio for this source
                    target_audio = target_sources[i, source_idx].cpu().numpy().T  # [n_samples, 2]
                    separated_audio = separated_sources[i, source_idx].cpu().numpy().T  # [n_samples, 2]
                    
                    # Save files
                    target_path = os.path.join(output_dir, f"{base_name}_source_{source_idx}_target.wav")
                    output_path = os.path.join(output_dir, f"{base_name}_source_{source_idx}_separated.wav")
                    
                    sf.write(target_path, target_audio, 48000)
                    sf.write(output_path, separated_audio, 48000)
                    
                    # Calculate SI-SNR for this source
                    target_tensor = torch.from_numpy(target_audio.T).float()  # [2, n_samples]
                    separated_tensor = torch.from_numpy(separated_audio.T).float()  # [2, n_samples]
                    
                    si_snr_val = sisnr_calculator(separated_tensor, target_tensor)
                    
                    # Calculate Multi-Mel-SNR for this source (average over channels)
                    # Convert to mono for multi_mel_snr (it expects 1D array)
                    target_mono = target_tensor.mean(dim=0).cpu().numpy()  # [n_samples]
                    separated_mono = separated_tensor.mean(dim=0).cpu().numpy()  # [n_samples]
                    multi_mel_snr_val = multi_mel_snr(target_mono, separated_mono, sr=48000)
                    
                    file_pairs.append((target_path, output_path, si_snr_val.item(), multi_mel_snr_val))
    
    return file_pairs


def calculate_metrics(file_pairs, batch_size=16):
    """Calculate SI-SNR, Multi-Mel-SNR, and FAD-CLAP metrics from file pairs."""
    print("--- Calculating SI-SNR and Multi-Mel-SNR for each pair ---")
    
    all_target_paths = []
    all_output_paths = []
    si_snr_values = []
    multi_mel_snr_values = []
    
    for pair in file_pairs:
        if len(pair) == 4:
            # New format with multi_mel_snr
            target_path, output_path, si_snr_val, multi_mel_snr_val = pair
        else:
            # Old format (backward compatibility)
            target_path, output_path, si_snr_val = pair
            multi_mel_snr_val = None
        
        print(f"{target_path}|{output_path}|SI-SNR:{si_snr_val:.4f}", end="")
        if multi_mel_snr_val is not None:
            print(f"|Multi-Mel-SNR:{multi_mel_snr_val:.4f}")
        else:
            print()
        
        all_target_paths.append(target_path)
        all_output_paths.append(output_path)
        si_snr_values.append(si_snr_val)
        if multi_mel_snr_val is not None:
            multi_mel_snr_values.append(multi_mel_snr_val)
    
    # Calculate average SI-SNR
    avg_si_snr = np.mean(si_snr_values)
    print(f"\nAverage SI-SNR: {avg_si_snr:.4f} dB")
    
    # Calculate average Multi-Mel-SNR
    if multi_mel_snr_values:
        avg_multi_mel_snr = np.mean(multi_mel_snr_values)
        print(f"Average Multi-Mel-SNR: {avg_multi_mel_snr:.4f} dB")
    else:
        avg_multi_mel_snr = float('inf')
    
    print("\n--- Calculating FAD-CLAP for all target vs. all output files ---")
    if not all_target_paths:
        print("No valid file pairs found to calculate FAD-CLAP.")
        return {"si_snr": avg_si_snr, "multi_mel_snr": avg_multi_mel_snr, "fad_clap": float('inf')}
    
    try:
        print("Loading CLAP model...")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model.eval()
        print("CLAP model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load CLAP model. Please check internet connection. Error: {e}")
        return {"si_snr": avg_si_snr, "multi_mel_snr": avg_multi_mel_snr, "fad_clap": float('inf')}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nCalculating embeddings for all target files...")
    target_embeddings = get_clap_embeddings(all_target_paths, clap_model, clap_processor, device, batch_size)

    print("Calculating embeddings for all output files...")
    output_embeddings = get_clap_embeddings(all_output_paths, clap_model, clap_processor, device, batch_size)

    if target_embeddings.size > 0 and output_embeddings.size > 0:
        print("Calculating Frechet Audio Distance (FAD)...")
        fad_score = calculate_frechet_distance(target_embeddings, output_embeddings)
        if fad_score is not None:
            print(f"\nOverall FAD-CLAP Score: {fad_score:.4f}")
            return {"si_snr": avg_si_snr, "multi_mel_snr": avg_multi_mel_snr, "fad_clap": fad_score}
        else:
            print("\nCould not calculate FAD-CLAP score.")
            return {"si_snr": avg_si_snr, "multi_mel_snr": avg_multi_mel_snr, "fad_clap": float('inf')}
    else:
        print("\nCould not calculate FAD-CLAP due to issues with embedding generation.")
        return {"si_snr": avg_si_snr, "multi_mel_snr": avg_multi_mel_snr, "fad_clap": float('inf')}


def evaluate_msr_bench(
    model,
    lightning_module,
    dataset,
    device,
    window_duration=10.0,
    sr=48000,
    max_samples=None,
    shuffle_samples=True,
    calculate_fad_clap=False,
    fad_clap_batch_size=16,
    subset_indices=None,
    subset_metadata=None,
    precision=None
):
    """
    Evaluate model on MSRBench dataset using non-overlapping 10-second windows.
    
    Args:
        model: Trained model (from lightning_module.model)
        lightning_module: Lightning module (to access args.columns)
        dataset: MSRBench dataset (MultiAudioFullSongDataset)
        device: Device to run inference on
        window_duration: Duration of each window in seconds (default: 10.0)
        sr: Sampling rate (default: 48000)
        max_samples: Maximum number of samples to evaluate (None = all)
        shuffle_samples: If True, randomize sample order. If False, use original order.
                        Default: True (to avoid bias if dataset is ordered by class)
        calculate_fad_clap: If True, calculate FAD-CLAP metric (default: False)
        fad_clap_batch_size: Batch size for FAD-CLAP embedding calculation (default: 16)
    
    Returns:
        Dictionary with SI-SNR scores per source label and overall, and FAD-CLAP if enabled
    """
    
    model.eval()
    model.to(device)
    
    # Determine precision setting from config (to match training precision)
    use_autocast = False
    autocast_dtype = None
    if precision:
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
            # Unknown precision, default to no autocast
            print(f"Unknown precision setting '{precision}', using FP32 for inference")
    else:
        print("No precision setting provided, using FP32 precision for inference")
    
    # Get columns from lightning module
    columns = lightning_module.columns if hasattr(lightning_module, 'columns') else None
    if columns is None:
        raise ValueError("lightning_module.columns must be set to determine source index")
    
    # Window size in samples
    window_samples = int(window_duration * sr)
    
    # SI-SNR calculator
    si_snr_values = {label: [] for label in columns}
    all_si_snr_values = []
    
    # Multi-Mel-SNR calculator
    multi_mel_snr_values = {label: [] for label in columns}
    all_multi_mel_snr_values = []
    
    # Track SI-SNR by DT group for efficient grouping
    # DT groups: DT0 (Reference), DT1-DT4 (Analog/Acoustic), DT5-DT8 (Traditional Codecs), DT9-DT12 (Neural Codecs)
    dt_group_names = {
        'DT0': 'Reference',
        'DT1-DT4': 'Analog/Acoustic',
        'DT5-DT8': 'Traditional Codecs',
        'DT9-DT12': 'Neural Codecs'
    }
    si_snr_by_dt_group = {group: [] for group in dt_group_names.keys()}
    
    # Track SI-SNR by DT group AND stem (cross-tabulation)
    si_snr_by_dt_group_and_stem = {group: {label: [] for label in columns} for group in dt_group_names.keys()}
    
    # Track Multi-Mel-SNR by DT group
    multi_mel_snr_by_dt_group = {group: [] for group in dt_group_names.keys()}
    
    # Track Multi-Mel-SNR by DT group AND stem (cross-tabulation)
    multi_mel_snr_by_dt_group_and_stem = {group: {label: [] for label in columns} for group in dt_group_names.keys()}
    
    # Track number of files evaluated per stem
    files_evaluated = {label: 0 for label in columns}
    
    # Track DT group distribution (per sample, not per window)
    dt_group_distribution = {group: 0 for group in dt_group_names.keys()}
    total_samples_accounted = 0
    
    def extract_dt_group(clip_id: str) -> str:
        """Extract DT group from clip_id (e.g., '0_DT6' -> 'DT5-DT8')."""
        if not clip_id:
            return 'DT0'
        
        # Extract DT number (e.g., "0_DT6" -> "DT6" -> 6)
        parts = clip_id.split('_')
        dt_str = None
        for part in parts:
            if part.startswith('DT'):
                dt_str = part
                break
        
        if dt_str is None:
            return 'DT0'
        
        # Extract number
        try:
            dt_num = int(dt_str[2:])  # Remove "DT" prefix
        except ValueError:
            return 'DT0'
        
        # Map to group
        if dt_num == 0:
            return 'DT0'
        elif 1 <= dt_num <= 4:
            return 'DT1-DT4'
        elif 5 <= dt_num <= 8:
            return 'DT5-DT8'
        elif 9 <= dt_num <= 12:
            return 'DT9-DT12'
        else:
            return 'DT0'  # Fallback
    
    # FAD-CLAP storage
    fad_clap_enabled = calculate_fad_clap
    if fad_clap_enabled:
        try:
            from transformers import ClapModel, ClapProcessor
            clap_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Loading CLAP model for FAD-CLAP calculation...")
            clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            clap_model.eval()
            clap_model.to(clap_device)
            print("CLAP model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load CLAP model. FAD-CLAP will be disabled. Error: {e}")
            fad_clap_enabled = False
            clap_model = None
            clap_processor = None
            clap_device = None
    
    # Store audio for FAD-CLAP calculation (per source label)
    if fad_clap_enabled:
        fad_clap_targets = {label: [] for label in columns}
        fad_clap_predictions = {label: [] for label in columns}
    
    total_available = len(dataset)
    if subset_indices is not None:
        # Ensure indices are within range
        sample_indices = [idx for idx in subset_indices if 0 <= idx < total_available]
        if max_samples is not None:
            sample_indices = sample_indices[:min(max_samples, len(sample_indices))]
        num_samples = len(sample_indices)
        print("Using deterministic validation subset:")
        if subset_metadata:
            subset_seed = subset_metadata.get('seed')
            requested = subset_metadata.get('requested_size')
            print(f"  Seed: {subset_seed}, requested: {requested}, available: {total_available}")
        print(f"  Selected {num_samples} samples (indices match training subset)")
    else:
        num_samples = total_available if max_samples is None else min(max_samples, total_available)
        print("Checking dataset order...")
        sample_indices = list(range(num_samples))
        if shuffle_samples:
            # Check first 20 samples to see if they're all the same class
            first_labels = []
            for i in range(min(20, num_samples)):
                try:
                    sample = dataset[i]
                    source_label = sample.get('source_label', '')
                    first_labels.append(source_label)
                except Exception:
                    break
            
            # If all first 20 samples have the same label, dataset is likely ordered
            if len(set(first_labels)) <= 2 and len(first_labels) >= 10:
                print(f"⚠️  Dataset appears to be ordered by class (first 20 samples: {first_labels[:10]}...)")
                print("   Randomizing sample order to avoid bias...")
                random.shuffle(sample_indices)
            else:
                print("✓ Dataset appears to be randomized or mixed")
        else:
            print("Using original dataset order (no shuffling)")
    
    if subset_metadata:
        subset_info = dict(subset_metadata)
    else:
        subset_info = {}
    subset_info.update({
        'evaluated_samples': num_samples,
        'total_available_samples': total_available,
        'mode': 'subset' if subset_indices is not None else ('limited' if max_samples is not None else 'full'),
    })
    
    print(f"Evaluating on {num_samples} samples from MSRBench dataset...")
    print(f"Using {window_duration}-second non-overlapping windows")
    print(f"Model columns: {columns}")
    
    forward_pass_count = 0
    stats_interval = 1000
    
    with torch.no_grad():
        for sample_idx, idx in enumerate(tqdm(sample_indices, desc="Evaluating samples")):
            try:
                sample = dataset[idx]
                
                # Get mixture and source
                mixture = sample['mixture']['waveform']  # [channels, samples]
                source_gt = sample['source']['waveform']  # [channels, samples]
                source_label = sample['source_label']
                clip_id = sample.get('clip_id', '')
                
                # Extract DT group for this sample
                dt_group = extract_dt_group(clip_id)
                if dt_group not in dt_group_distribution:
                    dt_group_distribution[dt_group] = 0
                dt_group_distribution[dt_group] += 1
                total_samples_accounted += 1
                
                # Skip if source_label is not in model columns
                if source_label not in columns:
                    continue
                
                # Get the index of this source in the model output
                source_idx = columns.index(source_label)
                
                # Segment into non-overlapping windows
                mixture_length = mixture.shape[-1]
                source_length = source_gt.shape[-1]
                min_length = min(mixture_length, source_length)
                
                # Track if we evaluated at least one window from this file
                file_evaluated = False
                
                # Process each window
                window_start = 0
                while window_start + window_samples <= min_length:
                    # Extract window
                    mixture_window = mixture[:, window_start:window_start + window_samples]  # [channels, window_samples]
                    source_gt_window = source_gt[:, window_start:window_start + window_samples]  # [channels, window_samples]
                    
                    # Move to device and add batch dimension
                    mixture_window = mixture_window.unsqueeze(0).to(device)  # [1, channels, window_samples]
                    
                    # Run inference with correct precision
                    # Note: autocast automatically handles mixed precision - it uses FP16 for operations
                    # that benefit from it (matmuls, convolutions) and FP32 for operations that need
                    # higher precision (STFT, reductions). Inner autocast(enabled=False) blocks in the
                    # model will automatically force FP32 for those specific operations, matching training.
                    input_dict = {'mixture': mixture_window}
                    if hasattr(lightning_module, 'use_film_conditioning') and lightning_module.use_film_conditioning:
                        label_vector = lightning_module.label_vectors.flatten().unsqueeze(0).to(device)
                        input_dict['label_vector'] = label_vector
                    
                    if use_autocast and autocast_dtype is not None:
                        with torch.cuda.amp.autocast(dtype=autocast_dtype):
                            output_dict = model(input_dict)
                    else:
                        output_dict = model(input_dict)
                    separated_sources = output_dict['waveform']  # [1, n_sources, channels, window_samples]
                    
                    # Extract prediction for this source
                    prediction = separated_sources[0, source_idx]  # [channels, window_samples]
                    
                    # Move to CPU for metric calculation
                    prediction = prediction.cpu()
                    source_gt_window = source_gt_window.cpu()
                    
                    # Calculate SI-SNR
                    # si_snr expects [batch, channels, samples] or [channels, samples]
                    si_snr_val = si_snr(prediction, source_gt_window).mean()  # scalar
                    
                    # Calculate Multi-Mel-SNR
                    # Convert to mono numpy arrays for multi_mel_snr
                    prediction_mono = prediction.mean(dim=0).cpu().numpy()  # [window_samples]
                    source_gt_mono = source_gt_window.mean(dim=0).cpu().numpy()  # [window_samples]
                    multi_mel_snr_val = multi_mel_snr(source_gt_mono, prediction_mono, sr=sr)
                    
                    # Store results
                    si_snr_val_item = si_snr_val.item()
                    si_snr_values[source_label].append(si_snr_val_item)
                    all_si_snr_values.append(si_snr_val_item)
                    si_snr_by_dt_group[dt_group].append(si_snr_val_item)  # Track by DT group
                    si_snr_by_dt_group_and_stem[dt_group][source_label].append(si_snr_val_item)  # Track by DT group AND stem
                    
                    # Store Multi-Mel-SNR results
                    multi_mel_snr_values[source_label].append(multi_mel_snr_val)
                    all_multi_mel_snr_values.append(multi_mel_snr_val)
                    multi_mel_snr_by_dt_group[dt_group].append(multi_mel_snr_val)  # Track by DT group
                    multi_mel_snr_by_dt_group_and_stem[dt_group][source_label].append(multi_mel_snr_val)  # Track by DT group AND stem
                    
                    # Store audio for FAD-CLAP (keep on CPU to save memory)
                    if fad_clap_enabled:
                        # Store as numpy arrays to save memory
                        fad_clap_targets[source_label].append(source_gt_window.cpu().numpy())
                        fad_clap_predictions[source_label].append(prediction.cpu().numpy())
                    
                    # Track file evaluation
                    if not file_evaluated:
                        files_evaluated[source_label] += 1
                        file_evaluated = True
                    
                    forward_pass_count += 1
                    
                    # Print statistics every stats_interval forward passes
                    if forward_pass_count % stats_interval == 0:
                        print(f"\n{'='*100}")
                        print(f"Statistics after {forward_pass_count} forward passes:")
                        print(f"{'='*100}")
                        print(f"Files evaluated per stem:")
                        for label in sorted(columns):
                            count = files_evaluated[label]
                            print(f"  {label:15s}: {count:5d} files")
                        print(f"\nSI-SNR scores (per stem):")
                        for label in sorted(columns):
                            if label in si_snr_values and len(si_snr_values[label]) > 0:
                                avg = np.mean(si_snr_values[label])
                                std = np.std(si_snr_values[label])
                                num_windows = len(si_snr_values[label])
                                print(f"  {label:15s}: {avg:7.4f} ± {std:7.4f} dB ({num_windows:5d} windows)")
                        if len(all_si_snr_values) > 0:
                            overall_avg = np.mean(all_si_snr_values)
                            overall_std = np.std(all_si_snr_values)
                            print(f"\n  Overall:        {overall_avg:7.4f} ± {overall_std:7.4f} dB ({len(all_si_snr_values):5d} windows)")
                        
                        print(f"\nMulti-Mel-SNR scores (per stem):")
                        for label in sorted(columns):
                            if label in multi_mel_snr_values and len(multi_mel_snr_values[label]) > 0:
                                avg = np.mean(multi_mel_snr_values[label])
                                std = np.std(multi_mel_snr_values[label])
                                num_windows = len(multi_mel_snr_values[label])
                                print(f"  {label:15s}: {avg:7.4f} ± {std:7.4f} dB ({num_windows:5d} windows)")
                        if len(all_multi_mel_snr_values) > 0:
                            overall_avg_mmsnr = np.mean(all_multi_mel_snr_values)
                            overall_std_mmsnr = np.std(all_multi_mel_snr_values)
                            print(f"\n  Overall:        {overall_avg_mmsnr:7.4f} ± {overall_std_mmsnr:7.4f} dB ({len(all_multi_mel_snr_values):5d} windows)")
                        
                        # Print DT group statistics
                        print(f"\n{'='*100}")
                        print("SI-SNR Performance by Degradation Type Group")
                        print(f"{'='*100}")
                        print(f"\n{'DT Group':<20} {'Name':<25} {'SI-SNR (dB)':<20} {'Windows':<10}")
                        print("-" * 100)
                        
                        # Print overall first
                        if len(all_si_snr_values) > 0:
                            print(f"{'Overall':<20} {'All Groups':<25} "
                                  f"{overall_avg:6.4f} ± {overall_std:6.4f}    "
                                  f"{len(all_si_snr_values):<10}")
                        
                        # Print DT groups in order
                        dt_group_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
                        for dt_group in dt_group_order:
                            if len(si_snr_by_dt_group[dt_group]) > 0:
                                dt_avg = np.mean(si_snr_by_dt_group[dt_group])
                                dt_std = np.std(si_snr_by_dt_group[dt_group])
                                dt_windows = len(si_snr_by_dt_group[dt_group])
                                print(f"{dt_group:<20} {dt_group_names[dt_group]:<25} "
                                      f"{dt_avg:6.4f} ± {dt_std:6.4f}    "
                                      f"{dt_windows:<10}")
                        
                        print("-" * 100)
                        
                        # Print DT group AND stem cross-tabulation
                        print(f"\n{'='*100}")
                        print("SI-SNR Performance by Degradation Type Group AND Stem")
                        print(f"{'='*100}")
                        
                        # Calculate column widths
                        stem_labels_sorted = sorted(columns)
                        dt_group_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
                        
                        # Create header
                        header = f"{'DT Group':<20} {'Name':<25}"
                        for stem in stem_labels_sorted:
                            header += f" {stem:<12}"
                        header += f" {'Avg':<12}"
                        print(f"\n{header}")
                        print("-" * 100)
                        
                        # Print each DT group row
                        for dt_group in dt_group_order:
                            if len(si_snr_by_dt_group[dt_group]) > 0:
                                row = f"{dt_group:<20} {dt_group_names[dt_group]:<25}"
                                dt_group_stem_values = []
                                
                                # Get values for each stem
                                for stem in stem_labels_sorted:
                                    stem_values = si_snr_by_dt_group_and_stem[dt_group][stem]
                                    if len(stem_values) > 0:
                                        stem_avg = np.mean(stem_values)
                                        dt_group_stem_values.append(stem_avg)
                                        row += f" {stem_avg:11.4f}"
                                    else:
                                        row += f" {'---':<12}"
                                
                                # Add average across stems for this DT group
                                if len(dt_group_stem_values) > 0:
                                    dt_group_avg = np.mean(dt_group_stem_values)
                                    row += f" {dt_group_avg:11.4f}"
                                else:
                                    # Fallback to overall DT group average
                                    dt_avg = np.mean(si_snr_by_dt_group[dt_group])
                                    row += f" {dt_avg:11.4f}"
                                
                                print(row)
                        
                        # Print row with averages per stem (across all DT groups)
                        print("-" * 100)
                        avg_row = f"{'Avg per stem':<20} {'Across all DT':<25}"
                        for stem in stem_labels_sorted:
                            if stem in si_snr_values and len(si_snr_values[stem]) > 0:
                                stem_avg = np.mean(si_snr_values[stem])
                                avg_row += f" {stem_avg:11.4f}"
                            else:
                                avg_row += f" {'---':<12}"
                        # Overall average in last column
                        if len(all_si_snr_values) > 0:
                            avg_row += f" {overall_avg:11.4f}"
                        else:
                            avg_row += f" {'---':<12}"
                        print(avg_row)
                        print("-" * 100)
                        print("="*100)
                        
                        # Calculate and print FAD-CLAP if enabled
                        if fad_clap_enabled:
                            print(f"\nFAD-CLAP scores (per stem):")
                            fad_clap_scores = {}
                            for label in sorted(columns):
                                if label in fad_clap_targets and len(fad_clap_targets[label]) > 0:
                                    try:
                                        # Get embeddings for this label
                                        target_embeddings = get_clap_embeddings_from_arrays(
                                            fad_clap_targets[label], clap_model, clap_processor, clap_device, fad_clap_batch_size
                                        )
                                        pred_embeddings = get_clap_embeddings_from_arrays(
                                            fad_clap_predictions[label], clap_model, clap_processor, clap_device, fad_clap_batch_size
                                        )
                                        
                                        if target_embeddings.size > 0 and pred_embeddings.size > 0:
                                            fad_score = calculate_frechet_distance(target_embeddings, pred_embeddings)
                                            if fad_score is not None:
                                                fad_clap_scores[label] = fad_score
                                                print(f"  {label:15s}: {fad_score:7.4f}")
                                    except Exception as e:
                                        print(f"  {label:15s}: Error calculating FAD-CLAP: {e}")
                            
                            # Overall FAD-CLAP
                            try:
                                all_targets = []
                                all_predictions = []
                                for label in columns:
                                    if label in fad_clap_targets:
                                        all_targets.extend(fad_clap_targets[label])
                                        all_predictions.extend(fad_clap_predictions[label])
                                
                                if len(all_targets) > 0:
                                    target_embeddings = get_clap_embeddings_from_arrays(
                                        all_targets, clap_model, clap_processor, clap_device, fad_clap_batch_size
                                    )
                                    pred_embeddings = get_clap_embeddings_from_arrays(
                                        all_predictions, clap_model, clap_processor, clap_device, fad_clap_batch_size
                                    )
                                    
                                    if target_embeddings.size > 0 and pred_embeddings.size > 0:
                                        overall_fad = calculate_frechet_distance(target_embeddings, pred_embeddings)
                                        if overall_fad is not None:
                                            print(f"\n  Overall:        {overall_fad:7.4f}")
                            except Exception as e:
                                print(f"\n  Overall:        Error calculating FAD-CLAP: {e}")
                        
                        print(f"{'='*100}\n")
                    
                    # Move to next window
                    window_start += window_samples
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Calculate average SI-SNR and Multi-Mel-SNR per source label
    results = {
        'per_source': {},
        'overall': {},
        'per_dt_group': {},
        'files_evaluated': {},
        'subset': subset_info
    }
    
    for label in columns:
        if label in si_snr_values and len(si_snr_values[label]) > 0:
            avg_si_snr = np.mean(si_snr_values[label])
            std_si_snr = np.std(si_snr_values[label])
            results['per_source'][label] = {
                'mean_si_snr': float(avg_si_snr),
                'std_si_snr': float(std_si_snr),
                'num_windows': len(si_snr_values[label])
            }
        
        if label in multi_mel_snr_values and len(multi_mel_snr_values[label]) > 0:
            avg_multi_mel_snr = np.mean(multi_mel_snr_values[label])
            std_multi_mel_snr = np.std(multi_mel_snr_values[label])
            if label not in results['per_source']:
                results['per_source'][label] = {}
            results['per_source'][label]['mean_multi_mel_snr'] = float(avg_multi_mel_snr)
            results['per_source'][label]['std_multi_mel_snr'] = float(std_multi_mel_snr)
        
        results['files_evaluated'][label] = files_evaluated[label]
    
    # Overall average
    if len(all_si_snr_values) > 0:
        results['overall'] = {
            'mean_si_snr': float(np.mean(all_si_snr_values)),
            'std_si_snr': float(np.std(all_si_snr_values)),
            'num_windows': len(all_si_snr_values)
        }
    
    if len(all_multi_mel_snr_values) > 0:
        if 'overall' not in results:
            results['overall'] = {}
        results['overall']['mean_multi_mel_snr'] = float(np.mean(all_multi_mel_snr_values))
        results['overall']['std_multi_mel_snr'] = float(np.std(all_multi_mel_snr_values))
    
    # Calculate statistics per DT group
    for dt_group in dt_group_names.keys():
        if len(si_snr_by_dt_group[dt_group]) > 0 or len(multi_mel_snr_by_dt_group[dt_group]) > 0:
            results['per_dt_group'][dt_group] = {
                'name': dt_group_names[dt_group]
            }
            if len(si_snr_by_dt_group[dt_group]) > 0:
                avg_si_snr = np.mean(si_snr_by_dt_group[dt_group])
                std_si_snr = np.std(si_snr_by_dt_group[dt_group])
                results['per_dt_group'][dt_group]['mean_si_snr'] = float(avg_si_snr)
                results['per_dt_group'][dt_group]['std_si_snr'] = float(std_si_snr)
                results['per_dt_group'][dt_group]['num_windows'] = len(si_snr_by_dt_group[dt_group])
            
            if len(multi_mel_snr_by_dt_group[dt_group]) > 0:
                avg_multi_mel_snr = np.mean(multi_mel_snr_by_dt_group[dt_group])
                std_multi_mel_snr = np.std(multi_mel_snr_by_dt_group[dt_group])
                results['per_dt_group'][dt_group]['mean_multi_mel_snr'] = float(avg_multi_mel_snr)
                results['per_dt_group'][dt_group]['std_multi_mel_snr'] = float(std_multi_mel_snr)
    
    # Store DT group cross-tabulation data for final printing
    # Convert lists to means for storage efficiency
    results['per_dt_group_and_stem'] = {}
    for dt_group in dt_group_names.keys():
        results['per_dt_group_and_stem'][dt_group] = {}
        for stem in columns:
            has_data = False
            stem_data = {}
            if len(si_snr_by_dt_group_and_stem[dt_group][stem]) > 0:
                stem_data['mean_si_snr'] = float(np.mean(si_snr_by_dt_group_and_stem[dt_group][stem]))
                stem_data['num_windows'] = len(si_snr_by_dt_group_and_stem[dt_group][stem])
                has_data = True
            if len(multi_mel_snr_by_dt_group_and_stem[dt_group][stem]) > 0:
                stem_data['mean_multi_mel_snr'] = float(np.mean(multi_mel_snr_by_dt_group_and_stem[dt_group][stem]))
                has_data = True
            if has_data:
                results['per_dt_group_and_stem'][dt_group][stem] = stem_data
    
    # Store metadata for final printing
    results['metadata'] = {
        'columns': columns,
        'dt_group_names': dt_group_names
    }

    results['dt_group_distribution'] = {
        'counts': {group: int(count) for group, count in dt_group_distribution.items()},
        'total_samples': int(total_samples_accounted),
        'group_names': dt_group_names
    }
    
    # Calculate final FAD-CLAP scores if enabled
    if fad_clap_enabled:
        print("\nCalculating final FAD-CLAP scores...")
        results['fad_clap'] = {'per_source': {}, 'overall': None}
        
        for label in sorted(columns):
            if label in fad_clap_targets and len(fad_clap_targets[label]) > 0:
                try:
                    target_embeddings = get_clap_embeddings_from_arrays(
                        fad_clap_targets[label], clap_model, clap_processor, clap_device, fad_clap_batch_size
                    )
                    pred_embeddings = get_clap_embeddings_from_arrays(
                        fad_clap_predictions[label], clap_model, clap_processor, clap_device, fad_clap_batch_size
                    )
                    
                    if target_embeddings.size > 0 and pred_embeddings.size > 0:
                        fad_score = calculate_frechet_distance(target_embeddings, pred_embeddings)
                        if fad_score is not None:
                            results['fad_clap']['per_source'][label] = float(fad_score)
                except Exception as e:
                    print(f"Warning: Failed to calculate FAD-CLAP for {label}: {e}")
        
        # Overall FAD-CLAP
        try:
            all_targets = []
            all_predictions = []
            for label in columns:
                if label in fad_clap_targets:
                    all_targets.extend(fad_clap_targets[label])
                    all_predictions.extend(fad_clap_predictions[label])
            
            if len(all_targets) > 0:
                target_embeddings = get_clap_embeddings_from_arrays(
                    all_targets, clap_model, clap_processor, clap_device, fad_clap_batch_size
                )
                pred_embeddings = get_clap_embeddings_from_arrays(
                    all_predictions, clap_model, clap_processor, clap_device, fad_clap_batch_size
                )
                
                if target_embeddings.size > 0 and pred_embeddings.size > 0:
                    overall_fad = calculate_frechet_distance(target_embeddings, pred_embeddings)
                    if overall_fad is not None:
                        results['fad_clap']['overall'] = float(overall_fad)
        except Exception as e:
            print(f"Warning: Failed to calculate overall FAD-CLAP: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate source separation model on test set")
    parser.add_argument("--config_yaml_file", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (or use --wandb_id)")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb ID to load checkpoint from (alternative to --checkpoint_path)")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs", help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--fad_batch_size", type=int, default=16, help="Batch size for FAD-CLAP calculation")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--evaluate_msr_bench", action="store_true", help="Evaluate on MSRBench dataset instead of test set")
    parser.add_argument("--msr_bench_split", type=str, default="train", help="MSRBench split to evaluate on (default: train)")
    parser.add_argument("--window_duration", type=float, default=10.0, help="Window duration in seconds for MSRBench evaluation (default: 10.0)")
    parser.add_argument("--calculate_fad_clap", action="store_true", help="Calculate FAD-CLAP metric for MSRBench evaluation")
    parser.add_argument("--fad_clap_batch_size", type=int, default=16, help="Batch size for FAD-CLAP embedding calculation (default: 16)")
    parser.add_argument("--subset_size", type=int, default=None, help="Override number of MSRBench samples to evaluate (defaults to config msr_bench_max_files)")
    parser.add_argument("--subset_seed", type=int, default=42, help="Override seed for deterministic subset selection (defaults to config msr_bench_seed)")
    parser.add_argument("--wandb_project", type=str, default="MSR_Separation_eval", help="Weights & Biases project for logging")
    parser.add_argument("--wandb_entity", type=str, default="something_with_audio", help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional wandb run name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Weights & Biases logging mode")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Optional wandb tags")
    
    args = parser.parse_args()
    
    wandb_run = init_wandb_run(args, Path(args.config_yaml_file).stem)
    
    # Load configuration
    from src.utils import parse_yaml, inject_dataset_paths_into_config
    import socket
    import src.config_updates as config_updates_module
    
    config = parse_yaml(args.config_yaml_file)
    
    # Inject dataset paths from config (with MUSICA cluster detection if needed)
    hostname = socket.gethostname()
    
    dataset_path_4_sources = config.get('dataset_path_4_sources')
    dataset_path_8_sources = config.get('dataset_path_8_sources')
    dataset_path_msr_bench = config.get('dataset_path_msr_bench')
    
    if hostname.startswith('n'):
        # MUSICA cluster detected - update paths like config_updates.py does
        MUSICA_DATA_PATH = "/data/<username>/"
        dataset_path_4_sources = os.path.join(MUSICA_DATA_PATH, "msr_separation/mss4s_musica")
        dataset_path_8_sources = os.path.join(MUSICA_DATA_PATH, "msr_separation/mss8s_musica_v2_other_is_present")
        # dataset_path_msr_bench stays as is
    
    # Use paths from config (required)
    inject_dataset_paths_into_config(
        config,
        dataset_path_4_sources=dataset_path_4_sources,
        dataset_path_8_sources=dataset_path_8_sources,
        dataset_path_msr_bench=dataset_path_msr_bench
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine checkpoint path
    if args.wandb_id:
        # Construct checkpoint path from wandb_id
        from src.config_updates import CKPT_PATH
        # Try to infer config name from config file path
        config_name = Path(args.config_yaml_file).stem
        # Construct checkpoint path
        checkpoint_path = os.path.join(CKPT_PATH, config_name, args.wandb_id, "last.ckpt")
    elif args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        raise ValueError("Either --checkpoint_path or --wandb_id must be provided")

    checkpoint_path = download_checkpoint_from_wandb(
        wandb_id=args.wandb_id,
        checkpoint_path=checkpoint_path,
        project="MSR_Separation",
        entity="something_with_audio",
        artifact_name=f"checkpoint-{args.wandb_id}"
    )

    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize LightningModule architecture and load checkpoint weights
    lightning_module = initialize_config(config['lightning_module'])
    map_4stem_to_9stem = False  # config.get('map_4stem_to_9stem')
    init_new_stems_from_other = False  # config.get('init_new_stems_from_other', False)
    add_noise_to_new_stems = False  # config.get('add_noise_to_new_stems', False)
    full_ckpt = load_ckpt(
        checkpoint_path,
        lightning_module,
        map_4stem_to_9stem=map_4stem_to_9stem,
        init_new_stems_from_other=init_new_stems_from_other,
        add_noise_to_new_stems=add_noise_to_new_stems,
    )
    # Load EMA state if present
    if hasattr(lightning_module, 'use_ema') and lightning_module.use_ema and lightning_module.ema is not None:
        if 'ema_state_dict' in full_ckpt:
            lightning_module.ema.load_state_dict(full_ckpt['ema_state_dict'])
            print("Loaded EMA state from checkpoint")
        else:
            print("Warning: EMA enabled but no EMA state found in checkpoint. EMA will use current model weights.")
    
    # Move model to device BEFORE applying EMA (important for device consistency)
    lightning_module = lightning_module.to(device)
    
    model = lightning_module.model
    model.eval()
    
    # Verify LoRA parameters are loaded and active
    # LoRA works automatically during inference when merge_weights=False (default)
    # The forward pass in LoRALinear applies the LoRA adaptation on-the-fly
    lora_params_count = 0
    lora_params_with_grad = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params_count += 1
            if param.requires_grad:
                lora_params_with_grad += 1
    
    if lora_params_count > 0:
        print(f"LoRA parameters detected: {lora_params_count} LoRA parameters found")
        print(f"  - LoRA parameters with gradients: {lora_params_with_grad}")
        print("  - LoRA will be applied automatically during inference (merge_weights=False)")
        print("  - No explicit merging needed - LoRA adaptation computed in forward pass")
    else:
        # Check if LoRA was enabled in config but no parameters found
        model_config = config.get('lightning_module', {}).get('args', {}).get('model', {}).get('args', {})
        lora_config = model_config.get('lora_config', {})
        if lora_config.get('enabled', False):
            print("Warning: LoRA is enabled in config but no LoRA parameters found in model")
            print("  This might indicate LoRA was not applied during model initialization")
    
    # Apply EMA weights if enabled (for inference, we want to use EMA model)
    # IMPORTANT: EMA must be applied AFTER moving to device to ensure device consistency
    # EMA state was loaded above, so we can safely swap weights here
    if hasattr(lightning_module, 'use_ema') and lightning_module.use_ema and lightning_module.ema is not None:
        print("Applying EMA weights for inference...")
        # Verify EMA state was loaded
        if hasattr(lightning_module.ema, 'ema_params') and len(lightning_module.ema.ema_params) > 0:
            print(f"  EMA state loaded: {len(lightning_module.ema.ema_params)} parameters")
        else:
            print("  WARNING: EMA state appears to be empty - EMA may not have been loaded correctly!")
        lightning_module.ema.apply()
        print("EMA weights applied successfully")
    else:
        print("EMA not enabled or not available - using regular model weights")
    
    # Get precision setting from config to match training precision
    trainer_config = config.get('train', {}).get('trainer', {}).get('args', {})
    precision = trainer_config.get('precision', None)
    if precision:
        print(f"Detected precision setting from config: {precision}")
    else:
        print("No precision setting found in config, will use FP32 for inference")
    
    try:
        # MSRBench evaluation
        if args.evaluate_msr_bench:
            print("\n" + "="*80)
            print("MSRBench Evaluation")
            print("="*80)
            
            # Load MSRBench dataset
            print(f"Loading MSRBench dataset (split: {args.msr_bench_split})...")
            msr_bench_dataset = MultiAudioFullSongDataset(
                split=args.msr_bench_split,
                columns=None,  # Use default: mixture_audio and source_audio
                sr=48000,
                mono=False,
                root=dataset_path_msr_bench,
            )
            print(f"Dataset loaded with {len(msr_bench_dataset)} samples")
            
            # Derive subset parameters (match training configuration)
            lightning_args = config.get('lightning_module', {}).get('args', {})
            config_subset_size = lightning_args.get('msr_bench_max_files')
            config_subset_seed = lightning_args.get('msr_bench_seed', 42)
            
            subset_size = args.subset_size if args.subset_size is not None else config_subset_size
            subset_seed = args.subset_seed if args.subset_seed is not None else config_subset_seed
            
            if subset_size is None:
                raise ValueError(
                    "subset_size is undefined. Provide --subset_size or set lightning_module.args.msr_bench_max_files in the config."
                )
            
            subset_indices = select_msr_bench_subset(len(msr_bench_dataset), subset_size, subset_seed)
            subset_metadata = {
                'seed': subset_seed,
                'requested_size': int(subset_size)
            }
            
            # Evaluate
            results = evaluate_msr_bench(
                model=model,
                lightning_module=lightning_module,
                dataset=msr_bench_dataset,
                device=device,
                window_duration=args.window_duration,
                sr=48000,
                max_samples=args.max_samples,
                shuffle_samples=False,  # Deterministic subset order
                calculate_fad_clap=args.calculate_fad_clap,
                fad_clap_batch_size=args.fad_clap_batch_size,
                subset_indices=subset_indices,
                subset_metadata=subset_metadata,
                precision=precision
            )
            
            # Print final results
            print("\n" + "="*100)
            print("MSRBench Evaluation Results (Final)")
            print("="*100)
            print(f"\nFiles evaluated per stem:")
            for label in sorted(results['files_evaluated'].keys()):
                count = results['files_evaluated'][label]
                print(f"  {label:15s}: {count:5d} files")
            
            dt_distribution = results.get('dt_group_distribution')
            if dt_distribution:
                group_names = dt_distribution.get('group_names', {
                    'DT0': 'Reference',
                    'DT1-DT4': 'Analog/Acoustic',
                    'DT5-DT8': 'Traditional Codecs',
                    'DT9-DT12': 'Neural Codecs'
                })
                dt_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
                counts = dt_distribution.get('counts', {})
                total_samples = dt_distribution.get('total_samples', sum(counts.values()))
                print(f"\nDT group distribution (unique samples in subset):")
                print(f"  Total subset samples: {total_samples}")
                for dt_group in dt_order:
                    group_label = group_names.get(dt_group, dt_group)
                    print(f"  {dt_group:8s} ({group_label:20s}): {counts.get(dt_group, 0):4d}")
            
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
            
            # Print DT group results in a table format (averages across all stems)
            if results.get('per_dt_group'):
                print(f"\n{'='*100}")
                print("SI-SNR Performance by Degradation Type Group (Average across all stems)")
                print("="*100)
                
                # Create table header
                print(f"\n{'DT Group':<20} {'Name':<25} {'SI-SNR (dB)':<20} {'Windows':<10}")
                print("-" * 100)
                
                # Print overall first
                if results['overall']:
                    print(f"{'Overall':<20} {'All Groups':<25} "
                          f"{results['overall']['mean_si_snr']:6.4f} ± {results['overall']['std_si_snr']:6.4f}    "
                          f"{results['overall']['num_windows']:<10}")
                
                # Print DT groups in order
                dt_group_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
                for dt_group in dt_group_order:
                    if dt_group in results['per_dt_group']:
                        dt_data = results['per_dt_group'][dt_group]
                        print(f"{dt_group:<20} {dt_data['name']:<25} "
                              f"{dt_data['mean_si_snr']:6.4f} ± {dt_data['std_si_snr']:6.4f}    "
                              f"{dt_data['num_windows']:<10}")
                
                print("-" * 100)
                print("="*100)
                
                # Print DT group AND stem cross-tabulation
                print(f"\n{'='*100}")
                print("SI-SNR Performance by Degradation Type Group AND Stem")
                print("="*100)
                
                # Get metadata
                columns_meta = results.get('metadata', {}).get('columns', sorted(results['per_source'].keys()))
                dt_group_names_meta = results.get('metadata', {}).get('dt_group_names', {
                    'DT0': 'Reference',
                    'DT1-DT4': 'Analog/Acoustic',
                    'DT5-DT8': 'Traditional Codecs',
                    'DT9-DT12': 'Neural Codecs'
                })
                
                # Calculate column widths
                stem_labels_sorted = sorted(columns_meta)
                dt_group_order = ['DT0', 'DT1-DT4', 'DT5-DT8', 'DT9-DT12']
                
                # Create header
                header = f"{'DT Group':<20} {'Name':<25}"
                for stem in stem_labels_sorted:
                    header += f" {stem:<15}"
                header += f" {'Avg':<15}"
                print(f"\n{header}")
                print("-" * 100)
                
                # Print each DT group row
                for dt_group in dt_group_order:
                    if dt_group in results['per_dt_group']:
                        row = f"{dt_group:<20} {dt_group_names_meta[dt_group]:<25}"
                        dt_group_stem_values = []
                        
                        # Get values for each stem from stored results
                        for stem in stem_labels_sorted:
                            # Get mean for this DT group and stem combination
                            dt_stem_data = results.get('per_dt_group_and_stem', {}).get(dt_group, {}).get(stem)
                            if dt_stem_data:
                                stem_avg = dt_stem_data['mean_si_snr']
                                dt_group_stem_values.append(stem_avg)
                                row += f" {stem_avg:11.4f}" + (" "*5)
                            else:
                                row += (" " * 7) + f" {'---':<8}"
                        
                        # Add average across stems for this DT group
                        if len(dt_group_stem_values) > 0:
                            dt_group_avg = np.mean(dt_group_stem_values)
                            row += f" {dt_group_avg:11.4f}"
                        elif dt_group in results['per_dt_group']:
                            # Fallback to overall DT group average
                            dt_data = results['per_dt_group'][dt_group]
                            row += f" {dt_data['mean_si_snr']:11.4f}"
                        else:
                            row += f" {'---':<12}"
                        
                        print(row)
                
                # Print row with averages per stem (across all DT groups)
                print("-" * 100)
                avg_row = f"{'Avg per stem':<20} {'Across all DT':<25}"
                for stem in stem_labels_sorted:
                    if stem in results['per_source']:
                        stem_avg = results['per_source'][stem]['mean_si_snr']
                        avg_row += f" {stem_avg:11.4f}"
                    else:
                        avg_row += f" {'---':<12}"
                # Overall average in last column
                if results['overall']:
                    avg_row += f" {results['overall']['mean_si_snr']:11.4f}"
                else:
                    avg_row += f" {'---':<12}"
                print(avg_row)
                print("-" * 100)
                print("="*100)
            
            # Print FAD-CLAP results if calculated
            if 'fad_clap' in results:
                print(f"\nPer-source FAD-CLAP scores:")
                for label, score in sorted(results['fad_clap']['per_source'].items()):
                    print(f"  {label:15s}: {score:7.4f}")
                
                if results['fad_clap']['overall'] is not None:
                    print(f"\nOverall FAD-CLAP: {results['fad_clap']['overall']:.4f}")
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, "msr_bench_evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
            print("="*80)
            log_wandb_metrics(results, prefix="msr_bench")
        
        else:
            # Standard evaluation
            # Create test dataloader
            print("Creating test dataloader...")
            test_config = config['datamodule']['args']['val_dataloader'].copy()
            test_config['dataset']['args']['split'] = 'test'  # Use test split
            if args.max_samples:
                test_config['dataset']['args']['max_samples'] = args.max_samples

            test_dataset = initialize_config(test_config['dataset'])
            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                collate_fn=test_dataset.collate_fn,
                num_workers=test_config['num_workers'],
                pin_memory=True,
                shuffle=False
            )

            # Evaluate model
            print("Starting evaluation...")
            file_pairs = evaluate_model_on_dataset(
                model, test_dataloader, device, args.output_dir,
                n_sources=config['lightning_module']['args']['n_sources'],
                precision=precision
            )

            # Calculate metrics
            metrics = calculate_metrics(file_pairs, args.fad_batch_size)

            # Save results
            results_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"\nEvaluation complete!")
            print(f"Results saved to: {results_file}")
            print(f"SI-SNR: {metrics['si_snr']:.4f} dB")
            if 'multi_mel_snr' in metrics and metrics['multi_mel_snr'] != float('inf'):
                print(f"Multi-Mel-SNR: {metrics['multi_mel_snr']:.4f} dB")
            print(f"FAD-CLAP: {metrics['fad_clap']:.4f}")
            log_wandb_metrics(metrics, prefix="evaluation")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
