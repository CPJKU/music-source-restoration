# Finally GAN for Music Restoration

A multi-stage GAN-based music restoration system that progressively improves audio quality through spectral and waveform processing.

## Overview

The Finally GAN architecture consists of:

1. **Generator (`FinallyGenerator`)**:
   - Processes degraded stereo audio at 24kHz base rate
   - Extracts perceptual features using MERT SSL embeddings (frozen)
   - Applies spectral processing via SpectralUNet
   - Upsamples using HiFi-GAN style blocks conditioned on embeddings
   - Refines with WaveUNet and spectral masking
   - Optional upsampling head for 48kHz output (stage 3)

2. **Discriminators (`FinallyDiscriminatorBundle`)**:
   - Multi-scale Mel discriminators (5 scales for base + 5 for target) only one used, alternatively the next three.
   - Multi-scale STFT discriminators (3 scales for base + 3 for target)
   - Multi-period discriminators (5 periods: 2, 3, 5, 7, 11)
   - Multi-band discriminators (4 frequency bands)

3. **Three-Stage Training**:
   - **Stage 1**: LMOS loss only (warmup on base resolution)
   - **Stage 2**: LMOS + adversarial + feature matching
   - **Stage 3**: All losses + perceptual + upsampling head (48kHz)
   - **Stage 4**: All losses + perceptual + upsampling head (48kHz) + online noise augmentation with graphmophone noises

## Dataset

The training uses the **SonicMaster Dataset** with:
- Paired clean/degraded stereo audio samples
- Multiple degradation types (codec artifacts, noise, bandwidth limitation, etc.)
- Train/validation/test splits

**Dataset Location**: `/opt/scratch/HF_datasets/saved/SonicMasterDataset`
- Audio data: `/opt/scratch/HF_datasets/saved/SonicMasterDataset/train_val_test/`
- Latents (optional): `/opt/scratch/HF_datasets/saved/SonicMasterDataset/codicodec_latents/`

## Model Architecture Details

### Generator Components

```
Input (degraded 24kHz stereo) → [B, 2, T_base]
    ↓
MERT Embeddings (frozen) → [B, 768, T_emb]
    ↓
STFT Features → [B*2, 4, F, K]  # magnitude, real, imag, phase
    ↓
SpectralUNet → [B*2, 256, F, K]
    ↓ (pool & project)
Spectral Tokens → [B, 256, T_tokens]
    ↓ (fuse with embeddings)
HiFiUpsampler → [B, 2, T_base]
    ↓
WaveUNet Refinement → [B, 2, T_base]
    ↓
Spectral Masking → [B, 2, T_base]
    ↓ (optional stage 3)
Upsample Head → [B, 2, T_target]  # 48kHz output
```

### Model Statistics

- **Generator**: ~20M trainable parameters (MERT frozen: ~95M frozen)
- **Discriminator**: ~25M parameters
- **Total training**: ~45M parameters

## Training Configuration

### Default Stage Configuration

```python
Stage 1: 40,000 steps
  - Batch size: 8
  - Learning rate: 2e-4
  - LMOS weight: 1.0
  - No adversarial training
  - Base resolution only (24kHz)

Stage 2: 60,000 steps
  - Batch size: 6
  - Learning rate: 2e-4
  - LMOS weight: 20.0
  - GAN weight: 0.4
  - Feature matching weight: 20.0
  - Base resolution only (24kHz)

Stage 3: 80,000 steps
  - Batch size: 4
  - Learning rate: 1.5e-4
  - LMOS weight: 0.5
  - GAN weight: 5.0
  - Feature matching weight: 15.0
  - Perceptual weight: 1.0
  - Upsampling to 48kHz enabled
```

### Data Augmentation

Online degradation pipeline (optional, **disabled by default**):
- Downsampling/upsampling (16kHz ↔ 24kHz)
- High-pass filtering (100-150 Hz)
- Low-pass filtering (12-20 kHz, clamped to Nyquist/2.5)
- Colored noise addition (24-48 dB SNR)
- Gain adjustments (-5 to +2 dB)
- Channel shuffling
- Polarity inversion

**Note**: The SonicMaster dataset already contains degraded samples, so online degradation is optional. Enable it in the config if you want additional augmentation during training.

## Usage

### Data Testing

```bash
python -m restoration.sonicmaster_finally_gan.test_data_loading
```

### Model Testing

```bash
python -m restoration.sonicmaster_finally_gan.test_model
```

### Training

```bash
# Basic training with defaults
python -m restoration.sonicmaster_finally_gan.train

# Training with custom configs
python -m restoration.sonicmaster_finally_gan.train \
    --model-cfg configs/model_custom.json \
    --train-cfg configs/train_custom.json \
    --data-cfg configs/data_custom.json
```

### Custom Configuration Files

Create JSON files for custom configurations:

**model_config.json**:
```json
{
  "base_sample_rate": 24000,
  "target_sample_rate": 48000,
  "stft_n_fft": 2048,
  "stft_hop": 512,
  "spectral_channels": 256,
  "upsample_scales": [8, 8, 8],
  "embedding_backbone": "mert95m"
}
```

**train_config.json**:
```json
{
  "stages": [
    {
      "name": "stage1",
      "max_steps": 40000,
      "batch_size": 8,
      "lr": 0.0002,
      ...
    }
  ],
  "checkpoint_dir": "checkpoints/finally_gan"
}
```

**data_config.json**:
```json
{
  "segment_seconds": 10.0,
  "saved_dir": "/opt/scratch/HF_datasets/saved/SonicMasterDataset",
  "apply_online_degradation": false,
  "num_workers": 6,
  "val_split": "validation",
  "val_batches": 4
}
```

**Note**: Set `apply_online_degradation` to `true` if you want additional runtime augmentation, though the dataset already contains degraded samples.

## Loss Functions

### LMOS (Latent Music Objective Score)
- Combines MERT embedding distance with multi-resolution STFT loss
- Ensures perceptual similarity and spectral accuracy

### Adversarial Loss (Least-Squares GAN)
- Generator minimizes: `mean((D(fake) - 1)^2)`
- Discriminator minimizes: `mean((D(real) - 1)^2) + mean(D(fake)^2)`

### Feature Matching Loss
- L1 distance between discriminator intermediate features
- Stabilizes training and improves perceptual quality

### Human Feedback Surrogate (Stage 3)
- Combines:
  - LPAPS: Log-mel spectrogram perceptual distance
  - Spectral convergence: Frobenius norm of STFT difference
  - Onset-weighted loss: Emphasizes transient accuracy

## Checkpoints

Checkpoints are saved in `checkpoints/finally_gan/`:
- Format: `{stage_name}_step{step:07d}.pt`
- Contains:
  - Generator and discriminator state dicts
  - EMA state dict
  - Optimizer and scheduler states
  - Stage configuration
  - Training step

## Requirements

### Core Dependencies
- PyTorch >= 2.0
- torchaudio
- transformers (for MERT embeddings)
- datasets (HuggingFace)

### Optional Dependencies
- torch-audiomentations (for online degradations)
- dasp_pytorch (for compressor/reverb effects)
- wandb (for experiment tracking)

## Memory Requirements

- **Training (batch_size=8, stage 1)**: ~6-8 GB GPU
- **Training (batch_size=4, stage 3)**: ~10-12 GB GPU
- **Inference**: ~2-4 GB GPU

Reduce batch size if running out of memory.

## Performance Tips

1. **Mixed Precision**: Enabled by default (bf16 on CUDA)
2. **Gradient Accumulation**: Configurable per stage
3. **Num Workers**: Default 6, adjust based on CPU cores
4. **Pin Memory**: Enabled for faster GPU transfers
5. **EMA**: Exponential moving average for stable inference

## Mixture of Experts (MoE) System

### Overview

The MoE system extends the Finally GAN architecture with **instrument-specific experts** to achieve superior restoration quality. The approach:

1. **Train a general model** on the full SonicMaster dataset (all instruments)
2. **Fine-tune 8 specialized experts** from the general model, one per instrument category
3. **Optional: Train a classifier** for automatic instrument detection
4. **Inference**: Route audio to appropriate expert(s) based on instrument content

### Instrument Categories

The system supports 8 instrument-specific experts:
- **Piano**: Harmonic clarity, wide stereo imaging
- **Drums**: Transient preservation (critical), tight stereo
- **Vocals**: Spectral clarity, mono-focused
- **Bass**: Low-frequency focus, narrow stereo
- **Guitar**: Mid-frequency detail, moderate stereo
- **Strings**: Harmonic richness, wide stereo
- **Brass**: Spectral brightness, moderate stereo
- **Synth**: Full-range processing, flexible stereo

### Training Pipeline

#### Phase 1: General Model (60-80 hours on L40)

Train the base model on all instruments:

```bash
python -m restoration.sonicmaster_finally_gan.train
```

This produces a general checkpoint (e.g., `checkpoints/general_model.pt`) that can restore any instrument type with good quality.

**Expected Performance**:
- PESQ: 3.2-3.5
- SISDR: 15-18 dB
- Training time: 60-80 hours (single L40)

#### Phase 2: Expert Fine-Tuning (64-96 hours total, parallelizable)

Fine-tune instrument-specific experts from the general model. Each expert:
- **Freezes the spectral encoder** (reduces 45M → 15-20M trainable parameters)
- Applies **instrument-specific loss weights** optimized for that category
- Trains for **30k steps** (~8-12 hours per expert on L40)

**Train a single expert**:

```bash
python -m restoration.sonicmaster_finally_gan.train_expert \
    --instrument piano \
    --general-checkpoint checkpoints/general_model.pt \
    --max-steps 30000 \
    --batch-size 16
```

**Train all experts sequentially**:

```bash
python -m restoration.sonicmaster_finally_gan.train_expert \
    --instrument all \
    --general-checkpoint checkpoints/general_model.pt \
    --max-steps 30000 \
    --batch-size 16
```

**Train experts in parallel** (requires 8 GPUs):

```bash
# On 8 separate GPUs (or in screen/tmux sessions)
for instrument in piano drums vocals bass guitar strings brass synth; do
    CUDA_VISIBLE_DEVICES=$gpu_id python -m restoration.sonicmaster_finally_gan.train_expert \
        --instrument $instrument \
        --general-checkpoint checkpoints/general_model.pt \
        --max-steps 30000 \
        --batch-size 16 &
done
```

**Expected Performance** (per expert):
- PESQ: 3.5-3.9
- SISDR: 18-22 dB
- Improvement over general: +0.3-0.4 PESQ, +3-4 dB SISDR
- Training time: 8-12 hours per expert (single L40)

#### Phase 3: Classifier Training (Optional, 4-6 hours)

Train an instrument classifier for automatic routing:

```bash
python -m restoration.sonicmaster_finally_gan.train_classifier \
    --checkpoint checkpoints/general_model.pt \
    --max-steps 10000
```

The classifier is a lightweight MLP (768 → 256 → 128 → 8) trained on MERT embeddings.

### Expert Loss Weight Presets

Each instrument has optimized loss weights (see `expert_configs.py`):

```python
# Example: Drums (critical transient preservation)
ExpertLossWeights(
    w_lmos=20.0,      # Standard perceptual loss
    w_gan=5.0,        # Adversarial loss
    w_fm=15.0,        # Feature matching
    w_stft=2.0,       # STFT loss
    w_if=3.0,         # CRITICAL: Instantaneous frequency (transients)
    w_gain=1.0,       # Gain consistency
    w_stereo=0.5      # Stereo imaging (drums are typically tight)
)

# Example: Vocals (spectral clarity)
ExpertLossWeights(
    w_lmos=20.0,
    w_gan=5.0,
    w_fm=15.0,
    w_stft=3.0,       # HIGH: Spectral clarity for intelligibility
    w_if=1.0,
    w_gain=1.2,
    w_stereo=0.5      # LOW: Vocals are typically mono
)

# Example: Piano (harmonic richness + stereo width)
ExpertLossWeights(
    w_lmos=20.0,
    w_gan=5.0,
    w_fm=15.0,
    w_stft=2.0,
    w_if=1.0,
    w_gain=1.0,
    w_stereo=1.5      # HIGH: Piano has wide stereo imaging
)
```

Full presets for all 8 instruments are defined in `expert_configs.EXPERT_PRESETS`.

### Inference with MoE

The system supports **3 routing strategies**:

#### 1. Manual Routing

Explicitly specify the instrument:

```python
from restoration.sonicmaster_finally_gan.mixture_inference import create_mixture_system

# Load general + all experts
mixture = create_mixture_system(
    general_checkpoint="checkpoints/general_model.pt",
    expert_checkpoints={
        "piano": "checkpoints/expert_piano.pt",
        "drums": "checkpoints/expert_drums.pt",
        # ... other experts
    },
    routing_strategy="manual"
)

# Restore with specific expert
restored = mixture.restore(degraded_audio, instrument="piano")
```

**Use case**: When you know the instrument content (e.g., stem restoration, isolated instruments)

#### 2. Classifier-Based Routing

Automatically detect the instrument and route to the best expert:

```python
mixture = create_mixture_system(
    general_checkpoint="checkpoints/general_model.pt",
    expert_checkpoints={...},
    classifier_checkpoint="checkpoints/classifier.pt",
    routing_strategy="classifier",
    confidence_threshold=0.7  # Fall back to general if confidence < 0.7
)

# Automatic routing
restored = mixture.restore(degraded_audio)

# Check what was detected
instrument, confidence = mixture.classify_instrument(degraded_audio)
print(f"Detected: {instrument} (confidence: {confidence:.2f})")
```

**Use case**: Single-instrument recordings with unknown content

#### 3. Weighted Ensemble Routing

Blend top-k experts based on classification confidence:

```python
mixture = create_mixture_system(
    general_checkpoint="checkpoints/general_model.pt",
    expert_checkpoints={...},
    classifier_checkpoint="checkpoints/classifier.pt",
    routing_strategy="weighted",
    top_k=2,  # Blend top 2 experts
    temperature=1.0  # Softmax temperature for blending
)

# Ensemble restoration (e.g., 70% piano + 30% strings)
restored = mixture.restore(degraded_audio)
```

**Use case**: Mixed instrument content or uncertain classification

### File Structure

```
restoration/hifi_gan/
├── expert_configs.py          # Expert configurations and presets
├── train_expert.py            # Expert fine-tuning script
├── mixture_inference.py       # MoE inference system
├── train_classifier.py        # Classifier training (optional)
├── generator.py               # Generator architecture
├── discriminator.py           # Discriminator architectures
├── losses.py                  # Loss functions (including MusicPerceptualLoss)
├── trainer.py                 # Training loop
└── configs.py                 # Model/train/data configs
```

### Expected Timeline

| Phase | Task | Hardware | Time | Parallelizable |
|-------|------|----------|------|----------------|
| 1 | Train general model | 1x L40 | 60-80h | No |
| 2 | Train 8 experts | 1x L40 (seq) | 64-96h | Yes (8x L40 → 8-12h) |
| 3 | Train classifier | 1x L40 | 4-6h | No |
| **Total** | | **1x L40** | **128-182h** | **8x L40 → 72-98h** |

### Performance Comparison

| Model | PESQ | SISDR | Use Case |
|-------|------|-------|----------|
| General | 3.2-3.5 | 15-18 dB | Any instrument, good baseline |
| Expert (single) | 3.5-3.9 | 18-22 dB | Known instrument, best quality |
| MoE (classifier) | 3.4-3.8 | 17-21 dB | Unknown instrument, auto-routing |
| MoE (weighted) | 3.5-3.8 | 17-21 dB | Mixed content, ensemble |

### Design Rationale

**Why freeze the encoder?**
- The spectral encoder learns general audio features (harmonics, transients, noise patterns)
- These features transfer well across instruments
- Freezing reduces parameters (45M → 15-20M), speeds training, and prevents overfitting

**Why instrument-specific loss weights?**
- Different instruments have different perceptual priorities:
  - **Drums**: Transient accuracy is critical (high `w_if`)
  - **Vocals**: Spectral clarity for intelligibility (high `w_stft`)
  - **Piano**: Stereo imaging for spaciousness (high `w_stereo`)
  - **Bass**: Low-frequency accuracy (high `w_stft`, low `w_stereo`)
- Generic weights work "okay" for all, specialized weights excel for each

**Why MoE vs. single large model?**
- **Efficiency**: 8 experts (15-20M each) vs. 1 giant model (200M+)
- **Specialization**: Each expert focuses on instrument-specific features
- **Modularity**: Easy to add/update individual experts


### Troubleshooting

**Expert training OOM (Out of Memory)**:
```bash
# Reduce batch size
python -m restoration.sonicmaster_finally_gan.train_expert \
    --instrument piano \
    --batch-size 8  # Default is 16
```

**Classifier has low confidence**:
- Train longer (increase `--max-steps` from 10k to 20k)
- Lower `confidence_threshold` (default 0.7 → 0.5)
- Use `weighted` routing instead of `classifier`

**Expert performs worse than general**:
- Check loss weight presets in `expert_configs.EXPERT_PRESETS`
- Ensure `freeze_encoder=True` (freezing is critical for stability)
- Train longer (increase `--max-steps` from 30k to 50k)


## License

See parent repository LICENSE file.
