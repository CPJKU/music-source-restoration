# Music Source Restoration - Separation Part

## Setup

1. Create and activate a conda environment:
   ```bash
   conda create -n msr_separation python=3.10
   conda activate msr_separation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Stages

The BS-RoFormer model was trained in multiple stages:

### Stage 0: Pre-trained Checkpoint

We start with a pre-trained BS-RoFormer checkpoint trained for separating 4 stems. The checkpoint can be found in the [pre-trained models documentation](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md).

**Command:**
```bash
sbatch scripts/stage_0.sh
```

### Stage 1: Clean Mixtures Training

Trained the model on a 4-stem dataset (MoisesDB, Musdb, and DSD100) with stems "vocals", "bass", "drums", and "other". The model was trained on the task of separating clean mixtures (without augmentations). Only the transformer blocks of BS-RoFormer were trained using LoRA.

**Command:**
```bash
sbatch scripts/stage_1.sh
```

### Stage 2: Augmented Mixtures Training

Trained the model on the same 4-stem dataset but on the task of separating augmented stems from mastered mixtures (mixture created by combining augmented sources and then applying mastering). Again, only LoRA was used and only the transformer blocks were trained.

**Command:**
```bash
sbatch scripts/stage_2.sh
```

### Stage 3: 8-Stem Fine-tuning

Used the checkpoint from stage 2 to train the model on an 8-stem dataset (consisting of kwatcharasupat_musdb25, medleydb_v2, moisesdb, raw_stems, slakh2100). The mask-estimators of the model for the "old" stems were reused, while for the new stems they were initialized new. In this stage, only the "new" mask estimators are fine-tuned.

**Command:**
```bash
sbatch scripts/stage_3.sh
```

## Evaluation

### Evaluate on MSRBench

To evaluate the trained model on MSRBench:

```bash
sbatch scripts/evaluate_on_msrbench.sh
```

In this setup, the checkpoints of the different stages are stored in folders using the name of the Weights & Biases Run-ID (wandb_id). The wandb_id is used to locate the checkpoint. If it is not present, it will try to download the checkpoint from Weights & Biases (this, of course, only works if the checkpoint is present in your Weights & Biases project). 
