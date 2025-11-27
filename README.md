# Music Source Separation
Music source separation and restoration systems built for the MSR ICASSP Challenge 2025.

- Emmanouil Karystinaios (Restoration and data curation/collection)
- Tobias Morocutti (Source Separation)

## Project layout
- `msr_separation/`: BS-RoFormer-based separation pipeline (LoRA fine-tuning from 4 to 8 stems).
- `restoration/hifi_gan/`: Finally-style GAN restorer with MERT conditioning, multi-stage training, and instrument-specific Mixture-of-Experts.
- `restoration/diffusion_restorer.py`: Diffusion-based restorer utilities (STFT helpers, PQMF bands, SonicMaster loaders, training/inference loops).
- `LICENSE`: Repository license.

## Separation (BS-RoFormer)
- Environment: `conda create -n msr_separation python=3.10 && conda activate msr_separation` then `pip install -r msr_separation/requirements.txt`.
- Training stages (Slurm):  
  - Stage 0: warm start from pretrained BS-RoFormer — `sbatch scripts/stage_0.sh`  
  - Stage 1: clean mixtures, LoRA on transformer blocks — `sbatch scripts/stage_1.sh`  
  - Stage 2: augmented/mastered mixtures, LoRA — `sbatch scripts/stage_2.sh`  
  - Stage 3: 8-stem fine-tuning (reuses old mask estimators, trains new ones) — `sbatch scripts/stage_3.sh`
- Configs live in `msr_separation/config/`; adjust paths, wandb IDs, and Slurm log dirs before submitting.
- Evaluation on MSRBench: `sbatch scripts/evaluate_on_msrbench.sh`. Checkpoints are resolved via wandb run IDs and downloaded if missing.

## Restoration (Finally GAN + MoE)
- Architecture: multi-stage generator (SpectralUNet → HiFi-GAN upsampler → WaveUNet + spectral masking) with frozen MERT embeddings; discriminators span mel/STFT/period/band scales; optional 48 kHz upsampling head.
- Training stages (defaults): Stage 1 LMOS warmup → Stage 2 adds GAN + feature matching → Stage 3 adds perceptual + 48 kHz head → Stage 4 optional online noise augmentation.
- Data: SonicMasterDataset stereo pairs. Default saved location `/opt/scratch/HF_datasets/saved/SonicMasterDataset` (audio under `train_val_test/`, optional codec latents under `codicodec_latents/`).
- Quick commands (module path used in code is `restoration.sonicmaster_finally_gan`, files are under `restoration/hifi_gan/`):
  - Sanity checks: `python -m restoration.sonicmaster_finally_gan.test_data_loading` and `python -m restoration.sonicmaster_finally_gan.test_model`
  - Train general model: `python -m restoration.sonicmaster_finally_gan.train`
  - Custom configs: pass `--model-cfg/--train-cfg/--data-cfg` JSON paths; resume via `--resume-from`
  - Train instrument experts from general ckpt: `python -m restoration.sonicmaster_finally_gan.train_expert --instrument piano --general-checkpoint <ckpt>`
  - Train classifier (optional for routing): `python -m restoration.sonicmaster_finally_gan.train_classifier --checkpoint <general_ckpt>`
- Mixture-of-Experts inference: load general + expert checkpoints via `mixture_inference.create_mixture_system` and choose manual, classifier, or weighted routing. See `restoration/hifi_gan/README.md` for command variants, presets, and timelines.

## License
See `LICENSE`.
