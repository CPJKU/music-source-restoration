#!/bin/bash
#SBATCH --job-name "stage_3"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=4

#SBATCH -o <your-project-path>/slurmout/%j_%a.out  # TODO: change <your-project-path> to your project path
#SBATCH --threads-per-core=1

source $HOME/.bashrc

env

cd $HOME/msr_separation  # TODO: go to your project directory

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate msr_separation

srun --gpu-bind=none python -m src.train with roformer_8s_dataset_msr_bench_val musica only_finetune_new_masks_3e_3 mix_8stem_training lightning_module.args.optimizer.args.lr=3e-3 lightning_module.args.loss.args.l1_base_weight=100 lightning_module.args.loss.args.l1_max_weight=1000 lightning_module.args.loss.args.low_amplitude_penalty_weight=750 lightning_module.args.loss.args.si_snr_weight=10 finetune_wandb_id=<wandb_id_of_stage_2> train.trainer.args.max_epochs=60 wandb.name="stage_3"

echo "Job finished successfully"