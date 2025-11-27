#!/bin/bash
#SBATCH --job-name "stage_1"
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

srun --gpu-bind=none python -m src.train with roformer_4s_dataset_step_lr_scheduler musica lightning_module.args.use_ema=1 train.trainer.args.devices=4 lightning_module.args.optimizer.args.lr=8e-3 wandb.name="stage_1"

echo "Job finished successfully"