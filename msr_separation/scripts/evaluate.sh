#!/bin/bash
#SBATCH --job-name "evaluate"
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

srun --gpu-bind=none python -m src.train with roformer_8s_dataset_evaluate msr_bench_evaluation.wandb_id=<wandb_id_of_stage_3> wandb.name="evaluate_stage_3"

echo "Job finished successfully"