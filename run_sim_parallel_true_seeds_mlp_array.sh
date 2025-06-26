#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_125_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --job-name=gx2c_sim
#SBATCH --output=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

module purge
cd /scratch/asr655/neuroinformatics/GeneEx2Conn/

# Define seeds array
seeds=(42 1 2 3 4 5 6 7 8 9)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv \
  --overlay /scratch/asr655/main_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; \
  python -m sim.sim_run experiment_sbatch_config.yml --model_type dynamic_mlp --feature_type transcriptome --n_cvs 3 --null_model none --random_seed $seed"

echo "Job Over"