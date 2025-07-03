#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_125_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --job-name=gx2c_mlp_mps
#SBATCH --output=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A.out
#SBATCH --error=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A.err
#SBATCH --array=0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu
#GREENE GREENE_GPU_MPS=YES

module purge
cd /scratch/asr655/neuroinformatics/GeneEx2Conn/

# === Map SLURM_ARRAY_TASK_ID to seed ===
seeds=(1 2 3 4 5 6 7 8 9 42)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

echo "Running MLP with seed=${seed} for folds 0,1,2,3 in parallel under MPS"

# === Run 4 folds in parallel ===
for fold in 0 1 2 3; do
    singularity exec --nv \
      --overlay /scratch/asr655/main_env/overlay-50G-10M.ext3:ro \
      /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
      /bin/bash -c "source /ext3/env.sh; \
      python -m sim.sim_run experiment_config.yml \
      --model_type dynamic_mlp \
      --feature_type transcriptome \
      --random_seed $seed \
      --null_model none \
      --use_folds $fold \
      --n_cvs 5" & 
done

wait
echo "Job Complete for seed=${seed} with all folds"