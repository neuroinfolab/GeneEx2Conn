#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_125_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|a100
#SBATCH --job-name=gx2c_smt
#SBATCH --output=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

module purge
cd /scratch/asr655/neuroinformatics/GeneEx2Conn/

CONFIG_PATH="$1"

# === Parse config filename from first argument ===
if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <config_file.yml>"
  exit 1
fi

# === Map SLURM_ARRAY_TASK_ID to (seed) === 
seeds=(1 2 3 4 5 6 7 8 9 42)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
echo "Running job with seed=${seed}"

# === Run job ===
singularity exec --nv \
  --overlay /scratch/asr655/main_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; \
  python -m sim.sim_run $CONFIG_PATH \
  --model_type shared_transformer \
  --feature_type transcriptome \
  --random_seed $seed \
  --skip_cv false \
  --n_cvs 5"

echo "Job Complete"