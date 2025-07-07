#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=pr_125_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --job-name=gx2c_sim
#SBATCH --output=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/GeneEx2Conn/sim/logs/sim_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

module purge
cd /scratch/asr655/neuroinformatics/GeneEx2Conn/

singularity exec --nv \
  --overlay /scratch/asr655/main_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh;
  python -m sim.sim_run experiment_config.yml --model_type shared_transformer_cls \
  --feature_type transcriptome --random_seed 1 \
  --null_model none"

echo "Job Over"