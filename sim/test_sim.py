import sys
import os
relative_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print('relative_root_path', relative_root_path)
absolute_root_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn'
print('absolute_root_path', absolute_root_path)
sys.path.append(absolute_root_path)

from env.imports import *
from sim.sim import Simulation

sim = Simulation(
    species="c_elegans",
    feature_type='transcriptome',
    cv_type='random',
    model_type='dynamic_mlp',
    gpu_acceleration=True,
    # gene_list="innexins",
)

sim.load_data()