from env.imports import *
import sys

absolute_root_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn'
print('absolute_root_path', absolute_root_path)
sys.path.append(absolute_root_path)

from data.data_load import load_transcriptome, load_connectome, load_network_labels
from data.data_viz import plot_connectome, plot_transcriptome

X = load_transcriptome(null_model='none', parcellation='S400', hemisphere='both', omit_subcortical=False, gene_list='0.2', sort_genes='expression', impute_strategy='mirror_interpolate')
Y = load_connectome(parcellation='S400', hemisphere='both', omit_subcortical=False)

plot_transcriptome(
    null_model='none',
    parcellation='S400', 
    hemisphere='both',
    omit_subcortical=False,
    gene_list='0.2',
    sort_genes='expression',
    impute_strategy='mirror_interpolate', 
    jupyter=False
)

#import netneurotools
#from netneurotools.datasets import fetch_schaefer2018
#from netneurotools.plotting import plot_fsaverage
#from netneurotools.freesurfer import find_fsaverage_centroids
import neuromaps
from neuromaps import datasets, images, plotting

# Fetch fsaverage surface atlas
fsaverage = datasets.fetch_fsaverage(density='41k', verbose=1)

# Generate dummy cortical data (left and right hemispheres)
data_lh = np.random.rand(41962)  # vertices for fsaverage 41k
data_rh = np.random.rand(41962)  

# Plot and save
fig = plotting.plot_surf_template(
    (data_lh, data_rh),
    template='fsaverage',
    density='41k',
    surf='inflated',
    mask_medial=True,
    cmap='viridis'
)

# Save the figure
fig.savefig('/scratch/asr655/neuroinformatics/GeneEx2Conn/glass/test_surface_plot.png')