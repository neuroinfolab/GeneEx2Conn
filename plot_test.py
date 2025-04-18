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
from neuromaps import images, nulls
from neuromaps import datasets, plotting, resampling

# # Directory structure
# DATA_DIR = '/scratch/asr655/neuroinformatics/GeneEx2Conn/data'
# GLASS_DIR = '/scratch/asr655/neuroinformatics/GeneEx2Conn/glass'

# # Fetch ABAGEN PC1 annotation from neuromaps
# abagen_pc1 = datasets.fetch_annotation(source='abagen', desc='pc1', space='fsaverage', den='41k')
# print(abagen_pc1)

# # Load PC1 data
# # lh_pc1, rh_pc1 = abagen_pc1  # These are GIFTI image paths

# # Resample if necessary (we'll stick with fsaverage for this test)
# # Note: This assumes both hemispheres are in the correct format and resolution

# # Plot PC1 values on the fsaverage surface and save
# plotting.plot_surf(
#     (abagen_pc1),
#     template='fsaverage',
#     surf='inflated',
#     hemi='both',
#     layout='row',
#     show=False,
#     save=os.path.join(GLASS_DIR, 'abagen_pc1_fsaverage.png')
# )

# print("Saved PC1 visualization to:", os.path.join(GLASS_DIR, 'abagen_pc1_fsaverage.png'))

from neuromaps.datasets import fetch_annotation
from neuromaps.plotting import plot_hemispheres

# Set the path to your 'glass' directory
glass_dir = '/scratch/asr655/neuroinformatics/GeneEx2Conn/glass'

# Fetch the Abagen PC1 annotation in fsaverage space with 10k density
abagen_pc1 = fetch_annotation(source='abagen', desc='genepc1', space='fsaverage', den='10k')

# Plot the left and right hemispheres
fig, axes = plot_hemispheres(
    abagen_pc1,
    surf='inflated',
    colorbar=True,
    cmap='viridis',
    size=(800, 400)
)

# Save the figure to the 'glass' directory
output_path = os.path.join(glass_dir, 'abagen_pc1_surface.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
