[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# GeneEx2Conn
This repository contains code for [Predicting Functional Brain Connectivity with Context-Aware Deep Neural Networks](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8UuFrz8AAAAJ&citation_for_view=8UuFrz8AAAAJ:eQOLeE2rZwMC) published at the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). Experimental code and modeling approaches are implemented herein to address the fundamental problem of predicting region-to-region human brain functional connectivity from gene expression and spatial information.

## Repository overview
Core repository functionalities are detailed below:
- **`/sim`**  
  - `sim.py` → simulation class defining features, targets, cross-validation, and run tracking  
  - `null.py` → null spin brain map generation and evaluation 
- **`/data`**  
  - `data_load.py` → load datasets and auxiliary data  
  - `data_utils.py` → data utilities, core RegionPairDataset class used for training, and target augmentation helpers  
  - `cv_split.py` → train-test split and cross-validation classes  
  - `BHA2/`, `HCP/`, `UKBB/` → subset of population average connectomes and parcellated transcriptomes  
  - `enigma/` → subsetted gene lists with known biological functions, null spin test indices and error metrics  
- **`/models`**  
  - `rules_based.py`, `bilinear.py`, `pls.py`, `dynamic_mlp.py` → baseline methods
  - `smt.py` → FlashAttention based MHSA transformer architectures including the _Spatiomolecular Transformer (SMT)_
  - `smt_advanced.py` → SMT variants, incorporating auxiliary information and pretrained embeddings 
  - `smt_cross.py` → compressed mixed-attention architectures (_in development_)  
  - `/configs/` → hyperparameter sweep configs for each model  
  - `/saved_models/` → pretrained SMT models for each dataset  
  - `train_val.py` → global training/validation loop  
  - `/metrics/`  
    - `eval.py` → evaluation class with 32+ prediction metrics 
- **`/notebooks`**  
  - `NeurIPS/` → Jupyter notebooks for analysis and figure creation

## Sim class
The main experimental functionality of `GeneEx2Conn` is the sim class. The sim class enables users to run detailed experiments depending on the underlying research question. Below is a basic example of a sim run, which can be triggered within a notebook or by command line.
```
single_sim_run(
              dataset='HCP',
              parcellation='S456',
              omit_subcortical=False,
              hemisphere='both',
              feature_type=[{'transcriptome': None}], # input features
              gene_list='0.2', # genes to retain for transcriptome
              connectome_target='FC', # connectivity target
              cv_type='spatial', # train-test split
              random_seed=42,
              model_type='shared_transformer', # model selection
              )
```
See `sim/sim_run.py` for further details. _Note: All Jupyter notebooks must be run from the root directory._

## Cross-validation
Careful train-test splits and null testing are critical to account for spatial autocorrelation in population average brain maps. `/data/cv_split.py` implements 4 train-test split styles: random, spatial (based on euclidean coordinates), community (based on Louvain communities), schaefer (based on Schafer parcellation functional subnetworks). Random and spatial splits are visualized below for `random seed=42`.
<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/data/glass/random_cv.gif" height="325"></td>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/data/glass/spatial_cv.gif" height="325"></td>
    </tr>
    <tr>
      <td align="center"><strong>Four-fold random split example</strong></td>
      <td align="center"><strong>Four-fold spatial split example</strong></td>
    </tr>
  </table>
</div>
Training regions in gray; test regions in orange. Pairwise patterns are learned in the training set to reconstruct test-set pairwise edge strength.

## Encoder-decoder architectures
All models in this repository are implemented in PyTorch. Models generally follow the form `Y_i,j = decode(enc(x_i), enc(x_j))`.
- Encoders: PCA, PLS, low-rank projection, autoencoder, transformer
- Decoders: bilinear layer, inner product, MLP task head

The proposed transformer-based architecture in [Predicting Functional Brain Connectivity with Context-Aware Deep Neural Networks](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8UuFrz8AAAAJ&citation_for_view=8UuFrz8AAAAJ:eQOLeE2rZwMC) is the Spatiomolecular Transformer (SMT).
- `smt.py` implements the standard version of the SMT and SMT w/ [CLS] featuring TSS-aware tokenization, a spatial context CLS token, FlashAttention multi-head self-attention (MHSA), and an MLP task head.
- `smt_advanced.py` includes extensions to the base SMT by performing MHSA over encoded versions of the gene expression vectors, such as PCA, PLS, autoencoded embeddings. Attention pooling is used in most of these models to compress embeddings from the transformer block.
- `smt_cross.py` implements compressed versions of the SMT operating at single gene resolution with a smaller inputted gene list. These models are more NLP style, seeking to learn a grammar over a learned vocabulary of select genes. They benefit from fewer parameters and may use region-to-region cross-attention in the encoding phase.
<p align="center">
  <img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/data/glass/NeurIPS_summary.png" height="425"><br>
</p>

Models are optimized to minimize the mean-squared error of predictions with the target population average connectome. Pretrained models can be found in `models/saved_models`. An example of how to load a pretrained model can be found in `/notebooks/NeurIPS/NeurIPS_Fig5_embeddings.ipynb`. See the `save_model` argument in `single_sim_run()` for saving a new model. 

## Datasets & access
- **Gene expression data:** The Allen Human Brain Atlas represents the most spatially resolved human gene expression dataset to date. Raw data is available [here](https://portal.brain-map.org/). This repo relies heavily on the [_abagen_](https://abagen.readthedocs.io/en/stable/index.html) package for AHBA preprocessing including normalizing, aggregating, and interpolating raw data into desired parcellations. Due to the size of the gene expression matrices, a sample csv file used for model training is made available for the coarsest parcellation resolution in `/data/BHA2/iPA_183`.
- **Neuroimaging data:** Models can be fit to connectomes from several open source datasets. MPI-LEMON is pubicly accessible [here](https://github.com/compneurobilbao/bha2). Access to HCP can be requested [here](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). Access to UKBB can be requested [here](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/). Population average connectomes are made available in `/data`. For access to underlying individualized connectomes please reach out with the appropriate data use agreements.

## Environment setup
`GeneEx2Conn` is a multi-component scientific research repository. Minimal requirements for our code are available in `/env/GeneEx2Conn.yml` (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html for environment setup).


## Citation
```
@inproceedings{
  ratzan2025predicting,
  title={Predicting Functional Brain Connectivity with Context-Aware Deep Neural Networks},
  author={Alexander Ratzan and Sidharth Goel and Junhao Wen and Christos Davatzikos and Erdem Varol},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=iQoZv77o3g}
}
```

<!--
## Developer notes
- If doing any substantial development with the repo, the assumption is that there is a `/GeneEx2Conn_data` folder one directory up from where the repo is setup. If you don't have access to this folder or if there are issues in pulling data from here, reach out to asr655 for access to this folder.
- To use this repo outside of NYU HPC Greene, setup a conda environment using `env/GeneEx2Conn.yml` from 'env' directory. This should download all relevant packages and toolboxes besides ENIGMA.
- A few extra steps may be required to get `enigmatoolbox` working:
    - Manually install enigma: `git clone https://github.com/MICA-MNI/ENIGMA.git; cd ENIGMA; python setup.py install`
    - Replace all references to `error_bad_lines=False` in code with `on_bad_lines='skip'` following [this](https://stackoverflow.com/questions/69513799/pandas-read-csv-the-error-bad-lines-argument-has-been-deprecated-and-will-be-re).
    - In `enigmatoolbox/datasets/matrices/hcp_connectivity/` rename all `funcMatrix_with_ctx_schaefer_{100,200,300,400}` to `funcMatrix_with_sctx_schaefer_{100,200,300,400}`.


## Setting up Jupyter notebook on NYU HPC (Greene)

To setup your Jupyter notebook on the HPC cluster, follow the instructions here [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/open-ondemand-ood-with-condasingularity?authuser=0). The key difference is that instead of setting up your own Singularity image, you can point your Jupyter config to our existing Singularity image, which contains all the necessary python packages.

Specifically, all you need to do is 

```
mkdir -p ~/.local/share/jupyter/kernels
cd ~/.local/share/jupyter/kernels
cp -R /share/apps/mypy/src/kernel_template ./my_env # this should be the name of your Singularity env
cd ./my_env 
```

And then in the file called `python` modify the existing singularity command at the bottom of the file to be

```
OVERLAY_FILE=/scratch/asr655/main_env/overlay-50G-10M.ext3
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif

singularity exec $nv \
  --overlay $OVERLAY_FILE:rw \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; $cmd $args"
```

This tells your Jupyter notebook to load its environment from this Singularity image. Now, when you launch your Jupyter notebook from the [OnDemand interface](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0#h.n3n5bitemm3g) you can select the kernel that you just defined.


<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/data/glass/S400_LAMA1_expression.png" height="300"></td>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/data/glass/cv_split_s400_structural_connectivity_spatial_split_10-3.gif" height="300"></td>
    </tr>
    <tr>
      <td align="center"><strong>Normalized expression of LAMA1 across brain</strong></td>
      <td align="center"><strong>10-fold spatial split visualization</strong></td>
    </tr>
  </table>
</div>
-->
