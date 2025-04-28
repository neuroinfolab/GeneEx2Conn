# GLOBAL IMPORTS FILE

# general imports
import numpy as np
from numpy import linalg as LA
import pandas as pd
import sklearn
import math
import random 
import os
import pickle
import itertools
from itertools import combinations
import inspect
from collections import defaultdict
import importlib
import psutil
from collections import Counter
import ast
import time
import re
import ssl
import yaml
from functools import partial
import gc

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch
import umap
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D

# stats 
import scipy
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import entropy
from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, shapiro, levene
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator

# modeling
import lightgbm
import wandb
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, log_loss
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBRegressor, XGBRFRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from numpy.polynomial import Polynomial

# cuda, pytorch, GPU
import GPUtil
import cupy as cp
import torch
import torch.nn as nn
import torchmetrics
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import PearsonCorrCoef
from torch.nn.parallel import DistributedDataParallel as DDP

# gene expression
import scib
import scanpy as sc
import inmoose
from anndata import AnnData
from inmoose.pycombat import pycombat_norm, pycombat_seq

# network analysis
import networkx as nx
import matplotlib.patches as patches
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eig

# optimization
import scipy.optimize as optimize
from scipy.optimize import Bounds
from skopt.plots import plot_objective, plot_histogram

# gene ontology
import gseapy as gp

# neuro data and toolboxes
import abagen
import nichord
import nibabel as nib
import netneurotools
from enigmatoolbox.datasets import fetch_ahba
from enigmatoolbox.datasets import load_sc, load_sc_as_one, load_fc, load_fc_as_one
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.permutation_testing import rotate_parcellation
from nilearn import plotting
from nichord.chord import plot_chord
from nichord.glassbrain import plot_glassbrain
from nichord.combine import combine_imgs, plot_and_combine
from nichord.coord_labeler import get_idx_to_label
from netneurotools.datasets import fetch_schaefer2018
from netneurotools.plotting import plot_fsaverage
from netneurotools import datasets as nndata
from netneurotools import freesurfer as nnsurf
from netneurotools import stats as nnstats
from netneurotools.plotting import plot_point_brain