# Gene2Conn/imports.py

# general imports
import numpy as np
from numpy import linalg as LA
import pandas as pd
import sklearn
import math
import random 
import os
import pickle
from itertools import combinations
from collections import defaultdict
import importlib
import psutil

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# stats 
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import ttest_ind, shapiro, levene
from sklearn.metrics import make_scorer
from scipy.stats import randint, uniform

# modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
import xgboost
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
import lightgbm
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# cuda and GPU
import cupy as cp
import torch
import GPUtil

# network analysis
import networkx as nx
import matplotlib.patches as patches

# optimization
import scipy.optimize as optimize
from scipy.optimize import Bounds

# gene ontology
import gseapy as gp

# enigma toolbox and data
from enigmatoolbox.datasets import fetch_ahba
from enigmatoolbox.datasets import load_sc, load_sc_as_one, load_fc, load_fc_as_one
from nilearn import plotting
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
