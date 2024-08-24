import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import shutil

# get data
import tarfile
import urllib.request
from pathlib import Path
from zlib import crc32
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from scipy.stats import binom

# explore data
import matplotlib.gridspec as gridspec
import seaborn as sns 

np.random.seed(seed=82)

