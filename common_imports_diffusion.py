

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore
from tabulate import tabulate
from torch.utils.data import DataLoader

from configs import run_float64, save_dir, device
from beta_scheduling import select_beta_schedule
from models import select_model_diffusion
from datasets_toy import select_dataset
from utils_diffusion import select_samples_for_plot, plot_samples
from utils_diffusion import save_config_json, create_save_dir_training, create_hdf_file_training
from utils_diffusion import  save_animation, create_hdf_file, create_save_dir
from utils_io import parse_args


def set_seed(seed):   
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # if run_float64: torch.set_default_dtype(torch.float64)
    torch.set_default_dtype(torch.float64) if run_float64 else torch.set_default_dtype(torch.float32)

