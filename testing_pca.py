
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import WeightedPCA
from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
from torch import Tensor
import torch
from ioh import get_problem


# Define a simple test function
problem_id = 1  # Sphere function
problem_instance = 1 # Instance ID
dimension = 40

reduction_percent = 0.5  # Reduce to 50% of original dimensions
n_components = int(dimension * reduction_percent)

n_samples = 250*dimension  # Number of samples

random_seed = 1234 # Random seed for reproducibility


# Get BBOB problem
problem = get_problem(fid=problem_id, 
                      dimension=dimension, 
                      instance=problem_instance)

# Sample data from the problem
sampler = get_sampler('lhs')  # Latin Hypercube Sampling
X = sampler(dim=dimension, 
                n_samples=n_samples, 
                random_seed=random_seed)

# Get problem bounds
lb = problem.bounds.lb
ub = problem.bounds.ub


# Scale samples to the problem bounds
X = qmc.scale(X, lb, ub)

X_tensor = torch.tensor(X, dtype=torch.float64)

# Compute function values
y_values = np.array([problem(x) for x in X])
y_tensor = torch.tensor(y_values, dtype=torch.float64)












