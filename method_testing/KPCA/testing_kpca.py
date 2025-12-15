
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import ScikitKPCAWrapper
from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
#from torch import Tensor
#import torch
from ioh import get_problem


# Define a simple test function
problem_id = 1  # Sphere function
problem_instance = 1 # Instance ID
dimension = 20

reduction_percent = 0.5  # Reduce to 50% of original dimensions
#n_components = int(dimension * reduction_percent)
n_components = 2  # Target number of components

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

#X_tensor = torch.tensor(X, dtype=torch.float64)

# Compute function values
y_values = np.array([problem(x) for x in X])
#y_tensor = torch.tensor(y_values, dtype=torch.float64)

# Get rank-based weighting
weights = get_rank_based_weighting(method="inverse").compute_weights(values=y_values,
                                                                         decay=0.4)

# Initialize Weighted PCA
weighted_pca = ScikitKPCAWrapper(n_components=n_components,
                                 kernel='rbf',
                                 gamma=None,
                                 degree=3,
                                 coef0=1,
                                 random_state=random_seed)

# Fit and transform the data
X_reduced = weighted_pca.fit_transform(X, sample_weights=weights)

print(f"Reduced shape: {X_reduced.shape}")

# Save the model or results as needed
weighted_pca.save_model("models/weighted_kpca/weighted_kpca_model.joblib")

# Load the model
loaded_weighted_pca = ScikitKPCAWrapper.load_model("models/weighted_kpca/weighted_kpca_model.joblib")

if n_components == 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Function Value')
    plt.title('KPCA Dimensionality Reduction of BBOB Function Samples')
    plt.xlabel('KPCA Component 1')
    plt.ylabel('KPCA Component 2')
    plt.grid(True)
    plt.show()











