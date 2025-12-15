
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import ScikitPCAWrapper
from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
#from torch import Tensor
#import torch
from ioh import get_problem


# Define a simple test function
problem_id = 9  # Sphere function
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
weights = get_rank_based_weighting(method="logarithmic").compute_weights(values=y_values,
                                                                         decay=0.4)

# Initialize Weighted PCA
weighted_pca = ScikitPCAWrapper(n_components=n_components)

# Fit and transform the data
X_reduced = weighted_pca.fit_transform(X, sample_weights=weights)

print(f"Reduced shape: {X_reduced.shape}")

# Perform the inverse transform
#X_reconstructed = weighted_pca.inverse_transform(X_reduced)

# Compute reconstruction error
#reconstruction_error = (X_tensor - X_reconstructed).norm()
#print(f"Reconstruction error (Frobenius norm): {reconstruction_error.item()}")

# Verify the reduced dimensions
assert X_reduced.shape[1] == n_components, "Dimensionality reduction did not yield expected number of components."
print("Dimensionality reduction successful with rank-based weighting.")

# Save the model
weighted_pca.save_model(path="models/weighted_pca/pca_bbob9_model.joblib", 
                      overwrite=True)

weighted_pca_loaded = ScikitPCAWrapper.load_model(path="models/weighted_pca/pca_bbob9_model.joblib")


if n_components == 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Function Value')
    plt.title('PCA Dimensionality Reduction of BBOB Function Samples')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()











