
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import RandomEmbedding
#from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
#from torch import Tensor
#import torch
from ioh import get_problem


# Define a simple test function
problem_id = 1  # Sphere function
problem_instance = 1 # Instance ID
dimension = 8

reduction_percent = 0.5  # Reduce to 50% of original dimensions
#n_components = int(dimension * reduction_percent)
n_components = 2  # Target number of components

n_samples = 100*dimension  # Number of samples

random_seed = 45 # Random seed for reproducibility


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
#weights = get_rank_based_weighting(method="logarithmic").compute_weights(values=y_values,
#                                                                         decay=0.4)

# Initialize Weighted PCA
random_embedding = RandomEmbedding(n_components=n_components, random_state=random_seed)

random_embedding.fit(X, y=y_values)

# Fit and transform the data
X_reduced = random_embedding.transform(X)

print(f"Reduced shape: {X_reduced.shape}")



# Verify the reduced dimensions
assert X_reduced.shape[1] == n_components, "Dimensionality reduction did not yield expected number of components."
print("Dimensionality reduction successful with rank-based weighting.")

# Save the objects for later use
random_embedding.save_model(f"random_embedding_bbob_f{problem_id}_d{dimension}_inst{problem_instance}.pkl",
                            overwrite =True)


# Load the model back
loaded_model = RandomEmbedding.load_model(f"random_embedding_bbob_f{problem_id}_d{dimension}_inst{problem_instance}.pkl")


# Perform the inverse transform
X_reconstructed = loaded_model.inverse_transform(X_reduced)

# Compute reconstruction error
reconstruction_error = np.linalg.norm(X_reconstructed - X_reconstructed)
print(f"Reconstruction error (Frobenius norm): {reconstruction_error}")

if n_components == 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Function Value')
    plt.title('Dimensionality Reduction of BBOB Function Samples')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()











