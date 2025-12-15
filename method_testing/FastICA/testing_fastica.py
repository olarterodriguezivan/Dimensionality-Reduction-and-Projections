
r"""Test Isomap with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import WeightedFastICA, is_supervised_model
from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
from ioh import get_problem


print("Testing Weighted FastICA on BBOB functions with rank-based weighting."
      )

print("------------------------------------------------------------")
print("Using Weighted FastICA for dimensionality reduction...")
print("Is the model supervised?:", is_supervised_model(WeightedFastICA))

# Define a simple test function
problem_id = 22  # Sphere function
problem_instance = 1 # Instance ID
dimension = 20

reduction_percent = 0.25  # Reduce to 50% of original dimensions
#n_components = int(dimension * reduction_percent)
n_components = 2  # Target number of components 

n_samples = 100*dimension  # Number of samples

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

# Compute function values
y_values = np.array([problem(x) for x in X])

# Get rank-based weighting
weights = get_rank_based_weighting(method="logarithmic").compute_weights(values=y_values,
                                                                         decay=0.4)

# Initialize Weighted PCA
isomap_model = WeightedFastICA(n_components=n_components,
                                 random_state=random_seed,
                                 max_iter=1000,
                                 tol=1e-5
                                 )

# Fit and transform the data
X_reduced = isomap_model.fit_transform(X, sample_weights=weights)

print("Original shape:", X.shape)
print(f"Reduced shape: {X_reduced.shape}")


# Verify the reduced dimensions
assert X_reduced.shape[1] == n_components, "Dimensionality reduction did not yield expected number of components."
print("Dimensionality reduction successful with rank-based weighting.")

if n_components == 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Function Value')
    plt.title('IsoMap Dimensionality Reduction of BBOB Function Samples')
    plt.xlabel('IsoMap Component 1')
    plt.ylabel('IsoMap Component 2')
    plt.grid(True)
    plt.show()
