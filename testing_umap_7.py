
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
#from dimensionality_reduction import ParametricUMAPWrapper
#from weighting_premises import get_rank_based_weighting
from umap.parametric_umap import ParametricUMAP
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
from ioh import get_problem


# Define a simple test function
problem_id = 1  # Sphere function
problem_instance = 1 # Instance ID
dimension = 60

reduction_percent = 0.2 # Reduce to 50% of original dimensions
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
#weights = get_rank_based_weighting(method="logarithmic").compute_weights(values=y_values,
 #                                                                        decay=0.4)


# Initialize Weighted PCA
umap_model = ParametricUMAP(batch_size=64,
                            n_neighbors=15,
                            min_dist=0.1,
                            n_components=n_components,
                            random_state=random_seed)

# Fit and transform the data
#ivis_model.fit(X,Y=y_values)
umap_model.fit(X, y=y_values)
X_reduced:np.ndarray = umap_model.transform(X)

print(f"Reduced shape: {X_reduced.shape}", 
      umap_model.get_params())

if n_components == 2:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Function Value')
    plt.title('UMAP Dimensionality Reduction of BBOB Function Samples')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.show()
