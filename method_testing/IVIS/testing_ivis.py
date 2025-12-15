
r"""Test PCA with rank-based weighting on BBOB functions."""

import numpy as np
from dimensionality_reduction import IvisWrapper, is_supervised_model
#from weighting_premises import get_rank_based_weighting
from qmc_samplers import get_sampler
from scipy.stats import qmc
from typing import List, Union
from ioh import get_problem


print("Is the model supervised?:", is_supervised_model(IvisWrapper))

# Define a simple test function
problem_id = 1 # Gallagher 101 function
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
#weights = get_rank_based_weighting(method="logarithmic").compute_weights(values=y_values,
 #                                                                        decay=0.4)


# Initialize Weighted PCA
ivis_model =IvisWrapper(n_components=n_components,
                        k=15,
                        distance='euclidean',
                        epochs=1000,
                        n_epochs_without_progress=50,
                        model='szubert',
                        supervision_metric="mean_squared_logarithmic_error",
                        verbose=True)

# Fit and transform the data
#ivis_model.fit(X,Y=y_values)
ivis_model.fit(X, y=y_values)
X_reduced:np.ndarray = ivis_model.transform(X)

print(f"Reduced shape: {X_reduced.shape}")

# Save the model
ivis_model.save_model("models/ivis")

# Load the model
loaded_model= IvisWrapper()
loaded_model.load_model("models/ivis")


# Transform data using the loaded model
X_reduced_loaded = loaded_model.transform(X)


if n_components==2:
    import matplotlib.pyplot as plt

    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_values, cmap='viridis', s=5)
    plt.colorbar(label='Function Value')
    plt.title('IVIS Dimensionality Reduction with Rank-Based Weighting')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()











