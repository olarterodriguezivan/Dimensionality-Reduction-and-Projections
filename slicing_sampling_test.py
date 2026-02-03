# ela_embedding_sampling_fixed.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from ioh import get_problem
from ioh.iohcpp.problem import RealSingleObjective

# Scipy imports
from scipy import sparse as sp
from scipy.stats.qmc import scale
from scipy.linalg import pinv

# Import some sampling method
from qmc_samplers import get_sampler

# Import combinatorial utilities
from itertools import product
# Import JSON utilities
import json

from pathlib import Path

# Import warnings
import warnings
from warnings import warn




# pflacco imports
from pflacco.classical_ela_features import (
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_nbc,
    calculate_dispersion,
    calculate_information_content,
    calculate_pca
)

from pflacco.misc_features import calculate_fitness_distance_correlation
from sklearn.dummy import check_random_state
from sklearn.random_projection import sample_without_replacement


### ---------------------------------------------------------
### CONSTANTS
### ---------------------------------------------------------

MAIN_DIR = Path(__file__).parent.resolve()
FUNCTION_IDS = [1,8,11,16,20] # "Ideally" list(range(1, 25 + 1))  # BBOB functions 1 to 25
INSTANCE_IDS = [*range(15)]  # BBOB instances 0 to 14 (15 total)



# ---------------------------------------------------------

def _check_input_size(d:int,
                      D:int)->None:
    r"""
    Check if the target embedding dimension is less than the original problem dimension.
    Parameters
    ----------
    d : int
        The target embedding dimension.
    D : int
        The original problem dimension.
    
    Returns
    -------
    None

    Raises
    ------
    ValueError
        If d is not less than D.

    """

    if d >= D:
        raise ValueError(f"Target embedding dimension d={d} must be less than original problem dimension D={D}.")

### --------------------------------------------------------------------------------------------------------------------
### DISCLAIMER: The following function is adapted from scikit-learn's implementations
### --------------------------------------------------------------------------------------------------------------------

def _check_density(density, D):
    """Factorize density check according to Li et al."""
    if density == "auto":
        density = 1 / np.sqrt(D)

    elif density <= 0 or density > 1:
        raise ValueError("Expected density in range ]0, 1], got: %r" % density)
    return density



def _gaussian_random_matrix(d:int, 
                            D:int, 
                            random_state:int =None):
    """Generate a dense Gaussian random matrix.

    The components of the random matrix are drawn from

        N(0, 1.0 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : ndarray of shape (n_components, n_features)
        The generated Gaussian random matrix.

    See Also
    --------
    GaussianRandomProjection
    """
    _check_input_size(d, D)
    rng = check_random_state(random_state)
    components = rng.normal(
        loc=0.0, scale=1.0 / np.sqrt(d), size=(d, D)
    )
    return components

def _sparse_random_matrix(d:int,
                           D:int,
                             density="auto",
                               random_state=None):
    
    r"""Generalized Achlioptas random sparse matrix for random projection.

    Setting density to 1 / 3 will yield the original matrix by Dimitris
    Achlioptas while setting a lower value will yield the generalization
    by Ping Li et al.

    If we note :math:`s = 1 / density`, the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(d)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(d)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    density : float or 'auto', default='auto'
        Ratio of non-zero component in the random projection matrix in the
        range `(0, 1]`

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : {ndarray, sparse matrix} of shape (n_components, n_features)
        The generated Gaussian random matrix. Sparse matrix will be of CSR
        format.

    See Also
    --------
    SparseRandomProjection

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           https://cgi.di.uoa.gr/~optas/papers/jl.pdf

    """
    _check_input_size(d, D)
    density = _check_density(density, D)
    rng = check_random_state(random_state)

    if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (d, D)) * 2 - 1
        return 1 / np.sqrt(d) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(d):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(D, density)
            indices_i = sample_without_replacement(
                D, n_nonzero_i, random_state=rng
            )
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix(
            (data, indices, indptr), shape=(d, D)
        )

        return np.sqrt(1 / density) / np.sqrt(d) * components

### --------------------------------------------------------------------------------------------------------------------
### END OF DISCLAIMER
### --------------------------------------------------------------------------------------------------------------------

### ---------------------------------------------------------
### ELA feature extraction method
### ---------------------------------------------------------

# ---------------------------------------------------------
# ELA Feature Extraction
# ---------------------------------------------------------
def extract_ela_features(seed: int,
                         X: np.ndarray,
                         fX: np.ndarray,
                         dim: int,
                         fid: int,
                         inst_id: int) -> pd.DataFrame:

    problem = get_problem(fid, inst_id, dim)

    ela_meta = calculate_ela_meta(X, fX)
    ela_distr = calculate_ela_distribution(X, fX)
    ela_level = calculate_ela_level(X, fX, ela_level_quantiles=[0.1, 0.25, 0.5])
    nbc = calculate_nbc(X, fX)
    disp = calculate_dispersion(X, fX)
    ic = calculate_information_content(X, fX, seed=seed)
    pca = calculate_pca(X, fX)

    fdc = calculate_fitness_distance_correlation(
        X, fX, problem.optimum.y, minkowski_p=2.0
    )

    features = {
        **ela_meta,
        **ela_distr,
        **ela_level,
        **nbc,
        **disp,
        **ic,
        **pca,
        **fdc
    }

    return pd.DataFrame(features, index=[0])

def _check_matrix_type(matrix_type:str)->None:
    r"""
    Check if the random matrix type is valid.
    Parameters
    ----------
    matrix_type : str
        The type of random matrix to generate. Options are "gaussian" or "sparse".
    
    Returns
    -------
    None

    Raises
    ------
    ValueError
        If matrix_type is not one of the valid options.

    """

    valid_types = ["gaussian", "sparse"]
    if matrix_type not in valid_types:
        raise ValueError(f"Invalid matrix_type: {matrix_type}. Must be one of {valid_types}.")
    

def examples_sampling_function_with_lifting(d: int,
    D: int,
    n_samples: int,
    seed: int,
    problem_BBOB,
    constraints: Tuple[float, float] = (-5.0, 5.0),
    normalize_embedding: bool = True      
) -> Dict[str, np.ndarray]:
    r"""
    
    An example function that demonstrates sampling on slices of a high-dimensional problem,
    with lifting samples back to the original space for evaluation.
    
    Args:
    ----------
    d (int): Target embedding dimension (d < D).
    D (int): Original problem dimension.
    n_samples (int): Number of samples to generate in the low-dimensional space.
    seed (int): Random seed for reproducibility.
    problem_BBOB (RealSingleObjective): An instance of an IOH problem from the BBOB suite.
    constraints (Tuple[float, float]): Box constraints for the problem.
    normalize_embedding (bool):  Whether to normalize the embedding vectors to unit length and make the embedding orthonormal.
    
    Returns:
    -------
    Dict[str, np.ndarray]: A dictionary containing:
    """

    print("This is an example function with lifting.")

    # --- Checks ---
    _check_input_size(d=d, D=D)

    # Use the seed for reproducibility
    rng = np.random.default_rng(seed=seed)

    # --- Generate Gaussian embedding ---
    embedding_ = rng.standard_normal(size=(D, d)) 

    # Normalize rows to unit length
    embedding_ /= np.linalg.norm(embedding_, axis=0, keepdims=True)

    if normalize_embedding:
        # Orthonormalize using SVD
        U, _, Vt = np.linalg.svd(embedding_, full_matrices=False)
        embedding_ = U @ Vt

    # Get the pseudoinverse for lifting
    embedding_pinv = pinv(embedding_)  # (d, D)

    # --- Sample in low-dimensional space ---
    sampler = get_sampler("lhs")
    samples = sampler(
        dim=d,
        n_samples=n_samples,
        random_seed=seed
    )

    # Scale to [-5, 5]^d
    samples = scale(
        samples,
        l_bounds=constraints[0] * np.ones(d),
        u_bounds=constraints[1] * np.ones(d)
    )*np.sqrt(D/d)  # scale by sqrt(D/d) as per random projection theory

    # --- Lift to original space ---
    samples_up = samples @ embedding_.T  # (n, D)

    # Clip to box constraints
    samples_up = np.clip(samples_up, constraints[0], constraints[1])

    # Project back down (consistency check / diagnostics)
    samples_down = samples_up @ embedding_pinv.T   # (n, d)


    # --- Evaluate objective in original space ---
    fitness_values = np.array([problem_BBOB(x) for x in samples_up])
    print(f"Evaluated fitness values: {fitness_values}")
    print(f"Generated low-d samples shape: {samples.shape}")
    print(f"Generated high-D samples shape: {samples_up.shape}")

    return {
        "low_d_samples": samples_down,
        "high_D_samples": samples_up,
        "fitness_values": fitness_values,
        "embedding": embedding_,
        "pseudoinverse_embedding": embedding_pinv
    }

def sample_embedding_matrix(d: int,
                            D: int,
                            n_samples: int,
                            seed: int,
                            constraints: Tuple[float, float] = (-5.0, 5.0),
                            normalize_embedding: bool = True      
                            ) -> Dict[str, np.ndarray]:
    r"""
    
    A function that samples points in a low-dimensional embedding space and lifts them to the original high-dimensional space.
    
    Args
    ----------
    - d (int): Target embedding dimension (d < D).
    - D (int): Original problem dimension.
    - n_samples (int): Number of samples to generate in the low-dimensional space.
    - seed (int): Random seed for reproducibility
    - constraints (Tuple[float, float]): Box constraints for the problem.
    - normalize_embedding (bool):  Whether to normalize the embedding vectors to unit length and make the embedding orthonormal.
    
    Returns
    -------

    Dict[str, np.ndarray]: A dictionary containing:
    - "low_d_samples": Samples in the low-dimensional space (after projection back down).
    - "high_D_samples": Samples lifted to the high-dimensional space.
    - "embedding": The random embedding matrix used for projection.
    - "pseudoinverse_embedding": The pseudoinverse of the embedding matrix used for lifting.
    """

    print("This is an example function with lifting.")

    # --- Checks ---
    _check_input_size(d=d, D=D)

    # Use the seed for reproducibility
    rng = np.random.default_rng(seed=seed)

    # --- Generate Gaussian embedding ---
    embedding_ = rng.standard_normal(size=(D, d)) 

    # Normalize rows to unit length
    embedding_ /= np.linalg.norm(embedding_, axis=0, keepdims=True)

    if normalize_embedding:
        # Orthonormalize using SVD
        U, _, Vt = np.linalg.svd(embedding_, full_matrices=False)
        embedding_ = U @ Vt

    # Get the pseudoinverse for lifting
    embedding_pinv = pinv(embedding_)  # (d, D)

    # --- Sample in low-dimensional space ---
    sampler = get_sampler("lhs")
    samples = sampler(
        dim=d,
        n_samples=n_samples,
        random_seed=seed
    )

    # Scale to [-5, 5]^d
    samples = scale(
        samples,
        l_bounds=constraints[0] * np.ones(d),
        u_bounds=constraints[1] * np.ones(d)
    )*np.sqrt(D/d)  # scale by sqrt(D/d) as per random projection theory

    # --- Lift to original space ---
    samples_up = samples @ embedding_.T  # (n, D)

    # Clip to box constraints
    samples_up = np.clip(samples_up, constraints[0], constraints[1])

    # Project back down (consistency check / diagnostics)
    samples_down = samples_up @ embedding_pinv.T   # (n, d)


    print(f"Generated low-d samples shape: {samples.shape}")
    print(f"Generated high-D samples shape: {samples_up.shape}")

    return {
        "low_d_samples": samples_down,
        "high_D_samples": samples_up,
        "embedding": embedding_,
        "pseudoinverse_embedding": embedding_pinv
    }


def example_sampling_function(
    d: int,
    D: int,
    n_samples: int,
    seed: int,
    problem_BBOB,
    random_matrix_type: str = "gaussian",
    constraints: Tuple[float, float] = (-5.0, 5.0)  
) -> Dict[str, np.ndarray]:
    """
    An example function that demonstrates sampling on slices of a high-dimensional problem.

    Args:
    ----------
    d (int): Target embedding dimension (d < D).
    D (int): Original problem dimension.
    n_samples (int): Number of samples to generate in the low-dimensional space.
    seed (int): Random seed for reproducibility.
    problem_BBOB (RealSingleObjective): An instance of an IOH problem from the BBOB suite.

    """

    print("This is an example function.")

    # --- Checks ---
    _check_input_size(d=d, D=D)
    _check_matrix_type(random_matrix_type)

    rng = np.random.default_rng(seed)

    # --- Generate embedding ---
    if random_matrix_type == "sparse":
        embedding = _sparse_random_matrix(
            d=d,
            D=D,
            density="auto",
            random_state=seed
        ).toarray()   # force dense for consistency
    else:
        embedding = _gaussian_random_matrix(
            d=d,
            D=D,
            random_state=seed
        )

    print(f"Generated random embedding of shape: {embedding.shape}")  # (d, D)

    # --- Sample in low-dimensional space ---
    sampler = get_sampler("lhs")
    samples = sampler(
        dim=d,
        n_samples=n_samples,
        random_seed=seed
    )

    # Scale to [-5, 5]^d
    samples = scale(
        samples,
        l_bounds=constraints[0] * np.ones(d),
        u_bounds=constraints[1] * np.ones(d)
    )

    samples = samples*np.sqrt(D/d)  # scale by sqrt(d) as per random projection theory

    # --- Lift to original space ---
    # embedding: (d, D) -> pseudoinverse: (D, d)
    inv_embedding = pinv(embedding)

    # Map samples up: (n, d) -> (n, D) 
    # There's this requirement to lift samples back to the original space before evaluation
    samples_up = samples @ inv_embedding.T

    # Clip to box constraints
    samples_up = np.clip(samples_up, constraints[0], constraints[1])

    # --- Optional: project back down (consistency check / diagnostics) ---
    samples_down = samples_up @ embedding.T   # (n, d)

    # --- Evaluate objective in original space ---
    fitness_values = np.array([problem_BBOB(x) for x in samples_up])

    print(f"Evaluated fitness values: {fitness_values}")
    print(f"Generated low-d samples shape: {samples.shape}")
    print(f"Generated high-D samples shape: {samples_up.shape}")

    return {
        "low_d_samples": samples_down,
        "high_D_samples": samples_up,
        "fitness_values": fitness_values,
        "embedding": embedding,
        "pseudoinverse_embedding": inv_embedding  
    }

def determine_number_of_samples_per_slice(n:int,
                                          S:int)->int:
    r"""
    Determine the number of samples to draw per slice.

    Args
    ----------
    S (int): Number of slices.
    n (int): Total number of samples.

    Returns
    -------
    int: Number of samples per slice.
    """

    return max(2, n // S)

def compute_global_seed_array(n:int,
                              initial_seed:int)->List[int]:
    r"""
    Compute a list of global seeds for each dataset.

    Args
    ----------
    n (int): Total number of datasets.
    initial_seed (int): Initial seed value.

    Returns
    -------
    List[int]: List of global seeds.
    """

    return [initial_seed + i for i in range(n)]


if __name__ == "__main__":

    # Ambient dimension
    D = 20

    # Target embedding dimension
    d = 10

    # Total number of samples
    n_samples:int = 10*D

    # Total number of datasets to generate
    N = 40

    # Initial random seed
    initial_seed:int = 44

    # Number of slices
    n_slices = 2

    # Number of samples per slice
    n_samples_per_slice = determine_number_of_samples_per_slice(
        S=n_slices,
        n=n_samples
    )

    # Total number of datasets
    global_seeds = compute_global_seed_array(n=n_slices*N, initial_seed=initial_seed)

    # Get an array with the initial seed per group
    group_seeds = [*range(global_seeds[0], global_seeds[-1], n_slices)]

    SAVE_DIR = MAIN_DIR / "sampling_outputs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    
    for group_idx, seed_idx in enumerate(group_seeds):
        print(f"\n=== Generating datasets for group starting with seed {seed_idx} ===")

        group_list = []
        for slice_idx in range(n_slices):
            print(f"\n--- Sampling on slice {slice_idx + 1}/{n_slices} ---")
            resp = sample_embedding_matrix(d=d, D=D, n_samples=n_samples_per_slice, seed=seed_idx+slice_idx)
            group_list.append(resp)
            
        # Sample from function
        for fid,iid in product(FUNCTION_IDS,INSTANCE_IDS):
            problem = get_problem(fid, iid, D)
            print(f"\nEvaluating problem F{fid} instance {iid}")

            fit_val_list = []
            for ii, resp in enumerate(group_list):
                fitness_values = np.array([problem(x) for x in resp["high_D_samples"]])
                fit_val_list.append(fitness_values)
                print(f"Slice {ii + 1} computed")
            
            # Combine fitness values from all slices
            combined_fitness_values = np.concatenate(fit_val_list)
            combined_up_samples = np.vstack([resp["high_D_samples"] for resp in group_list])
            print(f"Combined up samples shape: {combined_up_samples.shape}")
            print(f"Combined fitness values shape: {combined_fitness_values.shape}")

            # Compute ELA features on combined samples
            ela_features_full = extract_ela_features(
                seed=seed_idx,
                X=combined_up_samples,
                fX=combined_fitness_values,
                dim=D,
                fid=fid,
                inst_id=iid
            )

            # Save the ELA features for the full set
            print(f"ELA features for full combined samples:\n{ela_features_full}\n")
            full_save_path = SAVE_DIR / f"f{fid}/iid_{iid}/group{group_idx}/full.csv"
            full_save_path.parent.mkdir(parents=True, exist_ok=True)
            ela_features_full.to_csv(full_save_path, index=False)

            # Compute ELA features on each slice separately
            for ii, resp in enumerate(group_list):
                ela_features_slice = extract_ela_features(
                    seed=seed_idx + ii,
                    X=resp["low_D_samples"],
                    fX=fit_val_list[ii],
                    dim=D,
                    fid=fid,
                    inst_id=iid
                )
                print(f"ELA features for slice {ii + 1}:\n{ela_features_slice}\n")
                slice_save_path = SAVE_DIR / f"f{fid}/iid_{iid}/group{group_idx}/slice{ii + 1}.csv"
                slice_save_path.parent.mkdir(parents=True, exist_ok=True)
                ela_features_slice.to_csv(slice_save_path, index=False)
