# From normal ela_feature_extraction.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count
from ioh import get_problem

# Use the random embedding module
from dimensionality_reduction import GaussianRandomEmbeddings

# pflacco imports
from pflacco.classical_ela_features import (
    # Classical ELA features
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_local,
    calculate_ela_curvate,
    calculate_ela_conv,

    # Cell mapping features
    calculate_cm_angle,
    calculate_cm_conv,
    calculate_cm_grad,

    # Linear model features
    calculate_limo,

    #Nearest better clustering
    calculate_nbc,

    # Dispersion features Lunacek and Whitley
    calculate_dispersion,

    # Information content features Muñoz et al.
    calculate_information_content,

    # PCA features
    calculate_pca
)

from pflacco.misc_features import (
    
    calculate_fitness_distance_correlation,
    calculate_gradient_features,
    calculate_hill_climbing_features,
    calculate_length_scales_features,
    calculate_sobol_indices_features)

# ---------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------
def read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def read_x_samples(file_path: Path) -> np.ndarray:
    return pd.read_csv(file_path).values

def save_csv(df: pd.DataFrame, out: Path):
    df.to_csv(out, index=False)


# ---------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------
def get_files(directory: Path, suffix: str) -> List[Path]:
    return list(directory.rglob(suffix))


def parse_common_parts(parts):
    """Extract dim, seed, samples, objective_type."""
    dim = int([p for p in parts if p.startswith("Dimension_")][0].split("_")[1])
    seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
    n_samples = int([p for p in parts if p.startswith("Samples_")][0].split("_")[1])
    objective_type = [p for p in parts if p in ["ELA_extraction", "reduction"]][0]
    return dim, seed, n_samples, objective_type


def distill_x_sample_list(file_list: List[Path]) -> Dict[Tuple, Path]:
    distilled = {}
    for f in file_list:
        try:
            dim, seed, n_samples, obj = parse_common_parts(f.parts)
            key = (dim, seed, n_samples, obj)
            distilled[key] = f
        except:
            print(f"Warning: skipping unparseable X file {f}")
    return distilled


def distill_y_sample_list(file_list: List[Path]) -> Dict[Tuple, Tuple[Path, int, int]]:
    distilled = {}
    for f in file_list:
        try:
            dim, seed, n_samples, obj = parse_common_parts(f.parts)
            func_id = int([p for p in f.parts if p.startswith("f_")][0].split("_")[1])
            inst_id = int([p for p in f.parts if p.startswith("id_")][0].split("_")[1])
            key = (dim, seed, n_samples, obj)
            distilled.setdefault(key, []).append((f, func_id, inst_id))
        except:
            print(f"Warning: skipping unparseable Y file {f}")
    return distilled


# ---------------------------------------------------------
# ELA Feature Extraction
# ---------------------------------------------------------
def extract_ela_features(seed: int, 
                         X: np.ndarray,
                           fX: np.ndarray,
                           dim:int,
                           fid: int,
                           inst_id: int) -> pd.DataFrame:
    
    # Instantiate a function object
    problem = get_problem(fid, inst_id, dim)


    ### CLASSICAL ELA FEATURES ###
    # Raw data is X and fX
    ela_meta = calculate_ela_meta(X, fX)
    ela_distr = calculate_ela_distribution(X, fX)
    ela_level = calculate_ela_level(X,fX, ela_level_quantiles=[0.1,0.25,0.5])

    # Require extra problem info and more samples
    #ela_local = calculate_ela_local(X, fX, problem,dim,lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ela_curvate = calculate_ela_curvate(X,fX,problem,dim,lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ela_conv = calculate_ela_conv(X, fX,problem)

    ### CELL MAPPING FEATURES ###
    #cm_angle = calculate_cm_angle(X, fX,lower_bound=-5.0, upper_bound=5.0)
    #cm_conv = calculate_cm_conv(X, fX, lower_bound=-5.0, upper_bound=5.0)
    #cm_grad = calculate_cm_grad(X, fX, lower_bound=-5.0, upper_bound=5.0)


    ### LINEAR MODEL FEATURES ###
    #limo = calculate_limo(X, fX, upper_bound=5.0, lower_bound=-5.0)

    ### NEAREST BETTER CLUSTERING ###
    nbc = calculate_nbc(X, fX)


    ### DISPERSION FEATURES ###
    disp = calculate_dispersion(X, fX)

    ### INFORMATION CONTENT FEATURES ###
    ic = calculate_information_content(X, fX, seed=seed)

    ### PCA FEATURES ###
    pca_features = calculate_pca(X, fX)


    # Use the miscellaneous features
    fdc = calculate_fitness_distance_correlation(X, fX, problem.optimum.y, minkowski_p=2.0)
    #grad_features = calculate_gradient_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #hc_features = calculate_hill_climbing_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ls_features = calculate_length_scales_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #sobol_features = calculate_sobol_indices_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)

    #return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **cm_angle, **cm_conv, **cm_grad, **limo}, index=[0])
    #return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **limo}, index=[0])
    return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **fdc}, index=[0])

# ---------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------
def worker_extract_and_save(args, **kwargs):
    key, x_file, y_file, func_id, inst_id, base_dir = args

    dim, seed, n_samples, obj_type = key

    X = read_x_samples(x_file)
    fX = read_csv(y_file)["fX"].values

    

    out_dir = base_dir / "ela_features_2" / obj_type / f"Dimension_{dim}" / f"seed_{seed}" \
              / f"Samples_{n_samples}" / f"f_{func_id}" / f"id_{inst_id}"

    
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.joinpath("ela_features_2.csv").exists():
        #print(f"Skipping existing: {out_dir}")
        # Open the existing file to check if it's valid
        #df = pd.read_csv(out_dir / "ela_features_2.csv")

        df = extract_ela_features(seed, X, fX, dim, func_id, inst_id)

        save_csv(df, out_dir / "ela_features_2.csv")
        print(f"Saved: {out_dir}")

        
        return True
    else:
        df = extract_ela_features(seed, X, fX, dim, func_id, inst_id)

        save_csv(df, out_dir / "ela_features.csv")
        print(f"Saved: {out_dir}")

        return True

def worker_extract_and_save_2(args, **kwargs):

    # Extract arguments
    bootstrap_rounds = kwargs.get("bootstrap_rounds", 30)
    seed_list = kwargs.get("seed_list", [*range(100,111)])
    reduction_ratio = kwargs.get("reduction_ratio", 0.5) # Dim reduction ratio

    key, x_file, y_file, func_id, inst_id, base_dir = args

    dim, seed, n_samples, obj_type = key

    # Get the multiplier of samples to dimension
    sample_multiplier = n_samples // dim

    # Get the target reduced dimension
    reduced_dim = max(2, int(dim * reduction_ratio))

    # Calculate the number of bootstrap samples needed to keep the "ratio" of some multiplier*dimension
    n_bootstrap_samples = reduced_dim * sample_multiplier

    X = read_x_samples(x_file)
    fX = read_csv(y_file)["fX"].values

    out_dir = base_dir / "ela_features_reduced" / obj_type / f"Dimension_{dim}" / f"seed_{seed}" \
              / f"Samples_{n_samples}" / f"f_{func_id}" / f"id_{inst_id}" / f"reduction_ratio_{reduction_ratio}"
    
    out_dir.mkdir(parents=True, exist_ok=True)


    for cur_seed in seed_list:
        # Instantiate a random number generator
        rng = np.random.default_rng(cur_seed)

        # Apply Gaussian Random Embedding
        gre = GaussianRandomEmbeddings(n_components=reduced_dim, random_state=rng.integers(0,10000, size=1)[0])
        X_reduced = gre.fit_transform(X, y=fX)

        # Save the model used for reduction
        gre.save_model(out_dir / "models" / f"random_embedding_seed_{cur_seed}.pkl", overwrite=True)

        # Perform bootstrap sampling on the reduced data
        for ii in range(bootstrap_rounds):

            # Bootstrap sampling
            bootstrap_indices = rng.choice(X_reduced.shape[0], size=n_bootstrap_samples, replace=True)
            X_bootstrap = X_reduced[bootstrap_indices]
            fX_bootstrap = fX[bootstrap_indices]

            # Extract ELA features on the reduced and bootstrapped data
            df_features = extract_ela_features(cur_seed, X_bootstrap, fX_bootstrap, reduced_dim, func_id, inst_id)

            # Save the features with a filename indicating the seed
            feature_file = out_dir / "features" / f'random_embedding_seed_{cur_seed}' / f"ela_features_reduced_seed_{cur_seed}_round_{ii}.csv"
            feature_file.parent.mkdir(parents=True, exist_ok=True)
            save_csv(df_features, feature_file.absolute().as_posix())
            print(f"Saved: {feature_file}")

    # df = extract_ela_features(seed, X, fX, dim, func_id, inst_id)
    # save_csv(df, out_dir / "ela_features_2.csv")
    # print(f"Saved: {out_dir}")

    return True

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    base_dir = Path(os.getcwd())
    sets_of_functions = [1,8,11,16,20]  # subset of BBOB functions for testing

    seed_list = [*range(100,111)]  # seeds 100 to 110 to seed a random projection
    bootstrap_rounds = 30 # Number of bootstrap rounds

    # resampling settings
    kwargs = {"seed_list": seed_list,
                "bootstrap_rounds": bootstrap_rounds,
                "compression_ratio": 0.5}

    
    #f1: Sphere Function (Separable)
    #f8: Rosenbrock Function (low-moderate conditioning, unimodal, non-separable)
    #f11: Discus Function (High conditioning, unimodal)
    #f16: Rastrigin Function (Multi-modal with adequate global structure)
    #f20: Schwefel Function (Multi-modal with weak global structure)

    x_files = get_files(base_dir, "*samples.csv")
    y_files = get_files(base_dir, "*evaluations.csv")

    x_dict = distill_x_sample_list(x_files)
    y_dict = distill_y_sample_list(y_files)

    # build list of tasks for multiprocessing
    tasks = []
    for key, x_file in x_dict.items():
        if key in y_dict:
            if key[0] <=40:  # only dimensions up to 40
                if key[2]/key[0]>=50:  # only if samples/dim >=50
                    for (y_file, func_id, inst_id) in y_dict[key]:
                        if func_id in sets_of_functions:
                            tasks.append((key, x_file, y_file, func_id, inst_id, base_dir))

    print(f"Found {len(tasks)} feature extraction tasks.")

    # run multiprocessing
    #n_proc = max(1, cpu_count()//2 - 1)
    n_proc = 1
    print(f"Using {n_proc} processes...")

    if n_proc == 1:
        for task in tasks:
            worker_extract_and_save_2(task, **kwargs)
    else:
        with Pool(n_proc) as pool:
            pool.map(worker_extract_and_save_2, tasks, **kwargs)


if __name__ == "__main__":
    main()