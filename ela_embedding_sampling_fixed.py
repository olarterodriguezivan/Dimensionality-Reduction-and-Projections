# ela_embedding_sampling_fixed.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from ioh import get_problem

# Random embedding
from dimensionality_reduction import GaussianRandomEmbeddings

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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def read_csv(file_path: Path) -> pd.DataFrame:
    file_path = Path(file_path)
    return pd.read_csv(file_path)

def read_x_samples(file_path: Path) -> np.ndarray:
    file_path = Path(file_path)
    return pd.read_csv(file_path).values

def save_csv(df: pd.DataFrame, out: Path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


# ---------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------
def get_files(directory: Path, suffix: str) -> List[Path]:
    return list(directory.rglob(suffix))


def parse_common_parts(parts):
    dim = int([p for p in parts if p.startswith("Dimension_")][0].split("_")[1])
    seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
    n_samples = int([p for p in parts if p.startswith("Samples_")][0].split("_")[1])
    objective_type = [p for p in parts if p in ["ELA_extraction", "reduction"]][0]
    return dim, seed, n_samples, objective_type


def distill_x_sample_list(file_list: List[Path]) -> Dict[Tuple, Path]:
    distilled = {}
    for f in file_list:
        try:
            key = parse_common_parts(f.parts)
            distilled[key] = f
        except Exception:
            print(f"Warning: skipping unparseable X file {f}")
    return distilled


def distill_y_sample_list(file_list: List[Path]) -> Dict[Tuple, List[Tuple[Path, int, int]]]:
    distilled = {}
    for f in file_list:
        try:
            dim, seed, n_samples, obj = parse_common_parts(f.parts)
            func_id = int([p for p in f.parts if p.startswith("f_")][0].split("_")[1])
            inst_id = int([p for p in f.parts if p.startswith("id_")][0].split("_")[1])
            key = (dim, seed, n_samples, obj)
            distilled.setdefault(key, []).append((f, func_id, inst_id))
        except Exception:
            print(f"Warning: skipping unparseable Y file {f}")
    return distilled


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


# ---------------------------------------------------------
# Worker (random embedding + bootstrap)
# ---------------------------------------------------------
def worker_extract_and_save_2(args, **kwargs):

    bootstrap_rounds = kwargs.get("bootstrap_rounds", 30)
    seed_list = kwargs.get("seed_list", list(range(100, 111)))
    reduction_ratio = kwargs.get("reduction_ratio", 0.5)

    key, x_file, y_file, func_id, inst_id, base_dir = args
    dim, seed, n_samples, obj_type = key

    sample_multiplier = max(1, n_samples // dim)
    reduced_dim = max(2, int(dim * reduction_ratio))
    n_bootstrap_samples = reduced_dim * sample_multiplier

    X = read_x_samples(x_file)
    fX = read_csv(y_file)["fX"].values

    out_dir = (
        base_dir / "ela_features_reduced_2" / obj_type /
        f"D_{dim}" / f"s_{seed}" /
        f"N_{n_samples}" / f"f_{func_id}" /
        f"id_{inst_id}" / f"r_ratio_{reduction_ratio}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for cur_seed in seed_list:

        gre = GaussianRandomEmbeddings(
            n_components=reduced_dim,
            random_state=cur_seed
        )
        X_reduced = gre.fit_transform(X, y=fX)
        gre.save_model(model_dir / f"random_embedding_seed_{cur_seed}.pkl", overwrite=True)

        max_bs = min(n_bootstrap_samples, X_reduced.shape[0])

        feature_dir = out_dir / "features" / f"random_embedding_seed_{cur_seed}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        for ii in range(bootstrap_rounds):
            rng_embed = np.random.default_rng(seed=1000 + ii)
            idx = rng_embed.choice(X_reduced.shape[0], size=max_bs, replace=False)
            Xb = X_reduced[idx]
            fXb = fX[idx]

            feature_file = feature_dir / f"round_{ii}.csv"

            df = extract_ela_features(cur_seed, Xb, fXb, reduced_dim, func_id, inst_id)

            save_csv(df, feature_file)

    return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    base_dir = Path(os.getcwd()).absolute()

    sets_of_functions = [1, 8, 11, 16, 20]
    seed_list = list(range(100, 111))
    bootstrap_rounds = 30

    kwargs = {
        "seed_list": seed_list,
        "bootstrap_rounds": bootstrap_rounds,
        "reduction_ratio": 0.5
    }

    x_files = get_files(base_dir, "*samples.csv")
    y_files = get_files(base_dir, "*evaluations.csv")

    x_dict = distill_x_sample_list(x_files)
    y_dict = distill_y_sample_list(y_files)

    tasks = []
    for key, x_file in x_dict.items():
        if key in y_dict:
            dim, _, n_samples, _ = key
            if dim < 40 and n_samples / dim >= 50:
                for y_file, func_id, inst_id in y_dict[key]:
                    if func_id in sets_of_functions:
                        tasks.append((key, x_file, y_file, func_id, inst_id, base_dir))

    print(f"Found {len(tasks)} tasks.")

    n_proc = min(cpu_count(), 4)
    if n_proc == 1:
        for task in tasks:
            worker_extract_and_save_2(task, **kwargs)
    else:
        with Pool(n_proc) as pool:
            func = partial(worker_extract_and_save_2, **kwargs)
            pool.map(func, tasks)


if __name__ == "__main__":
    main()
