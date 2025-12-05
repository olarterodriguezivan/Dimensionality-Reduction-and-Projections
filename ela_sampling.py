from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count

# pflacco imports
from pflacco.classical_ela_features import (
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_nbc,
    calculate_dispersion,
    calculate_information_content,
    calculate_pca
)

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
def extract_ela_features(seed: int, X: np.ndarray, fX: np.ndarray) -> pd.DataFrame:
    ela_meta = calculate_ela_meta(X, fX)
    ela_distr = calculate_ela_distribution(X, fX)
    nbc = calculate_nbc(X, fX)
    disp = calculate_dispersion(X, fX)
    ic = calculate_information_content(X, fX, seed=seed)
    pca_features = calculate_pca(X, fX)

    return pd.DataFrame({**ela_meta, **ela_distr, **nbc, **disp, **ic, **pca_features}, index=[0])


# ---------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------
def worker_extract_and_save(args):
    key, x_file, y_file, func_id, inst_id, base_dir = args

    dim, seed, n_samples, obj_type = key

    X = read_x_samples(x_file)
    fX = read_csv(y_file)["fX"].values

    df = extract_ela_features(seed, X, fX)

    out_dir = base_dir / "ela_features" / obj_type / f"Dimension_{dim}" / f"seed_{seed}" \
              / f"Samples_{n_samples}" / f"f_{func_id}" / f"id_{inst_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csv(df, out_dir / "ela_features.csv")
    print(f"Saved: {out_dir}")

    return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    base_dir = Path(os.getcwd())

    x_files = get_files(base_dir, "*samples.csv")
    y_files = get_files(base_dir, "*evaluations.csv")

    x_dict = distill_x_sample_list(x_files)
    y_dict = distill_y_sample_list(y_files)

    # build list of tasks for multiprocessing
    tasks = []
    for key, x_file in x_dict.items():
        if key in y_dict:
            for (y_file, func_id, inst_id) in y_dict[key]:
                tasks.append((key, x_file, y_file, func_id, inst_id, base_dir))

    print(f"Found {len(tasks)} feature extraction tasks.")

    # run multiprocessing
    n_proc = max(1, 4)
    print(f"Using {n_proc} processes...")

    with Pool(n_proc) as pool:
        pool.map(worker_extract_and_save, tasks)


if __name__ == "__main__":
    main()
