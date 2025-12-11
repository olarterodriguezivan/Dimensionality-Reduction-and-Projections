from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from qmc_samplers import get_sampler
from scipy.stats.qmc import scale
import os
from pathlib import Path

"""
sampling_pyflacco.py

Generate samples from the BBOB suite and return/save a dataset of
samples + function values using pyflacco (with COCO backend).

This module tries to use pyflacco's BBOB utilities if available.
If pyflacco does not expose a direct BBOB helper, it instructs how
to install needed packages.

Dependencies (install as needed):
    pip install pyflacco cocoex numpy pandas

Usage example:
    python sampling_pyflacco.py --fun 1 --inst 1 --dim 5 --n 1000 --out samples.csv
"""



def generate_x_samples(
                       dim: int,
                       n_samples: int,
                       lower: float = -5.0,
                       upper: float = 5.0,
                       sampler_name: str = "lhs",
                       seed: int | None = None):
    """
    Generate samples in [lower, upper]^dim and evaluate the specified BBOB function.

    This function attempts to use pyflacco's BBOB wrapper. If pyflacco does not
    provide a direct wrapper in your installation, you need cocoex (COCO python)
    available and then you can evaluate problems via cocoex and create a
    pyflacco feature object manually.

    Returns:
        X: (n_samples, dim) numpy array of inputs
    """
    

    sampler = get_sampler(sampler_name)
    X_unit = sampler(dim=dim, n_samples=n_samples, random_seed=seed)
    X = scale(X_unit, l_bounds=lower, u_bounds=upper)
    return X


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate X_samples."
    )

    parser.add_argument(
        "--dim", type=int, required=True, help="Dimensionality of the problem."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples to generate."
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output name CSV file to save samples."
    )
    parser.add_argument(
        "--sampler", type=str, default="lhs", help="Sampler to use: 'halton', 'lhs', 'monte_carlo', 'sobol'."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for sampling."
    )
    parser.add_argument(
            "--dataset_type", type=str, default="reduction", choices=["ELA_extraction","reduction"], help="Dataset type: 'bbob' (default)."
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_samples = generate_x_samples(
        dim=args.dim,
        n_samples=args.n,
        sampler_name=args.sampler,
        seed=args.seed
    )

    save_path = Path(os.path.join(os.getcwd(),"x_samples",args.dataset_type,f"Dimension_{args.dim}",f"seed_{args.seed}",f"Samples_{args.n}"))

    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(X_samples, columns=[f"x{i+1}" for i in range(args.dim)])
    df.to_csv(save_path.joinpath(args.out), index=False)
    print(f"Saved {args.n} samples of dimension {args.dim} to {save_path / args.out}")

