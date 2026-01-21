import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

## =============================
## GECCO Conference Settings
## =============================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('fast') # Optimization: Use faster rendering style

## =============================
## CONSTANT CONFIGURATION
## =============================
DATASET_SIZE: int = 200
REDUCTION_RATIO: float = 0.25
FUNCTION_IDS: list = [1, 8, 11, 16, 20]
INSTANCE_IDS: list = [*range(15)]
DATASET_200_CONSIDERED_SEEDS = [*range(1001, 1041)]
DATASET_2000_CONSIDERED_SEEDS = [*range(2001, 2041)]

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_plots_sequential_2")

def choose_full_dataset_file(data_size: int) -> str:
    if data_size == 200: return "complete_data_2.csv"
    if data_size == 2000: return "complete_data_generated.parquet"
    raise ValueError("Unsupported DATASET_SIZE")

def choose_reduced_feature_file(data_size: int, ratio: float) -> str:
    mapping = {
        (200, 0.25): "reduced_1_200_0.25.parquet",
        (200, 0.5): "reduced_1_200_0.5.parquet",
        (2000, 0.25): "reduced_2_2000_0.25.parquet",
        (2000, 0.5): "reduced_2_2000_0.5.parquet"
    }
    return mapping.get((data_size, ratio))

def load_dataset_as_pd_df(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists(): raise FileNotFoundError(f"{file_path} not found")
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    seeds = DATASET_200_CONSIDERED_SEEDS if DATASET_SIZE == 200 else DATASET_2000_CONSIDERED_SEEDS
    df = df[df['seed_lhs'].isin(seeds)].copy()
    runtime_cols = [c for c in df.columns if 'runtime' in c.lower()]
    return df.drop(columns=runtime_cols)[df['function_idx'].isin(FUNCTION_IDS)]

def save_single_plot(feature_name, fid, iid, seed, full_val, red_data, red_oneshot, save_path):
    """Worker function to handle the plotting of a single feature."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Violin Plot
    sns.violinplot(data=red_data, x="embedding_seed", y=feature_name, 
                   inner="quartile", cut=0, density_norm='width', ax=ax, color="lightblue")
    
    # One-shot markers
    sns.stripplot(data=red_oneshot, x="embedding_seed", y=feature_name, 
                  color="darkorange", size=8, jitter=False, ax=ax, label="One-shot reduced")

    # Reference Line
    ax.axhline(full_val, color="darkgreen", linestyle="--", linewidth=2, label="Full Dimensional Value")

    ax.set_title(f"{feature_name} | fid={fid}, iid={iid}, rr={REDUCTION_RATIO}, seed={seed}")
    ax.set_ylabel(feature_name)
    
    # Legend deduplication
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.savefig(save_path / f"violin_{feature_name}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    # Load and Pre-process once
    df_full = process_dataframe(load_dataset_as_pd_df(choose_full_dataset_file(DATASET_SIZE)))
    df_red = process_dataframe(load_dataset_as_pd_df(choose_reduced_feature_file(DATASET_SIZE, REDUCTION_RATIO)))
    
    # Assume one-shot file selection logic exists as per original
    oneshot_file = choose_reduced_feature_file(DATASET_SIZE, REDUCTION_RATIO).replace("reduced_", "reduced_oneshot_")
    df_oneshot = process_dataframe(load_dataset_as_pd_df(oneshot_file))

    features = [c for c in df_red.columns if c not in [
        "embedding_seed", "round", "reduction_ratio", "instance_idx", 
        "function_idx", "n_samples", "seed_lhs", "dimension"
    ]]

    # Optimization: Group data to minimize filtering inside loops
    grouped_red = df_red.groupby(['function_idx', 'instance_idx', 'seed_lhs'])
    
    # Use ProcessPoolExecutor for parallel rendering
    with ProcessPoolExecutor() as executor:
        for (fid, iid, seed), red_chunk in grouped_red:
            # Quick filter for specific subsets
            full_row = df_full[(df_full['function_idx'] == fid) & 
                               (df_full['instance_idx'] == iid) & 
                               (df_full['seed_lhs'] == seed)]
            
            oneshot_chunk = df_oneshot[(df_oneshot['function_idx'] == fid) & 
                                       (df_oneshot['instance_idx'] == iid) & 
                                       (df_oneshot['seed_lhs'] == seed)]

            if full_row.empty: continue

            # Create Directory
            path = SAVE_FIGURE_DIRECTORY / f"n{DATASET_SIZE}/rr{REDUCTION_RATIO}/fid{fid}/iid{iid}/seed{seed}"
            path.mkdir(parents=True, exist_ok=True)

            # Submit tasks to the pool
            for feat in features:
                executor.submit(
                    save_single_plot, feat, fid, iid, seed, 
                    full_row[feat].values[0], red_chunk, oneshot_chunk, path
                )
                print(f"Queued: fid{fid}_iid{iid}_seed{seed}_{feat}")

if __name__ == "__main__":
    main()