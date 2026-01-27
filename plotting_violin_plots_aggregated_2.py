import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib

from typing import Tuple, List, Optional


## =============================
## GECCO Conference Settings for plots
## =============================

# Set font to be consistent before importing pyplot
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Import pyplot
import matplotlib.pyplot as plt

## =============================
## CONSTANT CONFIGURATION
## =============================

FUNCTION_IDS:list = [1, 8, 11, 16, 20]  # Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_seed_aggregation")


#/% First import the datasets (just keep in mind to make the labelling consistent)
# ELA Features on Full-Dataset without Projections


def choose_full_dataset_file(data_size:int) -> str:
    r"""
    This function chooses the appropriate full dataset file based on the data size.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).

    Returns
    --------------
    str: The filename of the full dataset corresponding to the given data size.

    """
    if data_size == 200:
        return "complete_data_2.csv"
    elif data_size == 2000:
        return "complete_data_generated.csv"
    else:
        raise ValueError("Unsupported DATASET_SIZE")

def choose_reduced_feature_file(data_size:int, reduction_ratio:float) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size and reduction ratio.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size and reduction ratio.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return "reduced_1_200_0.25.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return "reduced_1_200_0.5.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_2_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_2_2000_0.5.parquet"
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")
    
def choose_reduced_feature_file_one_shot(data_size:int, reduction_ratio:float) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file for one-shot reduction based on the data size and reduction ratio.   

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    
    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size and reduction ratio.
    
    """

    if data_size == 200 and reduction_ratio == 0.25:
        return "reduced_oneshot_1_200_0.25.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return "reduced_oneshot_1_200_0.5.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_oneshot_2_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_oneshot_2_2000_0.5.parquet"
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def load_dataset_as_pd_df(file_path:str) -> pd.DataFrame:
    r"""
    Load a dataset from a given file path into a pandas DataFrame.

    Args
    --------------
    file_path (str): The path to the dataset file.
    
    Returns
    --------------
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """

    # Check first the file exists
    if Path(file_path).exists is False:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .parquet or .csv file.")
    
    return df

def filter_considered_seeds(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Filter the DataFrame to include only the considered seeds based on the DATASET_SIZE.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame containing a 'seed_lhs' column.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The filtered DataFrame containing only the considered seeds.
    """

    if dataset_size == 200:
        considered_seeds = DATASET_200_CONSIDERED_SEEDS
    elif dataset_size == 2000:
        considered_seeds = DATASET_2000_CONSIDERED_SEEDS
    else:
        raise ValueError("Unsupported DATASET_SIZE")

    filtered_df = df[df['seed_lhs'].isin(considered_seeds)].copy()
    return filtered_df

def erase_runtime_columns(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Erase runtime-related columns from the DataFrame.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    
    Returns
    --------------
    pd.DataFrame: The DataFrame with runtime-related columns removed.
    """

    runtime_columns = [col for col in df.columns if 'runtime' in col.lower()]
    df_cleaned = df.drop(columns=runtime_columns)
    return df_cleaned

def select_only_required_function_ids(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Select only the rows corresponding to the required function IDs.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame containing a 'function_idx' column.
    
    Returns
    --------------
    pd.DataFrame: The DataFrame filtered to include only the required function IDs.
    """

    filtered_df = df[df['function_idx'].isin(FUNCTION_IDS)].copy()
    return filtered_df


def process_dataframe(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Process the DataFrame by filtering considered seeds and erasing runtime columns.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The processed DataFrame.
    """

    df_filtered = filter_considered_seeds(df, dataset_size)
    df_processed = erase_runtime_columns(df_filtered)
    df_processed = select_only_required_function_ids(df_processed)
    return df_processed

def plot_relative_bias_violin(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    df_reduced_oneshot: pd.DataFrame,
    reduction_ratio: float,
    data_size: int,
    feature_name: str,
    function_id: int = 1,
    instance_id_list: Optional[list[int]] = None,
    seed_lhs_list: Optional[list[int]] = None,
    eps: float = 1e-9,
    show_fig: bool = False,
    fig_size: Tuple[int, int] = (14, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot relative bias:
        (f_reduced - f_full) / (|f_full| + eps)
    """

    if instance_id_list is None:
        instance_id_list = INSTANCE_IDS

    if seed_lhs_list is None:
        if data_size == 200:
            seed_lhs_list = DATASET_200_CONSIDERED_SEEDS
        elif data_size == 2000:
            seed_lhs_list = DATASET_2000_CONSIDERED_SEEDS
        else:
            raise ValueError("Unsupported data_size")

    key_cols = ["function_idx", "instance_idx", "seed_lhs"]

    # --- Full data ---
    mask = (
        (df_full["function_idx"] == function_id)
        & (df_full["instance_idx"].isin(instance_id_list))
        & (df_full["seed_lhs"].isin(seed_lhs_list))
    )
    df_full_f = df_full.loc[mask, key_cols + [feature_name]]

    # --- Reduced data ---
    mask = (
        (df_reduced["function_idx"] == function_id)
        & (df_reduced["instance_idx"].isin(instance_id_list))
        & (df_reduced["seed_lhs"].isin(seed_lhs_list))
    )
    df_red_f = df_reduced.loc[
        mask,
        key_cols + ["embedding_seed", feature_name],
    ]

    # --- One-shot reduced ---
    mask = (
        (df_reduced_oneshot["function_idx"] == function_id)
        & (df_reduced_oneshot["instance_idx"].isin(instance_id_list))
        & (df_reduced_oneshot["seed_lhs"].isin(seed_lhs_list))
    )
    df_red1_f = df_reduced_oneshot.loc[
        mask,
        key_cols + ["embedding_seed", feature_name],
    ]

    # --- Reduced ↔ Full ---
    df_bias = df_red_f.merge(
        df_full_f,
        on=key_cols,
        suffixes=("_red", "_full"),
        how="left",
    )

    df_bias["relative_bias"] = (
        (df_bias[f"{feature_name}_red"] - df_bias[f"{feature_name}_full"])
        / (np.abs(df_bias[f"{feature_name}_full"]) + eps)
    )

    # --- One-shot ↔ Full ---
    df_bias1 = df_red1_f.merge(
        df_full_f,
        on=key_cols,
        suffixes=("_red", "_full"),
        how="left",
    )

    df_bias1["relative_bias"] = (
        (df_bias1[f"{feature_name}_red"] - df_bias1[f"{feature_name}_full"])
        / (np.abs(df_bias1[f"{feature_name}_full"]) + eps)
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=df_bias,
        x="embedding_seed",
        y="relative_bias",
        cut=0,
        inner="quartile",
        density_norm="width",
        ax=ax,
        color="lightblue",
        label="Reduced",
    )

    sns.stripplot(
        data=df_bias1,
        x="embedding_seed",
        y="relative_bias",
        color="darkorange",
        size=7,
        jitter=False,
        ax=ax,
        label="One-shot reduced",
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

    ax.set_title(
        f"Relative bias | {feature_name}, "
        f"fid={function_id}, "
        f"reduction_ratio={reduction_ratio}, "
        f"n_samples={data_size}"
    )
    ax.set_xlabel("Embedding seed")
    ax.set_ylabel("Relative bias")

    ax.set_ylim(ymin=-1.25, ymax=1.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    if show_fig:
        plt.show()

    return fig, ax


def main( )->None:
    # Load the dataset
    full_dataset_file_2000 = choose_full_dataset_file(2000)
    df_full_ela_2000 = process_dataframe(load_dataset_as_pd_df(full_dataset_file_2000),2000)


    reduced_feature_file_2000_05 = choose_reduced_feature_file(2000, 0.5)
    df_reduced_ela_2000_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_2000_05),2000)

    reduced_feature_file_oneshot_2000_05 = choose_reduced_feature_file_one_shot(2000, 0.5)
    df_reduced_ela_oneshot_2000_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot_2000_05),2000)

    
    
    all_feature_names_1 = [col for col in df_reduced_ela_2000_05.columns if col not in ["embedding_seed",
                                                                              "round",
                                                                              "reduction_ratio",
                                                                              "instance_idx",
                                                                              "function_idx",
                                                                              "n_samples",
                                                                              "seed_lhs",
                                                                              "dimension"]]
    
    # Perform plotting here with a loop
    for function_id in FUNCTION_IDS:
        for feature_name in all_feature_names_1:
            print(
                f"Plotting for feature: {feature_name}, "
                f"function_id: {function_id}"
            )

            fig, ax = plot_relative_bias_violin(
                df_full=df_full_ela_2000,
                df_reduced=df_reduced_ela_2000_05,
                df_reduced_oneshot=df_reduced_ela_oneshot_2000_05,
                reduction_ratio=0.5,
                data_size=2000,
                feature_name=feature_name,
                function_id=function_id,
                instance_id_list=INSTANCE_IDS,
                seed_lhs_list=DATASET_2000_CONSIDERED_SEEDS,
                eps=1e-12,
                show_fig=False,
                fig_size=(14, 6),       
            )
            
                # Build directory hierarchy cleanly
            figure_path = (
                SAVE_FIGURE_DIRECTORY
                / f"fid{function_id}"
                / f"reduction_0.5"
                / f"data_size_2000"

            )

            figure_path.mkdir(parents=True, exist_ok=True)

            # Final filename
            figure_file = figure_path / f"violin_plot_aggregated_{feature_name}.pdf"

            if not fig is None:
                fig.savefig(figure_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
            
        

if __name__ == "__main__":    main()