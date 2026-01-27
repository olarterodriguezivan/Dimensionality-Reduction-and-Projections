import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib

from typing import Tuple, List


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
#FUNCTION_IDS:list = [20]  # Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_plots_phase_2")


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

def plot_violin_plots_unbiased_2(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    df_reduced_oneshot: pd.DataFrame,
    feature_name: str,
    reduction_ratio: float,
    data_size: int,
    function_id: int,
    instance_id: int, 
    seed_lhs: int,
    show_fig: bool = False,
    fig_size: Tuple = (14, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plots an aggregated violin plot (collapsed over embeddings) for the given
    function ID and instance ID, using full, reduced, and one-shot reduced data.
    """

    # --- Filter data ---
    df_full_filtered = df_full[
        (df_full["function_idx"] == function_id)
        & (df_full["instance_idx"] == instance_id)
        & (df_full["seed_lhs"] == seed_lhs)
    ]

    df_reduced_filtered = df_reduced[
        (df_reduced["function_idx"] == function_id)
        & (df_reduced["instance_idx"] == instance_id)
        & (df_reduced["seed_lhs"] == seed_lhs)
    ]

    df_reduced_oneshot_filtered = df_reduced_oneshot[
        (df_reduced_oneshot["function_idx"] == function_id)
        & (df_reduced_oneshot["instance_idx"] == instance_id)
        & (df_reduced_oneshot["seed_lhs"] == seed_lhs)
    ]

    # --- Extract feature values ---
    original_feature = df_full_filtered[[feature_name]]

    if original_feature.shape[0] != 1:
        raise ValueError("Original feature must contain exactly one row.")

    full_value = original_feature.iloc[0][feature_name]
    reduced_values = df_reduced_filtered[feature_name].values
    oneshot_values = df_reduced_oneshot_filtered[feature_name].values

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    # Violin plot for reduced data (collapsed over embeddings)
    sns.violinplot(
        y=reduced_values,
        inner="quartile",
        cut=0,
        density_norm="width",
        color="lightblue",
        ax=ax,
    )

    # Overlay one-shot reduced samples
    sns.stripplot(
        y=oneshot_values,
        color="darkorange",
        size=8,
        #jitter=True,
        ax=ax,
        label="One-shot reduced",
    )

    # Reference line for full-dimensional result
    ax.axhline(
        full_value,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label="Full Dimensional Feature Value",
    )

    ax.set_title(
        f"{feature_name} | fid={function_id}, instance={instance_id}, seed={seed_lhs}, "
        f"reduction_ratio={reduction_ratio}, n_samples={data_size}"
    )
    ax.set_ylabel(feature_name)
    ax.set_xticks([])

    # Legend (deduplicated)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    if show_fig:
        plt.show()

    return fig, ax


def plot_violin_plots_biased_2(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    df_reduced_oneshot: pd.DataFrame,
    feature_name: str,
    reduction_ratio: float,
    data_size: int,
    function_id: int,
    instance_id_list: List[int],
    show_fig: bool = False,
    fig_size: Tuple = (14, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Bias violin plots:
    - x-axis: instance index
    - violins aggregate over (seeds × embeddings)
    - bias computed per (function, instance, feature, seed)
    """

    # --- Select relevant rows ---
    df_full_f = df_full[
        (df_full["function_idx"] == function_id)
        & (df_full["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs", feature_name]]

    df_red_f = df_reduced[
        (df_reduced["function_idx"] == function_id)
        & (df_reduced["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs", feature_name]]

    df_red_os_f = df_reduced_oneshot[
        (df_reduced_oneshot["function_idx"] == function_id)
        & (df_reduced_oneshot["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs", feature_name]]

    # --- Merge reduced with full to compute bias ---
    merged = df_red_f.merge(
        df_full_f,
        on=["instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    merged["relative_bias"] = (
        (merged[f"{feature_name}_reduced"]
        - merged[f"{feature_name}_full"])
        / (np.abs(merged[f"{feature_name}_full"]) + EPSILON)
    )

    merged_os = df_red_os_f.merge(
        df_full_f,
        on=["instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    merged_os["relative_bias"] = (
        (merged_os[f"{feature_name}_reduced"]
        - merged_os[f"{feature_name}_full"])
        / (np.abs(merged_os[f"{feature_name}_full"]) + EPSILON)
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=merged,
        x="instance_idx",
        y="relative_bias",
        inner="quartile",
        cut=0,
        density_norm="width",
        color="lightblue",
        ax=ax,
    )

    sns.stripplot(
        data=merged_os,
        x="instance_idx",
        y="relative_bias",
        color="darkorange",
        size=6,
        jitter=True,
        ax=ax,
        label="One-shot relative bias",
    )

    # Zero-bias reference
    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Unbiased (0)",
    )

    ax.set_title(
        f"Relative Bias of {feature_name} | fid={function_id}, "
        f"reduction_ratio={reduction_ratio}, n_samples={data_size}"
    )
    ax.set_xlabel("Instance index")
    ax.set_ylabel("Relative Bias")

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

    full_dataset_file_200 = choose_full_dataset_file(200)
    df_full_ela_200 = process_dataframe(load_dataset_as_pd_df(full_dataset_file_200),200)

    reduced_feature_file_2000_05 = choose_reduced_feature_file(2000, 0.5)
    df_reduced_ela_2000_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_2000_05),2000)

    reduced_feature_file_2000_025 = choose_reduced_feature_file(2000, 0.25)
    df_reduced_ela_2000_025 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_2000_025),2000)

    reduced_feature_file_oneshot_2000_05 = choose_reduced_feature_file_one_shot(2000, 0.5)
    df_reduced_ela_oneshot_2000_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot_2000_05),2000)

    reduced_feature_file_oneshot_2000_025 = choose_reduced_feature_file_one_shot(2000, 0.25)
    df_reduced_ela_oneshot_2000_025 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot_2000_025),2000)

    reduced_feature_file_200_05 = choose_reduced_feature_file(200, 0.5)
    df_reduced_ela_200_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_200_05),200)

    reduced_feature_file_200_025 = choose_reduced_feature_file(200, 0.25)
    df_reduced_ela_200_025 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_200_025),200)

    reduced_feature_file_oneshot_200_05 = choose_reduced_feature_file_one_shot(200, 0.5)
    df_reduced_ela_oneshot_200_05 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot_200_05),200)

    reduced_feature_file_oneshot_200_025 = choose_reduced_feature_file_one_shot(200, 0.25)
    df_reduced_ela_oneshot_200_025 = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot_200_025),200)

    #feature_name = "ic.eps_max"

    # Get all the feature names available
    all_feature_names = [col for col in df_reduced_ela_2000_025.columns if col not in ["embedding_seed",
                                                                              "round",
                                                                              "reduction_ratio",
                                                                              "instance_idx",
                                                                              "function_idx",
                                                                              "n_samples",
                                                                              "seed_lhs",
                                                                              "dimension"]]
    
    # Loop all over functions, instances, seeds, and features
    for function_id in FUNCTION_IDS:
        for feature_name in all_feature_names:

            print(
                f"Plotting for feature: {feature_name}, "
            )

            fig, ax = plot_violin_plots_biased_2(
                                                df_full_ela_200,
                                                df_reduced_ela_200_05,
                                                df_reduced_ela_oneshot_200_05,
                                                feature_name=feature_name,
                                                reduction_ratio=0.5,
                                                data_size=200,
                                                function_id=function_id,
                                                instance_id_list=INSTANCE_IDS,
                                                show_fig=False,
                                                fig_size=(16, 6),
                                            )
            
            # Build directory hierarchy cleanly
            figure_path = (
                SAVE_FIGURE_DIRECTORY
                / f"fid{function_id}"
                / f"reduction_ratio_0.5"
                / f"n_samples_200"

            )

            figure_path.mkdir(parents=True, exist_ok=True)

            # Final filename
            figure_file = figure_path / f"violin_plot_{feature_name}.pdf"

            if not fig is None:
                fig.savefig(figure_file, dpi=300, bbox_inches="tight")
                plt.close(fig)


if __name__ == "__main__":
    main()