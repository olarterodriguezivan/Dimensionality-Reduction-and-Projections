import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from typing import Tuple, List

# Import the Wasserstein distance and wilcoxon computation functions
from scipy.stats import wasserstein_distance, wilcoxon



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

#FUNCTION_IDS:list = [1, 8, 11, 16, 20]  # Function IDs to consider
FUNCTION_IDS:list = [*range(1, 25)]  # Function IDs to consider
#FUNCTION_IDS:list = [20]  # Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("tables_wasserstein_1_distances_slices_stats")

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25, 0.1]

MODE=2 # 1 to include PCA features, 2 to exclude them


if MODE == 1:
    EXCLUDED_COLUMNS = {
    "embedding_seed",
    "round",
    "reduction_ratio",
    "instance_idx",
    "function_idx",
    "n_samples",
    "seed_lhs",
    "dimension",
    "group_id",
    "slice_id",
    }
elif MODE == 2:
    EXCLUDED_COLUMNS = {
    "embedding_seed",
    "round",
    "reduction_ratio",
    "instance_idx",
    "function_idx",
    "n_samples",
    "seed_lhs",
    "dimension",
    "group_id",
    "slice_id",
    # PCA features
    "pca.expl_var.cor_init",
    "pca.expl_var.cor_x", 
    "pca.expl_var.cov_init", 
    "pca.expl_var.cov_x", 
    "pca.expl_var_PC1.cor_init", 
    "pca.expl_var_PC1.cor_x",
    "pca.expl_var_PC1.cov_init",
    "pca.expl_var_PC1.cov_x",
    }

    P_ADJUST = "none"


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
        return "reduced_oneshot_3_200_0.25.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return "reduced_oneshot_3_200_0.5.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return "reduced_oneshot_3_200_0.1.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_oneshot_3_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_oneshot_3_2000_0.5.parquet"
    elif data_size == 2000 and reduction_ratio == 0.1:
        return "reduced_oneshot_3_2000_0.1.parquet"
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def choose_reduced_feature_file_slice(data_size:int, reduction_ratio:float, slice_id:int) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size, reduction ratio, and slice ID.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    slice_id (int): The slice ID.

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size, reduction ratio, and slice ID.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.25")
    elif data_size == 2000 and reduction_ratio == 0.5:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.5")
    elif data_size == 2000 and reduction_ratio == 0.1:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.1")
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def choose_reduced_feature_file_slice_all_in(data_size:int, reduction_ratio:float, slice_id:int) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size, reduction ratio, and slice ID,
    which are the all in versions of the slice datasets (i.e. they have one slice/group pairing so all the samples lie 
    in the same slice).

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    slice_id (int): The slice ID.

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size, reduction ratio, and slice ID.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.25")
    elif data_size == 2000 and reduction_ratio == 0.5:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.5")
    elif data_size == 2000 and reduction_ratio == 0.1:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.1")
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

def process_slice_dataframe(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Process the slice DataFrame by filtering considered seeds and erasing runtime columns.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The processed DataFrame.
    """

    df_filtered = df.copy() # Slices do not need seed filtering
    df_processed = erase_runtime_columns(df_filtered)
    df_processed = select_only_required_function_ids(df_processed)
    return df_processed


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def load_all_datasets():
    datasets = {}


    for n_samples in DATA_SIZES:
    # Full dataset
        full_file = choose_full_dataset_file(n_samples)
        datasets[("full", n_samples, None)] = process_dataframe(
        load_dataset_as_pd_df(full_file), n_samples
        )


        # Reduced datasets
        for ratio in REDUCTION_RATIOS:
            
            if not ratio==0.1:
                reduced_file = choose_reduced_feature_file(n_samples, ratio)
                datasets[("reduced", n_samples, ratio)] = process_dataframe(
                load_dataset_as_pd_df(reduced_file), n_samples
                )


            oneshot_file = choose_reduced_feature_file_one_shot(n_samples, ratio)
            datasets[("oneshot", n_samples, ratio)] = process_dataframe(
            load_dataset_as_pd_df(oneshot_file), n_samples
            )
    
            # Slice datasets (only for n_samples=200)
            if n_samples == 200:
                slice_file = choose_reduced_feature_file_slice(n_samples, ratio, slice_id=0)
                datasets[("slices", n_samples, ratio)] = process_slice_dataframe(
                load_dataset_as_pd_df(slice_file), n_samples
                )
            
            if n_samples == 200:
                slice_all_in_file = choose_reduced_feature_file_slice_all_in(n_samples, ratio, slice_id=0)
                datasets[("slices_all_in", n_samples, ratio)] = process_slice_dataframe(
                load_dataset_as_pd_df(slice_all_in_file), n_samples
                )


    return datasets


# -----------------------------------------------------------------------------
# Wasserstein distance computations (per instance, per function, per feature)
# -----------------------------------------------------------------------------

def compute_wasserstein_distance(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    feature_name_list,
    function_id_list,
    instance_id_list
):

    results = []

    # Pre-group datasets
    g1 = dataset1.groupby(['function_idx', 'instance_idx'])
    g2 = dataset2.groupby(['function_idx', 'instance_idx'])

    for function_id in function_id_list:
        for instance_id in instance_id_list:

            key = (function_id, instance_id)

            if key not in g1.groups or key not in g2.groups:
                continue

            df1 = g1.get_group(key)
            df2 = g2.get_group(key)

            for feature_name in feature_name_list:

                data1 = df1[feature_name].values
                data2 = df2[feature_name].values

                distance = wasserstein_distance(data1, data2)

                results.append({
                    'function_id': function_id,
                    'instance_id': instance_id,
                    'feature_name': feature_name,
                    'wasserstein_distance': distance
                })

    return pd.DataFrame(results)

def compute_wasserstein_distance_slices(ref_dataset:pd.DataFrame,
                                        slice_dataset:pd.DataFrame,
                                        feature_name_list:List[str],
                                        function_id_list:List[int],
                                        instance_id_list:List[int]) -> Tuple[pd.DataFrame]:
    r"""
    Compute the Wasserstein distance for each slice of the slice dataset compared to the reference dataset.
    
    Args
    --------------
        ref_dataset (pd.DataFrame): The reference dataset to compare against.
        slice_dataset (pd.DataFrame): The dataset containing the slices to compare.
        feature_name_list (List[str]): The list of feature names to compute the Wasserstein distance for.
        function_id_list (List[int]): The list of function IDs to compute the Wasserstein distance for.
        instance_id_list (List[int]): The list of instance IDs to compute the Wasserstein distance for.

    Returns
    --------------
        Tuple[pd.DataFrame]: A tuple of DataFrames, each containing the computed Wasserstein distances for a general and combined slice ID.
    """

    assert all(feature_name in ref_dataset.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in ref_dataset"
    assert all(feature_name in slice_dataset.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in slice_dataset"
    assert 'function_idx' in ref_dataset.columns, "Column 'function_idx' not found in ref_dataset"
    assert 'function_idx' in slice_dataset.columns, "Column 'function_idx' not found in slice_dataset"
    assert 'instance_idx' in ref_dataset.columns, "Column 'instance_idx' not found in ref_dataset"
    assert 'instance_idx' in slice_dataset.columns, "Column 'instance_idx' not found in slice_dataset"
    assert 'slice_id' in slice_dataset.columns, "Column 'slice_id' not found in slice_dataset"

    # Filter the reference dataset for the specific function IDs and instance IDs
    ref_data = ref_dataset[(ref_dataset['function_idx'].isin(function_id_list)) & (ref_dataset['instance_idx'].isin(instance_id_list))]

    # Filter the slice dataset for the specific function IDs and instance IDs
    slice_data = slice_dataset[(slice_dataset['function_idx'].isin(function_id_list)) & (slice_dataset['instance_idx'].isin(instance_id_list))]

    # Extract the slice 0 and the others
    slice_0_data = slice_data[slice_data['slice_id'] == 0]
    other_slices_data = slice_data[slice_data['slice_id'] != 0]

    # Compute the Wasserstein distance for slice 0
    df_wasserstein_slice_0 = compute_wasserstein_distance(ref_data, slice_0_data, feature_name_list, function_id_list, instance_id_list)

    # Compute the Wasserstein distance for the other slices combined
    df_wasserstein_other_slices = compute_wasserstein_distance(ref_data, other_slices_data, feature_name_list, function_id_list, instance_id_list)  

    return df_wasserstein_slice_0, df_wasserstein_other_slices


def plot_violin_plots_wasserstein_distances_per_feature_function(df_wasserstein_full:pd.DataFrame,
                                                                        df_wasserstein_reduced_05:pd.DataFrame,
                                                                        df_wasserstein_reduced_025:pd.DataFrame,
                                                                        df_wasserstein_reduced_01:pd.DataFrame,
                                                                        df_wasserstein_slices_0_05_0:pd.DataFrame,
                                                                        df_wasserstein_slices_0_05_gen:pd.DataFrame,
                                                                        df_wasserstein_slices_0_025_0:pd.DataFrame,
                                                                        df_wasserstein_slices_0_025_gen:pd.DataFrame,
                                                                        df_wasserstein_slices_0_01_0:pd.DataFrame,
                                                                        df_wasserstein_slices_0_01_gen:pd.DataFrame,
                                                                        feature_name:str,
                                                                        function_id:int,
                                                                        ) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plot violin plots of Wasserstein distances for a specific feature and function ID across different dataset variants.

    Args
    --------------
        df_wasserstein_full (pd.DataFrame): DataFrame containing Wasserstein distances for the full dataset.
        df_wasserstein_reduced_05 (pd.DataFrame): DataFrame containing Wasserstein distances for the reduced dataset with 50% reduction.
        df_wasserstein_reduced_025 (pd.DataFrame): DataFrame containing Wasserstein distances for the reduced dataset with 25% reduction.
        df_wasserstein_reduced_01 (pd.DataFrame): DataFrame containing Wasserstein distances for the reduced dataset with 10% reduction.
        df_wasserstein_slices_0_05_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the slice dataset with 50% reduction.
        df_wasserstein_slices_0_05_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the slice dataset with 50% reduction.
        df_wasserstein_slices_0_025_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the slice dataset with 25% reduction.
        df_wasserstein_slices_0_025_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the slice dataset with 25% reduction.
        df_wasserstein_slices_0_01_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the slice dataset with 10% reduction.
        df_wasserstein_slices_0_01_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the slice dataset with 10% reduction.      
        feature_name (str): The name of the feature to plot.
        function_id (int): The function ID to plot.
    
    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects containing the plot.
    """

    def _filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
        filtered = df[
            (df["feature_name"] == feature_name)
            & (df["function_id"] == function_id)
        ].copy()
        filtered["dataset_variant"] = label
        filtered["group"] = "Wasserstein"
        return filtered

    combined_df = pd.concat(
        [
            _filter(df_wasserstein_full, "Full"),
            _filter(df_wasserstein_reduced_05, "Reduced 0.5"),
            _filter(df_wasserstein_reduced_025, "Reduced 0.25"),
            _filter(df_wasserstein_reduced_01, "Reduced 0.1"),
            _filter(df_wasserstein_slices_0_05_0, "Slices 0.5 (slice 0)"),
            _filter(df_wasserstein_slices_0_05_gen, "Slices 0.5 (gen)"),
            _filter(df_wasserstein_slices_0_025_0, "Slices 0.25 (slice 0)"),
            _filter(df_wasserstein_slices_0_025_gen, "Slices 0.25 (gen)"),
            _filter(df_wasserstein_slices_0_01_0, "Slices 0.1 (slice 0)"),
            _filter(df_wasserstein_slices_0_01_gen, "Slices 0.1 (gen)"),
        ],
        ignore_index=True,
    )

    # Define a color palette
    palette = sns.color_palette("tab10", n_colors=combined_df["dataset_variant"].nunique())

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(
        x="group",
        y="wasserstein_distance",
        hue="dataset_variant",
        data=combined_df,
        inner="quartile",
        palette=palette,
        ax=ax,
        cut=0,
    )

    ax.set_title(
        f"Wasserstein Distance for Feature '{feature_name}', Function ID {function_id}"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Wasserstein Distance")

    # Remove x tick label
    ax.set_xticks([])

    # Improve legend placement
    ax.legend(
        title="Dataset Variant",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout()

    return fig, ax


# def heatmap_wasserstein_rankings(
#     df_wasserstein_full,
#     df_wasserstein_reduced_05,
#     df_wasserstein_reduced_025,
#     df_wasserstein_reduced_01,
#     df_wasserstein_slices_0_05_0,
#     df_wasserstein_slices_0_05_gen,
#     df_wasserstein_slices_0_025_0,
#     df_wasserstein_slices_0_025_gen,
#     df_wasserstein_slices_0_01_0,
#     df_wasserstein_slices_0_01_gen,
#     df_wasserstein_slices_all_in_0_05_0,
#     df_wasserstein_slices_all_in_0_05_gen,
#     df_wasserstein_slices_all_in_0_025_0,
#     df_wasserstein_slices_all_in_0_025_gen,
#     df_wasserstein_slices_all_in_0_01_0,
#     df_wasserstein_slices_all_in_0_01_gen,
#     feature_name_list,
#     function_id_list,
#     agg="median",
# ):

#     # ---------------------------------------------------------
#     # 1) Prepare long dataframe
#     # ---------------------------------------------------------

#     dataset_order = [
#         "Full (200 vs 2000)",
#         "Reduced 0.5",
#         "Reduced 0.25",
#         "Reduced 0.1",
#         "Sliced 0 – 0.5",
#         "Sliced gen – 0.5",
#         "Sliced 0 – 0.25",
#         "Sliced gen – 0.25",
#         "Sliced 0 – 0.1",
#         "Sliced gen – 0.1",
#         "Sliced 0 all in – 0.5",
#         "Sliced gen all in – 0.5",
#         "Sliced 0 all in – 0.25",
#         "Sliced gen all in – 0.25",
#         "Sliced 0 all in – 0.1",
#         "Sliced gen all in – 0.1",
#     ]

#     # 16 unique markers
#     markers = [
#         "o", "s", "D", "^", "v", "<", ">", "P",
#         "X", "*", "h", "H", "8", "p", "d", "|"
#     ]

#     # 16 distinct colors
#     dataset_colors = plt.get_cmap("tab20").colors[:len(dataset_order)]

#     dataset_to_marker = {
#         ds: markers[i] for i, ds in enumerate(dataset_order)
#     }

#     dataset_to_code = {
#         ds: i for i, ds in enumerate(dataset_order)
#     }

#     def prepare(df, label):
#         df2 = df.copy()
#         df2["dataset"] = label
#         return df2

#     df_all = pd.concat(
#         [
#             prepare(df_wasserstein_full, dataset_order[0]),
#             prepare(df_wasserstein_reduced_05, dataset_order[1]),
#             prepare(df_wasserstein_reduced_025, dataset_order[2]),
#             prepare(df_wasserstein_reduced_01, dataset_order[3]),
#             prepare(df_wasserstein_slices_0_05_0, dataset_order[4]),
#             prepare(df_wasserstein_slices_0_05_gen, dataset_order[5]),
#             prepare(df_wasserstein_slices_0_025_0, dataset_order[6]),
#             prepare(df_wasserstein_slices_0_025_gen, dataset_order[7]),
#             prepare(df_wasserstein_slices_0_01_0, dataset_order[8]),
#             prepare(df_wasserstein_slices_0_01_gen, dataset_order[9]),
#             prepare(df_wasserstein_slices_all_in_0_05_0, dataset_order[10]),
#             prepare(df_wasserstein_slices_all_in_0_05_gen, dataset_order[11]),
#             prepare(df_wasserstein_slices_all_in_0_025_0, dataset_order[12]),
#             prepare(df_wasserstein_slices_all_in_0_025_gen, dataset_order[13]),
#             prepare(df_wasserstein_slices_all_in_0_01_0, dataset_order[14]),
#             prepare(df_wasserstein_slices_all_in_0_01_gen, dataset_order[15]),
#         ],
#         ignore_index=True,
#     )

#     df_all = df_all[
#         df_all["feature_name"].isin(feature_name_list)
#         & df_all["function_id"].isin(function_id_list)
#     ]

#     # ---------------------------------------------------------
#     # 2) Aggregate
#     # ---------------------------------------------------------

#     df_agg = (
#         df_all
#         .groupby(["function_id", "feature_name", "dataset"])["wasserstein_distance"]
#         .agg(agg)
#         .reset_index()
#     )

#     # ---------------------------------------------------------
#     # 3) Rank (LOWER is better)
#     # ---------------------------------------------------------

#     df_ranked = (
#         df_agg
#         .sort_values(["function_id", "feature_name", "wasserstein_distance"])
#         .groupby(["function_id", "feature_name"], group_keys=False)
#         .apply(lambda x: x.assign(rank=np.arange(1, len(x) + 1)))
#     )

#     df_winner = df_ranked[df_ranked["rank"] == 1][
#         ["function_id", "feature_name", "dataset"]
#     ].rename(columns={"dataset": "winner"})

#     df_second = df_ranked[df_ranked["rank"] == 2][
#         ["function_id", "feature_name", "dataset"]
#     ].rename(columns={"dataset": "second"})

#     df_plot = df_winner.merge(
#         df_second,
#         on=["function_id", "feature_name"],
#         how="left",
#     )

#     df_plot["winner_code"] = df_plot["winner"].map(dataset_to_code)
#     df_plot["second_marker"] = df_plot["second"].map(dataset_to_marker)

#     # ---------------------------------------------------------
#     # 4) Pivot
#     # ---------------------------------------------------------

#     heatmap = df_plot.pivot(
#         index="function_id",
#         columns="feature_name",
#         values="winner_code",
#     )

#     heatmap = heatmap.reindex(index=function_id_list)
#     heatmap = heatmap[feature_name_list]

#     # ---------------------------------------------------------
#     # Plot
#     # ---------------------------------------------------------

#     cmap = ListedColormap(dataset_colors)
#     norm = BoundaryNorm(
#         np.arange(-0.5, len(dataset_colors) + 0.5, 1),
#         cmap.N,
#     )

#     fig, ax = plt.subplots(
#         figsize=(0.3 * heatmap.shape[1], 0.5 * heatmap.shape[0])
#     )

#     sns.heatmap(
#         heatmap,
#         cmap=cmap,
#         norm=norm,
#         linewidths=0.3,
#         cbar=False,
#         ax=ax,
#     )

#     # ---------------------------------------------------------
#     # Overlay runner-up markers
#     # ---------------------------------------------------------

#     feature_to_x = {f: i for i, f in enumerate(heatmap.columns)}
#     function_to_y = {f: i for i, f in enumerate(heatmap.index)}

#     for _, r in df_plot.iterrows():

#         if pd.isna(r["second_marker"]):
#             continue

#         if r["feature_name"] not in feature_to_x:
#             continue

#         if r["function_id"] not in function_to_y:
#             continue

#         ax.scatter(
#             feature_to_x[r["feature_name"]] + 0.5,
#             function_to_y[r["function_id"]] + 0.5,
#             marker=r["second_marker"],
#             s=60,
#             facecolors="none",
#             edgecolors="black",
#             linewidths=1,
#             zorder=10,
#         )

#     ax.set_xlabel("Feature")
#     ax.set_ylabel("Function")

#     # ---------------------------------------------------------
#     # Legends
#     # ---------------------------------------------------------

#     color_legend = ax.legend(
#         handles=[
#             Patch(color=dataset_colors[i], label=dataset_order[i])
#             for i in range(len(dataset_order))
#         ],
#         title="Best (lowest Wasserstein)",
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left",
#     )
#     ax.add_artist(color_legend)

#     ax.legend(
#         handles=[
#             Line2D(
#                 [0], [0],
#                 marker=m,
#                 color="black",
#                 linestyle="None",
#                 markerfacecolor="none",
#                 markersize=8,
#                 label=ds,
#             )
#             for ds, m in dataset_to_marker.items()
#         ],
#         title="Second-best",
#         bbox_to_anchor=(1.02, 0.1),
#         loc="upper left",
#     )

#     plt.tight_layout()
#     return fig, ax

def heatmap_wasserstein_rankings_2(combined_df: pd.DataFrame, 
                                   function_id_list:List[int],
                                   feature_name_list:List[str],
                                   agg:str="median") -> Tuple[plt.Figure, plt.Axes]:
    r"""
    A simplified version of the heatmap_wasserstein_rankings function that takes a single combined DataFrame as input.
    
    Args
    --------------
        combined_df (pd.DataFrame): A DataFrame containing columns 'function_id', 'feature_name', 'method', and 'wasserstein_distance'.
        feature_name_list (List[str]): The list of feature names to include in the heatmap.
        function_id_list (List[int]): The list of function IDs to include in the heatmap.
        agg (str): The aggregation method to use when averaging Wasserstein distances across instances. 
                   Must be either 'mean' or 'median'. Default is 'median'.
    
    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects containing the plot.
    """

    dataset_order = combined_df["method"].unique().tolist()

    # 16 unique markers
    markers = [
        "o", "s", "D", "^", "v", "<", ">", "P",
        "X", "*", "h", "H", "8", "p", "d", "|"
    ]

    # distinct colors
    # tab20 or tab10
    dataset_colors = plt.get_cmap("tab10").colors[:len(dataset_order)]

    dataset_to_marker = {
        ds: markers[i] for i, ds in enumerate(dataset_order)
    }

    dataset_to_code = {
        ds: i for i, ds in enumerate(dataset_order)
    }

    df_all = combined_df[
        combined_df["feature_name"].isin(feature_name_list)
        & combined_df["function_id"].isin(function_id_list)
    ]

    # ---------------------------------------------------------
    # 2) Aggregate
    # ---------------------------------------------------------

    df_agg = (
        df_all
        .groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
        .agg(agg)
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3) Rank (LOWER is better)
    # ---------------------------------------------------------

    df_ranked = (
        df_agg
        .sort_values(["function_id", "feature_name", "wasserstein_distance"])
        .groupby(["function_id", "feature_name"], group_keys=False)
        .apply(lambda x: x.assign(rank=np.arange(1, len(x) + 1)))
    )

    df_winner = df_ranked[df_ranked["rank"] == 1][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "winner"})

    df_second = df_ranked[df_ranked["rank"] == 2][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "second"})

    df_plot = df_winner.merge(
        df_second,
        on=["function_id", "feature_name"],
        how="left",
    )

    df_plot["winner_code"] = df_plot["winner"].map(dataset_to_code)
    df_plot["second_marker"] = df_plot["second"].map(dataset_to_marker)

    # ---------------------------------------------------------
    # 4) Pivot
    # ---------------------------------------------------------

    heatmap = df_plot.pivot(
        index="function_id",
        columns="feature_name",
        values="winner_code",
    )

    heatmap = heatmap.reindex(index=function_id_list)
    heatmap = heatmap[feature_name_list]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    cmap = ListedColormap(dataset_colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(dataset_colors) + 0.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(
        figsize=(0.3 * heatmap.shape[1], 0.5 * heatmap.shape[0])
    )

    sns.heatmap(
        heatmap,
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        cbar=False,
        ax=ax,
    )

    # ---------------------------------------------------------
    # Overlay runner-up markers
    # ---------------------------------------------------------

    feature_to_x = {f: i for i, f in enumerate(heatmap.columns)}
    function_to_y = {f: i for i, f in enumerate(heatmap.index)}

    for _, r in df_plot.iterrows():

        if pd.isna(r["second_marker"]):
            continue

        if r["feature_name"] not in feature_to_x:
            continue

        if r["function_id"] not in function_to_y:
            continue

        ax.scatter(
            feature_to_x[r["feature_name"]] + 0.5,
            function_to_y[r["function_id"]] + 0.5,
            marker=r["second_marker"],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Function")

    # ---------------------------------------------------------
    # Legends
    # ---------------------------------------------------------

    color_legend = ax.legend(
        handles=[
            Patch(color=dataset_colors[i], label=dataset_order[i])
            for i in range(len(dataset_order))
        ],
        title="Best (lowest Wasserstein)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(color_legend)

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker=m,
                color="black",
                linestyle="None",
                markerfacecolor="none",
                markersize=8,
                label=ds,
            )
            for ds, m in dataset_to_marker.items()
        ],
        title="Second-best",
        bbox_to_anchor=(1.02, 0.1),
        loc="upper left",
    )

    plt.tight_layout()
    return fig, ax



# def combine_wasserstein_results(list_of_dfs:List[pd.DataFrame], dataset_names:List[str]) -> pd.DataFrame:
#     r"""
#     This function combines multiple DataFrames containing Wasserstein distance results into a single DataFrame, 
#     adding a column to indicate the dataset/method corresponding to each row.

#     Args
#     --------------
#         list_of_dfs (List[pd.DataFrame]): A list of DataFrames to combine, each containing Wasserstein distance results.
#         dataset_names (List[str]): A list of names for each dataset.
    
#     Returns
#     --------------
#         pd.DataFrame: A combined DataFrame containing all Wasserstein distance results with an additional column indicating the dataset/method.
#     """
#     dfs = []
#     for df, name in zip(list_of_dfs, dataset_names):
#         d = df.copy()
#         d["method"] = name
#         dfs.append(d)

#     return pd.concat(dfs, ignore_index=True)


# def best_method_per_feature(df, agg="median") -> pd.DataFrame:
#     r"""
#     Determine best method per feature across functions.

#     Args
#     --------------
#         df (pd.DataFrame): A DataFrame containing columns 'function_id', 'feature_name', 'method', and 'wasserstein_distance'.
#         agg (str): The aggregation method to use when averaging Wasserstein distances across instances. 
#                    Must be either 'mean' or 'median'. Default is 'median'.
#     Returns
#     --------------
#         pd.DataFrame: A DataFrame containing the best method per feature.

#     """

#     assert agg in ["mean", "median"], "agg must be 'mean' or 'median'"

#     # aggregate across instances
#     df_agg = (
#         df.groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
#         .agg(agg)
#         .reset_index()
#     )

#     # rank per function-feature
#     df_agg["rank"] = (
#         df_agg
#         .groupby(["function_id", "feature_name"])["wasserstein_distance"]
#         .rank(method="dense", ascending=True)
#     )

#     # average rank across functions
#     df_feature = (
#         df_agg
#         .groupby(["feature_name", "method"])["rank"]
#         .mean()
#         .reset_index()
#     )

#     # determine winner
#     df_feature["final_rank"] = (
#         df_feature
#         .groupby("feature_name")["rank"]
#         .rank(method="dense", ascending=True)
#     )

#     return df_feature.sort_values(["feature_name", "final_rank"])

# def best_method_per_function(df: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
#     r"""
#     Determine best method per function across features.
#     Args
#     --------------
#         df (pd.DataFrame): A DataFrame containing columns 'function_id', 'feature_name', 'method', and 'wasserstein_distance'.
#         agg (str): The aggregation method to use when averaging Wasserstein distances across features.
    
#     Returns
#     --------------
#         pd.DataFrame: A DataFrame containing the best method per function.
#     """

#     assert agg in ["mean", "median"], "agg must be 'mean' or 'median'"

#     # aggregate across instances
#     df_inst = (
#         df.groupby(["function_id", "feature_name", "method", "instance_id"])["wasserstein_distance"]
#         .median()
#         .reset_index()
#     )

#     # aggregate across features
#     df_func = (
#         df_inst
#         .groupby(["function_id", "method"])["wasserstein_distance"]
#         .agg(agg)
#         .reset_index()
#     )

#     # rank per function
#     df_func["rank"] = (
#         df_func
#         .groupby("function_id")["wasserstein_distance"]
#         .rank(method="dense", ascending=True)
#     )

#     return df_func.sort_values(["function_id", "rank"])


# def significance_best_vs_second(df:pd.DataFrame) -> pd.DataFrame:
#     r"""
#     Compute significance of the difference between the best and second-best methods.

#     For each (function_id, feature_name) pair, we identify the best and second-best methods based on the median Wasserstein distance across instances.
#     We then perform a Wilcoxon signed-rank test to compare the Wasserstein distances of the best and second-best methods across instances, testing the alternative hypothesis that the best method has lower Wasserstein distances than the second-best method. 
#     The resulting DataFrame contains the function_id, feature_name, best_method, second_method, and p_value for each comparison.

#     Args
#     --------------
#         df (pd.DataFrame): A DataFrame containing columns 'function_id', 'feature_name', 'method', 'instance_id', and 'wasserstein_distance'.
    
#     Returns
#     --------------
#         pd.DataFrame: A DataFrame containing the function_id, feature_name, best_method, second_method, and p_value for each comparison between the best and second-best methods.
#     """

#     results = []

#     grouped = df.groupby(["function_id", "feature_name"])

#     for (f, g), group in grouped:

#         # aggregate across instances per method
#         med = (
#             group.groupby("method")["wasserstein_distance"]
#             .median()
#             .sort_values()
#         )

#         if len(med) < 2:
#             continue

#         best = med.index[0]
#         second = med.index[1]

#         g_best = group[group["method"] == best]
#         g_second = group[group["method"] == second]

#         merged = g_best.merge(
#             g_second,
#             on=["function_id", "instance_id", "feature_name"],
#             suffixes=("_best", "_second")
#         )

#         if len(merged) < 3:
#             continue

#         if merged["wasserstein_distance_best"].equals(merged["wasserstein_distance_second"]):
#             p = 1.0
#             stat = np.inf
#         else:
#             stat, p = wilcoxon(
#                 merged["wasserstein_distance_best"],
#                 merged["wasserstein_distance_second"],
#                 alternative="less",
#                 zero_method="zsplit",
#                 method="approx"
#             )

#         results.append({
#             "function_id": f,
#             "feature_name": g,
#             "best_method": best,
#             "second_method": second,
#             "p_value": p,
#             "statistic": stat
#         })

#     return pd.DataFrame(results)


# def global_method_ranking(df:pd.DataFrame) -> pd.DataFrame:
#     r"""
#     Compute the global ranking of methods based on their performance across all functions.

#     For each function, we determine the median Wasserstein distance for each method across features and instances, 
#     and rank the methods based on these median distances (lower is better).

#     We then compute the average rank of each method across all functions to obtain a global ranking of the methods.

#     Args
#     --------------
#         df (pd.DataFrame): A DataFrame containing columns 'function_id', 'method', and 'wasserstein_distance'.
    
#     Returns
#     --------------
#         pd.DataFrame: A DataFrame containing the global ranking of methods.
#     """

#     df_func = (
#         df.groupby(["function_id", "method"])["wasserstein_distance"]
#         .median()
#         .reset_index()
#     )

#     df_func["rank"] = (
#         df_func
#         .groupby("function_id")["wasserstein_distance"]
#         .rank(method="dense", ascending=True)
#     )

#     global_rank = (
#         df_func
#         .groupby("method")["rank"]
#         .mean()
#         .sort_values()
#     )

#     return global_rank

def combine_wasserstein_results(
    list_of_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> pd.DataFrame:
    r"""
    Combine multiple Wasserstein result dataframes into a single dataframe
    with a 'method' column.
    """
    if len(list_of_dfs) != len(dataset_names):
        raise ValueError("Length of list_of_dfs and dataset_names must match")

    combined = []
    for df_single, method_name in zip(list_of_dfs, dataset_names):
        tmp = df_single.copy()
        tmp["method"] = method_name
        combined.append(tmp)

    return pd.concat(combined, ignore_index=True)

from scipy.stats import friedmanchisquare, wilcoxon


def _aggregate_instances_per_feature(
    df: pd.DataFrame,
    agg: str = "median",
) -> pd.DataFrame:
    r"""
    Aggregate Wasserstein distances across instances for each
    (function_id, feature_name, method).

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg (str): Aggregation across instances, either 'mean' or 'median'.

    Returns
    --------------
        pd.DataFrame: Columns
            ['function_id', 'feature_name', 'method', 'wasserstein_distance'].
    """
    assert agg in ["mean", "median"], "agg must be 'mean' or 'median'"

    out = (
        df.groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
        .agg(agg)
        .reset_index()
    )
    return out


def best_method_per_function_rank_based(
    df: pd.DataFrame,
    agg_instances: str = "median",
    agg_ranks: str = "median",
) -> pd.DataFrame:
    r"""
    Determine the best method per function using rank aggregation across features.

    Pipeline
    --------------
    1. Aggregate Wasserstein distances across instances for each
       (function, feature, method).
    2. For each (function, feature), rank methods by Wasserstein distance
       (lower is better).
    3. Aggregate these ranks across features for each (function, method).
    4. Rank methods within each function based on the aggregated feature-rank score.

    This avoids aggregating raw Wasserstein distances across features, which may
    be on different scales.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances, 'mean' or 'median'.
        agg_ranks (str): Aggregation across feature-wise ranks, 'mean' or 'median'.

    Returns
    --------------
        pd.DataFrame: Columns
            ['function_id', 'method', 'aggregated_feature_rank', 'final_rank'].
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert agg_ranks in ["mean", "median"], "agg_ranks must be 'mean' or 'median'"

    # Step 1: aggregate over instances within each feature
    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    # Step 2: rank methods within each (function, feature)
    # smaller wasserstein_distance = better rank
    df_feat["feature_rank"] = (
        df_feat.groupby(["function_id", "feature_name"])["wasserstein_distance"]
        .rank(method="dense", ascending=True)
    )

    # Step 3: aggregate ranks across features
    df_func = (
        df_feat.groupby(["function_id", "method"])["feature_rank"]
        .agg(agg_ranks)
        .reset_index(name="aggregated_feature_rank")
    )

    # Step 4: final ranking per function
    df_func["final_rank"] = (
        df_func.groupby("function_id")["aggregated_feature_rank"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    return df_func.sort_values(["function_id", "final_rank", "method"]).reset_index(drop=True)


def best_method_per_feature_rank_based(
    df: pd.DataFrame,
    agg_instances: str = "median",
    agg_ranks: str = "mean",
) -> pd.DataFrame:
    r"""
    Determine the best method per feature across functions using rank aggregation.

    Pipeline
    --------------
    1. Aggregate Wasserstein distances across instances for each
       (function, feature, method).
    2. For each (function, feature), rank methods by Wasserstein distance.
    3. Aggregate these ranks across functions for each (feature, method).
    4. Rank methods within each feature.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances, 'mean' or 'median'.
        agg_ranks (str): Aggregation across functions, 'mean' or 'median'.

    Returns
    --------------
        pd.DataFrame: Columns
            ['feature_name', 'method', 'aggregated_function_rank', 'final_rank'].
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert agg_ranks in ["mean", "median"], "agg_ranks must be 'mean' or 'median'"

    # Step 1
    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    # Step 2
    df_feat["feature_rank"] = (
        df_feat.groupby(["function_id", "feature_name"])["wasserstein_distance"]
        .rank(method="dense", ascending=True)
    )

    # Step 3
    df_feature = (
        df_feat.groupby(["feature_name", "method"])["feature_rank"]
        .agg(agg_ranks)
        .reset_index(name="aggregated_function_rank")
    )

    # Step 4
    df_feature["final_rank"] = (
        df_feature.groupby("feature_name")["aggregated_function_rank"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    return df_feature.sort_values(["feature_name", "final_rank", "method"]).reset_index(drop=True)


def significance_best_vs_second_per_function_feature(
    df: pd.DataFrame,
    agg_instances: str = "median",
    p_adjust: str = "holm",
) -> pd.DataFrame:
    r"""
    For each (function, feature), test whether the top-ranked method is significantly
    better than the second-ranked method using a paired one-sided Wilcoxon test
    across instances.

    Notes
    --------------
    - Ranking is based on aggregated performance over instances for each method.
    - The significance test itself uses paired instance-level distances.
    - Smaller Wasserstein distance is better.
    - alternative='less' tests whether best < second.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances for ranking, 'mean' or 'median'.
        p_adjust (str): Currently supports 'holm' or 'none'.

    Returns
    --------------
        pd.DataFrame: One row per (function, feature), with
            best_method, second_method, p_value, adjusted_p_value, significant.
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert p_adjust in ["holm", "none"], "p_adjust must be 'holm' or 'none'"

    # For ranking the methods
    df_rank_source = _aggregate_instances_per_feature(df, agg=agg_instances)

    results = []

    for (function_id, feature_name), grp_rank in df_rank_source.groupby(["function_id", "feature_name"]):
        grp_rank = grp_rank.sort_values("wasserstein_distance", ascending=True)

        if grp_rank["method"].nunique() < 2:
            continue

        best_method = grp_rank.iloc[0]["method"]
        second_method = grp_rank.iloc[1]["method"]

        grp_raw = df[
            (df["function_id"] == function_id) &
            (df["feature_name"] == feature_name) &
            (df["method"].isin([best_method, second_method]))
        ].copy()

        best_df = grp_raw[grp_raw["method"] == best_method][
            ["instance_id", "wasserstein_distance"]
        ].rename(columns={"wasserstein_distance": "wd_best"})

        second_df = grp_raw[grp_raw["method"] == second_method][
            ["instance_id", "wasserstein_distance"]
        ].rename(columns={"wasserstein_distance": "wd_second"})

        merged = best_df.merge(second_df, on="instance_id", how="inner")

        # Need at least 2 paired observations for Wilcoxon to be meaningful
        if len(merged) < 2:
            p_value = np.nan
            statistic = np.nan
        else:
            try:
                statistic, p_value = wilcoxon(
                    merged["wd_best"].values,
                    merged["wd_second"].values,
                    alternative="less",
                    zero_method="zsplit",
                    method="approx",
                )
            except ValueError:
                # Happens for degenerate cases, e.g. all paired differences zero
                statistic, p_value = np.nan, np.nan

        results.append({
            "function_id": function_id,
            "feature_name": feature_name,
            "best_method": best_method,
            "second_method": second_method,
            "best_score": grp_rank.iloc[0]["wasserstein_distance"],
            "second_score": grp_rank.iloc[1]["wasserstein_distance"],
            "n_pairs": len(merged),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

    out = pd.DataFrame(results)

    if out.empty:
        out["adjusted_p_value"] = []
        out["significant"] = []
        return out

    if p_adjust == "none":
        out["adjusted_p_value"] = out["p_value"]
    else:
        out["adjusted_p_value"] = holm_adjust_pvalues(out["p_value"].values)

    out["significant"] = out["adjusted_p_value"] < 0.05
    return out.sort_values(["function_id", "feature_name"]).reset_index(drop=True)


def holm_adjust_pvalues(pvalues: np.ndarray) -> np.ndarray:
    r"""
    Holm step-down p-value adjustment.
    NaN values are preserved.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan, dtype=float)

    valid_mask = ~np.isnan(pvalues)
    valid_p = pvalues[valid_mask]

    if len(valid_p) == 0:
        return adjusted

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    m = len(sorted_p)

    holm_vals = np.empty(m, dtype=float)
    for i, p in enumerate(sorted_p):
        holm_vals[i] = (m - i) * p

    # enforce monotonicity
    holm_vals = np.maximum.accumulate(holm_vals)
    holm_vals = np.clip(holm_vals, 0.0, 1.0)

    # put back original order
    unsorted = np.empty(m, dtype=float)
    unsorted[order] = holm_vals

    adjusted[valid_mask] = unsorted
    return adjusted


def friedman_test_per_feature(
    df: pd.DataFrame,
    agg_instances: str = "median",
) -> pd.DataFrame:
    r"""
    Run a Friedman test for each feature across functions.

    Blocks = functions
    Treatments = methods
    Response = aggregated instance-level Wasserstein distance for that feature

    Returns one row per feature.
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"

    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    results = []

    for feature_name, grp in df_feat.groupby("feature_name"):
        pivot = grp.pivot_table(
            index="function_id",
            columns="method",
            values="wasserstein_distance",
            aggfunc="first"
        )

        # keep only complete blocks
        pivot = pivot.dropna(axis=0, how="any")

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            results.append({
                "feature_name": feature_name,
                "n_functions": pivot.shape[0],
                "n_methods": pivot.shape[1],
                "friedman_statistic": np.nan,
                "p_value": np.nan,
            })
            continue

        statistic, p_value = friedmanchisquare(*[pivot[c].values for c in pivot.columns])

        results.append({
            "feature_name": feature_name,
            "n_functions": pivot.shape[0],
            "n_methods": pivot.shape[1],
            "friedman_statistic": statistic,
            "p_value": p_value,
        })

    return pd.DataFrame(results).sort_values("feature_name").reset_index(drop=True)


# def compute_per_function_rankings(
#     list_of_dfs: List[pd.DataFrame],
#     dataset_names: List[str],
#     function_id_list: List[int],
#     feature_name_list: List[str],
#     agg: str = "mean"
# ) -> pd.DataFrame:
#     """
#     Compute per-function rankings of datasets based on Wasserstein distances.
#     """

#     if len(list_of_dfs) != len(dataset_names):
#         raise ValueError("Length of list_of_dfs and dataset_names must be the same")

#     if agg not in ["mean", "median"]:
#         raise ValueError("agg must be 'mean' or 'median'")

#     # -------------------------------------------------
#     # Combine datasets
#     # -------------------------------------------------

#     dfs = []
#     for df, name in zip(list_of_dfs, dataset_names):
#         df2 = df.copy()
#         df2["dataset"] = name
#         dfs.append(df2)

#     combined_df = pd.concat(dfs, ignore_index=True)

#     # -------------------------------------------------
#     # Filter functions and features
#     # -------------------------------------------------

#     df_all = combined_df[
#         (combined_df["function_id"].isin(function_id_list)) &
#         (combined_df["feature_name"].isin(feature_name_list))
#     ].copy()

#     # -------------------------------------------------
#     # Rank datasets per (function, feature)
#     # -------------------------------------------------

#     df_all["rank"] = (
#         df_all
#         .groupby(["function_id", "instance_id", "feature_name"])["wasserstein_distance"]
#         .rank(method="dense", ascending=True)
#     )

#     # -------------------------------------------------
#     # Aggregate ranks across features
#     # -------------------------------------------------

#     df_agg = (
#         df_all
#         .groupby(["function_id", "dataset"])["rank"]
#         .agg(agg)
#         .reset_index(name="aggregated_rank")
#     )

#     # -------------------------------------------------
#     # Final ranking per function
#     # -------------------------------------------------

#     df_agg["final_rank"] = (
#         df_agg
#         .groupby("function_id")["aggregated_rank"]
#         .rank(method="dense", ascending=True)
#         .astype(int)
#     )

#     df_agg = df_agg.sort_values(["function_id", "final_rank"])

#     # -------------------------------------------------
#     # Pivot output
#     # -------------------------------------------------

#     ranking_table = df_agg.pivot(
#         index="function_id",
#         columns="dataset",
#         values="final_rank"
#     )

#     return ranking_table




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    datasets = load_all_datasets()


    reference_df = datasets[("reduced", 200, 0.5)]
    all_feature_names = [
                            col for col in reference_df.columns
                            if col not in EXCLUDED_COLUMNS
    ]

    # Get the Wasserstein distances between full datasets of different sizes
    df_wasserstein_full = compute_wasserstein_distance(
        datasets[("full", 2000, None)],
        datasets[("full", 200, None)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    # df_wasserstein_reduced_05 = compute_wasserstein_distance(
    #     datasets[("full", 2000, None)],
    #     datasets[("oneshot", 200, 0.5)],
    #     all_feature_names,
    #     FUNCTION_IDS,
    #     INSTANCE_IDS
    # )

    # df_wasserstein_reduced_025 = compute_wasserstein_distance(
    #     datasets[("full", 2000, None)],
    #     datasets[("oneshot", 200, 0.25)],
    #     all_feature_names,
    #     FUNCTION_IDS,
    #     INSTANCE_IDS
    # )

    # df_wasserstein_reduced_01 = compute_wasserstein_distance(
    #     datasets[("full", 2000, None)],
    #     datasets[("oneshot", 200, 0.1)],
    #     all_feature_names,
    #     FUNCTION_IDS,
    #     INSTANCE_IDS
    # )

    df_wasserstein_slices_0_05_0, df_wasserstein_slices_0_05_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.5)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_0_025_0, df_wasserstein_slices_0_025_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.25)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_0_01_0, df_wasserstein_slices_0_01_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.1)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_05_0, df_wasserstein_slices_all_in_0_05_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.5)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_025_0, df_wasserstein_slices_all_in_0_025_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.25)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_01_0, df_wasserstein_slices_all_in_0_01_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.1)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_all_methods = combine_wasserstein_results(
    [
        df_wasserstein_full,
        df_wasserstein_slices_0_05_0,
        df_wasserstein_slices_0_025_0,
        df_wasserstein_slices_0_01_0,
        df_wasserstein_slices_all_in_0_05_0,
        df_wasserstein_slices_all_in_0_025_0,
        df_wasserstein_slices_all_in_0_01_0,
        #df_wasserstein_slices_0_05_gen,
        #df_wasserstein_slices_0_025_gen,
        #df_wasserstein_slices_0_01_gen,
        df_wasserstein_slices_all_in_0_05_gen,
        df_wasserstein_slices_all_in_0_025_gen,
        df_wasserstein_slices_all_in_0_01_gen,
    ],
    [
        "Full",
        "Slice0_05",
        "Slice0_025",
        "Slice0_01",
        "SliceAll0_05",
        "SliceAll0_025",
        "SliceAll0_01",
        #"SliceGen_05",
        #"SliceGen_025",
        #"SliceGen_01",
        "SliceGenAll_05",
        "SliceGenAll_025",
        "SliceGenAll_01",
    ]
)
    
    # 1) Best method per function overall, using rank aggregation across features
    df_best_per_function = best_method_per_function_rank_based(
        df_all_methods,
        agg_instances="median",
        agg_ranks="median",
    )

    print("\nBest method per function overall:")
    print(df_best_per_function.head(30))

    # 2) Best method per feature across functions
    df_best_per_feature = best_method_per_feature_rank_based(
        df_all_methods,
        agg_instances="median",
        agg_ranks="mean",
    )

    print("\nBest method per feature:")
    print(df_best_per_feature.head(30))

    # 3) Significance: best vs second-best for each function-feature pair
    df_significance = significance_best_vs_second_per_function_feature(
        df_all_methods,
        agg_instances="median",
        p_adjust=P_ADJUST,
    )

    print("\nSignificance results:")
    print(df_significance.head(30))

    # 4) Friedman test per feature across functions
    df_friedman = friedman_test_per_feature(
        df_all_methods,
        agg_instances="median",
    )

    print("\nFriedman per feature:")
    print(df_friedman.head(30))


    SAVE_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save results
    df_best_per_function.to_csv(SAVE_FIGURE_DIRECTORY / f"best_method_per_function_mode_{MODE}.csv", index=False)
    df_best_per_feature.to_csv(SAVE_FIGURE_DIRECTORY / f"best_method_per_feature_mode_{MODE}.csv", index=False)
    df_significance.to_csv(SAVE_FIGURE_DIRECTORY / f"significance_best_vs_second_mode_{MODE}.csv", index=False)
    df_friedman.to_csv(SAVE_FIGURE_DIRECTORY / f"friedman_per_feature_mode_{MODE}.csv", index=False)
    
    

    # The heatmap of best methods per function-feature combination is quite large, so we save it as a separate figure.
    fig, ax = heatmap_wasserstein_rankings_2(
        df_all_methods,
        FUNCTION_IDS,
        all_feature_names,
        agg="median"
    )

    fig.savefig(SAVE_FIGURE_DIRECTORY / f"wasserstein_ranking_heatmap_mode_{MODE}.pdf", dpi=300)

    # fig, ax = heatmap_wasserstein_rankings(
    # df_wasserstein_full,
    # df_wasserstein_reduced_05,
    # df_wasserstein_reduced_025,
    # df_wasserstein_reduced_01,
    # df_wasserstein_slices_0_05_0,
    # df_wasserstein_slices_0_05_gen,
    # df_wasserstein_slices_0_025_0,
    # df_wasserstein_slices_0_025_gen,
    # df_wasserstein_slices_0_01_0,
    # df_wasserstein_slices_0_01_gen,
    # df_wasserstein_slices_all_in_0_05_0,
    # df_wasserstein_slices_all_in_0_05_gen,
    # df_wasserstein_slices_all_in_0_025_0,
    # df_wasserstein_slices_all_in_0_025_gen,
    # df_wasserstein_slices_all_in_0_01_0,
    # df_wasserstein_slices_all_in_0_01_gen,
    # all_feature_names,
    # FUNCTION_IDS,
    # )

    # fig.savefig("wasserstein_ranking_heatmap.pdf", dpi=300)

    # Now we have all the Wasserstein distance DataFrames computed, we can proceed to plotting them.
    # for function_id in FUNCTION_IDS:
    #     for feature_name in all_feature_names:
    #         fig, ax = plot_violin_plots_wasserstein_distances_per_feature_function(
    #             df_wasserstein_full,
    #             df_wasserstein_reduced_05,
    #             df_wasserstein_reduced_025,
    #             df_wasserstein_reduced_01,
    #             df_wasserstein_slices_0_05_0,
    #             df_wasserstein_slices_0_05_gen,
    #             df_wasserstein_slices_0_025_0,
    #             df_wasserstein_slices_0_025_gen,
    #             df_wasserstein_slices_0_01_0,
    #             df_wasserstein_slices_0_01_gen,
    #             feature_name,
    #             function_id,
    #         )

    #         figure_path = (
    #         SAVE_FIGURE_DIRECTORY
    #         / f"function_id_{function_id}"
    #         / f"feature_{feature_name}"
    #         )

    #         figure_path.mkdir(parents=True, exist_ok=True)
    #         fig.savefig(
    #         figure_path / "violin_plot_comparison_all_variants.pdf",
    #         dpi=300,
    #         #bbox_inches="tight",
    #         )

    #         plt.close(fig)




    # # Get the differences between full datasets of different sizes
    # df_differences_full = compute_differences_full(
    #     datasets[("full", 2000, None)],
    #     datasets[("full", 200, None)],
    #     all_feature_names,
    #     agg="median"
    # )

    # # Get the differences between full and reduced datasets
    # df_differences_reduced_05 = compute_differences_in_reduced(
    #     datasets[("full", 2000, None)],
    #     #datasets[("reduced", 200, 0.5)],
    #     datasets[("oneshot", 200, 0.5)],
    #     all_feature_names,
    #     agg="median"
    # )

    # # Get the differences between full and reduced datasets
    # df_differences_reduced_025 = compute_differences_in_reduced(
    #     datasets[("full", 2000, None)],
    #     #datasets[("reduced", 200, 0.5)],
    #     datasets[("oneshot", 200, 0.25)],
    #     all_feature_names,
    #     agg="median"
    # )

    # df_differences_slices_0_05 = compute_differences_in_slices_0(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.5)],
    #     all_feature_names,
    #     agg="median"
    # )

    # df_differences_slices_gen_05 = compute_differences_in_slices_general(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.5)],
    #     all_feature_names,
    #     agg="median"
    # )

    # # Get the differences between full and slices datasets
    # df_differences_slices_0_025 = compute_differences_in_slices_0(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.25)],
    #     all_feature_names,
    #     agg="median"
    # )

    # df_differences_slices_gen_025 = compute_differences_in_slices_general(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.25)],
    #     all_feature_names,
    #     agg="median"
    # )

    #fig, ax = box_plots_of_differences(
    #        df_differences_full,
    #        df_differences_reduced_05,
    #        all_feature_names,
    #        function_id =20,
    #    )
    

    #plt.show()

    # Get the plot

    # for function_id in FUNCTION_IDS:
    #     for feature_name in all_feature_names:
    #         fig, ax = violin_plots_of_differences_global_2(
    #             df_differences_full,
    #             df_differences_reduced_05,
    #             df_differences_reduced_025,
    #             df_differences_slices_0_05,
    #             df_differences_slices_gen_05,
    #             df_differences_slices_0_025,
    #             df_differences_slices_gen_025,
    #             feature_name,
    #             function_id,
    #         )

    #         figure_path = (
    #         SAVE_FIGURE_DIRECTORY
    #         / f"function_id_{function_id}"
    #         / f"feature_{feature_name}"
    #         )

    #         figure_path.mkdir(parents=True, exist_ok=True)
    #         fig.savefig(
    #         figure_path / "violin_plot_comparison_all_variants.pdf",
    #         dpi=300,
    #         #bbox_inches="tight",
    #         )

    #         plt.close(fig)

    # for function_id in FUNCTION_IDS:
    #     fig, ax = box_plots_of_differences(
    #         df_differences_full,
    #         df_differences_reduced_05,
    #         all_feature_names,
    #         function_id =function_id,
    #     )

    #     figure_path = (
    #     SAVE_FIGURE_DIRECTORY
    #     / f"function_id_{function_id}"
    #     / f"comparison_full_vs_reduced"
    #     )

    #     figure_path.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(
    #     figure_path / "box_plot_comparison_full_vs_reduced.pdf",
    #     dpi=300,
    #     #bbox_inches="tight",
    #     )

    #     plt.close(fig)



    a = 1


    # for function_id in FUNCTION_IDS:
    #     print(f"Processing function ID: {function_id}")


    #     for feature_name in all_feature_names:
    #         print(f"Processing feature: {feature_name}")

    #     for n_samples in DATA_SIZES:
    #         for ratio in REDUCTION_RATIOS:
    #             print(f"Processing feature: {feature_name}")

    #             df_full = datasets[("full", n_samples, None)]
    #             df_reduced = datasets[("reduced", n_samples, ratio)]

    #             if n_samples == 200 and ratio == 0.25:
    #                 continue

    #             plot_color = "skyblue"


    #             #fig, ax = plot_violin_plots_biased_3_v2(
    #             #df_full,
    #             #df_reduced,
    #             #feature_name_list=all_feature_names,
    #             #reduction_ratio=ratio,
    #             #data_size=n_samples,
    #             #function_id=function_id,
    #             #instance_id_list=INSTANCE_IDS,
    #             #show_fig=False,
    #             #fig_size=(16, 3),
    #             #plot_color=plot_color,
    #             #)


    #             if fig is None:
    #                 continue


    #             figure_path = (
    #             SAVE_FIGURE_DIRECTORY
    #             / f"function_id_{function_id}"
    #             / f"reduction_ratio_{ratio}"
    #             / f"n_samples_{n_samples}"
    #             )
    #             figure_path.mkdir(parents=True, exist_ok=True)


    #             fig.savefig(
    #             figure_path / "violin_plot_corrected.pdf",
    #             dpi=300,
    #             bbox_inches="tight",
    #             )
    #             plt.close(fig)


if __name__ == "__main__":
    main()