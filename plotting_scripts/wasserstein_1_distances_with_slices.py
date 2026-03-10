import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib

from typing import Tuple, List

# Import the Wasserstein distance computation functions
from scipy.stats import wasserstein_distance


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
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_plots_wasserstein_distances")

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25, 0.1]


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

def compute_wasserstein_distance(dataset1:pd.DataFrame, 
                                 dataset2:pd.DataFrame, 
                                 feature_name_list:List[str],
                                 function_id_list:List[int],
                                 instance_id_list:List[int]) -> pd.DataFrame:
    r"""
    Compute the Wasserstein distance for a specific feature, function ID, and instance ID between two datasets.
    
    Args
    --------------
        dataset1 (pd.DataFrame): The first dataset.
        dataset2 (pd.DataFrame): The second dataset.
        feature_name_list (List[str]): The list of feature names to compute the Wasserstein distance for.
        function_id_list (List[int]): The list of function IDs to compute the Wasserstein distance for.
        instance_id_list (List[int]): The list of instance IDs to compute the Wasserstein distance for.

    Returns
    --------------
        pd.DataFrame: A DataFrame containing the computed Wasserstein distances for each feature, function ID, and instance ID combination.
    
    """

    assert all(feature_name in dataset1.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in dataset1"
    assert all(feature_name in dataset2.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in dataset2"
    assert 'function_idx' in dataset1.columns, "Column 'function_idx' not found"
    assert 'function_idx' in dataset2.columns, "Column 'function_idx' not found"
    assert 'instance_idx' in dataset1.columns, "Column 'instance_idx' not found"
    assert 'instance_idx' in dataset2.columns, "Column 'instance_idx' not found"


    # Initialize a DataFrame to store the results
    result_df = pd.DataFrame(columns=['function_id', 'instance_id', 'feature_name', 'wasserstein_distance'])

    # Loop over the feature names, function IDs, and instance IDs to compute the Wasserstein distance for each combination
    for feature_name in feature_name_list:
        for function_id in function_id_list:
            for instance_id in instance_id_list:
                data1 = dataset1[(dataset1['function_idx'] == function_id) & (dataset1['instance_idx'] == instance_id)][feature_name].values
                data2 = dataset2[(dataset2['function_idx'] == function_id) & (dataset2['instance_idx'] == instance_id)][feature_name].values

                distance = wasserstein_distance(data1, data2)

                # Create a DataFrame to store the result
                result_df = pd.concat([result_df, pd.DataFrame({
                    'function_id': [function_id],
                    'instance_id': [instance_id],
                    'feature_name': [feature_name],
                    'wasserstein_distance': [distance]
                })], ignore_index=True)

    return result_df

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
                                                                        df_wasserstein_slices_all_in_0_05_0:pd.DataFrame,
                                                                        df_wasserstein_slices_all_in_0_05_gen:pd.DataFrame,
                                                                        df_wasserstein_slices_all_in_0_025_0:pd.DataFrame,
                                                                        df_wasserstein_slices_all_in_0_025_gen:pd.DataFrame,
                                                                        df_wasserstein_slices_all_in_0_01_0:pd.DataFrame,
                                                                        df_wasserstein_slices_all_in_0_01_gen:pd.DataFrame,
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
        df_wasserstein_slices_all_in_0_05_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the all-in slice dataset with 50% reduction.
        df_wasserstein_slices_all_in_0_05_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the all-in slice dataset with 50% reduction.
        df_wasserstein_slices_all_in_0_025_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the all-in slice dataset with 25% reduction.
        df_wasserstein_slices_all_in_0_025_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the all-in slice dataset with 25% reduction.
        df_wasserstein_slices_all_in_0_01_0 (pd.DataFrame): DataFrame containing Wasserstein distances for slice 0 of the all-in slice dataset with 10% reduction.
        df_wasserstein_slices_all_in_0_01_gen (pd.DataFrame): DataFrame containing Wasserstein distances for the combined slices of the all-in slice dataset with 10% reduction.
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
            _filter(df_wasserstein_slices_all_in_0_05_0, "Slices All-In 0.5 (slice 0)"),
            _filter(df_wasserstein_slices_all_in_0_05_gen, "Slices All-In 0.5 (gen)"),
            _filter(df_wasserstein_slices_all_in_0_025_0, "Slices All-In 0.25 (slice 0)"),
            _filter(df_wasserstein_slices_all_in_0_025_gen, "Slices All-In 0.25 (gen)"),
            _filter(df_wasserstein_slices_all_in_0_01_0, "Slices All-In 0.1 (slice 0)"),
            _filter(df_wasserstein_slices_all_in_0_01_gen, "Slices All-In 0.1 (gen)"),
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

    df_wasserstein_reduced_05 = compute_wasserstein_distance(
        datasets[("full", 2000, None)],
        datasets[("oneshot", 200, 0.5)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_reduced_025 = compute_wasserstein_distance(
        datasets[("full", 2000, None)],
        datasets[("oneshot", 200, 0.25)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_reduced_01 = compute_wasserstein_distance(
        datasets[("full", 2000, None)],
        datasets[("oneshot", 200, 0.1)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

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

    # Now we have all the Wasserstein distance DataFrames computed, we can proceed to plotting them.
    for function_id in FUNCTION_IDS:
        for feature_name in all_feature_names:
            fig, ax = plot_violin_plots_wasserstein_distances_per_feature_function(
                df_wasserstein_full,
                df_wasserstein_reduced_05,
                df_wasserstein_reduced_025,
                df_wasserstein_reduced_01,
                df_wasserstein_slices_0_05_0,
                df_wasserstein_slices_0_05_gen,
                df_wasserstein_slices_0_025_0,
                df_wasserstein_slices_0_025_gen,
                df_wasserstein_slices_0_01_0,
                df_wasserstein_slices_0_01_gen,
                df_wasserstein_slices_all_in_0_05_0,
                df_wasserstein_slices_all_in_0_05_gen,
                df_wasserstein_slices_all_in_0_025_0,
                df_wasserstein_slices_all_in_0_025_gen,
                df_wasserstein_slices_all_in_0_01_0,
                df_wasserstein_slices_all_in_0_01_gen,
                feature_name,
                function_id,
            )

            figure_path = (
            SAVE_FIGURE_DIRECTORY
            / f"function_id_{function_id}"
            / f"feature_{feature_name}"
            )

            figure_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
            figure_path / "violin_plot_comparison_all_variants.pdf",
            dpi=300,
            #bbox_inches="tight",
            )

            plt.close(fig)




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