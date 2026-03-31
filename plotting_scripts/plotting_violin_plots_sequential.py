import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib

from typing import Tuple


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

DATASET_SIZE:int = 200  # Number of samples in the dataset
REDUCTION_RATIO:float = 0.25  # Reduction ratio used
#FUNCTION_IDS:list = [1, 8, 11, 16, 20]  # Function IDs to consider
FUNCTION_IDS:list = [*range(1, 25)]  # Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_plots_sequential")


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
        return "complete_data_generated.parquet"
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
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_oneshot_3_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_oneshot_3_2000_0.5.parquet"
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

def filter_considered_seeds(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Filter the DataFrame to include only the considered seeds based on the DATASET_SIZE.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame containing a 'seed_lhs' column.
    
    Returns
    --------------
    pd.DataFrame: The filtered DataFrame containing only the considered seeds.
    """

    if DATASET_SIZE == 200:
        considered_seeds = DATASET_200_CONSIDERED_SEEDS
    elif DATASET_SIZE == 2000:
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


def process_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Process the DataFrame by filtering considered seeds and erasing runtime columns.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    
    Returns
    --------------
    pd.DataFrame: The processed DataFrame.
    """

    df_filtered = filter_considered_seeds(df)
    df_processed = erase_runtime_columns(df_filtered)
    df_processed = select_only_required_function_ids(df_processed)
    return df_processed

def plot_violin_plots_unbiased(df_full:pd.DataFrame,
                      df_reduced:pd.DataFrame,
                      df_reduced_oneshot:pd.DataFrame,
                      feature_name:str,
                      function_id:int = 1,
                      instance_id:int = 0,
                      seed_lhs:int = 1001,
                      show_fig:bool = False,
                      fig_size:Tuple = (14,6)) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    
    r"""
    Plots violin plots for the given function ID, instance ID, and seed LHS using the full, reduced, and one-shot reduced DataFrames.

    Args
    --------------
    df_full (pd.DataFrame): The full dataset DataFrame.
    df_reduced (pd.DataFrame): The reduced dataset DataFrame.
    df_reduced_oneshot (pd.DataFrame): The one-shot reduced dataset DataFrame.
    feature_name (str): The name of the feature to plot.
    function_id (int): The function ID to filter on.
    instance_id (int): The instance ID to filter on.
    seed_lhs (int): The seed LHS to filter on.
    show_fig (bool): Whether to display the figure immediately.
    fig_size (Tuple): The size of the figure.
    """

    # Filter data for the specified function ID, instance ID, and seed LHS
    df_full_filtered = df_full[
        (df_full['function_idx'] == function_id) &
        (df_full['instance_idx'] == instance_id) &
        (df_full['seed_lhs'] == seed_lhs)
    ]

    df_reduced_filtered = df_reduced[
        (df_reduced['function_idx'] == function_id) &
        (df_reduced['instance_idx'] == instance_id) &
        (df_reduced['seed_lhs'] == seed_lhs)
    ]

    df_reduced_oneshot_filtered = df_reduced_oneshot[
        (df_reduced_oneshot['function_idx'] == function_id) &
        (df_reduced_oneshot['instance_idx'] == instance_id) &
        (df_reduced_oneshot['seed_lhs'] == seed_lhs)
    ]

    # Extract feature(s)
    original_feature = df_full_filtered[[feature_name]]
    reduced_feature = df_reduced_filtered[[feature_name]]
    reduced_feature_oneshot = df_reduced_oneshot_filtered[[feature_name]]


     # Check if original feature has exactly one row

    if original_feature.shape[0] != 1:
        raise ValueError("Original feature must contain exactly one row.")

    # --- Long-format reduced data with embedding seed ---
    reduced_long = df_reduced_filtered[["embedding_seed"]].join(reduced_feature)
    reduced_long = reduced_long.melt(
        id_vars="embedding_seed",
        var_name="feature",
        value_name="value",
    )

    reduced_long_oneshot = df_reduced_oneshot_filtered[["embedding_seed"]].join(reduced_feature_oneshot)
    reduced_long_oneshot = reduced_long_oneshot.melt(
        id_vars="embedding_seed",
        var_name="feature",
        value_name="value",
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    # --- Violin plot for reduced data ---
    sns.violinplot(
        data=reduced_long,
        x="embedding_seed",
        y="value",
        inner="quartile",
        cut=0,
        density_norm='width',
        ax=ax,
        color="lightblue",
    )

    # --- Overlay stripplot for one-shot reduced data ---
    sns.stripplot(
    data=reduced_long_oneshot,
    x="embedding_seed",
    y="value",
    color="darkorange",
    size=8,
    jitter=False,
    ax=ax,
    label="One-shot reduced",
    )

    # --- Reference line from original feature ---
    for col in original_feature.columns:
        ax.axhline(
            original_feature.iloc[0][col],
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label="Full Dimensional Feature Value",
        )

    ax.set_title(
        f"{feature_name} | fid={function_id}, instance={instance_id}, reduction_ratio={REDUCTION_RATIO}, seed_lhs={seed_lhs}, n_samples={DATASET_SIZE}"
    )
    ax.set_xlabel("Projection Identifier (embedding seed)")
    ax.set_ylabel(feature_name)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    if show_fig is True:
        plt.show()

    return fig, ax
    
    


# Create sample data or load your dataset
# df = pd.read_csv('your_data.csv')

# Sample data for demonstration
#data = {
#    'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'] * 10,
#    'values': [1, 2, 3, 2, 3, 4, 3, 4, 5] * 10
#}
#df = pd.DataFrame(data)

# Create violin plot
#plt.figure(figsize=(10, 6))
#sns.violinplot(data=df, x='category', y='values')
#plt.title('Violin Plot')
#plt.xlabel('Category')
#plt.ylabel('Values')
#plt.show()

# Alternative with multiple variables
# sns.violinplot(data=df, x='category', y='values', hue='another_column')


def main( )->None:
    # Load the dataset
    full_dataset_file = choose_full_dataset_file(DATASET_SIZE)
    df_full_ela = process_dataframe(load_dataset_as_pd_df(full_dataset_file))

    reduced_feature_file = choose_reduced_feature_file(DATASET_SIZE, REDUCTION_RATIO)
    df_reduced_ela = process_dataframe(load_dataset_as_pd_df(reduced_feature_file))

    reduced_feature_file_oneshot = choose_reduced_feature_file_one_shot(DATASET_SIZE, REDUCTION_RATIO)
    df_reduced_ela_oneshot = process_dataframe(load_dataset_as_pd_df(reduced_feature_file_oneshot))

    if DATASET_SIZE == 200:
        considered_seeds = DATASET_200_CONSIDERED_SEEDS
    elif DATASET_SIZE == 2000:
        considered_seeds = DATASET_2000_CONSIDERED_SEEDS
    else:
        raise ValueError("Unsupported DATASET_SIZE")

    #feature_name = "ela_meta.lin_simple.adj_r2"
    #feature_name = "ela_distr.kurtosis"
    #feature_name = "ela_distr.number_of_peaks"
    #feature_name = "ela_meta.lin_simple.intercept"
    #feature_name = "nbc.nb_fitness.cor"
    #feature_name = "ela_level.mmce_qda_25"
    #feature_name = "ic.eps_max"
    #feature_name = "pca.expl_var.cor_x"
    #feature_name = "ic.eps_ratio"
    #feature_name = "pca.expl_var_PC1.cor_x"
    #feature_name = "disp.diff_median_25"

    # Get all the feature names available
    all_feature_names = [col for col in df_reduced_ela.columns if col not in ["embedding_seed",
                                                                              "round",
                                                                              "reduction_ratio",
                                                                              "instance_idx",
                                                                              "function_idx",
                                                                              "n_samples",
                                                                              "seed_lhs",
                                                                              "dimension"]]
    
    # Loop all over functions, instances, seeds, and features
    for function_id in FUNCTION_IDS:
        for instance_id in INSTANCE_IDS:
            for seed_idx in considered_seeds:
                for feature_name in all_feature_names:

                    print(
                        f"Plotting for feature: {feature_name}, "
                        f"function_id: {function_id}, instance_id: {instance_id}, seed: {seed_idx}"
                    )

                    fig, ax = plot_violin_plots_unbiased(
                        df_full_ela,
                        df_reduced_ela,
                        df_reduced_ela_oneshot,
                        feature_name=feature_name,
                        function_id=function_id,
                        instance_id=instance_id,
                        seed_lhs=seed_idx,
                        show_fig=False,
                        fig_size=(14, 6),
                    )

                    # Build directory hierarchy cleanly
                    figure_path = (
                        SAVE_FIGURE_DIRECTORY
                        / f"n{DATASET_SIZE}"
                        / f"rr{REDUCTION_RATIO}"
                        / f"fid{function_id}"
                        / f"iid{instance_id}"
                        / f"seed{seed_idx}"
                    )

                    figure_path.mkdir(parents=True, exist_ok=True)

                    # Final filename
                    figure_file = figure_path / f"violin_{feature_name}.pdf"

                    fig.savefig(figure_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)



    




if __name__ == "__main__":    main()