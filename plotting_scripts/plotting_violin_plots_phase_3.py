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
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_violin_plots_phase_3_2_mod_changed")

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25]


EXCLUDED_COLUMNS = {
"embedding_seed",
"round",
"reduction_ratio",
"instance_idx",
"function_idx",
"n_samples",
"seed_lhs",
"dimension",
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
            reduced_file = choose_reduced_feature_file(n_samples, ratio)
            datasets[("reduced", n_samples, ratio)] = process_dataframe(
            load_dataset_as_pd_df(reduced_file), n_samples
            )


            oneshot_file = choose_reduced_feature_file_one_shot(n_samples, ratio)
            datasets[("oneshot", n_samples, ratio)] = process_dataframe(
            load_dataset_as_pd_df(oneshot_file), n_samples
            )


    return datasets

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


def plot_violin_plots_biased_3(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    df_reduced_oneshot: pd.DataFrame,
    feature_name_list: List[str],
    reduction_ratio: float,
    data_size: int,
    function_id: int,
    instance_id_list: List[int],
    show_fig: bool = False,
    fig_size: Tuple = (14, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Bias violin plots:
    - x-axis: feature names
    - violins aggregate over (instances × seeds × embeddings)
    - relative bias computed per (function, instance, feature, seed)
    """

    # --- Filter relevant rows ---
    df_full_f = df_full[
        (df_full["function_idx"] == function_id)
        & (df_full["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs"] + feature_name_list]

    df_red_f = df_reduced[
        (df_reduced["function_idx"] == function_id)
        & (df_reduced["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs"] + feature_name_list]

    df_red_os_f = df_reduced_oneshot[
        (df_reduced_oneshot["function_idx"] == function_id)
        & (df_reduced_oneshot["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs"] + feature_name_list]

    # --- Merge ---
    merged = df_red_f.merge(
        df_full_f,
        on=["instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    merged_os = df_red_os_f.merge(
        df_full_f,
        on=["instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    # --- Compute relative bias per feature (long format) ---
    records = []
    records_os = []

    for feat in feature_name_list:
        rb = (
            merged[f"{feat}_reduced"] - merged[f"{feat}_full"]
        ) / (np.abs(merged[f"{feat}_full"]) + EPSILON)

        records.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "relative_bias": rb,
                }
            )
        )

        rb_os = (
            merged_os[f"{feat}_reduced"] - merged_os[f"{feat}_full"]
        ) / (np.abs(merged_os[f"{feat}_full"]) + EPSILON)

        records_os.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "relative_bias": rb_os,
                }
            )
        )

    df_bias = pd.concat(records, ignore_index=True)
    df_bias_os = pd.concat(records_os, ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=df_bias,
        x="feature",
        y="relative_bias",
        inner="quartile",
        cut=0,
        density_norm="width",
        color="lightblue",
        ax=ax,
    )

    sns.stripplot(
        data=df_bias_os,
        x="feature",
        y="relative_bias",
        color="darkorange",
        size=0.8,
        jitter=True,
        ax=ax,
        label="One-shot relative bias",
    )

    # Zero reference
    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Unbiased (0)",
    )

    ax.set_title(
        f"Normalized feature distribution shift | fid={function_id}, "
        f"reduction_ratio={reduction_ratio}, n_samples={data_size}"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Feature shift")

    # Set the limits to improve visualization
    ax.set_ylim(ymax=1.0, ymin=-1.0)

    ax.tick_params(axis="x", rotation=90)

    #handles, labels = ax.get_legend_handles_labels()
    #unique = dict(zip(labels, handles))
    ax.legend([], [])


    if show_fig:
        plt.show()

    return fig, ax

def plot_violin_plots_biased_3_v2(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    feature_name_list: List[str],
    reduction_ratio: float,
    data_size: int,
    function_id: int,
    instance_id_list: List[int],
    show_fig: bool = False,
    fig_size: Tuple = (14, 6),
    plot_color: str = "lightblue",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Bias violin plots:
    - x-axis: feature names
    - violins aggregate over (instances × seeds × embeddings)
    - relative bias computed per (function, instance, feature, seed)
    """

    # --- Filter relevant rows ---
    df_full_f = df_full[
        (df_full["function_idx"] == function_id)
        & (df_full["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs"] + feature_name_list]

    df_red_f = df_reduced[
        (df_reduced["function_idx"] == function_id)
        & (df_reduced["instance_idx"].isin(instance_id_list))
    ][["instance_idx", "seed_lhs"] + feature_name_list]

    # --- Merge ---
    merged = df_red_f.merge(
        df_full_f,
        on=["instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    # --- Compute relative bias per feature (long format) ---
    records = []

    for feat in feature_name_list:
        rb = (
            merged[f"{feat}_reduced"] - merged[f"{feat}_full"]
        ) / (np.abs(merged[f"{feat}_full"]) + EPSILON)

        records.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "relative_bias": rb,
                }
            )
        )



    df_bias = pd.concat(records, ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=df_bias,
        x="feature",
        y="relative_bias",
        inner="quartile",
        cut=0,
        density_norm="width",
        color=plot_color,
        ax=ax,
    )

    # sns.stripplot(
    #     data=df_bias,
    #     x="feature",
    #     y="relative_bias",
    #     color="grey",
    #     size=0.8,
    #     jitter=True,
    #     ax=ax,
    #     label="Relative bias samples",
    # )

    # Zero reference
    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Unbiased (0)",
        # The next command is for transparency of the line
        alpha=0.3,
    )

    ax.set_title(
        f"Normalized feature distribution shift | fid={function_id}, "
        f"$r={reduction_ratio}$, $S={data_size}$"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Feature shift")

    # Set the limits to improve visualization
    ax.set_ylim(ymax=1.0, ymin=-1.0)

    ax.tick_params(axis="x", rotation=90)

    #handles, labels = ax.get_legend_handles_labels()
    #unique = dict(zip(labels, handles))
    ax.legend([], [])

    # Hide the legend
    ax.legend().set_visible(False)



    if show_fig:
        plt.show()

    return fig, ax


def plot_violin_plots_biased_3_v2_inverted(
    df_full: pd.DataFrame,
    df_reduced: pd.DataFrame,
    feature_name: str,
    reduction_ratio: float,
    data_size: int,
    function_id_list: List[int],
    instance_id_list: List[int],
    show_fig: bool = False,
    fig_size: Tuple = (14, 6),
    plot_color: str = "lightblue",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Bias violin plots:
    - x-axis: function ids
    - violins aggregate over (instances × seeds)
    - relative bias computed per (function, instance, seed)
    """

    EPSILON = 1e-12

    # --- Filter relevant rows ---
    df_full_f = df_full[
        df_full["function_idx"].isin(function_id_list)
        & df_full["instance_idx"].isin(instance_id_list)
    ][["function_idx", "instance_idx", "seed_lhs", feature_name]]

    df_red_f = df_reduced[
        df_reduced["function_idx"].isin(function_id_list)
        & df_reduced["instance_idx"].isin(instance_id_list)
    ][["function_idx", "instance_idx", "seed_lhs", feature_name]]

    # --- Merge full vs reduced ---
    merged = df_red_f.merge(
        df_full_f,
        on=["function_idx", "instance_idx", "seed_lhs"],
        suffixes=("_reduced", "_full"),
    )

    # --- Compute relative bias ---
    merged["relative_bias"] = (
        merged[f"{feature_name}_reduced"]
        - merged[f"{feature_name}_full"]
    ) / (np.abs(merged[f"{feature_name}_full"]) + EPSILON)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=fig_size)

    sns.violinplot(
        data=merged,
        x="function_idx",
        y="relative_bias",
        inner="quartile",
        cut=0,
        density_norm="width",
        color=plot_color,
        ax=ax,
    )

    sns.stripplot(
        data=merged,
        x="function_idx",
        y="relative_bias",
        color="grey",
        size=0.8,
        jitter=True,
        ax=ax,
        label="Relative bias samples",
    )

    # Zero reference
    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(
        f"Normalized feature shift ({feature_name}) | "
        f"fids={function_id_list}, "
        f"reduction_ratio={reduction_ratio}, "
        f"n_samples={data_size}"
    )
    ax.set_xlabel("Function ID")
    ax.set_ylabel("Normalized feature shift")

    #ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="x")

    if show_fig:
        plt.show()

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


    for function_id in FUNCTION_IDS:
        print(f"Processing function ID: {function_id}")


        for feature_name in all_feature_names:
            print(f"Processing feature: {feature_name}")


        for n_samples in DATA_SIZES:
            for ratio in REDUCTION_RATIOS:
                print(f"Processing feature: {feature_name}")



                df_full = datasets[("full", n_samples, None)]
                df_reduced = datasets[("reduced", n_samples, ratio)]

                if n_samples == 200 and ratio == 0.25:
                    continue

                plot_color = "skyblue"


                fig, ax = plot_violin_plots_biased_3_v2(
                df_full,
                df_reduced,
                feature_name_list=all_feature_names,
                reduction_ratio=ratio,
                data_size=n_samples,
                function_id=function_id,
                instance_id_list=INSTANCE_IDS,
                show_fig=False,
                fig_size=(16, 3),
                plot_color=plot_color,
                )


                if fig is None:
                    continue


                figure_path = (
                SAVE_FIGURE_DIRECTORY
                / f"function_id_{function_id}"
                / f"reduction_ratio_{ratio}"
                / f"n_samples_{n_samples}"
                )
                figure_path.mkdir(parents=True, exist_ok=True)


                fig.savefig(
                figure_path / "violin_plot_corrected.pdf",
                dpi=300,
                bbox_inches="tight",
                )
                plt.close(fig)


if __name__ == "__main__":
    main()