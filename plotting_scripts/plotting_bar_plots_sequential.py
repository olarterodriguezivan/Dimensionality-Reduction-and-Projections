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

#FUNCTION_IDS:list = [1, 8, 11, 16, 20]  # Function IDs to consider
FUNCTION_IDS:list = [1,8,11,16,20]# Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_histogram_sequential")


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


def plot_bar_plots_unbiased(
    df_full_2000: pd.DataFrame,
    df_full_200: pd.DataFrame,
    df_reduced_2000_05: pd.DataFrame,
    df_reduced_200_05: pd.DataFrame,
    df_reduced_2000_025: pd.DataFrame,
    df_reduced_200_025: pd.DataFrame,
    df_reduced_oneshot_2000_05: pd.DataFrame,
    df_reduced_oneshot_200_05: pd.DataFrame,
    df_reduced_oneshot_2000_025: pd.DataFrame,
    df_reduced_oneshot_200_025: pd.DataFrame,
    feature_name: str,
    function_id: int = 1,
    instance_id: int = 0,
    show_fig: bool = False,
    fig_size: Tuple = (16, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Bar-plot visualization of ALL reduced and one-shot reduced datasets
    without aggregation or grouping by embedding_seed.

    Full references are computed as MEAN over all seeds.
    """

    # ------------------------------------------------------------
    # Helper to filter and tag dataframe
    # ------------------------------------------------------------
    def _prepare(df, label, kind):
        df_f = df[
            (df["function_idx"] == function_id)
            & (df["instance_idx"] == instance_id)
        ].copy()

        df_f["value"] = df_f[feature_name]
        df_f["dataset"] = label
        df_f["kind"] = kind
        return df_f[["value", "dataset", "kind"]]

    # ------------------------------------------------------------
    # Collect all reduced datasets
    # ------------------------------------------------------------
    reduced_dfs: List[pd.DataFrame] = [
        _prepare(df_reduced_2000_05,  "R=0.5 | N=2000", "reduced"),
        _prepare(df_reduced_200_05,   "R=0.5 | N=200",  "reduced"),
        _prepare(df_reduced_2000_025, "R=0.25 | N=2000","reduced"),
        _prepare(df_reduced_200_025,  "R=0.25 | N=200", "reduced"),
    ]

    reduced_os_dfs: List[pd.DataFrame] = [
        _prepare(df_reduced_oneshot_2000_05,  "R=0.5 | N=2000", "oneshot"),
        _prepare(df_reduced_oneshot_200_05,   "R=0.5 | N=200",  "oneshot"),
        _prepare(df_reduced_oneshot_2000_025, "R=0.25 | N=2000","oneshot"),
        _prepare(df_reduced_oneshot_200_025,  "R=0.25 | N=200", "oneshot"),
    ]

    df_reduced_all = pd.concat(reduced_dfs + reduced_os_dfs, ignore_index=True)
    df_reduced_all["run_id"] = df_reduced_all.index.astype(str)

    # ------------------------------------------------------------
    # Full reference values (SAFE, SEED-INDEPENDENT)
    # ------------------------------------------------------------
    df_full_2000_f = df_full_2000[
        (df_full_2000["function_idx"] == function_id)
        & (df_full_2000["instance_idx"] == instance_id)
    ]

    df_full_200_f = df_full_200[
        (df_full_200["function_idx"] == function_id)
        & (df_full_200["instance_idx"] == instance_id)
    ]

    if df_full_2000_f.empty or df_full_200_f.empty:
        raise ValueError(
            "Full dataset is empty after filtering by function_idx / instance_idx."
        )

    full_2000_val = df_full_2000_f[feature_name].mean()
    full_200_val = df_full_200_f[feature_name].mean()

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    # Bars = reduced
    sns.barplot(
        data=df_reduced_all[df_reduced_all["kind"] == "reduced"],
        x="run_id",
        y="value",
        hue="dataset",
        dodge=False,
        ax=ax,
    )

    # Points = one-shot reduced
    sns.stripplot(
        data=df_reduced_all[df_reduced_all["kind"] == "oneshot"],
        x="run_id",
        y="value",
        hue="dataset",
        dodge=False,
        size=7,
        marker="o",
        ax=ax,
    )

    # Reference lines (MEAN over seeds)
    ax.axhline(
        full_2000_val,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Full N=2000 (mean)",
    )

    ax.axhline(
        full_200_val,
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Full N=200 (mean)",
    )

    # ------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------
    ax.set_title(
        f"{feature_name} | fid={function_id}, instance={instance_id}"
    )
    ax.set_xlabel("Experiment index (all datasets)")
    ax.set_ylabel(feature_name)
    ax.set_xticks([])

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=2)

    if show_fig:
        plt.show()

    return fig, ax


def plot_histograms_unbiased(
    df_full_2000: pd.DataFrame,
    df_full_200: pd.DataFrame,
    df_reduced_2000_05: pd.DataFrame,
    df_reduced_200_05: pd.DataFrame,
    df_reduced_2000_025: pd.DataFrame,
    df_reduced_200_025: pd.DataFrame,
    df_reduced_oneshot_2000_05: pd.DataFrame,
    df_reduced_oneshot_200_05: pd.DataFrame,
    df_reduced_oneshot_2000_025: pd.DataFrame,
    df_reduced_oneshot_200_025: pd.DataFrame,
    feature_name: str,
    function_id: int = 1,
    instance_id: int = 0,
    bins: int = 30,
    show_fig: bool = False,
    fig_size: Tuple = (16, 6),
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Histogram-only visualization of ALL datasets (full, reduced, one-shot).
    No aggregation, no grouping, no seed bias.
    """

    # ------------------------------------------------------------
    # Helper: filter by function + instance
    # ------------------------------------------------------------
    def _filter(df):
        return df[
            (df["function_idx"] == function_id)
            & (df["instance_idx"] == instance_id)
        ][feature_name]

    # ------------------------------------------------------------
    # Collect all series
    # ------------------------------------------------------------
    data_series = {
        "Full $S=2000$": _filter(df_full_2000),
        "Full $S=200$": _filter(df_full_200),
        "Reduced $r=0.5$ | $S=2000$": _filter(df_reduced_2000_05),
        "Reduced $r=0.5$ | $S=200$": _filter(df_reduced_200_05),
        "Reduced $r=0.25$ | $S=2000$": _filter(df_reduced_2000_025),
        "Reduced $r=0.25$ | $S=200$": _filter(df_reduced_200_025),
        "One-shot $r=0.5$ | $S=2000$": _filter(df_reduced_oneshot_2000_05),
        "One-shot $r=0.5$ | $S=200$": _filter(df_reduced_oneshot_200_05),
        "One-shot $r=0.25$ | $S=2000$": _filter(df_reduced_oneshot_2000_025),
        "One-shot $r=0.25$ | $S=200$": _filter(df_reduced_oneshot_200_025),
    }

    # Remove empty datasets
    data_series = {k: v for k, v in data_series.items() if not v.empty}

    if not data_series:
        raise ValueError("No data available after filtering.")

    # ------------------------------------------------------------
    # Shared binning (important!)
    # ------------------------------------------------------------
    
    all_values = np.concatenate([v.values for v in data_series.values()])

    try:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    except ValueError as e:
        print("Error computing histogram bin edges:", e)
        return None, None

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    for label, values in data_series.items():
        sns.histplot(
            values,
            bins=bin_edges,
            stat="density",
            element="bars", #step
            fill=True,
            linewidth=2,
            ax=ax,
            label=label,
        )

    ax.set_title(
        f"{feature_name} | fid={function_id}, instance={instance_id}"
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(ncol=2)

    if show_fig:
        plt.show()

    return fig, ax

def plot_histograms_unbiased_rows_by_reduction(
    df_full_2000: pd.DataFrame,
    df_full_200: pd.DataFrame,
    df_reduced_2000_05: pd.DataFrame,
    df_reduced_200_05: pd.DataFrame,
    df_reduced_2000_025: pd.DataFrame,
    df_reduced_200_025: pd.DataFrame,
    df_reduced_oneshot_2000_05: pd.DataFrame,
    df_reduced_oneshot_200_05: pd.DataFrame,
    df_reduced_oneshot_2000_025: pd.DataFrame,
    df_reduced_oneshot_200_025: pd.DataFrame,
    feature_name: str,
    function_id: int = 1,
    instance_id: int = 0,
    bins: int = 30,
    show_fig: bool = False,
    fig_size: Tuple = (16, 12),
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Histogram visualization split by reduction ratio r.
    Subplots:
      (1) Full
      (2) r = 0.5
      (3) r = 0.25
    """

    # ------------------------------------------------------------
    # Helper: filter by function + instance
    # ------------------------------------------------------------
    def _filter(df):
        return df[
            (df["function_idx"] == function_id)
            & (df["instance_idx"] == instance_id)
        ][feature_name]

    # ------------------------------------------------------------
    # Organize data per subplot (per r)
    # ------------------------------------------------------------
    groups = [
        (
            "Full (no reduction)",
            {
                "Full $S=2000$": _filter(df_full_2000),
                "Full $S=200$": _filter(df_full_200),
            },
        ),
        (
            r"Reduced / One-shot ($r = 0.5$)",
            {
                "Reduced $S=2000$": _filter(df_reduced_2000_05),
                "Reduced $S=200$": _filter(df_reduced_200_05),
                "One-shot $S=2000$": _filter(df_reduced_oneshot_2000_05),
                "One-shot $S=200$": _filter(df_reduced_oneshot_200_05),
            },
        ),
        (
            r"Reduced / One-shot ($r = 0.25$)",
            {
                "Reduced $S=2000$": _filter(df_reduced_2000_025),
                "Reduced $S=200$": _filter(df_reduced_200_025),
                "One-shot $S=2000$": _filter(df_reduced_oneshot_2000_025),
                "One-shot $S=200$": _filter(df_reduced_oneshot_200_025),
            },
        ),
    ]

    # Remove empty series
    for _, d in groups:
        for k in list(d.keys()):
            if d[k].empty:
                del d[k]

    # ------------------------------------------------------------
    # Shared binning across ALL data
    # ------------------------------------------------------------
    all_values = np.concatenate(
        [
            v.values
            for _, group in groups
            for v in group.values()
            if not v.empty
        ]
    )

    if all_values.size == 0:
        raise ValueError("No data available after filtering.")

    try:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    except ValueError as e:
        print("Error computing histogram bin edges:", e)
        return None, None

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)

    for ax, (title, series_dict) in zip(axes, groups):
        for label, values in series_dict.items():
            sns.histplot(
                values,
                bins=bin_edges,
                stat="density",
                element="bars",
                fill=True,
                linewidth=1.5,
                ax=ax,
                label=label,
            )

        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend(ncol=2)

    axes[-1].set_xlabel(feature_name)

    fig.suptitle(
        f"{feature_name} | fid={function_id}, instance={instance_id}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show_fig:
        plt.show()

    return fig, axes


def plot_histograms_unbiased_rows_by_reduction(
    df_full_2000: pd.DataFrame,
    df_full_200: pd.DataFrame,
    df_reduced_2000_05: pd.DataFrame,
    df_reduced_200_05: pd.DataFrame,
    df_reduced_2000_025: pd.DataFrame,
    df_reduced_200_025: pd.DataFrame,
    df_reduced_oneshot_2000_05: pd.DataFrame,
    df_reduced_oneshot_200_05: pd.DataFrame,
    df_reduced_oneshot_2000_025: pd.DataFrame,
    df_reduced_oneshot_200_025: pd.DataFrame,
    feature_name: str,
    function_id: int = 1,
    instance_id: int = 0,
    bins: int = 30,
    show_fig: bool = False,
    fig_size: Tuple = (16, 12),
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Histogram visualization split by reduction ratio r.
    Subplots:
      (1) Full
      (2) r = 0.5
      (3) r = 0.25
    """

    # ------------------------------------------------------------
    # Helper: filter by function + instance
    # ------------------------------------------------------------
    def _filter(df):
        return df[
            (df["function_idx"] == function_id)
            & (df["instance_idx"] == instance_id)
        ][feature_name]

    # ------------------------------------------------------------
    # Organize data per subplot (per r)
    # ------------------------------------------------------------
    groups = [
        (
            "Full (no reduction)",
            {
                "Full $S=2000$": _filter(df_full_2000),
                "Full $S=200$": _filter(df_full_200),
            },
        ),
        (
            r"Reduced / One-shot ($r = 0.5$)",
            {
                "Reduced $S=2000$": _filter(df_reduced_2000_05),
                "Reduced $S=200$": _filter(df_reduced_200_05),
                "One-shot $S=2000$": _filter(df_reduced_oneshot_2000_05),
                "One-shot $S=200$": _filter(df_reduced_oneshot_200_05),
            },
        ),
        (
            r"Reduced / One-shot ($r = 0.25$)",
            {
                "Reduced $S=2000$": _filter(df_reduced_2000_025),
                "Reduced $S=200$": _filter(df_reduced_200_025),
                "One-shot $S=2000$": _filter(df_reduced_oneshot_2000_025),
                "One-shot $S=200$": _filter(df_reduced_oneshot_200_025),
            },
        ),
    ]

    # Remove empty series
    for _, d in groups:
        for k in list(d.keys()):
            if d[k].empty:
                del d[k]

    # ------------------------------------------------------------
    # Shared binning across ALL data
    # ------------------------------------------------------------
    all_values = np.concatenate(
        [
            v.values
            for _, group in groups
            for v in group.values()
            if not v.empty
        ]
    )

    if all_values.size == 0:
        raise ValueError("No data available after filtering.")

    try:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    except ValueError as e:
        print("Error computing histogram bin edges:", e)
        return None, None

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)

    for ax, (title, series_dict) in zip(axes, groups):
        for label, values in series_dict.items():
            sns.histplot(
                values,
                bins=bin_edges,
                stat="density",
                element="bars",
                fill=True,
                linewidth=1.5,
                ax=ax,
                label=label,
            )

        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend(ncol=2)

    axes[-1].set_xlabel(feature_name)

    fig.suptitle(
        f"{feature_name} | fid={function_id}, instance={instance_id}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show_fig:
        plt.show()

    return fig, axes


def plot_histograms_unbiased_rows_by_reduction_2(
    df_full_2000: pd.DataFrame,
    df_full_200: pd.DataFrame,
    df_reduced_2000_05: pd.DataFrame,
    df_reduced_200_05: pd.DataFrame,
    df_reduced_2000_025: pd.DataFrame,
    df_reduced_200_025: pd.DataFrame,
    df_reduced_oneshot_2000_05: pd.DataFrame,
    df_reduced_oneshot_200_05: pd.DataFrame,
    df_reduced_oneshot_2000_025: pd.DataFrame,
    df_reduced_oneshot_200_025: pd.DataFrame,
    feature_name: str,
    function_id: int = 1,
    instance_id: int = 0,
    bins: int = 30,
    show_fig: bool = False,
    fig_size: Tuple = (16, 12),
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Histogram visualization split by reduction ratio r.
    Subplots:
      (1) Full
      (2) r = 0.5
      (3) r = 0.25
    """

    # ------------------------------------------------------------
    # Helper: filter by function + instance
    # ------------------------------------------------------------
    def _filter(df):
        return df[
            (df["function_idx"] == function_id)
            & (df["instance_idx"] == instance_id)
        ][feature_name]

    # ------------------------------------------------------------
    # Organize data per subplot (per r)
    # ------------------------------------------------------------
    groups = [
        (
            "Full (no reduction)",
            {
                "$S=2000$": _filter(df_full_2000),
                "$S=200$": _filter(df_full_200),
            },
        ),
        (
            r"$r = 0.5$",
            {
                "$S=2000$": _filter(df_reduced_oneshot_2000_05),
                "$S=200$": _filter(df_reduced_oneshot_200_05),
            },
        ),
        (
            r"$r = 0.25$",
            {
               
                "$S=2000$": _filter(df_reduced_oneshot_2000_025),
                "$S=200$": _filter(df_reduced_oneshot_200_025),
            },
        ),
    ]

    # Remove empty series
    for _, d in groups:
        for k in list(d.keys()):
            if d[k].empty:
                del d[k]

    # ------------------------------------------------------------
    # Shared binning across ALL data
    # ------------------------------------------------------------
    all_values = np.concatenate(
        [
            v.values
            for _, group in groups
            for v in group.values()
            if not v.empty
        ]
    )

    if all_values.size == 0:
        raise ValueError("No data available after filtering.")

    try:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    except ValueError as e:
        print("Error computing histogram bin edges:", e)
        return None, None

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)

    for ax, (title, series_dict) in zip(axes, groups):
        for label, values in series_dict.items():
            sns.histplot(
                values,
                bins=bin_edges,
                stat="density",
                element="bars",
                fill=True,
                linewidth=1.5,
                ax=ax,
                label=label,
            )

        ax.set_title(title)
        ax.set_ylabel("Density")
        #ax.legend(ncol=2)

    axes[-1].set_xlabel(feature_name)

    fig.suptitle(
        f"{feature_name} | fid={function_id}, instance={instance_id}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show_fig:
        plt.show()

    return fig, axes

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

    feature_name = "ic.eps_max"
    # Test the plotting function for a specific feature, function, instance
    # fig, ax = plot_histograms_unbiased(
    #     df_full_ela_2000,
    #     df_full_ela_200,
    #     df_reduced_ela_2000_05,
    #     df_reduced_ela_200_05,
    #     df_reduced_ela_2000_025,
    #     df_reduced_ela_200_025,
    #     df_reduced_ela_oneshot_2000_05,
    #     df_reduced_ela_oneshot_200_05,
    #     df_reduced_ela_oneshot_2000_025,
    #     df_reduced_ela_oneshot_200_025,
    #     feature_name=feature_name,
    #     function_id=16,
    #     instance_id=1,
    #     show_fig=True,
    #     fig_size=(16, 6),
    #     bins=100,
    # )

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
        for instance_id in INSTANCE_IDS:
            for feature_name in all_feature_names:

                print(
                    f"Plotting for feature: {feature_name}, "
                    f"function_id: {function_id}, instance_id: {instance_id}"
                )

                fig, ax = plot_histograms_unbiased_rows_by_reduction_2(
                                                    df_full_ela_2000,
                                                    df_full_ela_200,
                                                    df_reduced_ela_2000_05,
                                                    df_reduced_ela_200_05,
                                                    df_reduced_ela_2000_025,
                                                    df_reduced_ela_200_025,
                                                    df_reduced_ela_oneshot_2000_05,
                                                    df_reduced_ela_oneshot_200_05,
                                                    df_reduced_ela_oneshot_2000_025,
                                                    df_reduced_ela_oneshot_200_025,
                                                    feature_name=feature_name,
                                                    function_id=function_id,
                                                    instance_id=instance_id,
                                                    show_fig=False,
                                                    fig_size=(16, 6),
                                                    bins=100,
                                                )
                
                 # Build directory hierarchy cleanly
                figure_path = (
                    SAVE_FIGURE_DIRECTORY
                    / f"fid{function_id}"
                    / f"iid{instance_id}"

                )

                figure_path.mkdir(parents=True, exist_ok=True)

                # Final filename
                figure_file = figure_path / f"histogram_{feature_name}_separate.pdf"

                if not fig is None:
                    fig.savefig(figure_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)

    # if DATASET_SIZE == 200:
    #     considered_seeds = DATASET_200_CONSIDERED_SEEDS
    # elif DATASET_SIZE == 2000:
    #     considered_seeds = DATASET_2000_CONSIDERED_SEEDS
    # else:
    #     raise ValueError("Unsupported DATASET_SIZE")

    # #feature_name = "ela_meta.lin_simple.adj_r2"
    # #feature_name = "ela_distr.kurtosis"
    # #feature_name = "ela_distr.number_of_peaks"
    # #feature_name = "ela_meta.lin_simple.intercept"
    # #feature_name = "nbc.nb_fitness.cor"
    # #feature_name = "ela_level.mmce_qda_25"
    # #feature_name = "ic.eps_max"
    # #feature_name = "pca.expl_var.cor_x"
    # #feature_name = "ic.eps_ratio"
    # ic.eps_s
    # disp.diff_mean_10
    # ela_meta.quad_simple.adj_r2
    # #feature_name = "pca.expl_var_PC1.cor_x"
    # #feature_name = "disp.diff_median_25"

    # # Get all the feature names available
    # all_feature_names = [col for col in df_reduced_ela.columns if col not in ["embedding_seed",
    #                                                                           "round",
    #                                                                           "reduction_ratio",
    #                                                                           "instance_idx",
    #                                                                           "function_idx",
    #                                                                           "n_samples",
    #                                                                           "seed_lhs",
    #                                                                           "dimension"]]
    
    # # Loop all over functions, instances, seeds, and features
    # for function_id in FUNCTION_IDS:
    #     for instance_id in INSTANCE_IDS:
    #         for seed_idx in considered_seeds:
    #             for feature_name in all_feature_names:

    #                 print(
    #                     f"Plotting for feature: {feature_name}, "
    #                     f"function_id: {function_id}, instance_id: {instance_id}, seed: {seed_idx}"
    #                 )

    #                 fig, ax = plot_violin_plots_unbiased(
    #                     df_full_ela,
    #                     df_reduced_ela,
    #                     df_reduced_ela_oneshot,
    #                     feature_name=feature_name,
    #                     function_id=function_id,
    #                     instance_id=instance_id,
    #                     seed_lhs=seed_idx,
    #                     show_fig=False,
    #                     fig_size=(14, 6),
    #                 )

    #                 # Build directory hierarchy cleanly
    #                 figure_path = (
    #                     SAVE_FIGURE_DIRECTORY
    #                     / f"n{DATASET_SIZE}"
    #                     / f"rr{REDUCTION_RATIO}"
    #                     / f"fid{function_id}"
    #                     / f"iid{instance_id}"
    #                     / f"seed{seed_idx}"
    #                 )

    #                 figure_path.mkdir(parents=True, exist_ok=True)

    #                 # Final filename
    #                 figure_file = figure_path / f"violin_{feature_name}.pdf"

    #                 fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    #                 plt.close(fig)



    




if __name__ == "__main__":    main()