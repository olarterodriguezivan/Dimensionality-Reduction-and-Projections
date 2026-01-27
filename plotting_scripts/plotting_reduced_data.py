# ============================================================
# Loading necessary libraries
# ============================================================

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# For scipy
from scipy.stats import ttest_ind, f


sns.set_context("talk")
sns.set_style("white")

EPS = 1e-8

# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_1_PATH = Path("complete_data_generated.parquet").resolve()
DATASET_2_PATH = Path("complete_data_2.csv").resolve()

OUTPUT_DIR = Path("ela_heatmaps").resolve()

# ============================================================
# CLEANING UTILITIES
# ============================================================

def remove_runtime_sourcefile_name_features(df: pd.DataFrame) -> pd.DataFrame:
    runtime_features = [
        col for col in df.columns
        if ("costs_runtime" in col or "source_file" in col)
    ]
    return df.drop(columns=runtime_features)


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_features_by_lhs_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over seed_lhs and compute mean/std
    per (function_idx, instance_idx).
    """

    group_cols = ["function_idx", "instance_idx"]
    feature_cols = [c for c in df.columns if c not in group_cols + ["seed_lhs"]]

    aggregated_df = (
        df
        .groupby(group_cols)[feature_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    aggregated_df.columns = [
        col if isinstance(col, str) else f"{col[0]}_{col[1]}"
        for col in aggregated_df.columns
    ]

    return aggregated_df


# ============================================================
# COMPARISON PIPELINE
# ============================================================

def align_ela_datasets(df_small: pd.DataFrame,
                       df_large: pd.DataFrame) -> pd.DataFrame:
    return df_small.merge(
        df_large,
        on=["function_idx_", "instance_idx_"],
        suffixes=("_small", "_large")
    )


def extract_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Extract ELA feature base names while excluding metadata features.
    """

    EXCLUDE_SUBSTRINGS = {
        "dimension",
        "n_samples",
    }

    features = set()

    for col in df.columns:
        if not col.endswith("_mean_small"):
            continue

        base = col.replace("_mean_small", "")

        if any(excl in base for excl in EXCLUDE_SUBSTRINGS):
            continue

        features.add(base)

    return sorted(features)



def add_comparison_metrics(df: pd.DataFrame,
                           features: list[str]) -> pd.DataFrame:
    df = df.copy()

    for feat in features:
        df[f"{feat}_mean_rel_diff"] = (
            (df[f"{feat}_mean_large"] - df[f"{feat}_mean_small"]) /
            (df[f"{feat}_mean_large"].abs() + EPS)
        )

        df[f"{feat}_std_ratio"] = (
            df[f"{feat}_std_small"] /
            (df[f"{feat}_std_large"] + EPS)
        )

    return df


def compute_global_limits(df: pd.DataFrame,
                          features: list[str],
                          metric: str):

    vals = np.concatenate([
        df[f"{feat}_{metric}"].values
        for feat in features
    ])

    if metric == "mean_rel_diff":
        vmax = np.nanpercentile(np.abs(vals), 95)
        return -vmax, vmax

    return (
        np.nanpercentile(vals, 5),
        np.nanpercentile(vals, 95),
    )

# ============================================================
# STATISTICAL TESTS (RAW DATA)
# ============================================================

def welch_ttest_per_cell(df_small: pd.DataFrame,
                         df_large: pd.DataFrame,
                         feature: str) -> pd.DataFrame:
    """
    Welch two-sample t-test on feature means per (function, instance).
    """

    records = []

    for (f_idx, i_idx), g_small in df_small.groupby(["function_idx", "instance_idx"]):
        g_large = df_large.query(
            "function_idx == @f_idx and instance_idx == @i_idx"
        )

        x = g_small[feature].dropna().values
        y = g_large[feature].dropna().values

        if len(x) < 2 or len(y) < 2:
            t_stat, pval = np.nan, np.nan
        else:
            t_stat, pval = ttest_ind(x, y, equal_var=False)

        records.append({
            "function_idx": f_idx,
            "instance_idx": i_idx,
            "t_stat": t_stat,
            "p_value": pval,
        })

    return pd.DataFrame(records)


def ftest_variance_per_cell(df_small: pd.DataFrame,
                            df_large: pd.DataFrame,
                            feature: str) -> pd.DataFrame:
    """
    F-test on variance reduction (small vs large samples).
    """

    records = []

    for (f_idx, i_idx), g_small in df_small.groupby(["function_idx", "instance_idx"]):
        g_large = df_large.query(
            "function_idx == @f_idx and instance_idx == @i_idx"
        )

        x = g_small[feature].dropna().values
        y = g_large[feature].dropna().values

        if len(x) < 2 or len(y) < 2:
            f_stat, pval = np.nan, np.nan
        else:
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)

            f_stat = var_x / (var_y + EPS)

            dfn, dfd = len(x) - 1, len(y) - 1
            pval = 1.0 - f.cdf(f_stat, dfn, dfd)

        records.append({
            "function_idx": f_idx,
            "instance_idx": i_idx,
            "f_stat": f_stat,
            "p_value": pval,
        })

    return pd.DataFrame(records)


# ============================================================
# PLOTTING
# ============================================================

def plot_feature_heatmap(df: pd.DataFrame,
                         feature: str,
                         metric: str,
                         vmin: float,
                         vmax: float,
                         cmap: str,
                         center: float | None,
                         save_dir: Path):

    data = df.pivot(
        index="function_idx_",
        columns="instance_idx_",
        values=f"{feature}_{metric}"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
    data,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    center=center,
    linewidths=0.3,
    annot=True,
    fmt=".2e" if metric == "mean_rel_diff" else ".2f",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8}
)

    plt.title(f"{feature} – {metric.replace('_', ' ')}")
    plt.xlabel("Instance index")
    plt.ylabel("Function index")

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{feature}_{metric}.png", dpi=200)
    plt.close()

def plot_statistical_heatmap(df: pd.DataFrame,
                             value_col: str,
                             title: str,
                             cmap: str,
                             center: float | None,
                             save_path: Path):

    data = df.pivot(
        index="function_idx",
        columns="instance_idx",
        values=value_col
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        data,
        cmap=cmap,
        center=center,
        linewidths=0.3,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title)
    plt.xlabel("Instance index")
    plt.ylabel("Function index")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    # Load datasets
    df1 = pd.read_parquet(DATASET_1_PATH)
    df2 = pd.read_csv(DATASET_2_PATH)

    # Clean
    df1 = remove_runtime_sourcefile_name_features(df1)
    df2 = remove_runtime_sourcefile_name_features(df2)

    # Filter dimension
    df1 = df1[df1["dimension"] != 40]
    df2 = df2[df2["dimension"] != 40]

    # Aggregate over seed_lhs
    df1_agg = aggregate_features_by_lhs_seed(df1)
    df2_agg = aggregate_features_by_lhs_seed(df2)

    # Align datasets
    df = align_ela_datasets(df1_agg, df2_agg)

    # Feature discovery
    features = extract_feature_names(df)

    # Metrics
    df = add_comparison_metrics(df, features)

    # Global limits
    mean_diff_limits = compute_global_limits(df, features, "mean_rel_diff")
    std_ratio_limits = compute_global_limits(df, features, "std_ratio")

    # Plot
    for feat in features:
        plot_feature_heatmap(
            df,
            feat,
            "mean_rel_diff",
            *mean_diff_limits,
            cmap="RdBu",
            center=0.0,
            save_dir=OUTPUT_DIR / "mean_rel_diff"
        )

        plot_feature_heatmap(
            df,
            feat,
            "std_ratio",
            *std_ratio_limits,
            cmap="viridis",
            center=None,
            save_dir=OUTPUT_DIR / "std_ratio"
        )

        # --- Welch t-test (mean shift) ---
        df_ttest = welch_ttest_per_cell(df1, df2, feat)

        plot_statistical_heatmap(
            df=df_ttest,
            value_col="t_stat",
            title=f"{feat} – Welch t-statistic",
            cmap="RdBu",
            center=0.0,
            save_path=OUTPUT_DIR / "t_test" / f"{feat}_t_stat.png"
        )

        plot_statistical_heatmap(
            df=df_ttest,
            value_col="p_value",
            title=f"{feat} – Welch t-statistic",
            cmap="RdBu",
            center=0.0,
            save_path=OUTPUT_DIR / "t_test" / f"{feat}_p_value.png"
        )

        # --- F-test (variance reduction) ---
        df_ftest = ftest_variance_per_cell(df1, df2, feat)

        # log-scale for interpretability
        df_ftest["log_f_stat"] = np.log10(df_ftest["f_stat"])

        plot_statistical_heatmap(
            df=df_ftest,
            value_col="log_f_stat",
            title=f"{feat} – log10 variance ratio (F-test)",
            cmap="viridis",
            center=0.0,
            save_path=OUTPUT_DIR / "f_test" / f"{feat}_log_f_stat.png"
        )

    print("✔ ELA stability heatmaps generated.")


if __name__ == "__main__":
    main()


