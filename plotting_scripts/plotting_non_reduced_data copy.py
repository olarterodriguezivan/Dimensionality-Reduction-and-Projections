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
DATASET_2_PATH = Path("reduced_data_org.parquet").resolve()

OUTPUT_DIR = Path("ela_heatmaps").resolve()


# ============================================================
# FEATURE FAMILIES
# ============================================================

FEATURE_FAMILIES = ["ela_meta",  "ela_level",
                    "ela_distr", "ic", "nbc", "pca", "disp","fitness_distance"]

def get_feature_family_from_prefix(feature_name: str) -> str:
    """
    Assign feature family based on starting prefix.
    """
    for prefix in FEATURE_FAMILIES:
        if feature_name.startswith(prefix):
            return prefix
    return "other"

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
# HELPER UTILITIES
# ============================================================
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def compute_feature_order(C: np.ndarray, features: list[str]) -> list[str]:
    """
    Compute a stable feature ordering using hierarchical clustering
    on absolute correlations.
    """

    # Distance = 1 - |corr|
    dist = 1.0 - np.abs(C)

    # --- CRITICAL FIXES ---
    # enforce symmetry
    dist = 0.5 * (dist + dist.T)

    # enforce zero diagonal
    np.fill_diagonal(dist, 0.0)

    # numerical safety
    dist = np.clip(dist, 0.0, 2.0)

    Z = linkage(squareform(dist), method="average")
    order = leaves_list(Z)

    return [features[i] for i in order]



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

def get_common_features(df_fi, feature_cols):
    valid_sets = []

    for emb_id, g in df_fi.groupby("embedding_seed"):
        X = g.sort_values("round")[feature_cols]
        valid = X.std(axis=0) > 1e-12
        valid_sets.append(set(X.columns[valid]))

    return sorted(set.intersection(*valid_sets))

def get_feature_columns(df):
    meta_cols = {
        "function_idx",
        "instance_idx",
        "embedding_seed",
        "round",
        "dimension",
        "n_samples",
    }
    return [c for c in df.columns if c not in meta_cols]



def compute_corr_from_embedded(
    df: pd.DataFrame,
    function_idx: int,
    instance_idx: int,
):

    df_fi = df[
        (df["function_idx"] == function_idx) &
        (df["instance_idx"] == instance_idx)
    ]

    meta_cols = {
        "function_idx",
        "instance_idx",
        "embedding_seed",
        "round",
        "dimension",
        "n_samples",
    }

    feature_cols = [c for c in df.columns if c not in meta_cols]

    # 🔑 compute common feature set
    common_features = get_common_features(df_fi, feature_cols)

    if len(common_features) < 2:
        raise ValueError("Not enough common features for correlation.")

    corr_matrices = []

    for emb_id, g in df_fi.groupby("embedding_seed"):

        X = g.sort_values("round")[common_features]

        if X.shape[0] < 2:
            continue

        C = np.corrcoef(X.values, rowvar=False)
        corr_matrices.append(C)

    if len(corr_matrices) == 0:
        raise ValueError("No valid correlation matrices computed.")

    C_mean = np.mean(corr_matrices, axis=0)

    return C_mean, common_features

def collapse_rounds(df_emb):
    feature_cols = get_feature_columns(df_emb)

    df_mu = (
        df_emb
        .groupby(["function_idx", "instance_idx", "embedding_seed"])[feature_cols]
        .mean()
        .reset_index()
    )

    return df_mu

def compute_stability(df_mu):
    feature_cols = get_feature_columns(df_mu)

    stats = []

    for (f, i), g in df_mu.groupby(["function_idx", "instance_idx"]):
        mu = g[feature_cols].mean()
        sd = g[feature_cols].std()

        cv = sd / (mu.abs() + 1e-8)

        for feat in feature_cols:
            stats.append({
                "function_idx": f,
                "instance_idx": i,
                "feature": feat,
                "cv": cv[feat],
            })

    return pd.DataFrame(stats)

def compute_bias(df_mu, df_ref):
    feature_cols = get_feature_columns(df_ref)

    df_mu_mean = (
        df_mu
        .groupby(["function_idx", "instance_idx"])[feature_cols]
        .mean()
        .reset_index()
    )

    df = df_mu_mean.merge(
        df_ref[["function_idx", "instance_idx"] + feature_cols],
        on=["function_idx", "instance_idx"],
        suffixes=("_emb", "_ref"),
    )

    records = []

    for feat in feature_cols:
        rel_bias = (
            df[f"{feat}_emb"] - df[f"{feat}_ref"]
        ) / (df[f"{feat}_ref"].abs() + 1e-8)

        tmp = df[["function_idx", "instance_idx"]].copy()
        tmp["feature"] = feat
        tmp["rel_bias"] = rel_bias
        records.append(tmp)

    return pd.concat(records, ignore_index=True)





def reference_identity_corr(features: list[str]) -> np.ndarray:
    """
    Reference correlation when only marginal ELA values exist.
    """
    return np.eye(len(features))


def plot_correlation_triplet(
    C_ref: np.ndarray,
    C_emb: np.ndarray,
    title: str,
    save_path: Path,
    diff_vmax: float = 0.4,
):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        C_ref,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar=True,
        ax=axes[0],
    )
    axes[0].set_title("Reference (Ambient)")

    sns.heatmap(
        C_emb,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar=True,
        ax=axes[1],
    )
    axes[1].set_title("Mean Embedded")

    sns.heatmap(
        C_emb - C_ref,
        cmap="coolwarm",
        center=0,
        vmin=-diff_vmax,
        vmax=diff_vmax,
        square=True,
        cbar=True,
        ax=axes[2],
    )
    axes[2].set_title("Difference (Embedded − Reference)")

    plt.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_correlation_triplet_with_labels(
    C_ref: np.ndarray,
    C_emb: np.ndarray,
    feature_order: list[str],
    title: str,
    save_path: Path,
    diff_vmax: float = 0.4,
):

    n_feat = len(feature_order)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(max(14, 0.25 * n_feat), max(8, 0.25 * n_feat))
    )

    def _plot(C, ax, title, vmax=1.0):
        sns.heatmap(
            C,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            square=True,
            xticklabels=feature_order,
            yticklabels=feature_order,
            cbar=True,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            ha="right",
            fontsize=6,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=6,
        )

    _plot(C_ref, axes[0], "Reference (Ambient)")
    _plot(C_emb, axes[1], "Mean Embedded")
    _plot(C_emb - C_ref, axes[2], "Difference", vmax=diff_vmax)

    plt.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_correlation_heatmap_with_labels(
    C: np.ndarray,
    feature_order: list[str],
    title: str,
    ax,
    vmax=1.0,
    label_fontsize=6,
):
    sns.heatmap(
        C,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=feature_order,
        yticklabels=feature_order,
        ax=ax,
        cbar=True,
    )

    ax.set_title(title)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="right",
        fontsize=label_fontsize,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=label_fontsize,
    )

def plot_stability_heatmap(df_cv, save_path):
    data = (
        df_cv
        .groupby(["function_idx", "feature"])["cv"]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="function_idx", values="cv")
    )

    plt.figure(figsize=(12, max(8, 0.25 * len(data))))
    sns.heatmap(
        data,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "Coefficient of variation"},
    )

    plt.title("ELA feature stability under embedding (CV)")
    plt.ylabel("ELA feature")
    plt.xlabel("Function index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_bias_heatmap(df_bias, save_path):
    data = (
        df_bias
        .groupby(["function_idx", "feature"])["rel_bias"]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="function_idx", values="rel_bias")
    )

    vmax = np.nanpercentile(np.abs(data.values), 95)

    plt.figure(figsize=(12, max(8, 0.25 * len(data))))
    sns.heatmap(
        data,
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=False,
        cbar_kws={"label": "Relative bias"},
    )

    plt.title("ELA feature bias under embedding")
    plt.ylabel("ELA feature")
    plt.xlabel("Function index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_bias_vs_stability(df_cv, df_bias, save_path):
    df = (
        df_cv
        .groupby("feature")["cv"]
        .mean()
        .reset_index()
        .merge(
            df_bias.groupby("feature")["rel_bias"].mean().abs().reset_index(),
            on="feature"
        )
    )

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x="rel_bias",
        y="cv",
    )

    plt.xlabel("Mean |relative bias|")
    plt.ylabel("Mean coefficient of variation")
    plt.title("ELA bias vs stability under embedding")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_stability_heatmaps_by_family(df_cv_fam, output_dir):
    """
    Plot stability (CV) heatmap for one feature family.
    """

    if df_cv_fam.empty:
        return

    data = (
        df_cv_fam
        .groupby(["function_idx", "feature"])["cv"]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="function_idx", values="cv")
    )

    plt.figure(figsize=(10, max(6, 0.3 * len(data))))

    sns.heatmap(
        data,
        cmap="viridis",
        cbar_kws={"label": "Coefficient of variation"},
        annot=True,                 # ← write values
        fmt=".2f",                  # ← CV formatting
        annot_kws={"size": 7},      # ← control font size
    )

    plt.title(f"ELA stability under embedding – family: {df_cv_fam['family'].iloc[0]}")
    plt.xlabel("Function index")
    plt.ylabel("ELA feature")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "stability.png", dpi=200)
    plt.close()


def plot_bias_heatmaps_by_family(df_bias_fam, output_dir):
    """
    Plot relative bias heatmap for one feature family.
    """

    if df_bias_fam.empty:
        return

    data = (
        df_bias_fam
        .groupby(["function_idx", "feature"])["rel_bias"]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="function_idx", values="rel_bias")
    )

    vmax = np.nanpercentile(np.abs(data.values), 95)

    plt.figure(figsize=(10, max(6, 0.3 * len(data))))

    sns.heatmap(
        data,
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Relative bias"},
        annot=True,                 # ← write values
        fmt=".2f",                  # ← CV formatting
        annot_kws={"size": 7},      # ← control font size
    )

    plt.title(f"ELA bias under embedding – family: {df_bias_fam['family'].iloc[0]}")
    plt.xlabel("Function index")
    plt.ylabel("ELA feature")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "bias.png", dpi=200)
    plt.close()


def plot_bias_vs_stability_by_family(
    df_cv_fam,
    df_bias_fam,
    output_dir,
):
    """
    Scatter plot of bias vs stability for one feature family.
    Each point corresponds to (feature, function),
    colored by function_idx.
    """

    if df_cv_fam.empty or df_bias_fam.empty:
        return

    # --- Aggregate over instances, keep function & feature ---
    df_cv_agg = (
        df_cv_fam
        .groupby(["function_idx", "feature"])["cv"]
        .mean()
        .reset_index()
    )

    df_bias_agg = (
        df_bias_fam
        .groupby(["function_idx", "feature"])["rel_bias"]
        .mean()
        .abs()
        .reset_index()
    )

    # --- Merge stability and bias ---
    df = df_cv_agg.merge(
        df_bias_agg,
        on=["function_idx", "feature"],
    )

    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="rel_bias",
        y="cv",
        hue="function_idx",
        palette="tab10",     # good default for ≤10 functions
        alpha=0.8,
        edgecolor="black",
    )

    plt.xlabel("Mean |relative bias|")
    plt.ylabel("Mean coefficient of variation")

    family = df_cv_fam["family"].iloc[0]
    plt.title(f"Bias vs stability – family: {family}")

    plt.legend(
        title="Function",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "bias_vs_stability.png", dpi=200)
    plt.close()







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

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df_ref = pd.read_parquet(DATASET_1_PATH)
    df_emb = pd.read_parquet(DATASET_2_PATH)

    df_ref = remove_runtime_sourcefile_name_features(df_ref)
    df_emb = remove_runtime_sourcefile_name_features(df_emb)

    # Only get dimension=20
    df_ref = df_ref[df_ref["dimension"] == 20]
    df_emb = df_emb[df_emb["dimension"] == 20]

    print("✔ Data loaded")
    print("Reference shape:", df_ref.shape)
    print("Embedded shape:", df_emb.shape)

    # ------------------------------------------------------------
    # Collapse rounds -> one ELA vector per embedding
    # ------------------------------------------------------------
    df_mu = collapse_rounds(df_emb)

    print("✔ Collapsed rounds")
    print("Collapsed shape:", df_mu.shape)

    # ------------------------------------------------------------
    # Compute stability (CV across embeddings)
    # ------------------------------------------------------------
    df_cv = compute_stability(df_mu)

    # ------------------------------------------------------------
    # Compute bias (relative to ambient reference)
    # ------------------------------------------------------------
    df_bias = compute_bias(df_mu, df_ref)

    print("✔ Stability and bias computed")

    # ------------------------------------------------------------
    # Assign feature families using PREFIXES
    # ------------------------------------------------------------
    df_cv["family"] = df_cv["feature"].apply(get_feature_family_from_prefix)
    df_bias["family"] = df_bias["feature"].apply(get_feature_family_from_prefix)

    # ------------------------------------------------------------
    # Optional: remove unassigned features
    # ------------------------------------------------------------
    df_cv = df_cv[df_cv["family"] != "other"]
    df_bias = df_bias[df_bias["family"] != "other"]

    print("✔ Feature families assigned")
    print("Feature counts per family:")
    print(df_cv["family"].value_counts())

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------
    stab_dir = OUTPUT_DIR / "stability_by_family"
    bias_dir = OUTPUT_DIR / "bias_by_family"
    scatter_dir = OUTPUT_DIR / "bias_vs_stability_by_family"

    # ------------------------------------------------------------
    # Plot stability & bias diagnostics per ELA feature family
    # ------------------------------------------------------------

    for family in sorted(FEATURE_FAMILIES):

        # --- Stability heatmap ---
        plot_stability_heatmaps_by_family(
            df_cv[df_cv["family"] == family],
            stab_dir / family
        )

        # --- Bias heatmap ---
        plot_bias_heatmaps_by_family(
            df_bias[df_bias["family"] == family],
            bias_dir / family
        )

        # --- Bias vs stability scatter ---
        plot_bias_vs_stability_by_family(
            df_cv[df_cv["family"] == family],
            df_bias[df_bias["family"] == family],
            scatter_dir / family
        )






if __name__ == "__main__":
    main()


