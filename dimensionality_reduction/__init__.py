# __init__.py

import importlib


UNSUPERVISED_MODELS = [
    "WeightedPCA",
    "TruncatedSVDWrapper",
    "IsomapWrapper",
    "LLEWrapper",
    "WeightedFastICA",
    "ScikitPCAWrapper",
    "ScikitKPCAWrapper"]

SUPERVISED_MODELS = [
    "IvisWrapper",
    "UMAPWrapper",
    "ParametricUMAPWrapper",
]

RANDOM_MODELS = [
    "RandomEmbedding"
]


def _safe_import(module_path, name=None):
    """
    Dynamically import a module or attribute.
    Returns None if the import fails.
    """
    try:
        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, name) if name else module
    except Exception:
        return None


def is_supervised_model(model) -> bool:
    if model is None:
        return False

    # Handle instances
    cls = model if isinstance(model, type) else model.__class__

    return cls.__name__ in SUPERVISED_MODELS

def is_random_model(model) -> bool:
    if model is None:
        return False

    # Handle instances
    cls = model if isinstance(model, type) else model.__class__

    return cls.__name__ in RANDOM_MODELS


# --- hard-coded models ---
WeightedPCA = _safe_import(".hard_coded_models.weighted_pca", "WeightedPCA")
RandomEmbedding = _safe_import(".hard_coded_models.random_embedding", "RandomEmbedding")

# --- scikit-learn models ---
IsomapWrapper = _safe_import(".scikit_learn_models.isomap", "IsomapWrapper")
WeightedFastICA = _safe_import(".scikit_learn_models.FastICA", "WeightedFastICA")
ScikitKPCAWrapper = _safe_import(".scikit_learn_models.KPCA", "WeightedKPCA")
ScikitPCAWrapper = _safe_import(".scikit_learn_models.PCA", "WeightedPCA")
LLEWrapper = _safe_import(".scikit_learn_models.lle", "LLEWrapper")
TruncatedSVDWrapper = _safe_import(".scikit_learn_models.truncated_SVD", "WeightedTruncatedSVD")

# --- IVIS model ---
IvisWrapper = _safe_import(".ivis.ivis", "IvisWrapper")

# --- UMAP model ---
UMAPWrapper = _safe_import(".UMAP.non_parametric_UMAP", "NonParametricUMAP")
ParametricUMAPWrapper = _safe_import(".UMAP.parametric_UMAP", "ParametricUMAPTransformer")

# ---- UMAP Torch extension ---
UMAPTorch = _safe_import(".UMAP_extensions.umap_torch", "UMAPTorch")
UMAPTorchSupervised = _safe_import(".UMAP_extensions.umap_torch_2", "SupervisedUMAP")
