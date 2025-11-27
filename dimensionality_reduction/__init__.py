# __init__.py

import importlib


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


# --- hard-coded models ---
WeightedPCA = _safe_import(".hard_coded_models.weighted_pca", "WeightedPCA")

# --- scikit-learn models ---
IsomapWrapper = _safe_import(".scikit_learn_models.isomap", "IsomapWrapper")
WeightedFastICA = _safe_import(".scikit_learn_models.FastICA", "WeightedFastICA")
ScikitKPCAWrapper = _safe_import(".scikit_learn_models.KPCA", "WeightedKPCA")
ScikitPCAWrapper = _safe_import(".scikit_learn_models.PCA", "WeightedPCA")
LLEWrapper = _safe_import(".scikit_learn_models.lle", "LLEWrapper")

# --- IVIS model ---
IvisWrapper = _safe_import(".ivis.ivis", "IvisWrapper")
