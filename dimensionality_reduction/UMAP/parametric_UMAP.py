import os
import numpy as np
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from dimensionality_reduction.UMAP.non_parametric_UMAP import DEFAULT_TARGET_METRICS
from pathlib import Path
from warnings import warn, catch_warnings, filterwarnings
import pickle
from typing import Union, Optional, Dict, Any
import json
from tensorflow import keras



class ParametricUMAPTransformer(ParametricUMAP):
    """
    Parametric UMAP transformer with scikit-learn compatible interface.
    """
    
    def __init__(self,
        batch_size=64,
        n_neighbors=15,
        min_dist=0.1,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_fcn=None,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        global_correlation_loss_weight=0,
        landmark_loss_fn=None,
        landmark_loss_weight=1.0,
        target_metric: str = 'l1',
        target_metric_kwds: dict = {},
        keras_fit_kwargs={},
        n_epochs: int = 1000,
        **kwargs):
    
        
        assert target_metric in DEFAULT_TARGET_METRICS, \
            f"target_metric must be one of {list(DEFAULT_TARGET_METRICS.keys())}, got {target_metric}."
        
        super().__init__(
            batch_size=batch_size,
            dims=dims,
            encoder=encoder,
            decoder=decoder,
            parametric_reconstruction=parametric_reconstruction,
            parametric_reconstruction_loss_weight=parametric_reconstruction_loss_weight,
            parametric_reconstruction_loss_fcn=parametric_reconstruction_loss_fcn,
            autoencoder_loss=autoencoder_loss,
            reconstruction_validation=reconstruction_validation,
            global_correlation_loss_weight=global_correlation_loss_weight,
            landmark_loss_weight=landmark_loss_weight,
            landmark_loss_fn=landmark_loss_fn,
            n_epochs=n_epochs,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            keras_fit_kwargs=keras_fit_kwargs,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **kwargs
        )

            
    
    def fit(self, X, y, 
            precomputed_distances=None, 
            landmark_positions=None, 
            **kwargs):
        """
        Fit the Parametric UMAP model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit.
        y : array-like of shape (n_samples,) or None
            The target labels for supervised UMAP. If None, unsupervised UMAP is performed.
        **kwargs : additional keyword arguments
            Additional arguments passed to the parent fit method.

        Returns:
        -------
        self : object
            Returns the instance itself.
        """

        return super().fit(X, y, 
                           precomputed_distances=precomputed_distances, 
                           landmark_positions=landmark_positions, 
                           **kwargs)
    
    def transform(self, X):
        """
        Transform new data into the learned embedding space.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data in the embedding space.
        """
        return super().transform(X, self.batch_size)
    
    def fit_transform(self, X, y, precomputed_distances=None, landmark_positions=None, **kwargs):
        """
        Fit the model to the data and transform it in one step.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit and transform.
        y : array-like of shape (n_samples,) or None
            The target labels for supervised UMAP. If None, unsupervised UMAP is performed.
        **kwargs : additional keyword arguments
            Additional arguments passed to the parent fit_transform method.
        Returns:
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data in the embedding space.
        """

        return super().fit_transform(X, 
                                     y, 
                                     precomputed_distances=precomputed_distances,
                                       landmark_positions=landmark_positions, 
                                       **kwargs)
    
    def inverse_transform(self, X):
        """
        Inverse transform data from the embedding space back to the original space.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_components)
            Data in the embedding space to inverse transform.
            
        Returns:
        --------
        X_original : array-like of shape (n_samples, n_features)
            Data transformed back to the original space.
        """
        return super().inverse_transform(X)
    

    # This is the get_config method from the base class
    def get_config(self):
        # Start from an EMPTY config, not super()
        return {
            "batch_size": self.batch_size,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "n_epochs": self.n_epochs,
            "target_metric": self.target_metric,
            "target_metric_kwds": self.target_metric_kwds,
        }


    
    def save_model(
        self,
        folder_path: Union[str, Path],
        overwrite: bool = True,
        exclude_raw_data: bool = False,
        verbose: bool = True,
    ) -> None:
        folder = Path(folder_path).expanduser().resolve()

        if folder.exists() and not overwrite:
            raise FileExistsError(f"{folder} exists and overwrite=False")

        folder.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # 1) Save ONLY the Keras submodels that are safe to serialize
        #    (DO NOT save self.parametric_model -> causes get_config error)
        # ------------------------------------------------------------
        keras_dir = folder / "keras"
        keras_dir.mkdir(exist_ok=True)

        if getattr(self, "encoder", None) is not None:
            enc_path = keras_dir / "encoder.keras"
            self.encoder.save(enc_path)
            if verbose:
                print(f"Saved encoder to {enc_path}")

        if getattr(self, "decoder", None) is not None:
            dec_path = keras_dir / "decoder.keras"
            self.decoder.save(dec_path)
            if verbose:
                print(f"Saved decoder to {dec_path}")

        # ------------------------------------------------------------
        # 2) Save UMAP base state WITHOUT any Keras objects attached
        # ------------------------------------------------------------
        state: Dict[str, Any] = dict(self.__dict__)  # shallow copy

        # Remove Keras objects that must never be pickled
        for k in ["encoder", "decoder", "parametric_model", "model"]:
            state.pop(k, None)

        # Optionally remove raw data (often huge)
        if exclude_raw_data:
            state.pop("_raw_data", None)
            if "knn_search_index" in state:
                try:
                    # knn_search_index can contain raw data too
                    ks = state["knn_search_index"]
                    if hasattr(ks, "_raw_data"):
                        ks = ks  # keep object but strip raw
                        ks._raw_data = None
                        state["knn_search_index"] = ks
                except Exception:
                    pass

        state_path = folder / "umap_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"Saved UMAP base state to {state_path}")

        # Small metadata file for debugging / future migrations
        meta = {"class": self.__class__.__name__, "format_version": 1}
        meta_path = folder / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"Saved metadata to {meta_path}")

    @classmethod
    def load_model(
        cls,
        folder_path: Union[str, Path],
        verbose: bool = True,
    ) -> "ParametricUMAPTransformer":
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Model folder not found: {folder}")

        # ------------------------------------------------------------
        # 1) Load UMAP base state (Python only)
        # ------------------------------------------------------------
        state_path = folder / "umap_state.pkl"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing {state_path}")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        obj = cls.__new__(cls)          # bypass __init__
        obj.__dict__.update(state)

        # ------------------------------------------------------------
        # 2) Load Keras submodels (encoder/decoder)
        # ------------------------------------------------------------
        keras_dir = folder / "keras"

        enc_path = keras_dir / "encoder.keras"
        if enc_path.exists():
            obj.encoder = keras.models.load_model(enc_path)
            if verbose:
                print(f"Loaded encoder from {enc_path}")
        else:
            obj.encoder = None

        dec_path = keras_dir / "decoder.keras"
        if dec_path.exists():
            obj.decoder = keras.models.load_model(dec_path)
            if verbose:
                print(f"Loaded decoder from {dec_path}")
        else:
            obj.decoder = None

        # ------------------------------------------------------------
        # 3) Rebuild parametric model if needed (optional but recommended)
        # ------------------------------------------------------------
        # Some codepaths require parametric_model; if you hit issues later,
        # rebuild it lazily in fit/transform, or rebuild here if you know how.
        obj.parametric_model = None

        return obj
        
