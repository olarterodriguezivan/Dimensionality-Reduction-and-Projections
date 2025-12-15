import numpy as np
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from dimensionality_reduction.UMAP.non_parametric_UMAP import DEFAULT_TARGET_METRICS
from pathlib import Path


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


    

    def save_model(self,
                   folder_path: str,
                   overwrite: bool = True,
                   exclude_raw_data: bool = False) -> None:

        folder_path:Path = Path(folder_path).expanduser().resolve()

        if folder_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"The folder {folder_path} already exists and overwrite=False."
                )
        else:
            folder_path.mkdir(parents=True, exist_ok=True)

        super().save(
            save_location=folder_path.as_posix(),
            verbose=True,
            exclude_raw_data=exclude_raw_data,
        )



    @classmethod
    def load_model(cls, folder_path: str) -> "ParametricUMAPTransformer":
        folder_path = Path(folder_path).expanduser().resolve()

        base_model = load_ParametricUMAP(
            save_location=folder_path.as_posix(),
            verbose=True,
        )

        # Create an uninitialized instance
        obj = cls.__new__(cls)

        # Copy internal state
        obj.__dict__.update(base_model.__dict__)

        return obj

        
