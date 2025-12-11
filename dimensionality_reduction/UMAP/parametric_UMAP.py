import numpy as np
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from dimensionality_reduction.UMAP.non_parametric_UMAP import DEFAULT_TARGET_METRICS
from typing import Optional
from joblib import load, dump
#from keras.models import load_model, save_model

class ParametricUMAPTransformer(ParametricUMAP):
    """
    Parametric UMAP transformer with scikit-learn compatible interface.
    """
    
    def __init__(self,
                 batch_size=128,
                 dims=2,
                 encoder=None,
                 decoder=None,
                 parametric_reconstruction=False,
                 parametric_reconstruction_loss_weight=0.0,
                 parametric_reconstruction_loss_fcn=None,
                 autoencoder_loss = None,
                 reconstruction_validation = None,
                 global_correlation_loss_weight = 0.0,
                 landmark_loss_weight = 0.0,
                 landmark_loss_fcn = None,
                    n_epochs=None,
                    negative_sample_rate=5,
                    learning_rate=1.0,
                    repulsion_strength=1.0,
                    target_metric="l1",
                    target_weight=0.5,
                    transform_seed=42,
                    **kwargs):
        
        if target_metric not in DEFAULT_TARGET_METRICS:
            raise ValueError(f"target_metric must be one of {DEFAULT_TARGET_METRICS}, got {target_metric}")
        
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
            landmark_loss_fcn=landmark_loss_fcn,
            n_epochs=n_epochs,
            negative_sample_rate=negative_sample_rate,
            learning_rate=learning_rate,
            repulsion_strength=repulsion_strength,
            target_metric=target_metric,
            target_weight=target_weight,
            transform_seed=transform_seed,
            **kwargs
        )

            
    
    def fit(self, X, y, **kwargs):
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

        return super().fit(X, y, **kwargs)
    
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
        return super().transform(X)
    
    def fit_transform(self, X, y=None, precomputed_distances=None, landmark_positions=None, **kwargs):
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
    

    def save_model(self, folder_path: str) -> None:
        """
        This is just a wrapper to save both the UMAP model and the Keras models.
        Parameters
        ----------
        folder_path : str
            Path to the folder where the model will be saved.
        """

        super().save(folder_path,
                     verbose=True,
                       save_format="h5")


    @staticmethod
    def load_model(folder_path: str) -> 'ParametricUMAPTransformer':
        """
        Load a ParametricUMAPTransformer model from disk.
        
        Parameters
        ----------
        folder_path : str
            Path to the folder where the model is saved.
            
        Returns
        -------
        model : ParametricUMAPTransformer
            The loaded ParametricUMAPTransformer model.
        """
        return load_ParametricUMAP(save_location=folder_path,
                                   verbose=True)
        
