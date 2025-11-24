import numpy as np
from typing import Optional, Union, Tuple, List
from copy import deepcopy, copy


"""
Wrapper for the ivis dimensionality reduction library.
"""

try:
    from ivis import Ivis
except ImportError:
    raise ImportError("ivis library not found. Install with: pip install ivis")


# This is the list of allowed regression metrics for supervised ivis
ALLOWED_REGRESSION_METRICS = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                              'mean_squared_logarithmic_error','cosine_similarity',"huber",
                               "log_cosh", "tversky", "dice" ]


class IvisWrapper(Ivis):
    """
    Wrapper class for the ivis dimensionality reduction algorithm.
    
    Ivis is a machine learning library for reducing dimensionality of very large datasets
    using Siamese Neural Networks.
    """

    
    def __init__(
        self,
        n_components: int = 2,
        k: int = 150,
        distance: str = 'pn',
        batch_size: int = 128,
        epochs: int = 1000,
        n_epochs_without_progress: int = 50,
        knn_distance_metric: str = 'angular',
        n_trees: int = 10,
        search_k: int = -1,
        precompute: bool = True,
        model: str = 'szubert',
        supervision_metric: str = 'cosine_similarity',
        supervision_weight: float = 0.5,
        annoy_index_path: Optional[str] = None,
        build_index_on_disk: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize ivis wrapper.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions in the embedding space.
        k : int, default=150
            The number of neighbours to retrieve for each point.
        distance : str, default='pn'
            The loss function used to train the neural network ('pn', 'euclidean', 'softmax_ratio_pn').
        batch_size : int, default=128
            The size of mini-batches used during gradient descent.
        epochs : int, default=1000
            The maximum number of epochs to train for.
        n_epochs_without_progress : int, default=50
            After n number of epochs without an improvement in the loss, terminate training.
        knn_distance_metric : str, default='angular'
            The distance metric used to compute KNN 
        n_trees : int, default=10
            The number of random projections trees built by Annoy to approximate KNN.
        search_k : int, default=-1
            The maximum number of nodes inspected during a tree search.
        precompute : bool, default=True
            Whether to pre-compute the trustworthiness and continuity.
        model : str, default='szubert'
            The neural network architecture to use ('maaten', 'hinton', 'szubert').
        supervision_metric : str, default='cosine_similarity'
            The loss function for supervised training (for regression).
        supervision_weight : float, default=0.5
            Weighting factor for the supervised loss component.
        annoy_index_path : str, optional
            The filepath of a pre-trained annoy index file.
        build_index_on_disk : bool, default=True
            Whether to build the annoy index on disk.
        verbose : int, default=1
            Controls the verbosity of the training process.
        """

        if supervision_metric not in ALLOWED_REGRESSION_METRICS:
            raise ValueError(f"supervision_metric must be one of {ALLOWED_REGRESSION_METRICS}, got {supervision_metric}.")
        
        if model not in ['maaten', 'hinton', 'szubert']:
            raise ValueError("model must be one of ['maaten', 'hinton', 'szubert']")
        

        super().__init__(
            embedding_dims=n_components,
            k=k,
            distance=distance,
            batch_size=batch_size,
            epochs=epochs,
            n_epochs_without_progress=n_epochs_without_progress,
            knn_distance_metric=knn_distance_metric,
            n_trees=n_trees,
            search_k=search_k,
            precompute=precompute,
            model=model,
            supervision_metric=supervision_metric,
            supervision_weight=supervision_weight,
            annoy_index_path=annoy_index_path,
            build_index_on_disk=build_index_on_disk,
            verbose=verbose
        )
        
    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None, 
            weights:Optional[Union[np.ndarray,List]]=None) -> 'IvisWrapper':
        """
        Fit the ivis model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        Y : array-like, shape (n_samples,), optional
            Target values for supervised dimensionality reduction.
        weights : array-like, shape (n_samples,), optional
            Sample weights to use during training.
        Returns
        -------
        self : IvisWrapper
            Returns the instance itself.
        """

        # Set weights
        self.weights = weights

        # Compute mean for centering
        self._mean = np.mean(X, axis=0)

        X_centered = X - self._mean

        # Use weights if provided
        if self.weights is not None:
            X_mod = X_centered* self.weights.reshape(-1, 1)
        else:
            X_mod = X_centered
            

        # Transform 
        if Y is not None:
            self._model.fit(X_mod,Y)
        else:
            self._model.fit(X_mod)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimensional space.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, embedding_dims)
            Transformed data in the embedding space.
        """

        # Center data
        X_mod = X - self._mean

        # Return transformed data
        return self._model.transform(X_mod)
        
    def fit_transform(self, X: np.ndarray, 
                      Y: Optional[np.ndarray] = None,
                      weights:Optional[Union[np.ndarray,List]]=None) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        Y : array-like, shape (n_samples,), optional
            Target values for supervised dimensionality reduction.
        weights : array-like, shape (n_samples,), optional
            Sample weights to use during training.
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, embedding_dims)
            Transformed data in the embedding space.
        """
        # Set weights
        self.weights = weights

        # Compute mean for centering
        self._mean = np.mean(X, axis=0)

        X_centered = X - self._mean

        # Use weights if provided
        if self.weights is not None:
            X_mod = X_centered * self.weights.reshape(-1, 1)
        else:
            X_mod = X_centered
        
        # Fit and transform
        if Y is not None:
            return self._model.fit_transform(X_mod, Y, shuffle_mode=True)
        else:
            return self._model.fit_transform(X_mod, shuffle_mode=True)
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path where to save the model.
        """
        self._model.save_model(filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        self._model.load_model(filepath)
    
    @property
    def model(self) -> Ivis:
        """
        Get the underlying ivis model (deepcopy).
        
        Returns
        -------
        model : Ivis
            The underlying ivis model instance.
        """
        return deepcopy(self._model)

    @property
    def model_params(self) -> dict:
        """
        Get the model parameters.
        
        Returns
        -------
        params : dict
            Dictionary of model parameters.
        """
        return self._model.get_params()
    
    @property
    def weights(self) -> Optional[np.ndarray]:
        """
        Get the sample weights used during training.
        
        Returns
        -------
        weights : np.ndarray or None
            Sample weights if provided during initialization, else None.
        """
        return self._weights
    
    @weights.setter
    def weights(self, value: Optional[Union[np.ndarray, List]]) -> None:
        """
        Set the sample weights.
        
        Parameters
        ----------
        value : np.ndarray or list or None
            Sample weights to set.
        """
        if value is not None:
            if isinstance(value, list):
                self._weights = np.array(value)
            elif isinstance(value, np.ndarray):
                self._weights = value
            else:
                raise ValueError("Weights must be a numpy array, list, or None.")
        else:
            self._weights = None
    
    @weights.deleter
    def weights(self) -> None:
        """
        Delete the sample weights.
        """
        self._weights = None