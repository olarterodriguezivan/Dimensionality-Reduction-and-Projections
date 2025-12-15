import numpy as np
from typing import Optional, Union, Tuple, List
from pathlib import Path
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

    Note that this wrapper is just meant for supervised ivis, where regression targets are provided.
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
        **kwargs
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
        
         # Store init parameters for serialization
        self._init_params = dict(
            n_components=n_components,
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
            verbose=verbose,
            **kwargs
        )
        

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
            verbose=verbose,
            **kwargs
        )
        
    def fit(self, 
            X: np.ndarray, 
            y:np.ndarray) -> 'IvisWrapper':
        """
        Fit the ivis model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values for supervised dimensionality reduction.
        Returns
        -------
        self : IvisWrapper
            Returns the instance itself.
        """

        assert len(X.shape) == 2, "X must be a 2D array."
        assert y is None or len(y.shape) == 1, "Y must be a 1D array or None."

    
        
        super().fit(X,y)

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

        # Return transformed data
        return super().transform(X)
    
        
    def fit_transform(self, 
                      X: np.ndarray, 
                      y: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Parameters
        ---------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values for supervised dimensionality reduction.
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, embedding_dims)
            Transformed data in the embedding space.
        """
        return self.fit(X, y).transform(X)
    
    def to_dict(self) -> dict:
        """
        Convert the wrapper (excluding the trained TF model) to a JSON-safe dict.
        """
        state = {
            "class": self.__class__.__name__,
            "init_params": deepcopy(self._init_params),
            "mean": None,
        }

        if hasattr(self, "_mean") and self._mean is not None:
            state["mean"] = self._mean.tolist()

        return state

    @classmethod
    def from_dict(cls, state: dict) -> "IvisWrapper":
        """
        Reconstruct an IvisWrapper from a dict.
        """
        obj = cls(**state["init_params"])

        if state.get("mean") is not None:
            obj._mean = np.asarray(state["mean"])

        return obj
    

    def save_model(self, folder_path: str,
                   save_format: str = "tf",
                   overwrite=True) -> None:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        folder_path : str
            Path to the folder where the model will be saved.
        """

        folder_path = Path(folder_path).absolute().as_posix()

        # Check if the parent folder exists
        parent_folder = Path(folder_path).parent

        if not overwrite and Path(folder_path).exists():
            raise FileExistsError(f"The folder {folder_path} already exists and overwrite is set to False.")
        
        if not parent_folder.exists():
            parent_folder.mkdir(parents=True, exist_ok=True)

        super().save_model(folder_path,
                           save_format=save_format,
                           overwrite=overwrite)
    
    def load_model(self, filepath: str) -> 'IvisWrapper':
        """
        Load a trained model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the file from which the model will be loaded.
            
        Returns
        -------
        model : IvisWrapper
            The loaded ivis model.
        """
        

        filepath = Path(filepath).absolute().as_posix()

        loaded = Ivis().load_model(filepath)

        # Copy ivis internals into *this* instance
        self.__dict__.update(loaded.__dict__)

        return self

    
    
    