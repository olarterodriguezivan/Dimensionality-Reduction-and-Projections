from sklearn.manifold import Isomap
import numpy as np
from typing import Optional, Union
from joblib import load, dump
from pathlib import Path

class IsomapWrapper(Isomap):
    """
    Wrapper class for scikit-learn's Isomap dimensionality reduction.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        radius: Optional[float] = None,
        eigen_solver: str = 'auto',
        tol: float = 0,
        max_iter: Optional[int] = None,
        path_method: str = 'auto',
        neighbors_algorithm: str = 'auto',
        n_jobs: Optional[int] = None,
        metric: Union[str, callable] = 'minkowski',
        p: int = 2,
        metric_params: Optional[dict] = None
    ):
        """
        Initialize Isomap wrapper.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of coordinates for the manifold
        n_neighbors : int, default=5
            Number of neighbors to consider for each point
        radius : float, optional
            Limiting distance of neighbors to return
        eigen_solver : str, default='auto'
            Eigenvalue decomposition strategy
        tol : float, default=0
            Convergence tolerance passed to arpack or lobpcg
        max_iter : int, optional
            Maximum number of iterations for the arpack solver
        path_method : str, default='auto'
            Method to use in finding shortest path
        neighbors_algorithm : str, default='auto'
            Algorithm to use for nearest neighbors search
        n_jobs : int, optional
            Number of parallel jobs to run
        metric : str or callable, default='minkowski'
            Metric to use for distance computation
        p : int, default=2
            Parameter for the Minkowski metric
        metric_params : dict, optional
            Additional keyword arguments for the metric function
        """
        
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            radius=radius,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            metric=metric,
            p=p,
            metric_params=metric_params
        )
        self.is_fitted = False
    
    def fit(self, 
            X: np.ndarray,
            y:Optional[np.ndarray]=None,
            sample_weights:Optional[np.ndarray]=None) -> Isomap:
        """
        Fit the model from data in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency.
        weights : array-like, shape (n_samples,), optional
            Sample weights to use during fitting (not used in Isomap)
            
        Returns:
        --------
        self : object
        """

        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        else:
            sample_weights = np.asarray(sample_weights).ravel()


        # Transform the data
        sample_weights = sample_weights/np.sum(sample_weights)  # Normalize weights
        
        # Store the weights
        self._sample_weights = sample_weights

        # Center the data
        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().fit(X_weighted, y)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X into the embedded space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        # Center the data
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().transform(X_weighted)
    
    def fit_transform(self, 
                      X: np.ndarray,
                      sample_weights:Optional[np.ndarray]=None) -> np.ndarray:
        """
        Fit the model from data in X and transform X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """

        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])

        # Transform the data
        sample_weights = sample_weights/np.sum(sample_weights)  # Normalize weights
        # Store the weights
        self._sample_weights = sample_weights

        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        self.is_fitted = True

        return super().fit_transform(X_weighted)
    
    def _get_state(self):
        return {
            "init_params": self.get_params(deep=False),
            "_mean": self._mean.tolist(),
            "_sample_weights": self._sample_weights.tolist(),
            "embedding_": self.embedding_.tolist(),
            "dist_matrix_": self.dist_matrix_.tolist(),
            "nbrs_": self.nbrs_,
            "is_fitted": self.is_fitted,
        }

    def _set_state(self, state):
        self._mean = np.array(state["_mean"])
        self._sample_weights = np.array(state["_sample_weights"])
        self.embedding_ = np.array(state["embedding_"])
        self.dist_matrix_ = np.array(state["dist_matrix_"])
        self.nbrs_ = state["nbrs_"]
        self.is_fitted = state["is_fitted"]
    
    # ---------------------------------------
    # Saving and loading methods
    # ---------------------------------------

    def save_model(self, path: Union[str, Path],
                   overwrite: bool = True)-> None:
        """Save the entire wrapper (reducer + model + state)."""

        # Ensure path is a Path object
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"The file {path} already exists and overwrite is set to False.")
        
        # Make sure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        dump(self, path)
    
    @staticmethod
    def load_model(path: str) -> 'IsomapWrapper':
        """Load a wrapper from disk."""
        return load(path)
    
    