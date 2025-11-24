from sklearn.manifold import Isomap
import numpy as np
from typing import Optional, Union

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

        X_scaled = X_centered * np.diag(sample_weights)

        return super().fit(X_scaled, y)
    
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

        X_scaled = X_centered * np.diag(self._sample_weights)

        return super().transform(X_scaled)
    
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

        X_scaled = X_centered * sample_weights.reshape(-1, 1)

        self.is_fitted = True

        return super().fit_transform(X_scaled)
    
    