from sklearn.manifold import Isomap
import numpy as np
from typing import Optional, Union

class IsomapWrapper:
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
        self.isomap = Isomap(
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
            weights:Optional[np.ndarray]=None) -> 'IsomapWrapper':
        """
        Fit the model from data in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        weights : array-like, shape (n_samples,), optional
            Sample weights to use during fitting (not used in Isomap)
            
        Returns:
        --------
        self : object
        """

        if weights is None:
            weights = np.ones(X.shape[0])

        # Transform the data
        weights = weights/np.sum(weights)  # Normalize weights

        # Store the weights
        self._weights = weights

        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_scaled = X_centered * weights.reshape(-1, 1)



        self.isomap.fit(X_scaled)
        self.is_fitted = True
        return self
    
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
            raise ValueError("Model must be fitted before transforming data.")
        return self.isomap.transform(X)
    
    def fit_transform(self, 
                      X: np.ndarray,
                      weights:Optional[np.ndarray]=None) -> np.ndarray:
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

        if weights is None:
            weights = np.ones(X.shape[0])

        # Transform the data
        weights = weights/np.sum(weights)  # Normalize weights

        # Store the weights
        self._weights = weights

        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_scaled = X_centered * weights.reshape(-1, 1)

        self.is_fitted = True

        return self.isomap.fit_transform(X_scaled)
    
    def get_params(self) -> dict:
        """Get parameters for this estimator."""
        return self.isomap.get_params()
    
    def set_params(self, **params) -> 'IsomapWrapper':
        """Set the parameters of this estimator."""
        self.isomap.set_params(**params)
        return self
    
    @property
    def embedding_(self) -> np.ndarray:
        """Stores the embedding vectors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return self.isomap.embedding_
    
    @property
    def kernel_pca_(self):
        """KernelPCA object used to implement Isomap."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return self.isomap.kernel_pca_
    
    @property
    def nbrs_(self):
        """Stores the nearest neighbors instance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return self.isomap.nbrs_
    
    @property
    def dist_matrix_(self) -> np.ndarray:
        """Stores the geodesic distance matrix of training data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return self.isomap.dist_matrix_