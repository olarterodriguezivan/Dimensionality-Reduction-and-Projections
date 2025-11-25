import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from typing import Optional


class LLEWrapper(LocallyLinearEmbedding):
    def __init__(self, n_components: int = 2,
                 n_neighbors: int = 5,
                 method: str = 'standard',
                 eigen_solver: Optional[str] = None,
                 tol: float = 1e-6,
                 max_iter: int = 100,
                 n_jobs: Optional[int] = None):
        r"""
        Wrapper for sklearn's Locally Linear Embedding (LLE)

        Args:
            n_components: Number of dimensions to reduce to
            n_neighbors: Number of neighbors to consider for each point
            method: LLE method to use ('standard', 'modified', 'hessian', 'ltsa')
            eigen_solver: Eigenvalue solver to use ('auto', 'arpack', 'dense')
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations for the eigenvalue solver
            n_jobs: Number of parallel jobs to run
        """
        super().__init__(n_components=n_components,
                         n_neighbors=n_neighbors,
                         method=method,
                         eigen_solver=eigen_solver,
                         tol=tol,
                         max_iter=max_iter,
                         n_jobs=n_jobs)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None) -> 'LLEWrapper':
        
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        elif not isinstance(sample_weights, np.ndarray):
            sample_weights = np.array(sample_weights).ravel()
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)

        # Store sample weights
        self._sample_weights = sample_weights.copy()

        # Center the data with weights
        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean
        
        # Apply weights
        X_weighted = X_centered * np.diag(self._sample_weights)

        # Fit the LLE model on weighted data
        super().fit(X_weighted, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Center the data with weights
        X_centered = X - self._mean
        
        # Apply weights
        X_weighted = X_centered * np.diag(self._sample_weights)
        
        return super().transform(X_weighted)
    
    def fit_transform(self, X, y = None, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y, sample_weights=sample_weights).transform(X)
    
