import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from joblib import load, dump
from typing import Optional, Union
from pathlib import Path


class LLEWrapper(LocallyLinearEmbedding):
    def __init__(self, n_components: int = 2,
                 n_neighbors: int = 5,
                 method: str = 'standard',
                 eigen_solver: Optional[str] = 'auto',
                 tol: float = 1e-6,
                 max_iter: int = 100,
                 hessian_tol: float = 1e-4,
                 modified_tol: float = 1e-4,
                 neighbors_algorithm: str = 'auto',
                 random_state: Optional[int] = 1234,
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
                         hessian_tol=hessian_tol,
                         modified_tol=modified_tol,
                         neighbors_algorithm=neighbors_algorithm,
                         random_state=random_state,
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
        X_weighted = X_centered * self._sample_weights[:, None]

        # Fit the LLE model on weighted data
        super().fit(X_weighted, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Center the data with weights
        X_centered = X - self._mean
        
        # Apply weights
        X_weighted = X_centered * self._sample_weights[:, None]
        
        return super().transform(X_weighted)
    
    def fit_transform(self, X, y = None, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        
        
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
        X_weighted = X_centered * self._sample_weights[:, None]

        return super().fit_transform(X_weighted, y)
    
    def _get_state(self):
        return {
            "init_params": self.get_params(deep=False),
            "_mean": self._mean.tolist(),
            "_sample_weights": self._sample_weights.tolist(),
            "embedding_": self.embedding_.tolist(),
            "reconstruction_error_": self.reconstruction_error_,
            "nbrs_": self.nbrs_,
        }

    def _set_state(self, state):
        self._mean = np.array(state["_mean"])
        self._sample_weights = np.array(state["_sample_weights"])
        self.embedding_ = np.array(state["embedding_"])
        self.reconstruction_error_ = state["reconstruction_error_"]
        self.nbrs_ = state["nbrs_"]
    
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
    def load_model(path: Union[str, Path]) -> 'LLEWrapper':
        """Load a wrapper from disk."""
        return load(path)
    
