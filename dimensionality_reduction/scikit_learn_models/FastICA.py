import numpy as np
from numpy.linalg import eigh, pinv
from scipy.linalg import sqrtm
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA

class WeightedFastICA(FastICA):
    def __init__(self, n_components: Optional[int] = None,
                 algorithm: str = 'parallel',
                 whiten: bool = True,
                 fun: str = 'logcosh',
                 fun_args: Optional[dict] = None,
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 w_init: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None):
        
        r"""
        Weighted Fast Independent Component Analysis (FastICA) implementation
        
        
        Args:
            n_components: Number of components to extract
            algorithm: 'parallel' or 'deflation' algorithm
            whiten: Whether to whiten the data before applying ICA
            fun: The functional form of the G function used in the approximation to neg-entropy
            fun_args: Additional arguments to pass to the functional form
            max_iter: Maximum number of iterations during fit
            tol: Tolerance for convergence
            w_init: Initial guess for the unmixing matrix
            random_state: Seed for random number generator
        """
        
        super().__init__(n_components=n_components,
                         algorithm=algorithm,
                         whiten=whiten,
                         fun=fun,
                         fun_args=fun_args,
                         max_iter=max_iter,
                         tol=tol,
                         w_init=w_init,
                         random_state=random_state)
        
        # Initialize sample weights
        self._sample_weights: np.ndarray = None
        self._mean: np.ndarray = None

        # Initialize a scaler for later use
        self._scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None) -> 'WeightedFastICA':
        
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

        # Fit the FastICA model on weighted data
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

    