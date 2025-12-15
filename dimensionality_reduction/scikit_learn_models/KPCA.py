import numpy as np
from sklearn.decomposition import KernelPCA
from pathlib import Path
from typing import Optional, Union
from joblib import load, dump

class WeightedKPCA(KernelPCA):
    """
    Kernel PCA with sample weights support.
    
    Extends scikit-learn's KernelPCA to support weighted samples.
    """

    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)
        self._sample_weights = None
        self._mean = None
    
    def fit(self, X, y=None, sample_weights=None)-> 'WeightedKPCA':
        """
        Fit the model with X using weighted samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
        sample_weights : array-like of shape (n_samples,), default=None
            Sample weights. If None, uniform weights are used.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = np.asarray(X)
        
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        else:
            sample_weights = np.asarray(sample_weights).ravel()
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)

        # Store the weights
        self._sample_weights = sample_weights

         # Center the data
        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().fit(X_weighted, y)
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        """
        Transform X into the kernel PCA space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            Transformed data.
        """
        X = np.asarray(X)

        # Uncenter the data
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().transform(X_weighted)
    
    def fit_transform(self, X, y=None, sample_weights=None):
        """
        Fit the model and apply dimensionality reduction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
        sample_weights : array-like of shape (n_samples,), default=None
            Sample weights. If None, uniform weights are used.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y=y, sample_weights=sample_weights)
        return self.transform(X)
    

    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in the reduced space.
            
        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            Data in the original space.
        """
        # Inverse transform is not straightforward for KernelPCA.
        # This is a placeholder implementation and may not yield accurate results.
        raise NotImplementedError("Inverse transform is not implemented for 'this' WeightedKPCA.")

    def _get_state(self):
        return {
            "init_params": self.get_params(deep=False),
            "_mean": self._mean.tolist(),
            "_sample_weights": self._sample_weights.tolist(),
            "alphas_": self.alphas_.tolist(),
            "lambdas_": self.lambdas_.tolist(),
            "X_fit_": self.X_fit_.tolist(),
        }

    def _set_state(self, state):
        self._mean = np.array(state["_mean"])
        self._sample_weights = np.array(state["_sample_weights"])
        self.alphas_ = np.array(state["alphas_"])
        self.lambdas_ = np.array(state["lambdas_"])
        self.X_fit_ = np.array(state["X_fit_"])

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
    def load_model(path: Union[str, Path]) -> 'WeightedKPCA':
        """Load a wrapper from disk."""
        return load(path)