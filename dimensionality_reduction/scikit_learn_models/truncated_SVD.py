from sklearn.decomposition import TruncatedSVD
from joblib import load, dump
import numpy as np
from typing import Optional, Union
from pathlib import Path

class WeightedTruncatedSVD(TruncatedSVD):
    def __init__(self, n_components=None,
                 algorithm='randomized',
                    n_iter=5,
                    random_state=None,
                    n_oversamples=10,
                    power_iteration_normalizer='auto',
                    tol=1e-12,
                  **kwargs):
        
        r"""
        Weighted Truncated SVD with sample weights.
        Extends scikit-learn's TruncatedSVD to support weighted samples.
        Parameters
        ----------
        n_components : int, default=None
            Number of components to keep.
        algorithm : {'randomized', 'arpack'}, default='randomized'
            SVD solver to use.
        n_iter : int, default=5
            Number of iterations for the randomized SVD solver.
        random_state : int, RandomState instance or None, default=None
            Random state for reproducibility.
        n_oversamples : int, default=10
            Additional number of random vectors to sample the range of X.
        power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
            Power iteration normalizer to use.
        **kwargs : dict
            Additional arguments passed to sklearn.decomposition.TruncatedSVD
        
        Returns
        -------
        self : object
            Unfitted estimator.
        """

        super().__init__(n_components=n_components,
                         algorithm=algorithm,
                         n_iter=n_iter,
                         random_state=random_state,
                         n_oversamples=n_oversamples,
                         power_iteration_normalizer=power_iteration_normalizer,
                         tol=tol,
                         **kwargs)
        
        # Initialize sample weights
        self._sample_weights: np.ndarray = None
        self._mean: np.ndarray = None

    
    def fit(self, X, y=None, sample_weights=None):
        """
        Fit the Truncated SVD model with optional sample weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        
        if sample_weights is None:
            sample_weights = np.ones_like(X.shape[0])
        else:
            sample_weights = np.asarray(sample_weights).ravel()
        
        # Normalize weights (for sanity of implementation)
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
        Transform data to the truncated SVD space.
        
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

        # Scale the data with sample weights
        X_weighted = X_centered * self._sample_weights[:, None]

        return super().transform(X_weighted)
    
    def fit_transform(self, X, y=None, sample_weights=None):
        """
        Fit the model and apply dimensionality reduction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            Transformed data.
        """

        X = np.asarray(X)
        
        if sample_weights is None:
            sample_weights = np.ones_like(X.shape[0])
        else:
            sample_weights = np.asarray(sample_weights).ravel()
        
        # Normalize weights (for sanity of implementation)
        sample_weights = sample_weights / np.sum(sample_weights)

        # Store the weights
        self._sample_weights = sample_weights

        # Center the data
        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().fit_transform(X_weighted, y)
    

    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in the reduced space.
            
        Returns
        -------
        X_original : array of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        X = np.asarray(X)

        # Perform the inverse transformation
        X_reconstructed = super().inverse_transform(X)

        # Unscale the data with sample weights
        X_unscaled = X_reconstructed * (self._sample_weights ** -1)[:, None]

        # Re-add the mean
        X_original = X_unscaled + self._mean

        return X_original
    
    def _get_state(self):
        return {
            "init_params": self.get_params(deep=False),
            "_mean": self._mean.tolist(),
            "_sample_weights": self._sample_weights.tolist(),
            "components_": self.components_.tolist(),
            "singular_values_": self.singular_values_.tolist(),
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_features_in_": self.n_features_in_,
        }

    def _set_state(self, state):
        self._mean = np.array(state["_mean"])
        self._sample_weights = np.array(state["_sample_weights"])
        self.components_ = np.array(state["components_"])
        self.singular_values_ = np.array(state["singular_values_"])
        self.explained_variance_ = state["explained_variance_"]
        self.explained_variance_ratio_ = state["explained_variance_ratio_"]
        self.n_features_in_ = state["n_features_in_"]
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
    def load_model(path: str) -> 'WeightedTruncatedSVD':
        """Load a wrapper from disk."""
        return load(path)