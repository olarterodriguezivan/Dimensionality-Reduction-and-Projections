from sklearn.decomposition import TruncatedSVD
from joblib import load, dump
import numpy as np


class WeightedTruncatedSVD(TruncatedSVD):
    def __init__(self, n_components=None,
                 algorithm='randomized',
                    n_iter=5,
                    random_state=None,
                    n_oversamples=10,
                    power_iteration_normalizer='auto',
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
                         **kwargs)

    
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
        return self.fit(X, y=y, sample_weights=sample_weights).transform(X)
    

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
    
    # ---------------------------------------
    # Saving and loading methods
    # ---------------------------------------

    def save_model(self, path: str)-> None:
        """Save the entire wrapper (reducer + model + state)."""
        dump(self, path)

    @staticmethod
    def load_model(path: str) -> 'WeightedTruncatedSVD':
        """Load a wrapper from disk."""
        return load(path)