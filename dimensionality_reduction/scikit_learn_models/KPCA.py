import numpy as np
from sklearn.decomposition import KernelPCA
from joblib import load, dump

class WeightedKPCA(KernelPCA):
    """
    Kernel PCA with sample weights support.
    
    Extends scikit-learn's KernelPCA to support weighted samples.
    """
    
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

        X_weighted = X_centered * np.diag(self._sample_weights)

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

        X_weighted = X_centered * np.diag(self._sample_weights)

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
    
    # ---------------------------------------
    # Saving and loading methods
    # ---------------------------------------

    def save_model(self, path: str)-> None:
        """Save the entire wrapper (reducer + model + state)."""
        dump(self, path)

    @staticmethod
    def load(path: str) -> 'WeightedKPCA':
        """Load a wrapper from disk."""
        return load(path)