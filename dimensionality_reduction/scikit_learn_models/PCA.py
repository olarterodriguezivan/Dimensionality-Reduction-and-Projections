from sklearn.decomposition import PCA as SklearnPCA
import numpy as np

class WeightedPCA(SklearnPCA):
    """
    PCA with support for sample weights.
    
    Inherits from sklearn.decomposition.PCA and adds weighted fitting capability.
    
    Parameters
    ----------
    n_components : int, float or None, default=None
        Number of components to keep.
    **kwargs : dict
        Additional arguments passed to sklearn.decomposition.PCA
    """
    
    def __init__(self, n_components=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        
    def fit(self, X, y=None, sample_weights=None):
        """
        Fit the PCA model with optional sample weights.
        
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

        # Transform data to the principal component space.
        X = np.asarray(X)

        # Uncenter the data
        X_centered = X - self._mean

        # Scale the data with sample weights
        X_weighted = X_centered * np.diag(self._sample_weights)

        return super().transform(X_weighted)
    
    def fit_transform(self, X, y=None, sample_weight=None):
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
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y=y, sample_weight=sample_weight)
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
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """

        # Perform the inverse transformation
        X_reconstructed = super().inverse_transform(X)

        # Unscale the data with sample weights
        X_unscaled = X_reconstructed * np.diag(np.power(self._sample_weights,-1))

        # Re-add the mean
        X_original = X_unscaled + self._mean
        return X_original
        

    
