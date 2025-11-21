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
        
    def fit(self, X, y=None, sample_weight=None):
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
        
        if sample_weight is None:
            # Use parent class fit method
            return super().fit(X, y)
        
        # Weighted PCA
        sample_weight = np.asarray(sample_weight)
        # Normalize weights
        sample_weight = sample_weight / sample_weight.sum()

        # Store the weights as an attribute
        self._sample_weight = sample_weight
        
        # Compute weighted mean
        self.mean_ = np.average(X, axis=0, weights=sample_weight)
        
        # Center data
        X_centered = X - self.mean_
        
        # Compute weighted covariance matrix
        X_weighted = X_centered * np.sqrt(sample_weight)[:, np.newaxis]
        
        # Use parent's _fit method on weighted centered data
        # We need to temporarily set mean_ to zero to avoid double centering
        temp_mean = self.mean_.copy()
        self.mean_ = np.zeros_like(temp_mean)
        super().fit(X_weighted, y)
        self.mean_ = temp_mean
        
        return self
    
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


    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return super().transform(X)
    
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
        return super().inverse_transform(X)
    

    @property
    def sample_weights(self):
        """
        Get the sample weights used during fitting.
        
        Returns
        -------
        sample_weight : ndarray of shape (n_samples,) or None
            Sample weights if provided during fitting, else None.
        """
        return self._sample_weights 
    

    @sample_weights.setter
    def sample_weights(self, weights):
        """
        Set the sample weights.
        
        Parameters
        ----------
        weights : array-like of shape (n_samples,)
            Sample weights to set.
        """
        self._sample_weights = np.asarray(weights)
