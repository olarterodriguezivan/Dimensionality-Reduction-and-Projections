import numpy as np
from sklearn.decomposition import KernelPCA

class WeightedKPCA(KernelPCA):
    """
    Kernel PCA with sample weights support.
    
    Extends scikit-learn's KernelPCA to support weighted samples.
    """
    
    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the model with X using weighted samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, uniform weights are used.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1D array")
            if len(sample_weight) != len(X):
                raise ValueError("sample_weight must have same length as X")
            
            # Normalize weights
            sample_weight = sample_weight / sample_weight.sum()
            
            # Apply weights by duplicating/resampling
            # For kernel methods, we weight the kernel matrix
            self._sample_weight = sample_weight
        else:
            self._sample_weight = None
            
        return super().fit(X, y)
    
    def _get_kernel(self, X, Y=None):
        """
        Compute weighted kernel matrix.
        """
        K = super()._get_kernel(X, Y)
        
        if hasattr(self, '_sample_weight') and self._sample_weight is not None:
            # Weight the kernel matrix
            W = np.sqrt(self._sample_weight)
            if Y is None:
                K = W[:, np.newaxis] * K * W[np.newaxis, :]
            else:
                K = W[:, np.newaxis] * K
                
        return K