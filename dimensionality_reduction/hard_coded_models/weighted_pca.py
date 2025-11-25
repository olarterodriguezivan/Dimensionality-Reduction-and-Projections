import torch
from torch import Tensor
from torch.optim import Adam, SGD
from typing import Optional, List, Tuple
from torch.functional import block_diag

from sklearn.base import BaseEstimator, TransformerMixin


class WeightedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_components: int):
        r"""
        Weighted Principal Component Analysis (PCA) base

        Args:
            n_components: Number of principal components to retain
        """
        
        self.n_components = n_components

        self.components_: Optional[Tensor] = None
        self.unbiased_mean_: Optional[Tensor] = None
        self.biased_mean_: Optional[Tensor] = None
        self.weights_: Optional[Tensor] = None

    def fit(self, X: Tensor, weights: Optional[Tensor] = None) -> 'WeightedPCA':
        
        # Get number of samples and features
        _, n_features = X.shape

        # Handle weights
        if weights is None:
            weights = torch.ones(n_features, device=X.device)
        
        # Store weights
        self.weights_ = weights
        
        # Normalize weights
        self.weights_ = self.weights_ / self.weights_.sum()

        # Compute unbiased mean and center data
        self.unbiased_mean_ = X.mean(dim=0)

        X_centered = X - self.unbiased_mean_

        # Scale data by weights
        X_scaled = torch.diag(self.weights_) @ X_centered 

        # Compute biased mean
        self.biased_mean_ = X_scaled.mean(dim=0)

        # Get the svd
        _, _, Vt = torch.pca_lowrank(X_scaled, q=self.n_components,
                                     center=True,
                                     niter=10)
        
        # store components
        self.components_ = Vt[:, :self.n_components]
        
        return self

    def transform(self, X: Tensor) -> Tensor:
        if self.components_ is None or self.unbiased_mean_ is None or self.biased_mean_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        X_centered = X - self.unbiased_mean_- self.biased_mean_

        return X_centered @ self.components_
    
    def inverse_transform(self, X_reduced: Tensor) -> Tensor:
        if self.components_ is None or self.unbiased_mean_ is None or self.biased_mean_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        # Step 1: undo projection
        X_unbiased = X_reduced @ self.components_.T

        # Step 2: add back biased mean
        X_centered = X_unbiased + self.biased_mean_

        # Step 4: add back the unbiased mean
        return X_centered + self.unbiased_mean_

    def fit_transform(self, X: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        self.fit(X, weights)
        return self.transform(X)