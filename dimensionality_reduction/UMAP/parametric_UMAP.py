import numpy as np
from umap import ParametricUMAP
from sklearn.base import BaseEstimator, TransformerMixin

class ParametricUMAPTransformer(BaseEstimator, TransformerMixin):
    """
    Parametric UMAP transformer with scikit-learn compatible interface.
    """
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 metric='euclidean', random_state=None, **kwargs):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs
        self.umap_model = None
        
    def fit(self, X, y=None):
        """
        Fit the Parametric UMAP model.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (ignored)
            
        Returns:
        self : object
        """
        self.umap_model = ParametricUMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            **self.kwargs
        )
        self.umap_model.fit(X)
        return self
    
    def transform(self, X):
        """
        Transform X using the fitted Parametric UMAP model.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if self.umap_model is None:
            raise ValueError("Model must be fitted before transform")
        return self.umap_model.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform X.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (ignored)
            
        Returns:
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)