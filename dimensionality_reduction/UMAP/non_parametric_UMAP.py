from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import umap

class NonParametricUMAP(BaseEstimator, TransformerMixin):
    """
    Non-parametric UMAP implementation extending scikit-learn's transformer interface.
    
    This implementation uses the standard UMAP algorithm without neural network
    parameterization, making it non-parametric.
    """
    
    def __init__(self, n_neighbors=15, n_components=2, metric='euclidean', 
                 min_dist=0.1, spread=1.0, random_state=None, **kwargs):
        """
        Initialize Non-parametric UMAP.
        
        Parameters:
        -----------
        n_neighbors : int, default=15
            Number of nearest neighbors to use
        n_components : int, default=2
            Number of dimensions for embedding
        metric : str, default='euclidean'
            Distance metric to use
        min_dist : float, default=0.1
            Minimum distance between embedded points
        spread : float, default=1.0
            Scale of embedded points
        random_state : int or None, default=None
            Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.random_state = random_state
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        """
        Fit the UMAP model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = check_array(X, accept_sparse=False)
        
        # Initialize UMAP with non-parametric settings
        self.umap_model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric=self.metric,
            min_dist=self.min_dist,
            spread=self.spread,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit the model
        self.embedding_ = self.umap_model_.fit_transform(X)
        
        return self
    
    def transform(self, X):
        """
        Transform new data using the fitted UMAP model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if not hasattr(self, 'umap_model_'):
            raise ValueError("Model must be fitted before transform")
            
        X = check_array(X, accept_sparse=False)
        return self.umap_model_.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (ignored)
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).embedding_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'n_neighbors': self.n_neighbors,
            'n_components': self.n_components,
            'metric': self.metric,
            'min_dist': self.min_dist,
            'spread': self.spread,
            'random_state': self.random_state
        }
        if deep:
            params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in ['n_neighbors', 'n_components', 'metric', 'min_dist', 
                      'spread', 'random_state']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self