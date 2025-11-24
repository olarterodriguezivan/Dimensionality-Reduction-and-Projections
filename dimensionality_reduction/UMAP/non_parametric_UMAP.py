from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import umap

class NonParametricUMAP(umap.UMAP):
    """
    Non-parametric UMAP implementation extending scikit-learn's transformer interface.
    
    This implementation uses the standard UMAP algorithm without neural network
    parameterization, making it non-parametric.
    """
    
    def __init__(self, 
                    n_neighbors=15,
                    n_components=2,
                    metric="euclidean",
                    metric_kwds=None,
                    output_metric="euclidean",
                    output_metric_kwds=None,
                    n_epochs=None,
                    learning_rate=1.0,
                    init="spectral",
                    min_dist=0.1,
                    spread=1.0,
                    low_memory=True,
                    n_jobs=-1,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1.0,
                    repulsion_strength=1.0,
                    negative_sample_rate=5,
                    transform_queue_size=4.0,
                    a=None,
                    b=None,
                    random_state=None,
                    angular_rp_forest=False,
                    target_n_neighbors=-1,
                    target_metric="categorical",
                    target_metric_kwds=None,
                    target_weight=0.5,
                    transform_seed=42,
                    transform_mode="embedding",
                    force_approximation_algorithm=False,
                    verbose=False,
                    tqdm_kwds=None,
                    unique=False,
                    densmap=False,
                    dens_lambda=2.0,
                    dens_frac=0.3,
                    dens_var_shift=0.1,
                    output_dens=False,
                    disconnection_distance=None,
                    precomputed_knn=(None, None, None),
    ):
        """
        Initialize Non-parametric UMAP.
        
        Parameters:
        -----------
        n_neighbors : int
            The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        n_components : int
            The dimension of the space to embed into.
        metric : str or callable
            The metric to use to compute distances in high dimensional space.
        min_dist : float
            The effective minimum distance between embedded points.
        spread : float
            The effective scale of embedded points.
        random_state : int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose : bool
            Whether to print progress messages during optimization.
        tqdm_kwds : dict or None
            Keyword arguments to pass to the tqdm progress bar.
        densmap : bool
            Whether to use DensMAP extension.
        dens_lambda : float
            DensMAP regularization strength.
        dens_frac : float
            Fraction of points used to estimate local density.
        dens_var_shift : float
            Variance shift for density estimation.
        output_dens : bool
            Whether to output density estimates alongside embeddings.
        disconnection_distance : float or None
            Distance threshold for disconnecting points in the embedding.
        precomputed_knn : tuple or None
            Precomputed KNN graph (distances, indices, n_neighbors) or None.
        
        """
        
        # Call the parent constructor
        super().__init__(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            spread=spread,
            random_state=random_state,
            metric_kwds=metric_kwds,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            low_memory=low_memory,
            n_jobs=n_jobs,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            transform_queue_size=transform_queue_size,
            a=a,
            b=b,
            angular_rp_forest=angular_rp_forest,
            target_n_neighbors=target_n_neighbors,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            target_weight=target_weight,
            transform_seed=transform_seed,
            transform_mode=transform_mode,
            force_approximation_algorithm=force_approximation_algorithm,
            verbose=verbose,
            tqdm_kwds=tqdm_kwds,
            unique=unique,
            densmap=densmap,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
            output_dens=output_dens,
            disconnection_distance=disconnection_distance,
            precomputed_knn=precomputed_knn,
        )
        
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
        
    
    def transform(self, 
                  X:np.ndarray):
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