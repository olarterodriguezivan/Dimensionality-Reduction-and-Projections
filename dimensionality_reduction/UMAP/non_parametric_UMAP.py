from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import umap
from typing import Optional
from joblib import load, dump

# CONSTANTS
DEFAULT_TARGET_METRICS = {"l1", "l2"}
class NonParametricUMAP(umap.UMAP):
    r"""
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
                    target_metric="l1",
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
        target_metric : str
            The metric to use for target space when performing supervised UMAP.
        target_n_neighbors : int
            The number of neighbors to consider in the target space.
        target_weight : float
            The weighting factor between the data space and target space.
        transform_mode : {'embedding', 'graph', 'spectral'}
            The mode to use for transforming new data.
        force_approximation_algorithm : bool
            Whether to force the use of the approximation algorithm for large datasets.
        tqdm_kwds : dict or None
            Keyword arguments to pass to the tqdm progress bar.
        unique : bool
            Whether to ensure unique embeddings for identical points.
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
        
        See Also:
        --------
        umap.UMAP : Original UMAP implementation.

        Raises:
        ------
        ValueError
            If target_metric is not in DEFAULT_TARGET_METRICS.
        
        Returns:
        -------
        self : object
            Unfitted NonParametricUMAP instance.S
        
        """

        if target_metric not in DEFAULT_TARGET_METRICS:
            raise ValueError(f"target_metric must be one of {DEFAULT_TARGET_METRICS}, got {target_metric}")
        
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
        
    def fit(self, X, y, **kwargs):
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

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
        super().fit(X,y,ensure_all_finite=True, **kwargs)
        return self
    
    def transform(self, X):
        """
        Transform new data into the UMAP embedding space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to transform.
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return super().transform(X)
    
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the UMAP model to the data and transform it in one step.
        
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

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
        return super().fit_transform(X,y= y, ensure_all_finite=True, **kwargs)
    
    def inverse_transform(self, X):
        """
        Inverse transform data from the UMAP embedding space back to the original space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_components)
            Data in the UMAP embedding space.
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Data transformed back to the original space.
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return super().inverse_transform(X)
    
    # Saving and loading methods
    # ---------------------------------------
    def save_model(self, path: str) -> None:
        """
        Save the entire NonParametricUMAP model to disk.
        
        Parameters:
        -----------
        path : str
            Path to the file where the model will be saved.
        """
        dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'NonParametricUMAP':
        """
        Load a NonParametricUMAP model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the file where the model is saved.
            
        Returns:
        --------
        model : NonParametricUMAP
            Loaded NonParametricUMAP model.
        """
        return load(path)