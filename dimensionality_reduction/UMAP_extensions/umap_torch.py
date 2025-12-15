import torch
import numpy as np
from typing import Optional, Union, Tuple
import torch.nn as nn
from torchdr import UMAP as TorchUMAP
from joblib import load, dump


class UMAPTorch(TorchUMAP):
    """PyTorch-based UMAP implementation for dimensionality reduction."""
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        lr: float = 1.0,
        optimizer: Union[str, torch.optim.Optimizer] = 'SGD',
        optimizer_kwargs: dict = {},
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_kwargs: dict = {},
        init: str = 'pca',
        init_scaling: float = 1.0,
        min_grad_norm: float = 1e-7,
        max_iter: int = 200,
        device: Optional[Union[str, torch.device]] = None,
        backend: str = 'faiss',
        verbose: bool = False,
        random_state: Optional[int] = None,
        max_iter_affinity: int = 100,
        metric_in='sqeuclidean',
        negative_sample_rate: int = 5,
        check_interval: int = 50,
        discard_NNs: bool = False,
        compile:bool = False,
        **kwargs
    ):
        r"""
        TorchUMAP constructor.

        Parameters
        ----------
        n_components : int, optional (default=2)
            The dimension of the space to embed into.
        n_neighbors : int, optional (default=15)
            The size of local neighborhood (in terms of number of neighboring sample points) used for manifold
            approximation. Larger values result in more global views of the manifold, while smaller values
            result in more local data being preserved. In general values should be in the range 2 to 200.
        min_dist : float, optional (default=0.1)
            The effective minimum distance between embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the manifold are drawn closer together, while
            larger values will result in a more even dispersal of points. The value should be set
            between 0.0 and 0.99.
        spread : float, optional (default=1.0)
            The effective scale of embedded points. In combination with `min_dist` this determines how clustered
            the embedded points are.
        a : float, optional (default=None)
            More specific parameters controlling the embedding. If None, reasonable values will be
            determined automatically.
        b : float, optional (default=None)
            More specific parameters controlling the embedding. If None, reasonable values will be
            determined automatically.
        lr : float, optional (default=1.0)
            The initial learning rate for the embedding optimization.
        optimizer : str or torch.optim.Optimizer, optional (default='SGD')
            The optimizer to use for the embedding optimization. Can be either a string (one of '
            SGD', 'Adam', 'RMSprop') or a PyTorch optimizer instance.
        optimizer_kwargs : dict, optional (default={})
            Additional keyword arguments to pass to the optimizer constructor if a string is provided
            for the `optimizer` parameter.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional (default=None)
            A learning rate scheduler to use during optimization.
        scheduler_kwargs : dict, optional (default={})
            Additional keyword arguments to pass to the scheduler constructor if a scheduler is provided.
        init : str, optional (default='spectral')
            How to initialize the low dimensional embedding. Options are 'spectral', 'random', or
            a numpy array of shape (n_samples, n_components) giving the initial embedding.
        init_scaling : float, optional (default=1.0)
            Scaling factor applied to the initial embedding.
        min_grad_norm : float, optional (default=1e-7)
            If the gradient norm during optimization falls below this value, optimization will stop.
        max_iter : int, optional (default=200)
            The number of training epochs to be used in optimizing the low dimensional embedding.
        device : str or torch.device, optional (default=None)
            The device to use for training. If None, will use 'cuda' if available else
            'cpu'.
        backend : str, optional (default='faiss')
            The nearest neighbor backend to use. One of 'faiss', 'pynndesc', or 'annoy'.
        verbose : bool, optional (default=False)
            Whether to print progress messages during optimization.
        random_state : int, optional (default=None)
            Random seed used for reproducibility.
        max_iter_affinity : int, optional (default=100)     
            The number of epochs to use when optimizing the fuzzy simplicial set
            representation of the data.
        metric_in : str, optional (default='sqeuclidean')
            The metric to use for the input data.
        negative_sample_rate : int, optional (default=5)
            The number of negative edge/1-simplex samples to use per positive edge/1-simple
            during optimization.
        check_interval : int, optional (default=50)
            How often to check for convergence during optimization.
        discard_NNs : bool, optional (default=False)
            Whether to discard the nearest neighbors after constructing the fuzzy simplicial set.
        compile : bool, optional (default=False)
            Whether to compile the model using torch.compile for potentially improved performance.
        **kwargs : additional keyword arguments
            Additional keyword arguments passed to the parent class.
        
        """    
    
        
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            max_iter_affinity=max_iter_affinity,
            metric_in=metric_in,
            negative_sample_rate=negative_sample_rate,
            check_interval=check_interval,
            discard_NNs=discard_NNs,
            compile=compile,
            **kwargs
        )
    
    
    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sample_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Fit the model to the data and transform it in one step.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit and transform.
        y : array-like of shape (n_samples,) or None
            The target labels for supervised UMAP. If None, unsupervised UMAP is performed.
        
        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data in the embedding space.
        """

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

        X_weighted = X_centered * self._sample_weights[:, None]

        return super().fit_transform(X_weighted, y)
    

    def save_model(self, filepath: str) -> None:
        """
        Save the UMAP model to a file.
        
        Parameters:
        -----------
        filepath : str
            The path to the file where the model will be saved.
        """
        dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> "UMAPTorch":
        """
        Load a UMAP model from a file.
        
        Parameters:
        -----------
        filepath : str
            The path to the file from which the model will be loaded.
        
        Returns:
        --------
        model : UMAPTorch
            The loaded UMAPTorch instance.
        """
        return load(filepath)
        