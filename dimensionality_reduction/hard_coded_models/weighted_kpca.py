import torch
from torch import Tensor
from typing import Optional
import gpytorch
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class WeightedKPCA(BaseEstimator, TransformerMixin):
    """
    Weighted Kernel Principal Component Analysis (Weighted KPCA) using GPyTorch and PyTorch
    
    Parameters:
    -----------
        n_components: Number of principal components to retain
        kernel: Kernel type ('rbf', 'linear', 'polynomial', 'cosine')
        lengthscale: Lengthscale parameter for RBF and Matern kernels
        device: Device to run computations on ('cpu' or 'cuda')
        **kwargs: Additional kernel parameters (e.g., degree for polynomial kernel)
    """
    
    def __init__(self, 
                 n_components:int=2, 
                 kernel:str='cosine', 
                 lengthscale:float=1.0,
                 device='cpu',
                 **kwargs):
        
        degree = kwargs.get('degree', 3)
        
        self.n_components = n_components
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.degree = degree
        self.weights = None
        self.device = device
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        if value not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        
        # Ensure that CUDA is available if selected
        if value == 'cuda' and not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = value

    @property 
    def lengthscale(self)-> float:
        return self._lengthscale
    
    @lengthscale.setter
    def lengthscale(self, value: float):
        if value <= 0:
            raise ValueError("Lengthscale must be positive.")
        self._lengthscale = value
    
    @property
    def kernel(self) -> str:
        return self._kernel
    
    @kernel.setter
    def kernel(self, value: str):
        valid_kernels = ['rbf', 'linear', 'polynomial', 'cosine']
        if value not in valid_kernels:
            raise ValueError(f"Kernel must be one of {valid_kernels}")
        self._kernel = value
    
    @property
    def degree(self) -> int:
        return self._degree
    
    @degree.setter
    def degree(self, value: int):
        if value <= 0:
            raise ValueError("Degree must be a positive integer.")
        self._degree = value
    
    def kernel_object(self):
        """Return the GPyTorch kernel object based on the kernel type"""
        return self._get_kernel()
    
    def _get_kernel(self)->gpytorch.kernels.Kernel:
        """Create GPyTorch kernel based on kernel type"""
        if self.kernel == 'rbf':
            return gpytorch.kernels.RBFKernel()
        elif self.kernel == 'linear':
            return gpytorch.kernels.LinearKernel()
        elif self.kernel == 'polynomial':
            return gpytorch.kernels.PolynomialKernel(power=self.degree)
        elif self.kernel == 'cosine':
            return gpytorch.kernels.CosineKernel()
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def fit(self, 
            X: Tensor, 
            weights: Optional[Tensor] = None):
        """
        Fit the model with X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        weights : array-like, shape (n_samples,), optional
            Sample weights
        """
        # Convert to torch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            X = torch.tensor(X).float()

        # Get number of samples and features
        _, n_features = X.shape

        # Handle weights
        if weights is None:
            weights = torch.ones(n_features, device=self.device)
        else:
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Store weights
        self.weights_ = weights
        
        # Normalize weights
        self.weights_ = self.weights_ / self.weights_.sum()

        
        X = X.to(self.device)
        n_samples = X.shape[0]
        
        
        # Store training data
        self.X_fit_ = X
        
        # Create kernel and compute kernel matrix
        kernel:gpytorch.kernels.Kernel = self._get_kernel()
        
        if self.kernel in ['rbf']:
            kernel.lengthscale = self.lengthscale
        
        with torch.no_grad():
            K = kernel(X, X)
            
            # Apply weights to kernel#
            # nel matrix
            W = torch.diag(torch.sqrt(self.weights_))
            K_weighted = W @ K @ W
            
            # Center the weighted kernel matrix
            ones = torch.ones((n_samples, n_samples), device=self.device) / n_samples
            K_centered = K_weighted - ones @ K_weighted - K_weighted @ ones + ones @ K_weighted @ ones
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(K_centered)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Keep only positive eigenvalues and corresponding eigenvectors
            positive_idx = eigenvals > 1e-12
            self.eigenvalues_ = eigenvals[positive_idx]
            self.eigenvectors_ = eigenvecs[:, positive_idx]
            
            # Normalize eigenvectors
            for i in range(self.eigenvectors_.shape[1]):
                self.eigenvectors_[:, i] = (self.eigenvectors_[:, i] / 
                                          torch.sqrt(self.eigenvalues_[i]))
            
            # Keep only the requested number of components
            self.n_components_ = min(self.n_components, len(self.eigenvalues_))
            self.eigenvalues_ = self.eigenvalues_[:self.n_components_]
            self.eigenvectors_ = self.eigenvectors_[:, :self.n_components_]
            
            # Store kernel for transform
            self.kernel_ = kernel
        
        return self
    
    def transform(self, X):
        """
        Transform X to the reduced dimensional space
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        # Convert to torch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            X = torch.tensor(X).float()
        
        X = X.to(self.device)
        
        with torch.no_grad():
            # Compute kernel matrix between X and training data
            K = self.kernel_(X, self.X_fit_).evaluate()
            
            # Apply weights
            W = torch.diag(torch.sqrt(self.weights_))
            K_weighted = K @ W
            
            # Center the kernel matrix
            n_train = self.X_fit_.shape[0]
            ones_train = torch.ones(n_train, device=self.device) / n_train
            K_train = self.kernel_(self.X_fit_, self.X_fit_).evaluate()
            K_train_weighted = torch.diag(torch.sqrt(self.weights_)) @ K_train @ torch.diag(torch.sqrt(self.weights_))
            K_train_centered_sum = torch.sum(self.weights_.unsqueeze(0) * K_train_weighted, dim=1)
            
            K_centered = (K_weighted - 
                         torch.outer(torch.ones(X.shape[0], device=self.device), K_train_centered_sum) -
                         K @ torch.outer(ones_train, ones_train) @ W +
                         torch.outer(torch.ones(X.shape[0], device=self.device), ones_train) * torch.sum(K_train_centered_sum))
            
            # Project onto principal components
            result = K_centered @ self.eigenvectors_
        
        # Convert back to numpy if needed for sklearn compatibility
        return result.cpu().numpy()
    
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and transform X
        
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
        return self.fit(X, y).transform(X)