import torch
from torch import Tensor
from torch.optim import Adam, SGD
from typing import Optional, List, Tuple
from torch.functional import block_diag


class WeightedPCA:
    def __init__(self, n_components: int, lr: float = 0.01, n_iter: int = 1000, use_sgd: bool = False):
        self.n_components = n_components
        self.lr = lr
        self.n_iter = n_iter
        self.use_sgd = use_sgd
        self.components_: Optional[Tensor] = None
        self.mean_: Optional[Tensor] = None

    def fit(self, X: Tensor, weights: Optional[Tensor] = None) -> 'WeightedPCA':
        n_samples, n_features = X.shape

        if weights is None:
            weights = torch.ones(n_samples, device=X.device)

        weights = weights / weights.sum()

        self.mean_ = (weights.unsqueeze(1) * X).sum(dim=0)
        X_centered = X - self.mean_

        components = torch.randn(n_features, self.n_components, device=X.device, requires_grad=True)

        optimizer_class = SGD if self.use_sgd else Adam
        optimizer = optimizer_class([components], lr=self.lr)

        for _ in range(self.n_iter):
            optimizer.zero_grad()
            X_proj = X_centered @ components
            X_reconstructed = X_proj @ components.T
            loss = ((weights.unsqueeze(1) * (X_centered - X_reconstructed) ** 2).sum())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                components[:] = torch.linalg.qr(components).Q[:, :self.n_components]

        self.components_ = components.detach()
        return self

    def transform(self, X: Tensor) -> Tensor:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        self.fit(X, weights)
        return self.transform(X)