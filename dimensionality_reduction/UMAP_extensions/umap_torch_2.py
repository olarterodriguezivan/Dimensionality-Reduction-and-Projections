import torch
from torchdr import UMAP


import torch
from torchdr import UMAP


class SupervisedUMAP(UMAP):
    """
    TorchDR UMAP with regression supervision by reweighting edge sampling frequency.

    Key idea:
      TorchDR's UMAP uses a per-edge schedule (epochs_per_sample / epoch_of_next_sample)
      to decide how often each neighbor edge is updated (see UMAP.on_affinity_computation_end()).
      We modify that schedule based on y-similarity so that pairs with similar targets
      get updated more often (stronger attraction), and dissimilar targets get updated less.

    Notes:
      - This does NOT add an out-of-sample .transform() (TorchDR UMAP doesn't provide one).
      - y can be (n,) or (n, d_y). Works for regression.
    """

    def __init__(
        self,
        *args,
        target_weight: float = 0.5,         # alpha in [0,1]
        target_metric: str = "l2",          # "l2" or "l1"
        target_sigma: str | float = "auto", # "auto" or positive float
        target_scale: str = "standardize",  # "standardize" or "none"
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 <= target_weight <= 1.0):
            raise ValueError("target_weight must be in [0, 1].")
        if target_metric not in ("l2", "l1"):
            raise ValueError("target_metric must be 'l2' or 'l1'.")
        if target_scale not in ("standardize", "none"):
            raise ValueError("target_scale must be 'standardize' or 'none'.")
        if isinstance(target_sigma, (int, float)) and target_sigma <= 0:
            raise ValueError("target_sigma must be 'auto' or a positive float.")

        self.target_weight = float(target_weight)
        self.target_metric = target_metric
        self.target_sigma = target_sigma
        self.target_scale = target_scale
        self._sup_eps = float(eps)

        self._y_sup_ = None  # set during fit

    def _infer_device(self, X) -> torch.device:
        # TorchDR can move data internally, but for y-similarity computation we need a device.
        if isinstance(self.device, torch.device):
            return self.device
        if isinstance(self.device, str) and self.device not in (None, "auto"):
            return torch.device(self.device)
        if torch.is_tensor(X):
            return X.device
        return torch.device("cpu")

    def _prepare_y(self, y, device: torch.device) -> torch.Tensor:
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y[:, None]
        y = y.to(device=device, dtype=torch.float32)

        if self.target_scale == "standardize":
            mu = y.mean(dim=0, keepdim=True)
            std = y.std(dim=0, keepdim=True).clamp_min(self._sup_eps)
            y = (y - mu) / std

        return y

    def _fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        if y is None:
            raise ValueError(
                "SupervisedUMAPRegressor requires y (regression targets). "
                "If you want unsupervised, use torchdr.UMAP directly."
            )

        device = self._infer_device(X)
        self._y_sup_ = self._prepare_y(y, device=device)

        # Important: we don't rely on TorchDR's 'y' plumbing; we only need it for our override.
        return super()._fit_transform(X, y=None)

    @torch.no_grad()
    def on_affinity_computation_end(self):
        # Run TorchDR UMAP's original hook first: builds epochs_per_sample and epoch_of_next_sample.
        super().on_affinity_computation_end()

        if self._y_sup_ is None or self.target_weight == 0.0:
            return

        # These are created by the base implementation and used by UMAP's training step.
        # chunk_indices_: (chunk,)
        # NN_indices_:    (chunk, k)
        if not hasattr(self, "chunk_indices_") or not hasattr(self, "NN_indices_"):
            return
        if not hasattr(self, "epochs_per_sample") or not hasattr(self, "epoch_of_next_sample"):
            return

        y = self._y_sup_
        yi = y[self.chunk_indices_]              # (chunk, d_y)
        yj = y[self.NN_indices_]                 # (chunk, k, d_y)
        diff = yi.unsqueeze(1) - yj

        if self.target_metric == "l2":
            dist = torch.sqrt((diff * diff).sum(dim=-1) + self._sup_eps)  # (chunk, k)
        else:
            dist = diff.abs().sum(dim=-1)                                  # (chunk, k)

        if self.target_sigma == "auto":
            dpos = dist[dist > 0]
            sigma = torch.median(dpos) if dpos.numel() > 0 else dist.mean()
            sigma = sigma.clamp_min(self._sup_eps)
        else:
            sigma = torch.tensor(float(self.target_sigma), device=dist.device)

        # Similarity in [0,1], high when targets are close
        sim_y = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))

        # Convert TorchDR's schedule back into a proxy "edge strength":
        # smaller epochs_per_sample => updated more often => stronger.
        w_x = 1.0 / (self.epochs_per_sample + self._sup_eps)

        # Normalize both to comparable scale
        w_x = w_x / (w_x.max() + self._sup_eps)
        sim_y = sim_y / (sim_y.max() + self._sup_eps)

        alpha = self.target_weight
        w_eff = (1.0 - alpha) * w_x + alpha * sim_y

        # Turn back into a schedule; match mean so global training pace stays similar
        epochs_new = 1.0 / (w_eff + self._sup_eps)
        epochs_new = epochs_new * (self.epochs_per_sample.mean() / (epochs_new.mean() + self._sup_eps))

        self.epochs_per_sample.copy_(epochs_new)
        self.epoch_of_next_sample.copy_(epochs_new)
