from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
from typing import Union, Optional
from pathlib import Path
import numpy as np

class WeightedPCA(SklearnPCA):
    """
    PCA with support for sample weights.
    
    Inherits from sklearn.decomposition.PCA and adds weighted fitting capability.
    
    Parameters
    ----------
    n_components : int, float or None, default=None
        Number of components to keep.
    **kwargs : dict
        Additional arguments passed to sklearn.decomposition.PCA
    """
    
    def __init__(self, n_components=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

        # Initialize sample weights
        self._sample_weights: np.ndarray = None
        self._mean: np.ndarray = None

        # Initialize a scaler for later use
        self._scaler = MinMaxScaler()
        
    def fit(self, X, y=None, sample_weights=None):
        """
        Fit the PCA model with optional sample weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        
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

        return super().fit(X_weighted, y)
    
    def transform(self, X:np.ndarray) -> np.ndarray:

        # Transform data to the principal component space.
        X = np.asarray(X)

        # Uncenter the data
        X_centered = X - self._mean

        # Scale the data with sample weights
        X_weighted = X_centered * self._sample_weights[:, None]

        return super().transform(X_weighted)
    
    def fit_transform(self, X, y=None, sample_weights=None):
        """
        Fit the model and apply dimensionality reduction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y=y, sample_weights=sample_weights)
        return self.transform(X)


    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in the reduced space.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """

        # Perform the inverse transformation
        X_reconstructed = super().inverse_transform(X)

        # Unscale the data with sample weights
        X_unscaled = X_reconstructed * (self._sample_weights ** -1)[:, None]

        # Re-add the mean
        X_original = X_unscaled + self._mean
        return X_original
    

    # ---------------------------------------
    # Saving and loading methods
    # ---------------------------------------

    def save_model(self, 
                   path: Union[str, Path], 
                   overwrite: Optional[bool] = False) -> None:
        """
        Save model parameters and learned state explicitly.

        Parameters
        ----------
        path : str or Path
            File path where the model will be saved.
        overwrite : bool, default=False
            Whether to overwrite an existing file.
        """
        path = Path(path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. "
                "Use overwrite=True to overwrite."
            )

        state = {
            # constructor parameters
            "init_params": {
                "n_components": self.n_components,
                **self.get_params(deep=False),
            },

            # learned sklearn state
            "components_": self.components_.tolist(),
            "explained_variance_": self.explained_variance_.tolist(),
            "explained_variance_ratio_": self.explained_variance_ratio_.tolist(),
            "singular_values_": self.singular_values_.tolist(),
            "n_features_in_": self.n_features_in_,

            # custom state
            "_mean": self._mean.tolist(),
            "_sample_weights": self._sample_weights.tolist(),
        }

        dump(state, path)


    @staticmethod
    def load_model(path: Union[str, Path]) -> "WeightedPCA":
        """
        Load model parameters and learned state explicitly.

        Parameters
        ----------
        path : str or Path
            File path where the model is stored.

        Returns
        -------
        WeightedPCA
            Reconstructed PCA model.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file '{path}' does not exist.")

        state = load(path)

        # Reconstruct model
        init_params = state["init_params"]
        model = WeightedPCA(**init_params)

        # Restore learned sklearn attributes
        model.components_ = np.array(state["components_"])
        model.explained_variance_ = np.array(state["explained_variance_"])
        model.explained_variance_ratio_ = np.array(state["explained_variance_ratio_"])
        model.singular_values_ = np.array(state["singular_values_"])
        model.n_features_in_ = state["n_features_in_"]

        # Restore custom state
        model._mean = np.array(state["_mean"])
        model._sample_weights = np.array(state["_sample_weights"])

        return model
        

    
