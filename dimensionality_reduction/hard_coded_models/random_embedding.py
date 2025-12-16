import warnings

from pathlib import Path
import numpy as np

from numpy.random import default_rng, Generator, BitGenerator

from sklearn.base import BaseEstimator, TransformerMixin,  ClassNamePrefixFeaturesOutMixin

from sklearn.utils.validation import check_is_fitted

from typing import Optional, Union


class RandomEmbedding(BaseEstimator, 
                      ClassNamePrefixFeaturesOutMixin, 
                      TransformerMixin):
    r"""
    A scikit-learn compatible transformer that applies random embedding to reduce
    the dimensionality of input data."""

    def __init__(self, 
                 n_components: int = 2, 
                 random_state: Optional[Union[int, BitGenerator]] = None,
                 initialization_method:str = 'PCA'):
        """
        Initialize the RandomEmbedding transformer.

        Args:
            n_components (int): The target number of dimensions.
            random_state (Optional[Union[int, BitGenerator]]): Seed or random number generator for reproducibility.
            initialization_method (str): Method for initializing the embedding ('PCA' or 'QR').
        """
        self.n_components = n_components
        self.initialization_method = initialization_method
        
        self._embedding_matrix: Optional[np.ndarray] = None

        if isinstance(random_state, BitGenerator):
            self.random_state = Generator(random_state)
        else:
            # Assure the random `seed` is a large non-negative integer
            assert (random_state is None or isinstance(random_state, int) and random_state >= 0), \
                "`random_state` must be None or a non-negative integer."
            self.random_state = default_rng(seed=random_state)
        
        # Initialize the ambient dimension to None
        self._ambient_dimension: int = None
    

    def fit(self, 
            X: np.ndarray, 
            y: Optional[np.ndarray] = None) -> 'RandomEmbedding':
        """
        Fit the RandomEmbedding transformer to the data.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
            y (Optional[np.ndarray]): Ignored. This parameter exists for compatibility.

        Returns:
            RandomEmbedding: The fitted transformer.
        """
        self._ambient_dimension = X.shape[1]

        # Generate a random ambient_dimension x ambient_dimension matrix
        self._fit_embedding_matrix()


        
        return self
    
    def _fit_embedding_matrix(self) -> None:
        """
        Fit the embedding matrix based on the initialization method.
        """

        random_matrix = self.random_state.standard_normal(
            size=(self._ambient_dimension, self._ambient_dimension)
        )
        if self.initialization_method == 'PCA':
            # Use PCA-based initialization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            pca.fit(self.random_state.standard_normal(
                size=(100, self._ambient_dimension)
            ))
            self._embedding_matrix = pca.components_.T
        elif self.initialization_method == 'QR':
            # Use QR decomposition for orthogonal initialization
            q, _ = np.linalg.qr(self.random_state.standard_normal(
                size=(self._ambient_dimension, self.n_components)
            ))
            self._embedding_matrix = q[:, :self.n_components]
        else:
            raise ValueError("Invalid initialization method.")

    @property
    def n_components(self) -> int:
        """
        Get the number of components after fitting.

        Returns:
            int: The number of components.
        """
        
        return self._n_components
    
    @n_components.setter
    def n_components(self, value: int) -> None:
        """
        Set the number of components.

        Args:
            value (int): The target number of dimensions.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("n_components must be a positive integer.")
        self._n_components = value

    @property
    def initialization_method(self) -> str:
        """
        Get the initialization method.

        Returns:
            str: The initialization method.
        """
        return self._initialization_method
    
    @initialization_method.setter
    def initialization_method(self, value: str) -> None:
        """
        Set the initialization method.

        Args:
            value (str): The initialization method.
        """
        if value not in ['PCA', 'QR']:
            raise ValueError("initialization_method must be either 'PCA' or 'QR'.")
        
        self._initialization_method = value
