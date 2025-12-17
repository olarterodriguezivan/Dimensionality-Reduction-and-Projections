import warnings

import joblib

from pathlib import Path
import numpy as np

from scipy.sparse.linalg import svds

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
                 initialization_method:str = 'SVD'):
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
    

    def _fit(self, 
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
        return self._fit(X, y)
    
    def _transform(self, 
                  X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted random embedding.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data of shape (n_samples, n_components).
        """
        check_is_fitted(self, attributes=['_embedding_matrix', '_ambient_dimension'])
        
        if X.shape[1] != self._ambient_dimension:
            raise ValueError(f"Input data must have {self._ambient_dimension} features, "
                             f"but got {X.shape[1]} features.")
        
        # Project the data onto the embedding
        return np.matmul(X, self._inv_embedding)
    
    def transform(self, 
                  X: np.ndarray) -> np.ndarray:
        return self._transform(X)
    
    def fit_transform(self, X, y = None)->np.ndarray:
        """
        Fit the RandomEmbedding transformer and transform the data in one step.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
            y (Optional[np.ndarray]): Ignored. This parameter exists for compatibility.

        Returns:
            np.ndarray: The transformed data of shape (n_samples, n_components).
        """
        self._fit(X, y)
        return self._transform(X)
    
    def inverse_transform(self,
                          X_reduced: np.ndarray) -> np.ndarray:
        """
        Inverse transform the reduced data back to the original space.
        Args:
            X_reduced (np.ndarray): The reduced data of shape (n_samples, n_components).
        Returns:
            np.ndarray: The data in the original space of shape (n_samples, n_features).
        """

        check_is_fitted(self, attributes=['_embedding_matrix', '_ambient_dimension'])

        if X_reduced.shape[1] != self.n_components:
            raise ValueError(f"Input reduced data must have {self.n_components} features, "
                             f"but got {X_reduced.shape[1]} features.")
        
        return np.matmul(X_reduced, self._embedding_matrix)
    
    def _fit_embedding_matrix(self) -> None:
        """
        Fit the embedding matrix based on the initialization method.
        """

        from scipy.stats import ortho_group

        random_matrix = ortho_group.rvs(dim=self._ambient_dimension, random_state=self.random_state)
        random_matrix = np.dot(random_matrix,random_matrix.T)
        if self.initialization_method == 'SVD':
            # Use SVD to obtain an orthogonal matrix
            U, _, _ = svds(random_matrix, k=self.n_components)
            self._embedding_matrix = U.transpose()
        elif self.initialization_method == 'QR':
            # Use QR decomposition to obtain an orthogonal matrix
            Q, _ = np.linalg.qr(random_matrix)
            self._embedding_matrix = Q[:, :self.n_components].transpose()

        # Store a back-projection matrix if needed in future
        else:
            raise ValueError("initialization_method must be either 'SVD' or 'QR'.")
        
        self._inv_embedding = np.matmul(np.transpose(self._embedding_matrix), np.linalg.inv(np.matmul(self._embedding_matrix, np.transpose(self._embedding_matrix))))



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
        if value not in ['SVD', 'QR']:
            raise ValueError("initialization_method must be either 'SVD' or 'QR'.")
        
        self._initialization_method = value

    
    #---------------------------------------------------------
    # Scikit-learn compatibility
    def get_feature_names_out(self, input_features=None):
        return np.array([f"RandomEmbedding_{i}" for i in range(self.n_components)])
    
    #--------------------------------------------------------- 
    # Save and Load
    # ---------------------------------------------------------

    def save_model(self, 
                   file_path: Union[Path, str],
                   overwrite: bool = False) -> None:
        """
        Save the model to a file.

        Args:
            file_path (Path): The path to the file where the model will be saved.
        """

        # Ensure the file path is a Path object
        file_path = Path(file_path).absolute()

        if file_path.exists() and not overwrite:
            raise FileExistsError(f"The file {file_path} already exists. "
                                  "Use overwrite=True to overwrite it.")
        
        
        # Create parent directories if they do not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        

        joblib.dump(self, file_path)
    
    @classmethod
    def load_model(cls, 
                   file_path: Union[Path, str]) -> 'RandomEmbedding':
        """
        Load a model from a file.

        Args:
            file_path (Path): The path to the file from which the model will be loaded. 
        
        Returns:
            RandomEmbedding: The loaded model.
        """

        # Ensure the file path is a Path object
        file_path = Path(file_path).absolute()

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        model = joblib.load(file_path)

        if not isinstance(model, cls):
            raise TypeError(f"The loaded object is not of type {cls.__name__}.")
        
        return model