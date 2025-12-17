from sklearn.random_projection import GaussianRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from typing import Optional, Union

from pathlib import Path
import joblib

class GaussianRandomEmbeddings(GaussianRandomProjection, BaseEstimator, TransformerMixin):
    """
    Wrapper for scikit-learn's GaussianRandomProjection for dimensionality reduction.
    
    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.
    eps : float, default=0.1
        Precision parameter for automatic n_components calculation.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the embeddings.
    """
    
    def __init__(self, 
                 n_components:Union[str,int]=2, # This is the only change from the base class, isnce it's not of the interest of using the
                                                # Johnson-Lindenstrauss lemma here.
                 eps:float=0.1, 
                 random_state:Optional[Union[int, np.random.RandomState]]=None):

        # Initialize the GaussianRandomProjection
        super().__init__(n_components=n_components, 
                         eps=eps, 
                         random_state=random_state)
    


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
                   file_path: Union[Path, str]) -> 'GaussianRandomEmbeddings':
        """
        Load a model from a file.

        Args:
            file_path (Path): The path to the file from which the model will be loaded. 
        
        Returns:
            GaussianRandomEmbedding: The loaded model.
        """

        # Ensure the file path is a Path object
        file_path = Path(file_path).absolute()

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        model = joblib.load(file_path)

        if not isinstance(model, cls):
            raise TypeError(f"The loaded object is not of type {cls.__name__}.")
        
        return model
    
    