r"""
This module is just for implementation of Sturges' rule as a binning heuristic.
It can be used to determine the optimal number of bins for histogram-based methods.
"""

# Typing import
from typing import Optional, Tuple
# NumPy import
import numpy as np


class BinningHeuristic:
    r"""
    This class implements Sturges' rule for determining the optimal number of bins (equally bin edges)
    for histogram-based methods.
    """

    def __init__(self, 
                 method: str = 'sturges'):
        r"""
        Initialize the BinningHeuristic with the target variable y.

        Parameters
        ----------
        method : str, default='sturges'
            The binning heuristic method to use. The options are sturges, scott, freedman-diaconis, and sqrt.
        """
        self.method = method
        self._y = None
        self._n_bins = None
        self._variable_range = None

    @property
    def method(self) -> str:
        r"""
        Get the binning heuristic method.

        Returns
        -------
        method : str
            The binning heuristic method.
        """
        return self._method
    
    @method.setter
    def method(self, method: str):
        r"""
        Set the binning heuristic method.

        Parameters
        ----------
        method : str
            The binning heuristic method to use. The options are sturges, scott, freedman-diaconis, and sqrt.
        """
        valid_methods = ['auto','sturges', 'scott', 'rice', 'stone', 'doane', 'fd', 'sqrt']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid options are {valid_methods}.")
        self._method = method

    
    def fit(self,
            y:np.ndarray,
            arange:Optional[Tuple[float, float]] = None) -> 'BinningHeuristic':
        r"""
        Fit the binning heuristic to the target variable y.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target variable.
        arange : tuple of (float, float), optional
            The range of the variable to consider for binning. If None, the range of y is used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert y to a numpy array and flatten
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        
        if y.ndim > 1:
            y = y.flatten()
        
        if arange is None:
            arange = (np.min(y), np.max(y))
        else:
            if not (isinstance(arange, tuple) and len(arange) == 2):
                raise ValueError("arange must be a tuple of (min, max).")
        
        self.variable_range = arange

        # Store the target variable
        self._y = y.copy()

        # Determine the histogram bin edges using numpy's histogram_bin_edges
        self._bin_edges = np.histogram_bin_edges(y, bins=self.method, range=arange)

        return self
    
    def transform(self,
                  y:np.ndarray,
                  zero_index:Optional[bool]=True) -> np.ndarray:
        r"""
        Transform the target variable y into bin indices based on the fitted bin edges.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target variable to be binned.
        zero_index : bool
            If True, the bin indices will start from 0. If False, they will start from 1.

        Returns
        -------
        bin_indices : array-like of shape (n_samples,)
            The indices of the bins to which each value in y belongs.
        """
        if self._bin_edges is None:
            raise ValueError("The model must be fitted before calling transform().")
        
        # Convert y to a numpy array and flatten
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        
        if y.ndim > 1:
            y = y.flatten()
        
        # Digitize the values into bins
        bin_indices = np.digitize(y, bins=self._bin_edges) 

        if zero_index:
            bin_indices -= 1

        # Ensure bin indices are within valid range
        if zero_index:
            bin_indices = np.clip(bin_indices, 0, len(self._bin_edges) - 2)
        else:
            bin_indices = np.clip(bin_indices, 1,  len(self._bin_edges) - 1)

        return bin_indices
    
    def fit_transform(self,
                      y:np.ndarray,
                      arange:Optional[Tuple[float, float]] = None,
                      zero_index:bool = True) -> np.ndarray:
        r"""
        Fit the binning heuristic to the target variable y and transform it into bin indices.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target variable.
        arange : tuple of (float, float), optional
            The range of the variable to consider for binning. If None, the range of y is used.
        zero_index : bool, default=True
            If True, the bin indices will start from 0. If False, they will start from 1.

        Returns
        -------
        bin_indices : array-like of shape (n_samples,)
            The indices of the bins to which each value in y belongs.
        """
        self.fit(y, arange)
        return self.transform(y, zero_index=zero_index)
    
    def get_bin_centers(self) -> np.ndarray:
        r"""
        Get the centers of the bins determined by the binning heuristic.

        Returns
        -------
        bin_centers : array-like of shape (n_bins,)
            The centers of the bins.
        """
        if self._bin_edges is None:
            raise ValueError("The model must be fitted before accessing bin centers.")
        
        bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        return bin_centers
    
    @property
    def n_bins(self) -> int:
        r"""
        Get the number of bins determined by the binning heuristic.

        Returns
        -------
        n_bins : int
            The number of bins.
        """
        if self._bin_edges is None:
            raise ValueError("The model must be fitted before accessing n_bins.")
        
        return len(self._bin_edges) - 1
    
    @property
    def bin_edges(self) -> np.ndarray:
        r"""
        Get the bin edges determined by the binning heuristic.

        Returns
        -------
        bin_edges : array-like of shape (n_bins + 1,)
            The edges of the bins.
        """
        if self._bin_edges is None:
            raise ValueError("The model must be fitted before accessing bin_edges.")
        
        return self._bin_edges
    
    @property
    def variable_range(self) -> Tuple[float, float]:
        r"""
        Get the range of the target variable y.

        Returns
        -------
        data_range : tuple of (float, float)
            The minimum and maximum values of y.
        """
        if not hasattr(self, '_y'):
            raise ValueError("The model must be fitted before calling range().")
        
        return self._variable_range
    
    @variable_range.setter
    def variable_range(self, arange:Tuple[float, float]):
        r"""
        Set the range of the target variable y.

        Parameters
        ----------
        arange : tuple of (float, float)
            The minimum and maximum values of y.
        """
        if not (isinstance(arange, tuple) and len(arange) == 2):
            raise ValueError("arange must be a tuple of (min, max).")
        
        self._variable_range = arange
    
    







