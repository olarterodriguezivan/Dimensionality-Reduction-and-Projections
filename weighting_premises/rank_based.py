import numpy as np
import torch
from torch import Tensor
from typing import Union, Optional


class RankBasedWeighting:
    """
    Rank-based weighting scheme for feature importance or data points.
    
    Methods support different ranking strategies like linear, exponential, 
    logarithmic and inverse rank weighting.
    """
    
    def __init__(self, method: str = 'logarithmic'):
        """
        Initialize rank-based weighting.
        
        Args:
            method: Weighting method ('linear', 'logarithmic', 'exponential', 'inverse')
        """

        # Set the method using the property setter for validation
        self.method = method
    
    def linear_weights(self, ranks: np.ndarray) -> np.ndarray:
        """Linear rank weighting: w_i = (n - r_i + 1) / sum"""
        n = len(ranks)
        weights = (n - ranks + 1) / np.sum(n - ranks + 1)
        return weights
    
    def exponential_weights(self, ranks: np.ndarray, decay: float = 0.5) -> np.ndarray:
        """Exponential rank weighting: w_i = exp(-decay * r_i)"""
        weights = np.exp(-decay * ranks)
        return weights / np.sum(weights)
    
    def inverse_weights(self, ranks: np.ndarray) -> np.ndarray:
        """Inverse rank weighting: w_i = 1/r_i"""
        weights = 1.0 / ranks
        return weights / np.sum(weights)
    
    def logarithmic_weights(self, ranks: np.ndarray) -> np.ndarray:
        r"""Logarithmic rank weighting: w_i = ln n - ln r_i;
        where n is the total number of items and r_i is the rank of item i.
        This follows from the idea that higher ranked items should have significantly
        more weight, but the increase in weight should diminish for lower ranks.

        The idea is used in Raponi et al. (2020) "Principal Component Analysis Bayesian Optimization - PCABO".

        Also in the context of dimensionality reduction, this weighting scheme is used by
        Tanabe (2021) "Towards Exploratory Landscape Analysis for Large-scale Optimization: A Dimensionality Reduction Framework".
           """
        n = len(ranks)
        weights = np.log(n) - np.log(ranks)
        return weights / np.sum(weights)
    
    def compute_weights(self, values: Union[np.ndarray,Tensor], **kwargs) -> np.ndarray:
        """
        Compute rank-based weights for given values.
        
        Args:
            values: Input values to rank
            **kwargs: Additional parameters for weighting methods
            
        Returns:
            Normalized weights based on ranks
        """

        if isinstance(values, Tensor):
            values = values.cpu().numpy().ravel()

        ranks = np.argsort(np.argsort(-values)) + 1  # Higher values get lower ranks
        
        if self.method == 'linear':
            return self.linear_weights(ranks)
        elif self.method == 'exponential':
            decay = kwargs.get('decay', 0.5)
            return self.exponential_weights(ranks, decay)
        elif self.method == 'inverse':
            return self.inverse_weights(ranks)
        elif self.method == 'logarithmic':
            return self.logarithmic_weights(ranks)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    
    @property
    def method(self) -> str:
        return self._method
    
    @method.setter
    def method(self, value: str):
        valid_methods = ['linear', 'logarithmic', 'exponential', 'inverse']
        if value not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self._method = value