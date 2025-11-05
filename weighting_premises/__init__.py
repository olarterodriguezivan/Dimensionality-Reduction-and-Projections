from .rank_based import RankBasedWeighting


def get_rank_based_weighting(method: str = 'logarithmic') -> RankBasedWeighting:
    """
    Factory function to get a RankBasedWeighting instance.
    
    Args:
        method: Weighting method ('linear', 'logarithmic', 'exponential', 'inverse')
        
    Returns:
        An instance of RankBasedWeighting with the specified method.
    """
    return RankBasedWeighting(method=method)

