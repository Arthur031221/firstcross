"""FirstCross: solids from single-photon histograms by first-crossing likelihoods."""
from .config import Settings, load_config
from .solver import reconstruct, solve

__all__ = ['Settings', 'load_config', 'reconstruct', 'solve']
__version__ = '1.0.0'
