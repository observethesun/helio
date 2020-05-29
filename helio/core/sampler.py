"""Sampler class."""
import numpy as np

class R: # pylint: disable=invalid-name
    """Random values generator.

    Parameters
    ----------
    sampler : str or callable
        Random number sampler. If ```str``` then corresponding attribute of ```numpy.random```.
    kwargs : dict
        Parameters of distribution.

    Attributes
    ----------
    params : dict
         Parameters of distribution.
    """
    def __init__(self, sampler='random', **kwargs):
        self._sampler = getattr(np.random, sampler) if isinstance(sampler, str) else sampler
        self._params = kwargs

    @property
    def params(self):
        """Parameters of distribution."""
        return self._params

    def __call__(self, size):
        """Get samples."""
        return self._sampler(size=size, **self._params)
