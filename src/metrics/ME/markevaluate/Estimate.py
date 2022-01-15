"""Base class for Estimators."""


import numpy as np
from progress.bar import ShadyBar

from . import DataOrg as do


class Estimate():
    """Parent class for Population Estimators.

    Provides all class variables for children classes
    to work properly and catches possible Errors at
    one central point.
    """

    def __init__(
            self,
            data_org: do,
            orig: bool = False):
        """Initialize necessary structures."""
        self.cand: np.ndarray = data_org.cand_embds
        self.ref: np.ndarray = data_org.ref_embds
        self.k: int = data_org.k
        self.data: do = data_org
        self.orig: bool = orig
