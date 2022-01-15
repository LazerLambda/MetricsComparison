"""Implementation of the Petersen estimator for Mark-Evaluate."""

import numpy as np

from .Estimate import Estimate
from . import DataOrg as do


class Petersen(Estimate):
    """Computing the ME-Petersen-estimator.

    Class to provide the functions to compute the ME-Petersen-estimator.
    """

    def __mark(self) -> int:
        """Mark function.

        "During the marking step, we mark all samples
        inside at least one hypersphere of s[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        Complexity is O(n).

        Returns
        -------
        int
            number of marked samples
        """
        return len(self.ref) + self.data.bin_vec_cand.sum()

    def __capture(self) -> int:
        """Capture function.

        Capture every sample from the candidate set
        plus each ref sample in the respective hyper-
        sphere of a sample from the cand set.

        Complexity is O(n).

        Returns
        -------
        int
            number of captured samples
        """
        return len(self.cand) + self.data.bin_vec_ref.sum()

    def __recapture(self) -> int:
        """Recapture function.

        Recapture all samples from within
        the hypersphere of a sample from
        the respective opposite set.

        Complexity is O(n).

        Returns
        -------
        int
            number of recaptured samples
        """
        return (
            self.data.bin_vec_ref.sum() +
            self.data.bin_vec_cand.sum())

    def estimate(self) -> float:
        """Estimate function.

        Computes the ME-Petersen-estimator.

        Complexity is O(n)

        Returns
        -------
        float
            ME-Petersen-estimation of the population
        """
        c: int = self.__capture()
        m: int = self.__mark()
        r: int = self.__recapture()

        return c * m / r if r != 0 else 0
