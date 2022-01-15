"""Metric Base class."""

from ..Experiment import Experiment
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
import numpy as np


class Metric:
    """Class name must be the same as file name."""

    limits: tuple = (0, 0)

    def __init__(self, **kwargs):
        """Initialize."""
        self.name: str = None
        self.description: str = None
        self.submetrics: list = None
        self.id: bool = False
        self.color: Experiment() = dict()
        self.experiment: Experiment() = None

    def set_exp(self, exp: Experiment):
        """Set experiment as class variable.

        Change method to access data from experiment class.
        """
        self.experiment = exp

    def compute(self):
        """Compute results.

        Must be returned as a list of lists, depending on the
        number of important submetrics: [[result_1, result_2,...],...]
        """
        raise NotImplementedError("compute method is not implemented.")

    def get_id(self, ref: list, cand: list) -> dict:
        r"""Return id value for identical inputs.

        Params
        ------
        ref : list
            reference list of sentences
        cand : list
            candidate list of sentences

        Returns
        -------
        dict
            dictionary similar to dictionary
            returned from 'evaluate()'
        """
        raise NotImplementedError("get_id method is not implemented.")

    @staticmethod
    def check_input(ref: list, cand: list):
        """Check Input.

        Params
        ------
        ref : list
            reference list of sentences
        cand : list
            candidate list of sentences

        Raise
        -----
        Exception iff a inputs contain empyty sentences.
        """
        ref_check: list = list(map(lambda x: len(x) == 0, ref))
        cand_check: list = list(map(lambda x: len(x) == 0, cand))
        if True in ref_check or True in cand_check:
            zipped: list = list(zip(ref, cand))
            ref_check, cand_check =\
                np.asarray(ref_check), np.asarray(cand_check)
            index_ref: int = np.where(ref_check == True)[0]
            index_cand: int = np.where(cand_check == True)[0]

            index_ref = index_ref if len(index_ref) != 0 else [-1]
            index_cand = index_cand if len(index_cand) != 0 else [-1]

            index: int = index_ref[0] if index_ref[0] > index_cand[0] else index_cand[0]

            raise Exception(
                ("ERROR:\n\t'->A sentence pair includes an empty sentence at position {}:"
                "\n\tReference: {}\n\tCandidate: {}").format(index, zipped[index][0], zipped[index][1]))
