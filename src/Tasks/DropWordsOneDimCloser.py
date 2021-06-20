from .OneDim2 import OneDim2
from .DropWordsOneDim import DropWordsOneDim
from .Task import Task

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb

class DropWordsOneDimCloser(DropWordsOneDim, OneDim2, Task):


    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(DropWordsOneDimCloser, self).__init__(params=params)
        self.name = "dropped_words_specific"
        self.descr = "Dropped words focused on small impairments."


    def set_steps(self, steps : dict) -> Task:
        self.step_arr = np.flip(np.arange(0.05, 0.2, 0.05))
        return self