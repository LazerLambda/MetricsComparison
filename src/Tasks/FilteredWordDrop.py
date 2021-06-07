from __future__ import annotations

from .DropWordsTwoDim import DropWordsTwoDim
from .TwoDim import TwoDim
from .Task import Task

import numpy as np
import spacy


class FilteredWordDrops(DropWordsTwoDim, TwoDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(FilteredWordDrops, self).__init__(params=params)
        self.name = "drop_words_f"
        self.descr = "Dropped words in sentences with increasing degree."

    # overwrite
    def set_steps(self, steps : dict) -> FilteredWordDrops:
        step_snt : float = 1 / steps['snt']
        self.step_arr.append([1])
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_snt), np.array([1])))))
        return self