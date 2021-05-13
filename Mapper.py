import numpy as np

from typing import Tuple


class Mapper:

    def __init__(self, x : list):
        if len(x) == 0:
            raise Exception("ERROR: Empty array passed!")
        self.x : list = x
        self.key_names : list = []

    def mapping(self) -> np.ndarray:

        self.key_names = list(self.x[0].keys())
        values : np.ndayrray = np.asarray(list(map(lambda elem : list(elem.values()), self.x)))
        return values

    def description(self) -> str:
        return "Identity map"



class BLEURTMapper_1(Mapper):

    def mapping(self) -> np.ndarray:
        self.key_names = list(self.x[0].keys())
        print(np.asarray(list(map(lambda elem : np.average(list(elem.values()), axis=1), self.x))))
        # average over sentences
        values : np.ndayrray = np.asarray(list(map(lambda elem : np.average(list(elem.values()), axis=1), self.x)))
        return values

    def description(self) -> str:
        return "Average of all evaluated sentences"
