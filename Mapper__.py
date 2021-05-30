import numpy as np

from typing import Tuple


class Mapper:
    """
    TODO
    """

    def __init__(self, x : list):
        if len(x) == 0:
            raise Exception("ERROR: Empty array passed!")
        self.x : list = x
        self.key_names = list(self.x[0].keys())

    def mapping(self) -> np.ndarray:

        values : np.ndayrray = np.asarray(list(map(lambda elem : list(elem.values()), self.x)))
        return values

    def description(self) -> str:
        return ""



class AverageMapper(Mapper):
    """
    TODO
    """

    def mapping(self) -> np.ndarray:
        # average over sentences
        values : np.ndarray = np.asarray(list(map(lambda elem : np.average(list(elem.values()), axis=1), self.x)))
        return values
    
    def description(self) -> str:
        return "Average of all evaluated sentences"
