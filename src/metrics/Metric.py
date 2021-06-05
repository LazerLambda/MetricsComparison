from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
import numpy as np

class Metric:

    def __init__(self, name : str, description : str, submetrics : list):
        self.name : str = name
        self.description : str = description
        self.submetrics : list = submetrics

    def compute(self):
        pass

    @staticmethod
    def check_input(ref : list, cand : list):
        ref_check : list = list(map(lambda x : len(x) == 0, ref))
        cand_check : list = list(map(lambda x : len(x) == 0, cand))
        if True in ref_check or True in cand_check:
            zipped : list = list(zip(ref, cand))
            ref_check, cand_check = np.asarray(ref_check), np.asarray(cand_check)
            index_ref : int = np.where(ref_check == True)[0] 
            index_cand : int = np.where(cand_check == True)[0]

            index_ref = index_ref if len(index_ref) != 0 else [-1]
            index_cand  = index_cand if len(index_cand) != 0 else [-1]
            
            index : int = index_ref[0] if index_ref[0] > index_cand[0] else index_cand[0]

            raise Exception(("ERROR:\n\t'->A sentence pair includes an empty sentence at position {}:"
                "\n\tReference: {}\n\tCandidate: {}").format(index, zipped[index][0], zipped[index][1]))

    @staticmethod
    def concat():
        pass

    def init_dict(self):
        pass

    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            return {
                'color': None,
                'vmin' : None,
                'vmax' : None
            }
        
        return None