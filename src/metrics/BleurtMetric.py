from .Metric import Metric
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
from bleurt import score


import cmasher as cmr
import os
import seaborn as sns

class BleurtMetric(Metric):

    limits : tuple = (-1.5,1.5)

    def __init__(self, name : str, description : str, submetric : list):
        
        super(BleurtMetric, self).__init__(name, description, submetric)
        
        # path : str = "src/bleurt/bleurt/test_checkpoint" + os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path : str = "src/bleurt/bleurt/test_checkpoint"

        # path from parent folder of src
        self.scorer_bleurt: score.BleurtScorer = score.BleurtScorer(
            checkpoint=path)

        palette = sns.color_palette(None, 1)
        self.color : dict = {
            'BLEURT' : palette[0]
        }
        self.id : bool = False

    def get_id(self, ref :list, cand : list):
        raise Exception("ERROR:\n\t'-> BLEURT has to be computed.")


    def compute(self, ref : list, cand : list):
        super(BleurtMetric, self).check_input(ref, cand)
        return [self.scorer_bleurt.score(references=ref, candidates=cand)]

    # TODO remove
    @staticmethod
    def concat(acc, elem):
        for e in elem:
            if isinstance(e, dict):
                if 'BLEURT' in e.keys():
                    return {
                        'BLEURT': acc['BLEURT'] + e['BLEURT']
                    }

    def init_dict(self):
        return dict(zip(self._submetrics, [[] for _ in self._submetrics]))

    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            cmap = cmr.prinsenvlag
            return {
                'color': cmap,
                'vmin' : -1.5,
                'vmax' : 1.5
            }
        
        return None

        
