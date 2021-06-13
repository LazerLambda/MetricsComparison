from .Metric import Metric

from ..ME.markevaluate.MarkEvaluate import MarkEvaluate as ME
from ..Tasks.Task import Task
from ..Tasks.OneDim2 import OneDim2
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim

import seaborn as sns

class MEMetric(Metric):
    
    limits : tuple = (0,1.05)

    def __init__(self, name : str, description : str, submetric : list):
        
        super(MEMetric, self).__init__(name, description, submetric)

        self.ME_scorer : ME = ME(orig=False)
        self.submetrics : list = ['Petersen', 'Schnabel_qul', 'Schnabel_div', 'CAPTURE']

        palette = sns.color_palette(None, 4)

        self.color : dict = {
            'Petersen' : palette[0],
            'Schnabel_qul' : palette[1],
            'Schnabel_div' : palette[2],
            'CAPTURE' : palette[3]
        }
        self.id : bool = False

    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref)), [1] * len(ref)

    def compute(self, ref : list, cand : list):
        super(MEMetric, self).check_input(ref, cand)
        score : dict = self.ME_scorer.estimate(cand=cand, ref=ref)
        return ([score['Petersen']], [score['Schnabel_qul']], [score['Schnabel_div']], [score['CAPTURE']])

    @staticmethod
    def concat(acc, elem):
        pass

    def init_dict(self):
        return dict(zip(self._submetrics, [[] for _ in self._submetrics]))

    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            return {
                'color': None,
                'vmin' : 0,
                'vmax' : 1
            }
        
        return None