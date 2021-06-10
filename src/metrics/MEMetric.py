from .Metric import Metric

from ..ME.markevaluate import MarkEvaluate as ME

import seaborn as sns

class MEMetric(Metric):
    
    limits : tuple = (0,1.05)

    def __init__(self, name : str, description : str, submetric : list):
        
        super(MEMetric, self).__init__(name, description, submetric)

        self.ME : ME = ME(orig=False)

        palette = sns.color_palette(None, 3)

        self.color : dict = {
            'Petersen' : palette[0],
            'Schnabel' : palette[1],
            'CAPTURE' : palette[2]
        }
        self.id : bool = False

    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref))