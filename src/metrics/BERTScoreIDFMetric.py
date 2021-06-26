from .Metric import Metric
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
from bert_score import BERTScorer

import seaborn as sns
from itertools import chain


class BERTScoreIDFMetric(Metric):

    limits : tuple = (0,1.05)

    def __init__(self):
        super(BERTScoreIDFMetric, self).__init__()

        # Properties
        self.name: str = "BERTScore (idf)"
        self.description: str = "BERTScore with idf-weighting"
        self.submetrics: list = ["R", "P", "F1"]
        self.id : bool = True

        palette: sns.palettes._ColorPalette =\
            sns.color_palette(None, 3)
        
        self.scorer_bertscore: BERTScorer = None

        self.color : dict = {
            'R' : palette[0],
            'P' : palette[1],
            'F1' : palette[2]
        }

    def init_idf(self, ref: list):
        """Initialize idf-weights.
        
        Reference corpus is passed as list of lists of strings
        from which the idf-weigthts are computed.
        """
        assert len(ref) != 0,\
            "Reference list is empty."
        assert isinstance(ref, list),\
            f"Passed value: {ref} is not of type 'list'"
        assert isinstance(ref[0], list),\
            f"First item: {ref[0]} of ref is not of type list"

        ref: list = list(chain.from_iterable(ref))
        self.scorer_bertscore: BERTScorer = BERTScorer(
                    lang="en",
                    idf=True, idf_sents=ref)

    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref))


    def compute(self, ref : list, cand : list):
        assert len(ref) == len(cand)
        return self.scorer_bertscore.score(cand, ref)

    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            return {
                'color': 'plasma',
                'vmin' : 0,
                'vmax' : 1
            }
        
        return None