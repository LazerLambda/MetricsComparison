from .Metric import Metric
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim 
from bert_score import BERTScorer

import seaborn as sns
import spacy
from datasets import load_metric

class BLEUScoreMetric(Metric):

    limits : tuple = (0,1.05)

    def __init__(self):
        super(BLEUScoreMetric, self).__init__()

        self.name: str = "BLEU"
        self.description: str = "Bilingual evaluation understudy"
        self.submetrics: list = ["BLEU"]
        self.id : bool = True

        self.bleu_hggfc = load_metric("bleu")
        self.nlp = spacy.load("en_core_web_sm")

        palette: sns.palettes._ColorPalette =\
            sns.color_palette(None, 1)

        self.color : dict = {
            'BLEU' : palette[0]
        }


    def get_id(self, ref :list, cand : list):
        assert len(ref) == len(cand)
        return [[1]] # * len(cand)


    def compute(self, ref : list, cand : list):

        # For this experiment, only one reference sample is used
        assert len(ref) == len(cand)
        cand, ref = [[token.text for token in self.nlp(str(sent))] for sent in cand],\
        [[[token.text for token in self.nlp(str(sent))]] for sent in ref]
        score = self.bleu_hggfc.compute(predictions=cand, references=ref)
        return [[score['bleu']]]

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