from src.metrics.BERTScoreMetric import BERTScoreMetric
from src.metrics.BleurtMetric import BleurtMetric
from Experiment import Experiment
from src.Tasks.NegationSent import Negation_Sent
from src.Tasks.POSDrop import POSDrop
from src.Tasks.DropWordsTwoDim import DropWordsTwoDim
from src.Tasks.SwapWordsTwoDim import SwapWordsTwoDim
from src.Tasks.RepetitionTwoDim import RepetitionTwoDim




bm : BleurtMetric = BleurtMetric("BLEURT", "BLEURT without filtering", ['BLEURT'])
bsm : BERTScoreMetric = BERTScoreMetric("BERTScore", "BERTScore without filtering", ['P', 'R', 'F1'])

metrics : list = [bm, bsm]
tasks : list = [(Negation_Sent, ), (POSDrop, [('pos', 'ADJ')]), (DropWordsTwoDim,),  (SwapWordsTwoDim,),  (RepetitionTwoDim, )]

loc : str = ".examplesForVis_2021-06-08_12-41-01"
exp = Experiment(loc=loc)
exp.setup(tasks, metrics, data_specs={'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 10},steps={'steps': 1, 'txt': 1, 'snt' : 2})
exp.perturbate()
exp.evaluate()