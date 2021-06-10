from src.metrics.BERTScoreMetric import BERTScoreMetric
from src.metrics.BleurtMetric import BleurtMetric
from Experiment import Experiment
from src.Tasks.NegationSent import Negation_Sent
from src.Tasks.DropWordsTwoDim import DropWordsTwoDim
from src.Tasks.RepetitionTwoDim import RepetitionTwoDim
from src.Tasks.POSDrop import POSDrop
# from src.Tasks.FilteredWordDrop import FilteredWordDrop
from src.Tasks.SwapWordsTwoDim import SwapWordsTwoDim

from src.Plot import Plot
from src.PlotByMetric import PlotByMetric

import time


bm : BleurtMetric = BleurtMetric("BLEURT", "BLEURT without filtering", ['BLEURT'])
bsm : BERTScoreMetric = BERTScoreMetric("BERTScore", "BERTScore without filtering", ['P', 'R', 'F1'])

metrics : list = [bm, bsm]
tasks : list = [(DropWordsTwoDim, ), (SwapWordsTwoDim, ), (POSDrop, [('pos', 'ADJ')]), (RepetitionTwoDim,), (Negation_Sent, )]

loc : str = ".old_2021-06-10_11-42-45"
exp = Experiment(loc=loc, name="old")
exp.setup(tasks, metrics, data_specs={'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 4},steps={'steps': 1, 'txt': 2, 'snt' : 2}, pos_list=['ADJ', 'DET', 'VERB', 'NOUN'])

start_time = time.time()
exp.perturbate()
print("--- Perturbation took %s seconds ---" % (time.time() - start_time))

start_time = time.time()
exp.evaluate()
print("--- Evaluation took %s seconds ---" % (time.time() - start_time))
exp.plot([Plot], metrics)
