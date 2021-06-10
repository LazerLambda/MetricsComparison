from src.metrics.BERTScoreMetric import BERTScoreMetric
from src.metrics.BleurtMetric import BleurtMetric
from Experiment import Experiment
from src.Tasks.Negate2 import Negate2
from src.Tasks.POSDrop2 import POSDrop2
from src.Tasks.DropWordsOneDim import DropWordsOneDim
from src.Tasks.Repettions2 import Repetitions2
from src.Tasks.SwapWordsOneDim import SwapWordsOneDim
from src.Tasks.DropAndSwap import DropAndSwap

from src.Plot import Plot
from src.PlotByMetric import PlotByMetric

import time


bm : BleurtMetric = BleurtMetric("BLEURT", "BLEURT without filtering", ['BLEURT'])
bsm : BERTScoreMetric = BERTScoreMetric("BERTScore", "BERTScore without filtering", ['P', 'R', 'F1'])

metrics : list = [bm, bsm]
tasks : list = [(DropWordsOneDim, ), (SwapWordsOneDim, ), (DropAndSwap, ), (Repetitions2,), (Negate2, ), (POSDrop2,), ]

loc : str = ".das_2021-06-09_20-57-28"
exp = Experiment(name="das")
exp.setup(tasks, metrics, data_specs={'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 4},steps={'steps': 1, 'txt': 2, 'snt' : 2}, pos_list=['ADJ', 'DET', 'VERB', 'NOUN'])

start_time = time.time()
exp.perturbate()
print("--- Perturbation took %s seconds ---" % (time.time() - start_time))

start_time = time.time()
exp.evaluate()
print("--- Evaluation took %s seconds ---" % (time.time() - start_time))
exp.plot([Plot], metrics)
