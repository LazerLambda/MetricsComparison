from src.metrics.BERTScoreMetric import BERTScoreMetric
from src.metrics.BleurtMetric import BleurtMetric
from Experiment import Experiment
from src.Tasks.POSDrop2 import POSDrop2

from src.Plot import Plot
from src.PlotByMetric import PlotByMetric

import time


bm : BleurtMetric = BleurtMetric("BLEURT", "BLEURT without filtering", ['BLEURT'])
bsm : BERTScoreMetric = BERTScoreMetric("BERTScore", "BERTScore without filtering", ['P', 'R', 'F1'])

metrics : list = [bm, bsm]
tasks : list = [(POSDrop2, )]

loc : str = ".POS_2021-06-09_19-41-46"
exp = Experiment(name="POS")
exp.setup(tasks, metrics, data_specs={'name': 'cnn_dailymail', 'version' : '3.0.0', 'n' : 2},steps={'steps': 1, 'txt': 1, 'snt' : 2}, pos_list=['ADJ', 'DET', 'VERB'])

start_time = time.time()
exp.perturbate()
print("--- Perturbation took %s seconds ---" % (time.time() - start_time))

start_time = time.time()
exp.evaluate()
print("--- Evaluation took %s seconds ---" % (time.time() - start_time))
exp.plot([Plot], metrics)
