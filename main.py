"""Main.py script."""


from argparse import ArgumentParser

import time

from Experiment import Experiment

from src.metrics.BERTScoreMetric import BERTScoreMetric
from src.metrics.BleurtMetric import BleurtMetric
# from src.metrics.MEMetric import MEMetric
from src.metrics.BERTScoreIDFMetric import BERTScoreIDFMetric

from src.Tasks.Negate import Negate
from src.Tasks.POSDrop import POSDrop
from src.Tasks.DropWordsOneDim import DropWordsOneDim
from src.Tasks.Repetition import Repetition
from src.Tasks.SwapWordsOneDim import SwapWordsOneDim
from src.Tasks.DropAndSwap import DropAndSwap
from src.Tasks.Mix import Mix

from src.Plot import Plot
from src.PlotByMetric import PlotByMetric

if __name__ == "__main__":
    """ Main script

    Main script to run experiments.
    Passing of different command line arguments is supported:
        -n / --number : number of samples, int
        -s / --steps : number of steps, int
        -d / --dir : path to directory, str
        -t / --title : title of the experiment, str
        -sd / --seed : seed for sampling : int
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--number",
        default=5,
        type=int,
        dest="n",
        help="Number of texts.")
    parser.add_argument(
        "-s",
        "--steps",
        default=2,
        type=int,
        dest="steps",
        help="Amount of damaging steps.")
    parser.add_argument(
        "-d",
        "--dir",
        default=None,
        type=str,
        dest="dir",
        help="Already existing directory to continue computation.")
    parser.add_argument(
        "-t",
        "--title",
        default="exp",
        type=str,
        dest="title",
        help="Title for a new experiment. Will be the folder name.")
    parser.add_argument(
        "-sd",
        "--seed",
        default=None,
        type=int,
        dest="seed",
        help="Seed for sampling.")

    args = parser.parse_args()
    args = vars(args)

    bm: BleurtMetric = BleurtMetric()
    bsm: BERTScoreMetric = BERTScoreMetric()
    # mem: MEMetric = MEMetric()

    metrics: list = [bm, bsm]
    tasks: list = [
        (DropWordsOneDim, ),
        (SwapWordsOneDim, ),
        (DropAndSwap, ),
        (Repetition,),
        (Negate, ),
        (POSDrop,),
        (Mix, )]

    # loc : str = ".all_2021-06-10_16-17-08"
    # loc : str = ".all_ME"
    exp = Experiment(loc=args['dir'], name=args['title'], verbose=True)
    exp.setup(
        tasks,
        metrics,
        data_specs={
            'name': 'cnn_dailymail',
            'version': '3.0.0',
            'n': args['n'],
            'f': (lambda data, sample: data['train'][sample]['article']),
            'seed': args['seed']},
        steps={
            'steps': args['steps'],
            'txt': args['steps'],
            'snt': args['steps']},
        pos_list=['ADJ', 'DET', 'VERB', 'NOUN'])

    # access data here and setup BERTScore idf
    # exp.metrics = metrics

    start_time = time.time()
    exp.perturbate()
    print("--- Perturbation took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    exp.evaluate()
    print("--- Evaluation took %s seconds ---" % (time.time() - start_time))
    exp.plot([Plot])
