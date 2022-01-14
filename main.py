"""Main.py script."""


from argparse import ArgumentParser

import importlib
import time

from src.Experiment import Experiment

from src.Tasks.Negate import Negate
from src.Tasks.POSDrop import POSDrop
from src.Tasks.DropWordsOneDim import DropWordsOneDim
from src.Tasks.Repetition import Repetition
from src.Tasks.SwapWordsOneDim import SwapWordsOneDim

from src.Plot import Plot
from src.PlotByMetric import PlotByMetric

if __name__ == "__main__":
    """ Main script

    Main script to run experiments.
    Passing of different command line arguments is supported:
        -m / --metric : name of metric-class in the metrics.custom_metrics folder
        -n / --number : number of samples, int
        -s / --steps : number of steps, int
        -d / --dir : path to directory, str
        -t / --title : title of the experiment, str
        -sd / --seed : seed for sampling : int
        -st / --sentence : set, if sample should consists of sentences instead of texts
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--metrics_class",
        required = True,
        type=str,
        dest="m",
        help="Name of metric-class in the metrics.custom_metrics folder.")
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
    parser.add_argument(
        "-st",
        "--sentence",
        default=False,
        action='store_true',
        dest="st",
        help="Set to sample sentences instead of texts.")
    parser.add_argument(
        "-sc",
        "--scale-down",
        default=None,
        type=int,
        dest="sc",
        help="Scale sample down (if computational costs are too high).")

    args = parser.parse_args()
    args = vars(args)

    # Import Custom Metric dynamically
    metric_class: type = getattr(
        importlib.import_module("src.metrics.custom_metrics." + args['m']),
        args['m'])
    instance: any = metric_class()

    tasks: list = [
        (DropWordsOneDim, ),
        (SwapWordsOneDim, ),
        # (DropAndSwap, ),
        (Repetition,),
        (Negate, ),
        (POSDrop,),]
        # (Mix, )]
    metrics: list =  [instance]

    # loc : str = ".all_2021-06-10_16-17-08"
    # loc : str = ".all_ME"

    exp = Experiment(
        loc=args['dir'],
        name=args['title'],
        sentence=args['st'],
        verbose=True,
        scale_down=args['sc'])
    
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

    # Perturbate
    start_time = time.time()
    exp.perturbate()
    print("--- Perturbation took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    exp.evaluate()
    print("--- Evaluation took %s seconds ---" % (time.time() - start_time))
    # exp.plot([Plot])
