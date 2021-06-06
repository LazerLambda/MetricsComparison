from __future__ import annotations

from .Consts import CONSTS
import json
import functools
import numpy as np
import os
import pickle

from copy import deepcopy
from itertools import chain
from .Metrics import Metrics as mtrc


class ReduceData:

    def __init__(self, metrics: list):

        # initialization
        self.metrics: list = metrics
        self.plot_data: dict = {}
        self.data: dict = dict()

    def __apply_fun__(
            self,
            fun: callable,
            fun_str: str,
            combined_data: list,
            metric_list: list,
            i: int,
            degree: float):

        for metric in metric_list:

            acc_dict: list = []
            for sub_metric in mtrc.SUBMETRICS[metric]:
                number: float = fun(np.asarray(
                    combined_data[metric]['data'][i][1][sub_metric]))
                acc_dict.append((sub_metric, number))

            acc_dict: dict = dict(acc_dict)

            combined_data[metric][fun_str][i] = (degree, acc_dict)

    def __add_data__(self, data: tuple):

        self.steps: np.ndarray = np.unique([e[0] for e in data[1]])
        task: str = data[0]
        data = data[1]

        new_data: list = lambda: {
            'data': [None for step in self.steps],
            'min': [None for step in self.steps],
            'max': [None for step in self.steps],
            'mean': [None for step in self.steps],
            'median': [None for step in self.steps]
        }

        # create list of metrics to apply
        metric_reductions: list = []
        for metric in self.metrics:
            metric_reductions += metric['REDUCE']

        combined_data: list = [(name, new_data())
                               for name in metric_reductions]
        combined_data: dict = dict(combined_data)


        for i, degree in enumerate(self.steps):

            for metric in metric_reductions:
                value: dict = functools.reduce(
                    mtrc.REDUCE_DICT[metric], data[i][1], mtrc.INIT_DICT[metric])
                combined_data[metric]['data'][i] = (degree, value)

            # mean
            self.__apply_fun__(np.average, 'mean', combined_data,
                           metric_reductions, i, degree)

            # min
            self.__apply_fun__(np.min, 'min', combined_data,
                           metric_reductions, i, degree)

            # max
            self.__apply_fun__(np.max, 'max', combined_data,
                           metric_reductions, i, degree)

            # median
            self.__apply_fun__(np.median, 'median', combined_data,
                           metric_reductions, i, degree)

            self.data[task] = combined_data


    def add_data(self, task_list: list, folder_path: str):
        for task_name in task_list:
            file_path : str = os.path.join(folder_path, task_name + "_data.p")
            data_file = open(file_path, 'rb')
            data = pickle.load(data_file)
            print(data)
            data_file.close()
            self.__add_data__(data)

    def __get_data__(self, task: str, key: str, id: str):
        # x value as deterioration step, y value as different metrices
        plot_data = [(x, np.asarray(list(y.values())))
                     for x, y in self.data[task][key][id]]
        # unzipping values
        plot_data: list = list(zip(*plot_data))
        x_data, y_data = np.asarray(plot_data[0]), np.asarray(plot_data[1])
        return x_data, y_data

    def vis_ready(self):

        for _, task in enumerate(self.data.keys()):
            self.plot_data[task] = dict()
            for _, key in enumerate(self.data[task].keys()):

                x_data, y_data_mean = self.__get_data__(task, key, 'mean')
                # _, y_data = self.__get_data__(key, 'data')
                _, y_data_min = self.__get_data__(task, key, 'min')
                _, y_data_max = self.__get_data__(task, key, 'max')

                self.plot_data[task][key] = {
                    'description': CONSTS[key],
                    'x': x_data,
                    'y_mean': y_data_mean,
                    'y_min': y_data_min,
                    'y_max': y_data_max,
                    # 'y_data' : y_data
                }

    def export_results(self):
        results: dict = deepcopy(self.data)
        results_json: dict = dict()
        for task in self.data.keys():
            for key in results[task].keys():
                results_json[task] = dict()
                results_json[task][key] = results[task][key]
                del results_json[task][key]['data']
            results_json['type'] = task

        file_results = open('results.json', 'w')
        json.dump(results_json, file_results)
        file_results.close()
        del results
        del results_json