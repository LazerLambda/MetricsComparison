from __future__ import annotations

import numpy as np

from itertools import chain
from Apply import Apply

class VisData:

    # __slots__ = [
    #     'indices',
    #     'scores',
    #     'degree',
    #     'data',
    #     'min',
    #     'max',
    #     'mean',
    #     'median',
    #     'scatter',
    #     'name',
    #     'legend']

    # def __init__(self, degree : float, data : dict, indices : list, name : str):
    #     self.degree : float = degree
    #     self.scores : int = len(data)
    #     self.indices : list = indices
    #     self.data : dict = data
    #     self.name : str = name
    #     self.legend : list = data.values()

    # def reduce(self, apply : callable) -> VisData:
    #     applied : tuple = apply(self.data, self.indices)

    #     self.mean : float = applied[0]
    #     self.min : float = applied[1]
    #     self.median : float = applied[2]
    #     self.max : float = applied[3]
    #     self.scatter : float = applied[4]

    #     return self

    def __init__(self, data : tuple, metric_name : str, apply : callable):

        legend : list = data[1][0][1][0].keys()
        steps : np.ndarray = np.unique([e[0] for e in data[1]])
        new_data : list = [[] for _ in steps]
        
        index_steps : int = 0
        previous : float = steps[0]
        
        for elem in data[1]:
            if previous != elem[0]:
                index_steps += 1
                previous = steps[index_steps]

            if metric_name == "BERTScore":
                new_data[index_steps].append(elem[1][0])

            if metric_name == "BLEURT":
                new_data[index_steps].append(elem[1][1])

            if metric_name == "ME":
                new_data[index_steps].append(elem[1][2])

        new_data : list = zip(steps, new_data)
        print(list(new_data))

        for step in new_data:
            tmp : list = step[1]
            cumulated : list = []
            # TODO
            for e in tmp:
                for name in legend:
                    cumulated += e[legend]



example_data : tuple = ('word_drop',
 [(0.5,
   ({'P': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9961, 1.0000,
             0.9896, 1.0000, 0.9866, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9941, 0.9889],
     'R': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9932, 1.0000,
             0.9829, 1.0000, 0.9770, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9883, 0.9818],
     'F1': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9947, 1.0000,
             0.9862, 1.0000, 0.9818, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9912, 0.9854]},
    {'BLEURT': [0.8784090280532837,
      0.9097437262535095,
      0.9119447469711304,
      0.9101507663726807,
      0.8474463224411011,
      0.913699209690094,
      0.9103060960769653,
      0.6871554851531982,
      0.8657276630401611,
      0.48376816511154175,
      0.8905432224273682,
      0.460135817527771,
      0.8605005145072937,
      0.864649772644043,
      0.8711574673652649,
      0.9031224250793457,
      0.8720568418502808,
      0.8796214461326599,
      0.8731204271316528,
      0.4222739338874817,
      0.6011916995048523]},
    {'Peterson': 0.9994047619047619,
     'Schnabel': 0.9864864864864865,
     'CAPTURE': 0.9761904761904762},
    [20, 11, 9, 19, 7])),
  (0.5,
   ({'P': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9961, 1.0000,
             0.9896, 1.0000, 0.9866, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9941, 0.9889],
     'R': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9932, 1.0000,
             0.9829, 1.0000, 0.9770, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9883, 0.9818],
     'F1': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9947, 1.0000,
             0.9862, 1.0000, 0.9818, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9912, 0.9854]},
    {'BLEURT': [0.8784090280532837,
      0.9097437262535095,
      0.9119447469711304,
      0.9101507663726807,
      0.8474463224411011,
      0.913699209690094,
      0.9103060960769653,
      0.6871554851531982,
      0.8657276630401611,
      0.48376816511154175,
      0.8905432224273682,
      0.460135817527771,
      0.8605005145072937,
      0.864649772644043,
      0.8711574673652649,
      0.9031224250793457,
      0.8720568418502808,
      0.8796214461326599,
      0.8731204271316528,
      0.4222739338874817,
      0.6011916995048523]},
    {'Peterson': 0.9994047619047619,
     'Schnabel': 0.9864864864864865,
     'CAPTURE': 0.9761904761904762},
    [20, 11, 9, 19, 7])),
  (1.0,
   ({'P': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9961, 1.0000,
             0.9896, 1.0000, 0.9866, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9941, 0.9889],
     'R': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9932, 1.0000,
             0.9829, 1.0000, 0.9770, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9883, 0.9818],
     'F1': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9947, 1.0000,
             0.9862, 1.0000, 0.9818, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9912, 0.9854]},
    {'BLEURT': [0.8784090280532837,
      0.9097437262535095,
      0.9119447469711304,
      0.9101507663726807,
      0.8474463224411011,
      0.913699209690094,
      0.9103060960769653,
      0.6871554851531982,
      0.8657276630401611,
      0.48376816511154175,
      0.8905432224273682,
      0.460135817527771,
      0.8605005145072937,
      0.864649772644043,
      0.8711574673652649,
      0.9031224250793457,
      0.8720568418502808,
      0.8796214461326599,
      0.8731204271316528,
      0.4222739338874817,
      0.6011916995048523]},
    {'Peterson': 0.9994047619047619,
     'Schnabel': 0.9864864864864865,
     'CAPTURE': 0.9761904761904762},
    [20, 11, 9, 19, 7])),
  (1.0,
   ({'P': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9961, 1.0000,
             0.9896, 1.0000, 0.9866, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9941, 0.9889],
     'R': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9932, 1.0000,
             0.9829, 1.0000, 0.9770, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9883, 0.9818],
     'F1': [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9947, 1.0000,
             0.9862, 1.0000, 0.9818, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
             1.0000, 0.9912, 0.9854]},
    {'BLEURT': [0.8784090280532837,
      0.9097437262535095,
      0.9119447469711304,
      0.9101507663726807,
      0.8474463224411011,
      0.913699209690094,
      0.9103060960769653,
      0.6871554851531982,
      0.8657276630401611,
      0.48376816511154175,
      0.8905432224273682,
      0.460135817527771,
      0.8605005145072937,
      0.864649772644043,
      0.8711574673652649,
      0.9031224250793457,
      0.8720568418502808,
      0.8796214461326599,
      0.8731204271316528,
      0.4222739338874817,
      0.6011916995048523]},
    {'Peterson': 0.9994047619047619,
     'Schnabel': 0.9864864864864865,
     'CAPTURE': 0.9761904761904762},
    [20, 11, 9, 19, 7]))])

if __name__ == "__main__":
    # for e in example_data[1]:
    #     data = e[1]
    #     indices = data[len(data) - 1]
    #     print(indices)
    #     test0 = VisData(e[0], data[0], indices, 'BLEURT').reduce(Apply.apply_BLEURT_det_only)
    #     test1 = VisData(e[0], data[0], indices, 'BLEURT').reduce(Apply.apply_BLEURT_all)
    #     print(test0.scatter)
    #     print(test1.scatter)

    test = VisData(example_data, 'BERTScore', lambda x : x)