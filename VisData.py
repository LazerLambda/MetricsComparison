from __future__ import annotations

import functools
import numpy as np
import json

from copy import deepcopy
from itertools import chain
from Apply import Apply

class VisData:

        def __init__(self, data : tuple, metric_name : str, apply : callable):

                # initialization 
                steps : np.ndarray = np.unique([e[0] for e in data[1]])
                new_data : list = lambda : {
                        'data' : [None for step in steps],
                        'min' : [None for step in steps],
                        'max' : [None for step in steps],
                        'mean' : [None for step in steps],
                        'median' : [None for step in steps]
                        }
                
                self.perturb_type : str = data[0]
                data = data[1]

                self.combined_data : dict = {
                        'BERTScore' : new_data(),
                        'BERTScore_f' : new_data(),
                        'BLEURT' : new_data(),
                        'BLEURT_f' : new_data(),
                        'ME' : new_data()
                }

                con_BERTScore   = lambda acc, elem : {\
                        'P' : acc['P'] + elem[0]['P'],\
                        'R' : acc['R'] + elem[0]['R'],\
                        'F1' : acc['F1'] + elem[0]['F1']}

                con_BERTScore_f  = lambda acc, elem : {\
                        'P' : np.concatenate(
                                (acc['P'], np.asarray(elem[0]['P'])[np.asarray(elem[3])] )
                                ),\
                        'R' : np.concatenate(
                                (acc['R'], np.asarray(elem[0]['R'])[np.asarray(elem[3])] )
                                ),\
                        'F1' : np.concatenate(
                                (acc['F1'], np.asarray(elem[0]['F1'])[np.asarray(elem[3])])
                                )
                        }

                con_BLEURT = lambda acc, elem : {\
                        'BLEURT': acc['BLEURT'] + elem[1]['BLEURT']
                        }

                con_BLEURT_f = lambda acc, elem : {\
                        'BLEURT': np.concatenate((acc['BLEURT'], np.asarray(elem[1]['BLEURT'])[np.asarray(elem[3])]))
                        }

                con_ME = lambda acc, elem : {\
                        'Petersen': acc['Petersen'] + [elem[2]['Petersen']],\
                        'Schnabel': acc['Schnabel'] + [elem[2]['Schnabel']],\
                        'CAPTURE': acc['CAPTURE'] + [elem[2]['CAPTURE']] 
                        }

                for i, degree in enumerate(steps):

                        # accumulate data
                        self.combined_data['BERTScore']['data'][i] = (degree, functools.reduce(con_BERTScore, data[i][1], {'P': [], 'R' : [], 'F1' : []}))
                        self.combined_data['BERTScore_f']['data'][i] = (degree, functools.reduce(con_BERTScore_f, data[i][1], {'P': [], 'R' : [], 'F1' : []}))
                        self.combined_data['BLEURT']['data'][i] = (degree, functools.reduce(con_BLEURT, data[i][1], {'BLEURT' : []}))
                        self.combined_data['BLEURT_f']['data'][i] = (degree, functools.reduce(con_BLEURT_f, data[i][1], {'BLEURT' : []}))
                        self.combined_data['ME']['data'][i] = (degree, functools.reduce(con_ME, data[i][1], {'Petersen' : [], 'Schnabel' : [], 'CAPTURE' : []}))

                        # mean
                        self.combined_data['BERTScore']['mean'][i] = (degree, {\
                                'P' : np.average(np.asarray(self.combined_data['BERTScore']['data'][i][1]['P'])),
                                'R' : np.average(np.asarray(self.combined_data['BERTScore']['data'][i][1]['R'])),
                                'F1': np.average(np.asarray(self.combined_data['BERTScore']['data'][i][1]['F1']))
                                })
                        self.combined_data['BERTScore_f']['mean'][i] = (degree, {\
                                'P' : np.average(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['P'])),
                                'R' : np.average(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['R'])),
                                'F1': np.average(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['F1']))
                                })
                        self.combined_data['BLEURT']['mean'][i] = (degree, {
                                'BLEURT' : np.average(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['BLEURT_f']['mean'][i] = (degree, {
                                'BLEURT' : np.average(np.asarray(self.combined_data['BLEURT_f']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['ME']['mean'][i] = (degree, {
                                'Petersen' : np.average(np.asarray(self.combined_data['ME']['data'][i][1]['Petersen'])),
                                'Schnabel' : np.average(np.asarray(self.combined_data['ME']['data'][i][1]['Schnabel'])),
                                'CAPTURE' : np.average(np.asarray(self.combined_data['ME']['data'][i][1]['CAPTURE']))
                                })

                        # minima
                        self.combined_data['BERTScore']['min'][i] = (degree, {\
                                'P' : np.min(np.asarray(self.combined_data['BERTScore']['data'][i][1]['P'])),
                                'R' : np.min(np.asarray(self.combined_data['BERTScore']['data'][i][1]['R'])),
                                'F1' : np.min(np.asarray(self.combined_data['BERTScore']['data'][i][1]['F1']))
                                })                     
                        self.combined_data['BERTScore_f']['min'][i] = (degree, {\
                                'P' : np.min(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['P'])),
                                'R' : np.min(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['R'])),
                                'F1' : np.min(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['F1']))
                                })
                        self.combined_data['BLEURT']['min'][i] = (degree, {
                                'BLEURT' : np.min(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['BLEURT_f']['min'][i] = (degree, {
                                'BLEURT' : np.min(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['ME']['min'][i] = (degree, {
                                'Petersen' : np.min(np.asarray(self.combined_data['ME']['data'][i][1]['Petersen'])),
                                'Schnabel' : np.min(np.asarray(self.combined_data['ME']['data'][i][1]['Schnabel'])),
                                'CAPTURE' : np.min(np.asarray(self.combined_data['ME']['data'][i][1]['CAPTURE']))
                                })

                        # maxima
                        self.combined_data['BERTScore']['max'][i] = (degree, {\
                                'P' : np.max(np.asarray(self.combined_data['BERTScore']['data'][i][1]['P'])),
                                'R' : np.max(np.asarray(self.combined_data['BERTScore']['data'][i][1]['R'])),
                                'F1' : np.max(np.asarray(self.combined_data['BERTScore']['data'][i][1]['F1']))
                                })                     
                        self.combined_data['BERTScore_f']['max'][i] = (degree, {\
                                'P' : np.max(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['P'])),
                                'R' : np.max(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['R'])),
                                'F1' : np.max(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['F1']))
                                })
                        self.combined_data['BLEURT']['max'][i] = (degree, {
                                'BLEURT' : np.max(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['BLEURT_f']['max'][i] = (degree, {
                                'BLEURT' : np.max(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['ME']['max'][i] = (degree, {
                                'Petersen' : np.max(np.asarray(self.combined_data['ME']['data'][i][1]['Petersen'])),
                                'Schnabel' : np.max(np.asarray(self.combined_data['ME']['data'][i][1]['Schnabel'])),
                                'CAPTURE' : np.max(np.asarray(self.combined_data['ME']['data'][i][1]['CAPTURE']))
                                })

                        # median
                        self.combined_data['BERTScore']['median'][i] = (degree, {\
                                'P' :  np.median(np.asarray(self.combined_data['BERTScore']['data'][i][1]['P'])),
                                'R' :  np.median(np.asarray(self.combined_data['BERTScore']['data'][i][1]['R'])),
                                'F1' : np.median(np.asarray(self.combined_data['BERTScore']['data'][i][1]['F1']))
                                })                     
                        self.combined_data['BERTScore_f']['median'][i] = (degree, {\
                                'P' :  np.median(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['P'])),
                                'R' :  np.median(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['R'])),
                                'F1' : np.median(np.asarray(self.combined_data['BERTScore_f']['data'][i][1]['F1']))
                                })
                        self.combined_data['BLEURT']['median'][i] = (degree, {
                                'BLEURT' : np.median(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['BLEURT_f']['median'][i] = (degree, {
                                'BLEURT' : np.median(np.asarray(self.combined_data['BLEURT']['data'][i][1]['BLEURT']))
                                })
                        self.combined_data['ME']['median'][i] = (degree, {
                                'Petersen' : np.median(np.asarray(self.combined_data['ME']['data'][i][1]['Petersen'])),
                                'Schnabel' : np.median(np.asarray(self.combined_data['ME']['data'][i][1]['Schnabel'])),
                                'CAPTURE' :  np.median(np.asarray(self.combined_data['ME']['data'][i][1]['CAPTURE']))
                                })
                
                self.steps = steps

        
        

        def export_results(self):
                results : dict = deepcopy(self.combined_data)
                results_json : dict = dict()
                for key in results.keys():
                        results_json[key] = results[key]
                        del results_json[key]['data']

                file_results = open('results.json', 'w') 
                json.dump(results_json, file_results)
                del results
                del results_json






example_data : tuple = ('word_drop',
 [(0.5,
   [({'P': [0.9770, 0.9760, 0.9803, 0.9295, 0.9693, 1.0000, 1.0000, 0.9908, 0.9851,
              1.0000, 1.0000, 0.9840, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
              0.9745, 1.0000, 0.9865, 0.9928, 0.9849, 1.0000, 1.0000, 0.9828, 1.0000,
              1.0000, 0.9736, 0.9877, 1.0000, 1.0000, 1.0000, 0.9693, 0.9941, 0.9533,
              1.0000, 1.0000, 1.0000, 0.9404, 0.9888, 0.9946, 0.9729, 1.0000, 1.0000,
              1.0000, 1.0000, 0.9740, 1.0000, 1.0000, 0.9883, 1.0000, 0.9699, 0.9510,
              1.0000, 1.0000],
      'R': [0.9764, 0.9808, 0.9844, 0.9622, 0.9817, 1.0000, 1.0000, 0.9941, 0.9924,
              1.0000, 1.0000, 0.9896, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
              0.9838, 1.0000, 0.9873, 0.9964, 0.9933, 1.0000, 1.0000, 0.9868, 1.0000,
              1.0000, 0.9852, 0.9926, 1.0000, 1.0000, 1.0000, 0.9757, 0.9937, 0.9599,
              1.0000, 1.0000, 1.0000, 0.9707, 0.9950, 0.9946, 0.9781, 1.0000, 1.0000,
              1.0000, 1.0000, 0.9805, 1.0000, 1.0000, 0.9879, 1.0000, 0.9844, 0.9752,
              1.0000, 1.0000],
      'F1': [0.9767, 0.9784, 0.9823, 0.9456, 0.9754, 1.0000, 1.0000, 0.9924, 0.9887,
              1.0000, 1.0000, 0.9868, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
              0.9791, 1.0000, 0.9869, 0.9946, 0.9891, 1.0000, 1.0000, 0.9848, 1.0000,
              1.0000, 0.9793, 0.9901, 1.0000, 1.0000, 1.0000, 0.9725, 0.9939, 0.9566,
              1.0000, 1.0000, 1.0000, 0.9553, 0.9919, 0.9946, 0.9755, 1.0000, 1.0000,
              1.0000, 1.0000, 0.9773, 1.0000, 1.0000, 0.9881, 1.0000, 0.9771, 0.9629,
              1.0000, 1.0000]},
     {'BLEURT': [0.871826708316803,
       0.8672036528587341,
       0.8317148089408875,
       0.2279856950044632,
       0.484746515750885,
       0.9054884910583496,
       0.8706612586975098,
       0.9189292788505554,
       0.806484580039978,
       0.8625785112380981,
       0.8891386389732361,
       0.8739809989929199,
       0.877536416053772,
       0.8891146183013916,
       0.8731096982955933,
       0.8439398407936096,
       0.8668380975723267,
       0.8582901954650879,
       0.9021390080451965,
       0.9110325574874878,
       0.7860539555549622,
       0.8623540997505188,
       0.6594355702400208,
       0.8602033257484436,
       0.8995091915130615,
       0.8637398481369019,
       0.9052300453186035,
       0.9265645742416382,
       0.8875113129615784,
       0.8379246592521667,
       0.9319407939910889,
       0.9183927774429321,
       0.867972195148468,
       0.904548704624176,
       0.8524854779243469,
       0.7100487947463989,
       0.8601565361022949,
       0.8937122821807861,
       0.8686271905899048,
       0.9379301071166992,
       0.8163343667984009,
       0.8553259968757629,
       0.8587245941162109,
       0.9290639162063599,
       0.907575786113739,
       0.8503203988075256,
       0.9089936017990112,
       0.6943812966346741,
       0.8928029537200928,
       0.8815011978149414,
       0.8396464586257935,
       0.8608013987541199,
       0.5925960540771484,
       0.6688219308853149,
       0.9113960862159729,
       0.8971254825592041]},
     {'Petersen': 0.9910714285714286,
      'Schnabel': 0.9910714285714286,
      'CAPTURE': 0.9821428571428571},
     [0,
      1,
      2,
      3,
      4,
      7,
      8,
      11,
      18,
      20,
      21,
      22,
      25,
      28,
      29,
      33,
      34,
      35,
      39,
      40,
      41,
      42,
      47,
      50,
      52,
      53]),
    ({'P': [1.0000, 1.0000, 0.9810, 1.0000, 1.0000, 1.0000, 0.9904, 1.0000, 0.9835,
              1.0000, 1.0000, 0.9747, 1.0000, 0.9007, 0.9892, 0.9820, 0.9835, 0.9879,
              0.9823, 1.0000],
      'R': [1.0000, 1.0000, 0.9796, 1.0000, 1.0000, 1.0000, 0.9936, 1.0000, 0.9896,
              1.0000, 1.0000, 0.9772, 1.0000, 0.9495, 0.9920, 0.9850, 0.9868, 0.9887,
              0.9812, 1.0000],
      'F1': [1.0000, 1.0000, 0.9803, 1.0000, 1.0000, 1.0000, 0.9920, 1.0000, 0.9865,
              1.0000, 1.0000, 0.9759, 1.0000, 0.9244, 0.9906, 0.9835, 0.9851, 0.9883,
              0.9818, 1.0000]},
     {'BLEURT': [0.8540278673171997,
       0.8415284156799316,
       0.8557943105697632,
       0.848135232925415,
       0.9079397320747375,
       0.8773334622383118,
       0.8059221506118774,
       0.8609747886657715,
       0.8664944767951965,
       0.8454383611679077,
       0.8974379301071167,
       0.6769760251045227,
       0.8642311096191406,
       0.35805267095565796,
       0.8710559606552124,
       0.870672881603241,
       0.8561930656433105,
       0.8648319244384766,
       0.8489693403244019,
       0.8486870527267456]},
     {'Petersen': 1.0, 'Schnabel': 1.0, 'CAPTURE': 0.975},
     [2, 6, 8, 11, 13, 14, 15, 16, 17, 18])]),
  (1.0,
   [({'P': [0.9832, 0.9787, 0.9803, 0.9282, 0.9892, 0.9895, 0.9913, 0.9865, 0.9950,
              0.9840, 0.9848, 0.9827, 0.9987, 0.9860, 0.9841, 0.9954, 0.9626, 0.9886,
              0.9723, 0.9874, 0.9781, 0.9849, 0.9872, 0.9841, 0.9773, 0.9904, 0.9285,
              0.9479, 0.9922, 0.9761, 0.8868, 0.9808, 0.9798, 0.9795, 0.9808, 0.9554,
              0.9830, 0.9896, 0.9915, 0.9444, 0.9700, 0.9931, 0.9862, 0.9497, 0.9829,
              0.9877, 0.9744, 0.9949, 0.9869, 0.9836, 0.9787, 0.9925, 0.9699, 0.9871,
              0.9552, 0.9870],
      'R': [0.9877, 0.9868, 0.9844, 0.9612, 0.9972, 0.9942, 0.9958, 0.9936, 0.9969,
              0.9828, 0.9859, 0.9905, 0.9988, 0.9940, 0.9892, 0.9979, 0.9673, 0.9896,
              0.9737, 0.9954, 0.9940, 0.9903, 0.9883, 0.9853, 0.9857, 0.9916, 0.9312,
              0.9758, 0.9968, 0.9750, 0.9439, 0.9894, 0.9863, 0.9863, 0.9817, 0.9736,
              0.9831, 0.9965, 0.9909, 0.9863, 0.9726, 0.9936, 0.9935, 0.9859, 0.9823,
              0.9898, 0.9729, 0.9959, 0.9906, 0.9865, 0.9840, 0.9954, 0.9844, 0.9862,
              0.9767, 0.9926],
      'F1': [0.9854, 0.9827, 0.9823, 0.9444, 0.9932, 0.9919, 0.9935, 0.9900, 0.9959,
              0.9834, 0.9853, 0.9866, 0.9988, 0.9900, 0.9866, 0.9966, 0.9650, 0.9891,
              0.9730, 0.9914, 0.9860, 0.9876, 0.9878, 0.9847, 0.9815, 0.9910, 0.9299,
              0.9617, 0.9945, 0.9756, 0.9144, 0.9851, 0.9830, 0.9829, 0.9812, 0.9644,
              0.9831, 0.9930, 0.9912, 0.9649, 0.9713, 0.9934, 0.9898, 0.9675, 0.9826,
              0.9888, 0.9737, 0.9954, 0.9887, 0.9851, 0.9813, 0.9940, 0.9771, 0.9867,
              0.9658, 0.9898]},
     {'BLEURT': [0.624107837677002,
       0.8156464695930481,
       0.8317148089408875,
       0.47416603565216064,
       0.6807964444160461,
       0.9100348353385925,
       0.790560781955719,
       0.8817466497421265,
       0.8948906064033508,
       0.8441402912139893,
       0.7496142983436584,
       0.7678824067115784,
       0.8792828321456909,
       0.8653877973556519,
       0.8249264359474182,
       0.8442618250846863,
       0.6416317224502563,
       0.8594216704368591,
       0.8713157176971436,
       0.877511739730835,
       0.6908235549926758,
       0.8059108853340149,
       0.8639296293258667,
       0.8435894250869751,
       0.8561351299285889,
       0.8587804436683655,
       0.8871808648109436,
       0.5544015169143677,
       0.828617513179779,
       0.8580859303474426,
       0.33419787883758545,
       0.7351886630058289,
       0.7506522536277771,
       0.9085596203804016,
       0.8576169013977051,
       0.8981044292449951,
       0.8567588329315186,
       0.8560529351234436,
       0.5980199575424194,
       0.9245481491088867,
       0.6152366995811462,
       0.8648635149002075,
       0.8718438744544983,
       0.9300558567047119,
       0.8833438754081726,
       0.8509871363639832,
       0.905039370059967,
       0.9016883969306946,
       0.8949985504150391,
       0.8827753663063049,
       0.7280495762825012,
       0.8600717782974243,
       0.5925960540771484,
       0.8714327812194824,
       0.6941704750061035,
       0.8998167514801025]},
     {'Petersen': 0.9910714285714286,
      'Schnabel': 0.9989177489177489,
      'CAPTURE': 0.9821428571428571},
     [0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55]),
    ({'P': [0.9713, 0.9933, 0.9895, 0.9963, 0.8927, 0.9840, 0.9913, 0.9908, 0.9907,
              0.9808, 0.9659, 0.9825, 0.9788, 0.9408, 0.9955, 0.9836, 0.9836, 0.9789,
              0.9886, 0.9698],
      'R': [0.9731, 0.9967, 0.9915, 0.9979, 0.9506, 0.9867, 0.9945, 0.9926, 0.9926,
              0.9802, 0.9751, 0.9855, 0.9913, 0.9843, 0.9979, 0.9885, 0.9853, 0.9835,
              0.9848, 0.9712],
      'F1': [0.9722, 0.9950, 0.9905, 0.9971, 0.9208, 0.9854, 0.9929, 0.9917, 0.9916,
              0.9805, 0.9704, 0.9840, 0.9850, 0.9620, 0.9967, 0.9860, 0.9844, 0.9812,
              0.9867, 0.9705]},
     {'BLEURT': [0.7738461494445801,
       0.8129750490188599,
       0.8620864748954773,
       0.8448581695556641,
       0.3093060851097107,
       0.8681425452232361,
       0.7851169109344482,
       0.8619651794433594,
       0.882185697555542,
       0.6573868989944458,
       0.6604447364807129,
       0.8457338809967041,
       0.7876018285751343,
       0.6600014567375183,
       0.8441046476364136,
       0.8343193531036377,
       0.7939693927764893,
       0.7490243911743164,
       0.8543717861175537,
       0.8290738463401794]},
     {'Petersen': 1.0, 'Schnabel': 1.0, 'CAPTURE': 0.975},
     [0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19])])])


if __name__ == "__main__":
        test = VisData(example_data, 'BERTScore', lambda x : x)
        assert len(example_data[1][0][1][0][0]['P'] + example_data[1][0][1][1][0]['P']) == len(test.combined_data['BERTScore']['data'][0][1]['P'])
        assert len(example_data[1][0][1][0][0]['R'] + example_data[1][0][1][1][0]['R']) == len(test.combined_data['BERTScore']['data'][0][1]['R'])
        assert len(example_data[1][0][1][0][0]['F1'] + example_data[1][0][1][1][0]['F1']) == len(test.combined_data['BERTScore']['data'][0][1]['F1'])
        # print(test.combined_data)
        test.export_results()