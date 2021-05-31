


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import src.Consts
import numpy as np
import torch
import unittest

from src.ReduceData import ReduceData
from src.Metrics import Metrics as mtrc



class ReduceData_Test(unittest.TestCase):

      example_data: tuple = ('word_drop',
            [(0.5,
                  [({'P': torch.Tensor([0.9770, 0.9760, 0.9803, 0.9295, 0.9693, 1.0000, 1.0000, 0.9908, 0.9851,
                        1.0000, 1.0000, 0.9840, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                        0.9745, 1.0000, 0.9865, 0.9928, 0.9849, 1.0000, 1.0000, 0.9828, 1.0000,
                        1.0000, 0.9736, 0.9877, 1.0000, 1.0000, 1.0000, 0.9693, 0.9941, 0.9533,
                        1.0000, 1.0000, 1.0000, 0.9404, 0.9888, 0.9946, 0.9729, 1.0000, 1.0000,
                        1.0000, 1.0000, 0.9740, 1.0000, 1.0000, 0.9883, 1.0000, 0.9699, 0.9510,
                        1.0000, 1.0000]),
                  'R': torch.Tensor([0.9764, 0.9808, 0.9844, 0.9622, 0.9817, 1.0000, 1.0000, 0.9941, 0.9924,
                        1.0000, 1.0000, 0.9896, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                        0.9838, 1.0000, 0.9873, 0.9964, 0.9933, 1.0000, 1.0000, 0.9868, 1.0000,
                        1.0000, 0.9852, 0.9926, 1.0000, 1.0000, 1.0000, 0.9757, 0.9937, 0.9599,
                        1.0000, 1.0000, 1.0000, 0.9707, 0.9950, 0.9946, 0.9781, 1.0000, 1.0000,
                        1.0000, 1.0000, 0.9805, 1.0000, 1.0000, 0.9879, 1.0000, 0.9844, 0.9752,
                        1.0000, 1.0000]),
                  'F1': torch.Tensor([0.9767, 0.9784, 0.9823, 0.9456, 0.9754, 1.0000, 1.0000, 0.9924, 0.9887,
                        1.0000, 1.0000, 0.9868, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                        0.9791, 1.0000, 0.9869, 0.9946, 0.9891, 1.0000, 1.0000, 0.9848, 1.0000,
                        1.0000, 0.9793, 0.9901, 1.0000, 1.0000, 1.0000, 0.9725, 0.9939, 0.9566,
                        1.0000, 1.0000, 1.0000, 0.9553, 0.9919, 0.9946, 0.9755, 1.0000, 1.0000,
                        1.0000, 1.0000, 0.9773, 1.0000, 1.0000, 0.9881, 1.0000, 0.9771, 0.9629,
                        1.0000, 1.0000])},
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
                  ({'P': torch.Tensor([1.0000, 1.0000, 0.9810, 1.0000, 1.0000, 1.0000, 0.9904, 1.0000, 0.9835,
                        1.0000, 1.0000, 0.9747, 1.0000, 0.9007, 0.9892, 0.9820, 0.9835, 0.9879,
                        0.9823, 1.0000]),
                  'R': torch.Tensor([1.0000, 1.0000, 0.9796, 1.0000, 1.0000, 1.0000, 0.9936, 1.0000, 0.9896,
                        1.0000, 1.0000, 0.9772, 1.0000, 0.9495, 0.9920, 0.9850, 0.9868, 0.9887,
                        0.9812, 1.0000]),
                  'F1': torch.Tensor([1.0000, 1.0000, 0.9803, 1.0000, 1.0000, 1.0000, 0.9920, 1.0000, 0.9865,
                        1.0000, 1.0000, 0.9759, 1.0000, 0.9244, 0.9906, 0.9835, 0.9851, 0.9883,
                        0.9818, 1.0000])},
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
                  [({'P': torch.Tensor([0.9832, 0.9787, 0.9803, 0.9282, 0.9892, 0.9895, 0.9913, 0.9865, 0.9950,
                              0.9840, 0.9848, 0.9827, 0.9987, 0.9860, 0.9841, 0.9954, 0.9626, 0.9886,
                              0.9723, 0.9874, 0.9781, 0.9849, 0.9872, 0.9841, 0.9773, 0.9904, 0.9285,
                              0.9479, 0.9922, 0.9761, 0.8868, 0.9808, 0.9798, 0.9795, 0.9808, 0.9554,
                              0.9830, 0.9896, 0.9915, 0.9444, 0.9700, 0.9931, 0.9862, 0.9497, 0.9829,
                              0.9877, 0.9744, 0.9949, 0.9869, 0.9836, 0.9787, 0.9925, 0.9699, 0.9871,
                              0.9552, 0.9870]),
                        'R': torch.Tensor([0.9877, 0.9868, 0.9844, 0.9612, 0.9972, 0.9942, 0.9958, 0.9936, 0.9969,
                              0.9828, 0.9859, 0.9905, 0.9988, 0.9940, 0.9892, 0.9979, 0.9673, 0.9896,
                              0.9737, 0.9954, 0.9940, 0.9903, 0.9883, 0.9853, 0.9857, 0.9916, 0.9312,
                              0.9758, 0.9968, 0.9750, 0.9439, 0.9894, 0.9863, 0.9863, 0.9817, 0.9736,
                              0.9831, 0.9965, 0.9909, 0.9863, 0.9726, 0.9936, 0.9935, 0.9859, 0.9823,
                              0.9898, 0.9729, 0.9959, 0.9906, 0.9865, 0.9840, 0.9954, 0.9844, 0.9862,
                              0.9767, 0.9926]),
                        'F1': torch.Tensor([0.9854, 0.9827, 0.9823, 0.9444, 0.9932, 0.9919, 0.9935, 0.9900, 0.9959,
                              0.9834, 0.9853, 0.9866, 0.9988, 0.9900, 0.9866, 0.9966, 0.9650, 0.9891,
                              0.9730, 0.9914, 0.9860, 0.9876, 0.9878, 0.9847, 0.9815, 0.9910, 0.9299,
                              0.9617, 0.9945, 0.9756, 0.9144, 0.9851, 0.9830, 0.9829, 0.9812, 0.9644,
                              0.9831, 0.9930, 0.9912, 0.9649, 0.9713, 0.9934, 0.9898, 0.9675, 0.9826,
                              0.9888, 0.9737, 0.9954, 0.9887, 0.9851, 0.9813, 0.9940, 0.9771, 0.9867,
                              0.9658, 0.9898])},
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
                  ({'P': torch.Tensor([0.9713, 0.9933, 0.9895, 0.9963, 0.8927, 0.9840, 0.9913, 0.9908, 0.9907,
                              0.9808, 0.9659, 0.9825, 0.9788, 0.9408, 0.9955, 0.9836, 0.9836, 0.9789,
                              0.9886, 0.9698]),
                        'R': torch.Tensor([0.9731, 0.9967, 0.9915, 0.9979, 0.9506, 0.9867, 0.9945, 0.9926, 0.9926,
                              0.9802, 0.9751, 0.9855, 0.9913, 0.9843, 0.9979, 0.9885, 0.9853, 0.9835,
                              0.9848, 0.9712]),
                        'F1': torch.Tensor([0.9722, 0.9950, 0.9905, 0.9971, 0.9208, 0.9854, 0.9929, 0.9917, 0.9916,
                              0.9805, 0.9704, 0.9840, 0.9850, 0.9620, 0.9967, 0.9860, 0.9844, 0.9812,
                              0.9867, 0.9705])},
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
    
      def test_general(self):
            test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE])
            task_list = [
                  "negated",
                  "pos_drop_adj",
                  "pos_drop_det",
                  "repetitions",
                  "word_drop",
                  "word_drop_every_sentence",
                  "word_swap",
                  "word_swap_every_sentence"]
            test.add_data(task_list, folder_path="src/output_9/")
            test.export_results()
            test.vis_ready()


      def test_len(self):
            test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE, mtrc.ME])
            test.__add_data__(self.example_data)

            len_fun_data_p_BERTScore = len(test.data['word_drop']['BERTScore']['data'][0][1]['P'])
            len_fun_data_r_BERTScore = len(test.data['word_drop']['BERTScore']['data'][0][1]['R'])
            len_fun_data_f1_BERTScore = len(test.data['word_drop']['BERTScore']['data'][0][1]['F1'])

            len_fun_data_BLEURT= len(test.data['word_drop']['BLEURT']['data'][0][1]['BLEURT'])

            len_fun_data_Schnabel_ME = len(test.data['word_drop']['ME']['data'][0][1]['Schnabel'])
            len_fun_data_Petersen_ME = len(test.data['word_drop']['ME']['data'][0][1]['Petersen'])
            len_fun_data_CAPTURE_ME = len(test.data['word_drop']['ME']['data'][0][1]['CAPTURE'])

            len_ex_data_p_BERTScore = len(self.example_data[1][0][1][0][0]['P'].tolist() + self.example_data[1][0][1][1][0]['P'].tolist())
            len_ex_data_r_BERTScore = len(self.example_data[1][0][1][0][0]['R'].tolist() + self.example_data[1][0][1][1][0]['R'].tolist())
            len_ex_data_f1_BERTScore = len(self.example_data[1][0][1][0][0]['F1'].tolist() + self.example_data[1][0][1][1][0]['F1'].tolist())

            len_ex_data_BLEURT = len(self.example_data[1][0][1][0][1]['BLEURT'] + self.example_data[1][0][1][1][1]['BLEURT'])
           
            len_ex_data_Schnabel_ME = len([self.example_data[1][0][1][0][2]['Schnabel']] + [self.example_data[1][0][1][1][2]['Schnabel']])
            len_ex_data_Petersen_ME = len([self.example_data[1][0][1][0][2]['Petersen']] + [self.example_data[1][0][1][1][2]['Petersen']])
            len_ex_data_CAPTURE_ME = len([self.example_data[1][0][1][0][2]['CAPTURE']] + [self.example_data[1][0][1][1][2]['CAPTURE']])

            self.assertEqual(len_fun_data_f1_BERTScore, len_ex_data_f1_BERTScore, msg="Check length of arrays, BERTScore F1")
            self.assertEqual(len_fun_data_r_BERTScore, len_ex_data_r_BERTScore, msg="Check length of arrays, BERTScore R")
            self.assertEqual(len_fun_data_p_BERTScore, len_ex_data_p_BERTScore, msg="Check length of arrays, BERTScore P")

            self.assertEqual(len_fun_data_BLEURT, len_ex_data_BLEURT, msg="Check length of arrays, BLEURT")

            self.assertEqual(len_fun_data_Schnabel_ME , len_ex_data_Schnabel_ME, msg="Check length of arrays, ME Schnabel")
            self.assertEqual(len_fun_data_Petersen_ME , len_ex_data_Petersen_ME, msg="Check length of arrays, ME Petersen")
            self.assertEqual(len_fun_data_CAPTURE_ME , len_ex_data_CAPTURE_ME, msg="Check length of arrays, ME CAPTURE")

      def test_len_f(self):
            test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE, mtrc.ME])
            test.__add_data__(self.example_data)

            indices_0 = np.asarray(self.example_data[1][0][1][0][3])
            indices_1 = np.asarray(self.example_data[1][0][1][1][3])

            len_fun_data_p_BERTScore = len(test.data['word_drop']['BERTScore_f']['data'][0][1]['P'])
            len_fun_data_r_BERTScore = len(test.data['word_drop']['BERTScore_f']['data'][0][1]['R'])
            len_fun_data_f1_BERTScore = len(test.data['word_drop']['BERTScore_f']['data'][0][1]['F1'])

            # print(test.data['word_drop']['BERTScore_f']['data'][0][1]['F1'])

            len_fun_data_BLEURT= len(test.data['word_drop']['BLEURT_f']['data'][0][1]['BLEURT'])

            len_ex_data_p_BERTScore =  len(np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['P'].tolist())[indices_0],   np.asarray(self.example_data[1][0][1][1][0]['P'].tolist())[indices_1])))
            len_ex_data_r_BERTScore =  len(np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['R'].tolist())[indices_0],   np.asarray(self.example_data[1][0][1][1][0]['R'].tolist())[indices_1])))
            len_ex_data_f1_BERTScore = len(np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['F1'].tolist())[indices_0], np.asarray(self.example_data[1][0][1][1][0]['F1'].tolist())[indices_1])))

            len_ex_data_BLEURT = len(np.concatenate((np.asarray(self.example_data[1][0][1][0][1]['BLEURT'])[indices_0], np.asarray(self.example_data[1][0][1][1][1]['BLEURT'])[indices_1])))

            self.assertEqual(len_fun_data_f1_BERTScore, len_ex_data_f1_BERTScore, msg="Check length of arrays, BERTScore F1")
            self.assertEqual(len_fun_data_r_BERTScore, len_ex_data_r_BERTScore, msg="Check length of arrays, BERTScore R")
            self.assertEqual(len_fun_data_p_BERTScore, len_ex_data_p_BERTScore, msg="Check length of arrays, BERTScore P")

            self.assertEqual(len_fun_data_BLEURT, len_ex_data_BLEURT, msg="Check length of arrays, BLEURT")

      def test_avg(self):
            test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE, mtrc.ME])
            test.__add_data__(self.example_data)

            ex_data_p_BERTScore = self.example_data[1][0][1][0][0]['P'].tolist() + self.example_data[1][0][1][1][0]['P'].tolist()
            ex_data_r_BERTScore = self.example_data[1][0][1][0][0]['R'].tolist() + self.example_data[1][0][1][1][0]['R'].tolist()
            ex_data_f1_BERTScore = self.example_data[1][0][1][0][0]['F1'].tolist() + self.example_data[1][0][1][1][0]['F1'].tolist()

            ex_data_BLEURT = self.example_data[1][0][1][0][1]['BLEURT'] + self.example_data[1][0][1][1][1]['BLEURT']
           
            ex_data_Schnabel_ME = [self.example_data[1][0][1][0][2]['Schnabel']] + [self.example_data[1][0][1][1][2]['Schnabel']]
            ex_data_Petersen_ME = [self.example_data[1][0][1][0][2]['Petersen']] + [self.example_data[1][0][1][1][2]['Petersen']]
            ex_data_CAPTURE_ME = [self.example_data[1][0][1][0][2]['CAPTURE']] + [self.example_data[1][0][1][1][2]['CAPTURE']]

            self.assertEqual(np.average(np.asarray(ex_data_p_BERTScore)), test.data['word_drop']['BERTScore']['mean'][0][1]['P'], msg="Check length of arrays, BERTScore P")
            self.assertEqual(np.average(np.asarray(ex_data_r_BERTScore)), test.data['word_drop']['BERTScore']['mean'][0][1]['R'], msg="Check length of arrays, BERTScore R")
            self.assertEqual(np.average(np.asarray(ex_data_f1_BERTScore)), test.data['word_drop']['BERTScore']['mean'][0][1]['F1'], msg="Check length of arrays, BERTScore F1")

            self.assertEqual(np.average(np.asarray(ex_data_BLEURT)), test.data['word_drop']['BLEURT']['mean'][0][1]['BLEURT'], msg="Check length of arrays, BLEURT")

            self.assertEqual(np.average(np.asarray(ex_data_Schnabel_ME)), test.data['word_drop']['ME']['mean'][0][1]['Schnabel'], msg="Check length of arrays, ME Schnabel")
            self.assertEqual(np.average(np.asarray(ex_data_Petersen_ME)), test.data['word_drop']['ME']['mean'][0][1]['Petersen'], msg="Check length of arrays, ME Petersen")
            self.assertEqual(np.average(np.asarray(ex_data_CAPTURE_ME)), test.data['word_drop']['ME']['mean'][0][1]['CAPTURE'], msg="Check length of arrays, ME CAPTURE")

      def test_avg_f(self):
            test = ReduceData([mtrc.BLEURT, mtrc.BERTSCORE, mtrc.ME])
            test.__add_data__(self.example_data)

            indices_0 = np.asarray(self.example_data[1][0][1][0][3])
            indices_1 = np.asarray(self.example_data[1][0][1][1][3])

            ex_data_p_BERTScore_f = np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['P'].tolist())[indices_0], np.asarray(self.example_data[1][0][1][1][0]['P'].tolist())[indices_1]))
            ex_data_r_BERTScore_f = np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['R'].tolist())[indices_0], np.asarray(self.example_data[1][0][1][1][0]['R'].tolist())[indices_1]))
            ex_data_f1_BERTScore_f = np.concatenate((np.asarray(self.example_data[1][0][1][0][0]['F1'].tolist())[indices_0], np.asarray(self.example_data[1][0][1][1][0]['F1'].tolist())[indices_1]))

            ex_data_BLEURT_f = np.concatenate((np.asarray(self.example_data[1][0][1][0][1]['BLEURT'])[indices_0], np.asarray(self.example_data[1][0][1][1][1]['BLEURT'])[indices_1]))


            self.assertEqual(np.average(np.asarray(ex_data_p_BERTScore_f)), test.data['word_drop']['BERTScore_f']['mean'][0][1]['P'], msg="Check length of arrays, BERTScore P")
            self.assertEqual(np.average(np.asarray(ex_data_r_BERTScore_f)), test.data['word_drop']['BERTScore_f']['mean'][0][1]['R'], msg="Check length of arrays, BERTScore R")
            self.assertEqual(np.average(np.asarray(ex_data_f1_BERTScore_f)), test.data['word_drop']['BERTScore_f']['mean'][0][1]['F1'], msg="Check length of arrays, BERTScore F1")

            self.assertEqual(np.average(np.asarray(ex_data_BLEURT_f)), test.data['word_drop']['BLEURT_f']['mean'][0][1]['BLEURT'], msg="Check length of arrays, BLEURT")


if __name__ == '__main__':
    unittest.main()