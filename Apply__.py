import numpy as np



class Apply:


    @staticmethod
    def apply_BERTScore_det_only(data : dict, indices : list) -> tuple:

        # (mean, min, median, max, scatter)
        return_tuple : tuple = ([], [], [], [], [])
        for key, values in data.items():

            # only with respect to the altered positions
            data_tmp : list = np.asarray(values)[indices]

            return_tuple[0].append(np.average(data_tmp))
            return_tuple[1].append(np.min(data_tmp))
            return_tuple[2].append(np.median(data_tmp))
            return_tuple[3].append(np.max(data_tmp))
            return_tuple[4].append(data_tmp)

        return return_tuple


    @staticmethod
    def apply_BERTScore_all(data : dict, indices : list) -> tuple:
        
        # (mean, min, median, max, scatter)
        return_tuple : tuple = ([], [], [], [], [])
        for key, values in data.items():

            data_tmp : list = np.asarray(values)

            return_tuple[0].append(np.average(data_tmp))
            return_tuple[1].append(np.min(data_tmp))
            return_tuple[2].append(np.median(data_tmp))
            return_tuple[3].append(np.max(data_tmp))
            return_tuple[4].append(data_tmp)

        return return_tuple


    @staticmethod
    def apply_ME(data : dict, indices : list) -> tuple:
        return ([data.values()], [data.values()], [data.values()], [data.values()])

    
    @staticmethod
    def apply_BLEURT_det_only(data : dict, indices : list) -> tuple:

        # (mean, min, median, max, scatter)
        return_tuple : tuple = ([], [], [], [], [])
        for key, values in data.items():
            # only with respect to the altered positions
            data_tmp : list = np.asarray(values)[indices]

            return_tuple[0].append(np.average(data_tmp))
            return_tuple[1].append(np.min(data_tmp))
            return_tuple[2].append(np.median(data_tmp))
            return_tuple[3].append(np.max(data_tmp))
            return_tuple[4].append(data_tmp)

        return return_tuple


    @staticmethod
    def apply_BLEURT_all(data : dict, indices : list) -> tuple:

        # (mean, min, median, max, scatter)
        return_tuple : tuple = ([], [], [], [], [])
        for key, values in data.items():
            # only with respect to the altered positions
            data_tmp : list = np.asarray(values)

            return_tuple[0].append(np.average(data_tmp))
            return_tuple[1].append(np.min(data_tmp))
            return_tuple[2].append(np.median(data_tmp))
            return_tuple[3].append(np.max(data_tmp))
            return_tuple[4].append(data_tmp)

        return return_tuple
