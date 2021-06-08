import math
import matplotlib.pyplot as plt


class Plot:

    __slots__ = ["task_list"]

    def __init__(self, task_list : list):
        self.task_list : list = task_list
        pass

    @staticmethod
    def __find_square(n : int) -> tuple:
        x, y = math.floor(math.sqrt(n)), math.ceil(math.sqrt(n))

        while True:
            if x * y < n:
                y += 1
            else:
                return x, y

    def plot(self) -> None:

        n_tasks : int = len(self.task_list)
        
        _, axes = plt.subplots(*self.__find_square(n_tasks))
        for task in self.task_list:
            task.plot(axes, task.descr)
            plt.show()

