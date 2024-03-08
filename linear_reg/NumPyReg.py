from abc import abstractmethod

from linear_reg.AbsReg import AbsReg
import numpy as np


class NumPyReg(AbsReg):
    @abstractmethod
    def get_pred(self, training_data, w, b):
        pass

    def get_mse(self, training_data, target_data, w, b):
        pred = self.get_pred(training_data, w, b)
        return np.mean((pred - target_data) ** 2)

    def get_gradients(self, training_data, target_data, w, b):
        pred = self.get_pred(training_data, w, b)
        error = pred - target_data
        total_w = np.dot(error, training_data) / self.n
        total_b = np.mean(error)
        return [total_w, total_b]
