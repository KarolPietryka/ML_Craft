from linear_reg.AbsReg import AbsReg
import numpy as np


class LinearRegNumPy(AbsReg):
    def get_mse(self, training_data, target_data, w, b):
        pred = w * training_data + b
        return np.mean((pred - target_data) ** 2)

    def get_gradients(self, training_data, target_data, w, b):
        pred = w * training_data + b
        error = pred - target_data
        total_w = np.dot(error, training_data) / self.n
        total_b = np.mean(error)
        return [total_w, total_b]
