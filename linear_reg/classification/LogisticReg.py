from linear_reg.NumPyReg import NumPyReg
import numpy as np


class LogisticReg(NumPyReg):
    def get_pred(self, training_data, w, b):
        return 1 / (1 + np.exp(-(w * training_data + b)))

    def predict(self, training_data):
        probabilities = self.get_pred(training_data, self.w, self.b)
        return np.where(probabilities >= 0.5, 1, 0)