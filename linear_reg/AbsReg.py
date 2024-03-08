from abc import ABC, abstractmethod
import time
import pandas as pd
import logging


class AbsReg(ABC):
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n = None
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.start_time = None
        self.data = []

    @abstractmethod
    def get_mse(self, training_data, target_data, w, b):
        pass

    @abstractmethod
    def get_gradients(self, training_data, target_data, w, b):
        pass

    def milestone(self, i, mse, w, b):
        self.logger.debug(f"MSE for iteration {i}: {mse}. w: {w}, b: {b}")

    def post_calc_callback(self):
        pass

    def fit(self, training_data, target_data):
        w = 0.0
        b = 0.0
        self.n = len(training_data)

        mse = self.get_mse(training_data, target_data, w, b)
        self.logger.info(f"Initial MSE: {mse}")
        self.start_time = time.time()
        self.logger.debug(f"Start time: {self.start_time}")
        for i in range(self.iterations):
            gradients = self.get_gradients(training_data, target_data, w, b)
            w, b = self.gradient_descent(gradients, w, b)
            if i % 10 == 0:
                mse = self.get_mse(training_data, target_data, w, b)
                self.milestone(i, mse, w, b)

        mse = self.get_mse(training_data, target_data, w, b)
        self.logger.info(f"Final w: {w}, and b: {b} with MSE: {mse}")

        self.post_calc_callback()


    def gradient_descent(self, gradients, w, b):
        w += -self.learning_rate * gradients[0]
        b += -self.learning_rate * gradients[1]
        return w, b
