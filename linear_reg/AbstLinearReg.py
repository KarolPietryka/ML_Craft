from abc import ABC, abstractmethod
import time
import pandas as pd


class AbsLinearReg(ABC):
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.n = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    @abstractmethod
    def get_mse(self, training_data, target_data, w, b):
        pass

    @abstractmethod
    def get_gradients(self, training_data, target_data, w, b):
        pass

    def fit(self, training_data, target_data):
        w = 0.0
        b = 0.0
        self.n = len(training_data)
        data = []

        mse = self.get_mse(training_data, target_data, w, b)
        print(f"Initial MSE: {mse}")
        start_time = time.time()
        for i in range(self.iterations):
            gradients = self.get_gradients(training_data, target_data, w, b)
            w, b = self.gradient_descent(gradients, w, b)
            if i % 10 == 0:
                mse = self.get_mse(training_data, target_data, w, b)
                print(f"MSE for iteration {i}: {mse}. w: {w}, b: {b}")
                data.append({'Iteration': i, 'MSE': mse, 'w': w, 'b': b, 'Total calc time': time.time() - start_time})

        mse = self.get_mse(training_data, target_data, w, b)
        print(f"Final w: {w}, and b: {b} with MSE: {mse}")

        df = pd.DataFrame(data)
        filename = f'{self.__class__.__name__}.csv'
        df.to_csv(filename, index=False)

    def gradient_descent(self, gradients, w, b):
        w += -self.learning_rate * gradients[0]
        b += -self.learning_rate * gradients[1]
        return w, b
