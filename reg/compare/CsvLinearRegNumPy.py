import time
import pandas as pd

from reg.linear.LinearRegNumPy import LinearRegNumPy

class CsvLinearRegNumPy(LinearRegNumPy):
    def milestone(self, i, mse, w, b):
        super(CsvLinearRegNumPy, self).milestone(i, mse, w, b)
        self.data.append({'Iteration': i, 'MSE': mse, 'w': w, 'b': b, 'Total calc time': time.time() - self.start_time})

    def post_calc_callback(self):
        df = pd.DataFrame(self.data)
        filename = f'{self.__class__.__name__}.csv'
        df.to_csv(filename, index=False)