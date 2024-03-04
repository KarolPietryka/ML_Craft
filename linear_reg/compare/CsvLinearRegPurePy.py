from linear_reg.LinearRegPurePy import LinearRegPurePy
import time
import pandas as pd

class CsvLinearRegPurePy(LinearRegPurePy):
    def milestone(self, i, mse, w, b):
        super(LinearRegPurePy, self).milestone(i, mse, w, b)
        self.data.append({'Iteration': i, 'MSE': mse, 'w': w, 'b': b, 'Total calc time': time.time() - self.start_time})

    def post_calc_callback(self):
        df = pd.DataFrame(self.data)
        filename = f'{self.__class__.__name__}.csv'
        df.to_csv(filename, index=False)