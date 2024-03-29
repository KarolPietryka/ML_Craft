import logging
import numpy as np
import pandas as pd
import os

from reg.compare.CsvLinearRegNumPy import CsvLinearRegNumPy
from reg.compare.CsvLinearRegPurePy import CsvLinearRegPurePy

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


CsvLinearRegNumPy().fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
CsvLinearRegPurePy().fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])


# File paths
numpy_file_path = f'{CsvLinearRegNumPy.__name__}.csv'
pure_py_file_path = f'{CsvLinearRegPurePy.__name__}.csv'

numpy_df = pd.read_csv(numpy_file_path)
pure_py_df = pd.read_csv(pure_py_file_path)

numpy_total_time = numpy_df.iloc[-1]['Total calc time']
pure_py_total_time = pure_py_df.iloc[-1]['Total calc time']

print(f"NumPy implementation total time: {numpy_total_time}")
print(f"Pure Python implementation total time: {pure_py_total_time}")
# Compare the times
if numpy_total_time < pure_py_total_time:
    faster_by = pure_py_total_time / numpy_total_time
    print(f"The NumPy implementation is {faster_by:.2f} times faster than the Pure Python implementation.")
else:
    faster_by = numpy_total_time / pure_py_total_time
    print(f"The Pure Python implementation is {faster_by:.2f} times faster than the NumPy implementation.")


# Delete the files
os.remove(numpy_file_path)
os.remove(pure_py_file_path)

print(f"Deleted {numpy_file_path} and {pure_py_file_path}.")