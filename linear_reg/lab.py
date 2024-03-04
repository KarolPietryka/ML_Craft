from linear_reg.LinearRegNumPy import LinearRegNumPy
import numpy as np
from linear_reg.LinearRegPurePy import LinearRegPurePy
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

LinearRegNumPy().fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
LinearRegPurePy().fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
