import numpy as np
from scipy.stats.qmc import LatinHypercube

x=np.array([1,2,3])
dec_value = 0.5
print(np.maximum(x-dec_value,0))