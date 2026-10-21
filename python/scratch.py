import numpy as np
from scipy.stats.qmc import LatinHypercube

x=np.array([[1,4,7],[2,5,8],[3,6,9]])
print(np.mean(x, axis=0))
