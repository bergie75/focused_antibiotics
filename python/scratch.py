import numpy as np
from scipy.stats.qmc import LatinHypercube

def poiss_mgf(t, lmbda):
    return np.exp(lmbda*(np.exp(t)-1))

print(poiss_mgf(np.log(1-0.5), 0.116))