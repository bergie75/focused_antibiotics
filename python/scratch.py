import numpy as np
from scipy.stats.qmc import LatinHypercube
from copy import deepcopy

rng = np.random.default_rng()
thing = [0,0.5,1]
cfg = {"Hella": "0.1-0.3", "Sus": 0.2, "Nah": [0.1,"0.5-0.6"], "Yo": ["0.1-0.8", 3]}

print(list(np.array([1,2,3,4])))

background_febrile_rash_cases = rng.poisson(lam=np.array([]))
print(np.sum(background_febrile_rash_cases))