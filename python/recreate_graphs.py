import numpy as np
import matplotlib.pyplot as plt

# parameters needed to calculate colonization/susceptibility status
S_0 = 0.77
C_0 = 0.23

def alpha_calc(f, col_prob, house_prob=0.25):
    return 1 - (1-col_prob*f/5)**5*(1-C_0*house_prob)

# [year_1, year_2, ..., year_10]
col_prob = 0.3
alpha_const = alpha_calc(0.3*np.ones(10), col_prob)
alpha = [alpha_calc(x, col_prob) for x in np.linspace(0.28, 0.09, 9)]
alpha.extend([alpha_calc(0.09, col_prob)])
beta = 0.43
years = list(range(1,11))

def P(t, alpha=alpha, beta=beta):
    P_out = 1
    for j in range(0, t+1):
        P_out *= (1-alpha[j]-beta)
    return P_out

def colonized(t, alpha=alpha, beta=beta):
    in_bracket = S_0*alpha[0]-beta*C_0
    for j in range(0, t):
        in_bracket += (S_0*alpha[j]-beta*C_0)/P(j, alpha=alpha, beta=beta)
    return C_0 + P(t-1, alpha=alpha, beta=beta)*in_bracket

x=[0]
x.extend(years)
y=[C_0*100]
z=[C_0*100]
y.extend([100*colonized(t-1) for t in years])
z.extend([100*colonized(t-1, alpha=alpha_const) for t in years])
plt.plot(x, y, "-s", color="blue", label="Intervention")
plt.plot(x,z, "-o", color="red", label="Comparator")
plt.plot(x, np.ones(11)*C_0*100, color="grey", linestyle="dashed", label="Baseline")
plt.xlim([0, 9])
plt.xticks([0,1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10])
plt.legend()
plt.grid(True)
plt.show()