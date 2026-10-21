import numpy as np
import matplotlib.pyplot as plt

# parameters needed to calculate colonization/susceptibility status
S_0 = 0.77
C_0 = 0.23
x_0 = np.array([S_0,C_0])

# [year_1, year_2, ..., year_10]
f_antibiotic = 0.28*np.ones(10)  # fraction given antibiotics each year
f_orig = np.linspace(0.28,0.09,10)
beta = 0.7
years = list(range(1,11))

def antibiotic_profile(start=0.28, end=0.09, mid=0.22, mid_index=5, total_years=10):
    if mid_index <=0:
        return np.linspace(start, end, total_years)
    else:
        my_arr = list(np.linspace(start, mid, mid_index))
        next_arr = list(np.linspace(mid, end, total_years-mid_index+1))
        my_arr.extend(next_arr[1:])

        return np.array(my_arr)

def year_step(x_t, t, f_antibiotic=f_antibiotic):
    _, C_t = x_t
    alpha = 1 - (1-0.22*f_antibiotic[t-1])*(1-0.7*C_t)
    A_t = np.array([[1-alpha,beta],[alpha,1-beta]])
    return np.matmul(A_t, x_t)

def results(x_0, f_antibiotic=f_antibiotic):
    res = []
    curr = x_0
    for t in years:
        curr = year_step(curr, t, f_antibiotic=f_antibiotic)
        res.append(curr)
    
    return np.array(res)

x=[0]
x.extend(years)
# plot a number of antibiotic impact scenarios
mid_target = 0.1744444
for mid_index in [2,3,4,0]:
    y=[x_0*100]
    f_anti = antibiotic_profile(mid=mid_target, mid_index=mid_index)
    y.extend(100*results(x_0, f_antibiotic=f_anti))
    if mid_index != 0:
        plt.plot(x, [a[1] for a in y], "-s", label=f"Reach year {mid_index+1}")
    else:
        plt.plot(x, [a[1] for a in y], "-s", label=f"Reach year linearly")
plt.plot(x, np.ones(11)*C_0*100, color="grey", linestyle="dashed", label="Baseline")
plt.xlim([0, 9])
plt.xticks([0,1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10])
plt.legend()
plt.grid(True)
plt.show()