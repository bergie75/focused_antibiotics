import numpy as np
from scipy.stats.qmc import LatinHypercube
import yaml
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
from copy import deepcopy

rng = np.random.default_rng()
par_names = ["norm_col_prob", "risk_factors", "dem_prob", "house_size",
            "house_trans", "food_acq", "animal_acq", "per_travel", "trips_per_yr",
            "per_hosp", "hosp_per_yr", "skip_to_H", "pr_pu_risk", "pr_pu_prob", "infection_probs",
            "start_anti_fracs", "start_col_dist", "num_years"]

def matrix_t(y,antibiotic_probs, pars):
    # load values from the config file
    # we will assume this is handed in as a vector and let something else do the config unpacking
    norm_col_prob, risk_factors, dem_prob, house_size, house_trans, food_acq, animal_acq, per_travel, trips_per_yr, per_hosp, hosp_per_yr, skip_to_H, pr_pu_risk, pr_pu_prob = pars

    # calculated values to be used in the transition matrix
    antibiotic_prob = [1-np.prod(1-x*np.dot(np.array(antibiotic_probs),np.array(risk_factors))) for x in norm_col_prob]
    house_acq = house_trans*(1-y[0]**house_size)  # y = [non-detect, low burden, high burden]
    proton_pump_factor = (1-pr_pu_prob) + pr_pu_prob*pr_pu_risk
    stay_ND = ((1-antibiotic_prob[0])*(1-house_acq)*(1-animal_acq)*(1-food_acq)*(1-per_travel)**trips_per_yr*(1-per_hosp)**hosp_per_yr)/proton_pump_factor
    no_acq_in_L = (1-antibiotic_prob[1])*(1-house_acq)*(1-animal_acq)*(1-food_acq)*(1-per_travel)**trips_per_yr*(1-per_hosp)**hosp_per_yr/proton_pump_factor
    stay_L = (1-dem_prob[0])*no_acq_in_L

    A_t = np.array([
        [stay_ND,dem_prob[0],0],
        [(1-skip_to_H)*(1-stay_ND),stay_L,dem_prob[1]],
        [skip_to_H*(1-stay_ND),(1-no_acq_in_L)*(1-dem_prob[0]),1-dem_prob[1]]
    ])

    return A_t

def count_ranges(x):
    num_ranges = 0
    y=deepcopy(x)
    if type(y) is list:
        for elem in y:
            num_ranges += count_ranges(elem)
    elif type(y) is str:
        # the element is of the form a-b, choose randomly from this interval
        num_ranges += 1
    
    return num_ranges

def replace_ranges(x, lhs_sample):
    y=deepcopy(x)
    if type(y) is list:
        for i,elem in enumerate(y):
            y[i] = replace_ranges(elem, lhs_sample)
        return y
    elif type(y) is str:
        # the element is of the form a-b, choose randomly from this interval
        r=float(lhs_sample.pop(0))
        a,b = y.split("-")
        a=float(a)
        b=float(b)
        return a+(b-a)*r
    else:
        return y

def run_sens_test(par_yaml, num_samples=1):
    par_file = os.path.join(os.getcwd(),"focused_antibiotics","parameters",f"{par_yaml}.yaml")
    with open(par_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    
    immutable_pars = [cfg[x] for x in par_names]
    num_ranges = count_ranges(immutable_pars)
    lhs_samples = LatinHypercube(d=num_ranges).random(n=num_samples)
    for i in range(0, num_samples):
        print(replace_ranges(immutable_pars, list(lhs_samples[i])))

def plotting_fn(input_val=None, sub_folder=None, filename=None):
    if input_val is not None:
        x=input_val
    elif filename is not None:
        cwd=os.getcwd()
        save_loc = os.path.join(cwd, "focused_antibiotics", "output")
        if sub_folder is not None:
            save_loc = os.path.join(save_loc, sub_folder)
        save_loc = os.path.join(save_loc, f"{filename}.npy")
        x = np.load(save_loc)

    yearly_antibiotics = x[3:-1,:]
    _, axs = plt.subplots(1,2)
    
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),x[0,:],label="Non-detectable", marker="o")
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),x[1,:],label="Low colonization", marker="x")
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),x[2,:],label="High colonization", marker="s")
    axs[0].set_title("ESBLE-E Colonization")
    axs[0].set_ylabel("Proportion of population")
    axs[0].set_xlabel("Year")
    axs[0].legend()

    axs[1].plot(range(0,yearly_antibiotics.shape[1]),x[-1,:],label="ESBLE-E infections", color="r", marker="*")
    axs[1].set_title("ESBL-E infections")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Proportion of population")
    plt.show()

def antibiotic_scenario(start_anti_fracs, num_years, mode="constant"):
    if mode == "constant":
        return np.stack((start_anti_fracs,)*num_years)
    else:
        raise ValueError("Invalid mode, scenario not implemented")

def run_simulation(pars, plot_results=True, save_results=False, sub_folder=None, filename=None, mode="constant"):
    x0 = np.array(pars[-2])
    start_anti_fracs = np.array(pars[-3])
    num_years = pars[-1]
    yearly_antibiotics = antibiotic_scenario(start_anti_fracs, num_years, mode=mode)
    x = np.zeros((1+len(x0)+len(start_anti_fracs),len(yearly_antibiotics)+1))
    infection_probs = np.array(pars[-4])
    x[:3,0] = x0
    x[3:-1,0] = start_anti_fracs
    for i,antibiotic_probs in enumerate(yearly_antibiotics):
        t=i+1 # deal with the offset between python and counting years
        A_t = matrix_t(x[:,t-1], antibiotic_probs, pars[:-4])  # we leave out the infection probability, matrix builder doesn't need it
        x[:3,t] = np.matmul(A_t,x[:3,t-1])
        x[3:-1,t] = antibiotic_probs
        x[-1:,t] = np.dot(x[:3,t], infection_probs) 

    if plot_results:
        # plt.plot(range(0,len(yearly_antibiotics)+1),x[0,:],label="ND")
        # plt.plot(range(0,len(yearly_antibiotics)+1),x[1,:],label="L")
        # plt.plot(range(0,len(yearly_antibiotics)+1),x[2,:],label="H")
        # plt.legend()
        # plt.show()
        plotting_fn(input_val=x)

    if save_results:
        cwd = os.getcwd()
        save_loc = os.path.join(cwd, "focused_antibiotics", "output")
        
        # useful functionality for sensitivity analysis
        if sub_folder is not None:
            save_loc = os.path.join(save_loc, sub_folder)
            if not os.path.exists(save_loc):
                os.mkdir(save_loc)
        
        if filename is None:
            datestring=str(dt.now())
            pieces = datestring.split(" ")
            subpieces = pieces[1].split(".")
            filename = "_".join(pieces[0].split("-"))+"_"+"_".join(subpieces[0].split(":"))+"_"+subpieces[1]
        
        save_loc = os.path.join(save_loc, filename)
        np.save(save_loc, x)
    
    return x, filename

if __name__ == "__main__":
    # unpack the values in the configuration file
    par_file = os.path.join(os.getcwd(),"focused_antibiotics","parameters","fixed.yaml")
    with open(par_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    pars = [cfg[x] for x in par_names]
    x,save_loc = run_simulation(pars,sub_folder="huh", save_results=False, plot_results=True)
    #run_sens_test("sens_test", num_samples=10)