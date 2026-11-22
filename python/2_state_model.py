import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import fsolve
import yaml
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
from copy import deepcopy

rng = np.random.default_rng()

antibiotic_modes = ["constant", "exponential", "linear", "zero_out"]

par_names = ["col_to_inf_risk", "norm_col_prob", "risk_factors", "dem_prob", "house_lmbda",
            "house_trans", "food_acq", "animal_acq", "travel_antibiotic_risk", "per_travel", "trips_lmbda",
            "per_hosp", "hosp_lmbda", "pr_pu_risk", "pr_pu_prob", "infection_probs",
            "start_anti_fracs", "start_col_dist", "num_years"]

full_names = {"col_to_inf_risk": "Risk ratio of developing infection while colonized and on antibiotics",
            "norm_col_prob": "Normalized prob. of increasing colonization status",
            "risk_factors": "Relative risk ratio of increasing colonization for antibiotics",
            "dem_prob": "Prob. of reducing colonization status",
            "house_lmbda": "Parameter for exponential distribution for house size",
            "house_trans": "Transmission prob. between household members",
            "food_acq": "Prob. of colonization by food",
            "animal_acq": "Prob. of colonization by animal contact",
            "per_travel": "Prob. of colonization per trip out of country",
            "travel_antibiotic_risk": "Risk factor for traveling and using antibiotics",
            "trips_lmbda": "Parameter for exponential distribution for trips taken",
            "per_hosp": "Prob. of colonization per hospital trip",
            "hosp_lmbda": "Parameter for exponential distribution for hospital visits",
            "pr_pu_risk": "Risk ratio of taking proton pump inhibitors",
            "pr_pu_prob": "Fraction of population taking proton pump inhibitors",
            "infection_probs": "Prob. of infection for colonization status",
            "start_anti_fracs": "Initial antibiotic prescription prob.",
            "start_col_dist": "Initial colonization status proportions",
            "num_years": "Years in simulation"}

def exp_mgf_help(t, lmbda):
    return lmbda/(lmbda-t)

def poiss_mgf(t, lmbda):
    return np.exp(lmbda*(np.exp(t)-1))

def poiss_mgf_simplify(s,lmbda):
    return np.exp(-s*lmbda)

def matrix_t(y,antibiotic_probs, pars, print_mat=False):
    # load values from the config file
    # we will assume this is handed in as a vector and let something else do the config unpacking
    _, norm_col_prob, risk_factors, dem_prob, house_lmbda, house_trans, food_acq, animal_acq, travel_antibiotic_risk, per_travel, trips_lmbda, per_hosp, hosp_lmbda, pr_pu_risk, pr_pu_prob = pars

    # calculated values to be used in the transition matrix
    antibiotic_prob = 1-np.prod(1-norm_col_prob*np.array(antibiotic_probs)*np.array(risk_factors))
    anti_impact = 1-np.prod(1-np.array(antibiotic_probs)*np.array(risk_factors))
    travel_anti_interaction = (1-anti_impact) + anti_impact*travel_antibiotic_risk
    travel_prob = 1-np.prod(np.array([poiss_mgf(np.log(1-min(1,travel_anti_interaction*per_travel[i])), trips_lmbda[i]) for i in range(0, len(per_travel))]))
    house_acq = house_trans*(1-poiss_mgf(np.log(y[0]), house_lmbda))
    proton_pump_factor = (1-pr_pu_prob) + pr_pu_prob*pr_pu_risk
    stay_ND = ((1-antibiotic_prob)*(1-house_acq)*(1-animal_acq)*(1-food_acq)*(1-travel_prob)*poiss_mgf(np.log((1-per_hosp)),hosp_lmbda))/proton_pump_factor

    A_t = np.array([
        [stay_ND,dem_prob],
        [(1-stay_ND),1-dem_prob]])

    if print_mat:
        print(A_t)
    return A_t

def par_equilibrium(y0, antibiotic_probs, pars):
    _, norm_col_prob, risk_factors, dem_prob, house_lmbda, house_trans, food_acq, animal_acq, travel_antibiotic_risk, per_travel, trips_lmbda, per_hosp, hosp_lmbda, pr_pu_risk, pr_pu_prob = pars
    antibiotic_prob = 1-np.prod(1-norm_col_prob*np.array(antibiotic_probs)*np.array(risk_factors))
    anti_impact = 1-np.prod(1-np.array(antibiotic_probs)*np.array(risk_factors))
    travel_anti_interaction = (1-anti_impact) + anti_impact*travel_antibiotic_risk
    travel_prob = 1-np.prod(np.array([poiss_mgf(np.log(1-min(1,travel_anti_interaction*per_travel[i])), trips_lmbda[i]) for i in range(0, len(per_travel))]))
    proton_pump_factor = (1-pr_pu_prob) + pr_pu_prob*pr_pu_risk
    nd_pref = ((1-antibiotic_prob)*(1-animal_acq)*(1-food_acq)*(1-travel_prob)*poiss_mgf(np.log((1-per_hosp)),hosp_lmbda))/proton_pump_factor
    eqn_rhs = lambda y: y*(1+dem_prob-nd_pref*(1-house_trans+house_trans*np.exp(-house_lmbda*(1-y)))) - dem_prob
    return fsolve(eqn_rhs, y0)

# LLM method, modified by me to include index information
def get_range_info(cfg):
    keys = []
    lows = []
    highs = []

    def visit(value, key, app_idx=""):
        if isinstance(value, list):
            for idx,item in enumerate(value):
                visit(item, key, app_idx=f"_{idx}")
        elif isinstance(value, str):
            parts = value.split("-")
            if len(parts) != 2:
                return

            try:
                low, high = map(float, parts)
            except ValueError:
                return

            keys.append(f"{key}{app_idx}")
            lows.append(low)
            highs.append(high)

    for key, value in cfg.items():
        visit(value, key)

    return keys, lows, highs

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

def run_sens_test(immutable_pars, run_name=None, num_samples=1000, mode="constant", dec_value=0.9, floor_frac=0.0, show_col=True):
    if run_name is None:
        datestring=str(dt.now())
        pieces = datestring.split(" ")
        subpieces = pieces[1].split(".")
        run_name = "_".join(pieces[0].split("-"))+"_"+"_".join(subpieces[0].split(":"))+"_"+subpieces[1]
    
    num_ranges = count_ranges(immutable_pars)  # how many ranges must be replaced by latin hypercube sampling
    run_infections = np.zeros((num_samples, 1+immutable_pars[-1]))  # final element is number of years in simulation
    run_colonizations = np.zeros((num_samples, 1+immutable_pars[-1]))  # final element is number of years in simulation
    lhs_samples = LatinHypercube(d=num_ranges).random(n=num_samples)
    sample_outputs = []
    for i in range(0, num_samples):
        temp_pars = replace_ranges(immutable_pars, list(lhs_samples[i]))
        sim_output,_ = run_simulation(temp_pars, plot_results=False, save_results=True, sub_folder=run_name, mode=mode, dec_value=dec_value, floor_frac=floor_frac)
        run_infections[i,:] = sim_output[-1,:]
        run_colonizations[i,:] = sim_output[1,:]
        sample_outputs.append(temp_pars)

    sens_plotting(run_colonizations, run_infections)

def sens_mag_test(immutable_pars_cfg, mode="constant", dec_value=0.9, floor_frac=0.0):
    immutable_pars = [immutable_pars_cfg[x] for x in par_names]
    changed_vars, changed_lows, changed_highs = get_range_info(immutable_pars_cfg)
    num_ranges = count_ranges(immutable_pars)  # how many ranges must be replaced by latin hypercube sampling
    sample_outputs = {}

    for k in range(0, num_ranges):
        low_sample = list(np.ones(num_ranges)*0.5)  # all elements are set to the median value
        high_sample = list(np.ones(num_ranges)*0.5)  # all elements are set to the median value
        low_sample[k] = 0  # we check the low end of the range for the low sample
        high_sample[k] = 1  # high end for the i=high
        var_changed = changed_vars[k]

        # low output
        changed_low = changed_lows[k]
        low_temp_pars = replace_ranges(immutable_pars, low_sample)
        low_sim_output,_ = run_simulation(low_temp_pars, plot_results=False, mode=mode, dec_value=dec_value, floor_frac=floor_frac)
        low_run_infections = low_sim_output[-1,:]
        low_run_colonizations = low_sim_output[1,:]

        # high output
        changed_high = changed_highs[k]
        high_temp_pars = replace_ranges(immutable_pars, high_sample)
        high_sim_output,_ = run_simulation(high_temp_pars, plot_results=False, mode=mode, dec_value=dec_value, floor_frac=floor_frac)
        high_run_infections = high_sim_output[-1,:]
        high_run_colonizations = high_sim_output[1,:]

        sample_outputs[var_changed] = [changed_low, changed_high, low_run_colonizations, high_run_colonizations, low_run_infections, high_run_infections]

    return sample_outputs        

def plot_sens_mag(sample_outputs, mode="mean", bar_labels=None):
    bar_outputs = {}
    for variable in sample_outputs.keys():
        changed_low, changed_high, low_run_colonizations, high_run_colonizations, low_run_infections, high_run_infections = sample_outputs[variable]
        infection_delta = np.abs(high_run_infections-low_run_infections)
        if mode == "mean":
            bar_outputs[variable] = np.mean(infection_delta)
        elif mode == "final":
            bar_outputs[variable] = infection_delta[-1]
        elif isinstance(mode, np.ndarray):
            bar_outputs[variable] = infection_delta*mode/np.sum(mode)

    if bar_labels is None:
        bar_labels = bar_outputs.keys()
    
    plt.bar(bar_labels, bar_outputs.values(), width=0.1)
    plt.xticks(rotation=45, ha="right")  # optional: improves readability
    plt.ylabel("Change in infections")
    plt.tight_layout()
    plt.show()

# def sens_plotting(run_vals, show_col):
#     num_runs, num_years = run_vals.shape
#     median_run = np.median(run_vals, axis=0)
#     if show_col:
#         plt.title("ESBL-E colonizations")
#     else:
#         plt.title("ESBL-E infections")
#     plt.xlabel("Year")
#     plt.ylabel("Proportion of population")
#     for i in range(0, num_runs):
#         plt.plot(range(0,num_years),run_vals[i,:],color="r",alpha=0.2)
#     plt.plot(range(0,num_years),median_run,color="black",marker="*")
#     plt.show()

def sens_plotting(run_cols, run_infs):
    _, axs = plt.subplots(1,2)
    median_cols = np.median(run_cols, axis=0)
    median_infs = np.median(run_infs, axis=0)
    num_runs, num_years = run_cols.shape

    for i in range(0, num_runs):
        axs[0].plot(range(0,num_years),run_cols[i,:],color="r",alpha=0.2)
        axs[1].plot(range(0,num_years),run_infs[i,:],color="r",alpha=0.2)

    axs[0].plot(range(0,num_years),median_cols,color="black",marker="*")
    axs[0].set_title("ESBLE-E Colonization")
    axs[0].set_ylabel("Proportion of population")
    axs[0].set_xlabel("Year")

    axs[1].plot(range(0,num_years),median_infs,color="black",marker="*")
    axs[1].set_title("ESBLE-E Infections")
    axs[1].set_xlabel("Year")

    plt.show()

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
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),x[1,:],label="Colonized", marker="x")
    #axs[0].plot(range(0,yearly_antibiotics.shape[1]),x[2,:],label="High colonization", marker="s")
    axs[0].set_title("ESBLE-E Colonization")
    axs[0].set_ylabel("Proportion of population")
    axs[0].set_xlabel("Year")
    axs[0].legend()

    axs[1].plot(range(0,yearly_antibiotics.shape[1]),x[-1,:],label="ESBLE-E infections", color="r", marker="*")
    axs[1].set_title("ESBL-E infections")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Proportion of population")
    plt.show()

def compare_two(x,y,scale_factor=1):
    yearly_antibiotics = x[3:-1,:]
    _, axs = plt.subplots(1,2)
    
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*x[0,:],label="Non-detectable 1", marker="o")
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*x[1,:],label="Colonization 1", marker="x")
    #axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*x[2,:],label="High colonization 1", marker="s")
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*y[0,:],label="Non-detectable 2", marker="o")
    axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*y[1,:],label="Colonization 2", marker="x")
    #axs[0].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*y[2,:],label="High colonization 2", marker="s")
    axs[0].set_title("ESBLE-E Colonization")
    axs[0].set_ylabel("Proportion of population")
    axs[0].set_xlabel("Year")
    axs[0].legend()

    axs[1].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*x[-1,:],label="ESBLE-E infections 1", color="r", marker="*")
    axs[1].plot(range(0,yearly_antibiotics.shape[1]),scale_factor*y[-1,:],label="ESBLE-E infections 2", color="black", marker="*")
    axs[1].set_title("ESBL-E infections")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Proportion of population")
    plt.show()

def antibiotic_scenario(start_anti_fracs, num_years, mode="constant", dec_value=0.7, floor_frac=0):
    if mode == "zero_out":
        return np.zeros((num_years, len(start_anti_fracs)))
    if mode == "constant":
        return np.stack((start_anti_fracs,)*num_years)
    if mode == "linear":
        x = np.zeros((num_years, len(start_anti_fracs)))
        x[0,:] = start_anti_fracs
        for i in range(1, num_years):
            x[i,:] = np.maximum(x[i-1,:]-dec_value,floor_frac*start_anti_fracs)
        return x
    if mode == "exponential":
        x = np.zeros((num_years, len(start_anti_fracs)))
        x[0,:] = start_anti_fracs
        for i in range(1, num_years):
            x[i,:] = dec_value*x[i-1,:]
        return x
    else:
        raise ValueError("Invalid mode, scenario not implemented")

def infections(x, antibiotic_probs, risk_factors, infection_probs, col_to_inf_risk):
    antibiotic_prob = 1-np.prod(1-np.array(antibiotic_probs)*np.array(risk_factors))
    mod_inf_probs = infection_probs*np.array([1,(1-antibiotic_prob)+col_to_inf_risk*antibiotic_prob])  # exposure to antibiotics doubles risk factor
    return np.dot(x, mod_inf_probs)

def run_simulation(pars, plot_results=True, save_results=False, sub_folder=None, filename=None, mode="constant", dec_value=0.7, floor_frac=0.0, print_mat=False):
    x0 = np.array(pars[-2])
    start_anti_fracs = np.array(pars[-3])
    num_years = pars[-1]
    col_to_inf_risk = pars[0]
    #norm_col_prob = pars[1]
    risk_factors = pars[2]
    yearly_antibiotics = antibiotic_scenario(start_anti_fracs, num_years, mode=mode, dec_value=dec_value, floor_frac=floor_frac)
    x = np.zeros((1+len(x0)+len(start_anti_fracs),len(yearly_antibiotics)+1))
    infection_probs = np.array(pars[-4])
    x[:2,0] = x0
    x[2:-1,0] = start_anti_fracs
    x[-1,0] = infections(x0, start_anti_fracs, risk_factors, infection_probs, col_to_inf_risk)  #np.dot(x0, infection_probs)
    for i,antibiotic_probs in enumerate(yearly_antibiotics):
        t=i+1 # deal with the offset between python and counting years
        A_t = matrix_t(x[:,t-1], antibiotic_probs, pars[:-4], print_mat=print_mat)  # we leave out the infection probability, matrix builder doesn't need it
        x[:2,t] = np.matmul(A_t,x[:2,t-1])
        x[2:-1,t] = antibiotic_probs
        x[-1:,t] = infections(x[:2,t], antibiotic_probs, risk_factors, infection_probs, col_to_inf_risk)  #np.dot(x[:2,t], infection_probs) 

    if plot_results:
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
    par_file = os.path.join(os.getcwd(),"focused_antibiotics","parameters","2_state.yaml")
    with open(par_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    pars = [cfg[x] for x in par_names]
    prop_y0 = cfg["start_col_dist"][0]
    prop_eq = par_equilibrium(prop_y0, np.array(pars[-3]), pars[:-4])
    
    print(f"Estimated equilibrium: [{prop_eq[0]},{1-prop_eq[0]}]")
    #y,_ = run_simulation(pars,sub_folder="default", save_results=False, plot_results=False, mode="zero_out")
    #y,_ = run_simulation(pars,sub_folder="default", save_results=False, plot_results=False, mode="exponential", dec_value=0.85)
    # x,_ = run_simulation(pars,sub_folder="default", save_results=False, plot_results=False, mode="constant")
    # y,_ = run_simulation(pars,sub_folder="default", save_results=False, plot_results=False, mode="linear", dec_value=0.01, floor_frac=0.2)
    # compare_two(x,y)

    sens_par_file = os.path.join(os.getcwd(),"focused_antibiotics","parameters","2_state_sensitivity.yaml")
    with open(sens_par_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    immutable_pars = [cfg[x] for x in par_names]
    run_sens_test(immutable_pars, num_samples=1000, mode="linear", dec_value=0.005, floor_frac=0.5, show_col=True)
    # sample_outs = sens_mag_test(cfg, mode="linear", dec_value=0.01, floor_frac=0.2)
    # plot_sens_mag(sample_outs, mode="final")