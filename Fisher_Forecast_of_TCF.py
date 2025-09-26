import os
from pathlib import Path
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import corner

####################################################################################################################################################
################################ Global Parameters #################################################################################################
####################################################################################################################################################
 
### --- Astrophysical parameters of the simulation --- ###
sim_params = ['ION_Tvir_MIN','R_BUBBLE_MAX','HII_EFF_FACTOR']
Fisher_Param_labels = ['$T_{Vir}$','$R_{Max}$','$\zeta$']

nparams = len(sim_params)

# parameter values for fiducial simulation
fid_vals = [50000, 15, 30] # Tvir, Rmax, HII 

# parameter ± for derivatives
delta_params = [5000, 5, 5] # Tvir, Rmax, HII  

noise_config = "AAstar_100hrs"

### --- simulation h5 files --- ###
# - clean sims - #
SIM_FID = '/data/cluster/lcrascal/SIM_data/SIM_FID/Lightcone_FID_400_Samples.h5'

SIM_HII_plus = '/data/cluster/lcrascal/SIM_data/SIM_HII_plus/Lightcone_HII_EFF_FACTOR_400_Samples_Plus.h5'
SIM_HII_minus = '/data/cluster/lcrascal/SIM_data/SIM_HII_minus/Lightcone_HII_EFF_FACTOR_400_Samples_Minus.h5'

SIM_Rmax_plus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_plus/Lightcone_R_BUBBLE_MAX_400_Samples_Plus.h5'  
SIM_Rmax_minus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_minus/Lightcone_R_BUBBLE_MAX_400_Samples_Minus.h5'

SIM_Tvir_plus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_plus/Lightcone_ION_Tvir_MIN_400_Samples_Plus.h5'
SIM_Tvir_minus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_minus/Lightcone_ION_Tvir_MIN_400_Samples_Minus.h5'

# - Noisy sims  - #
SIM_FID_Noisy = f'/data/cluster/lcrascal/SIM_data/SIM_FID/{noise_config}/Lightcone_FID_400_Samples_NOISE.h5'
SIM_FID_Obs = f'/data/cluster/lcrascal/SIM_data/SIM_FID/{noise_config}/Lightcone_FID_400_Samples_NOISE_SMOOTHING.h5'
SIM_FID_NoiseOnly = f'/data/cluster/lcrascal/SIM_data/SIM_FID/{noise_config}/Lightcone_FID_400_Samples_NOISE_ONLY_LC.h5'

simlist = [
    SIM_FID,           
    SIM_HII_plus,
    SIM_HII_minus,
    SIM_Rmax_plus,
    SIM_Rmax_minus,
    SIM_Tvir_plus,
    SIM_Tvir_minus
]

noisy_simlist = [
    SIM_FID_Noisy, 
    SIM_FID_Obs, 
    SIM_FID_NoiseOnly]

    
 ### --- parameters describing the simulation --- ###

with h5py.File(SIM_FID, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])  # Mpc/h
    box_dim = int(f['ngrid'][0])
    n_realisations = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')

L = box_length
####################################################################################################################################################
################################ Load Data and Create Dictionaries #################################################################################
####################################################################################################################################################
 

def load_tcf_data(h5file):
    """
    Load TCF and rvals from a given HDF5 file.
    """
    h5file = Path(h5file)
    with h5py.File(h5file, "r") as f:
        tcf = f["TCF_zidx0"][...]   # (400, 100)
        rvals = f["TCF_zidx0_rvals"][...]  # (100,)
    return {"tcf": tcf, "rvals": rvals}


# Dictionaries
TCF_clean_dict = {}
TCF_noisy_dict = {}

# Loop clean
for f in simlist:
    key = (
        Path(f).stem
        .replace("Lightcone_", "")
        .replace("_400_Samples", "")
    )
    TCF_clean_dict[key] = load_tcf_data(f)

# Loop noisy
for f in noisy_simlist:
    key = (
        Path(f).stem
        .replace("Lightcone_", "")
        .replace("_400_Samples", "")
    )
    TCF_noisy_dict[key] = load_tcf_data(f)


print(f"✅ Loaded {len(TCF_clean_dict)} clean sims, {len(TCF_noisy_dict)} noisy sims")


# checking and plots
def inspect_tcf_dict(tcf_dict, name="Dictionary"):
    """
    Print summary of a TCF dictionary.
    
    Parameters
    ----------
    tcf_dict : dict
        Dictionary with structure {key: {"tcf": array, "rvals": array}}.
    name : str
        Label for the dictionary in the printout.
    """
    print(f"\n📂 {name}: {len(tcf_dict)} entries")
    for key, dat in tcf_dict.items():
        tcf_shape = dat["tcf"].shape if "tcf" in dat else None
        r_shape   = dat["rvals"].shape if "rvals" in dat else None
        print(f"  {key:25s} → TCF {tcf_shape}, rvals {r_shape}")


inspect_tcf_dict(TCF_clean_dict, "Clean sims")
inspect_tcf_dict(TCF_noisy_dict, f"Noisy sims ({noise_config})")




####################################################################################################################################################
################################ Function to Compute Fisher Forecast ###############################################################################
####################################################################################################################################################


def compute_tcf_fisher(TCF_clean_dict, TCF_noisy_dict, 
    delta_params,
    sim_params, 
    max_idx
):
    """Compute Fisher matrix from TCF simulation files.""" 

    n_realisations = TCF_noisy_dict['FID_NOISE_SMOOTHING']['tcf'].shape[0]

    ############ Compute derivatives ############
    print("Computing Derivatives")
    rvals = TCF_clean_dict[f'{sim_params[0]}_Plus']['rvals']          # extract r vals (identical for all sims)           # extract r vals (identical for all sims)
    rvals_len = rvals.shape[0]                                        # length of r vals array
    derivatives = np.zeros((len(sim_params), n_realisations, rvals_len))    # set up derivatives array
    
    for idx, param in enumerate(sim_params):                                # loop over all params
        plus = TCF_clean_dict[f'{param}_Plus']['tcf']                 # plus param
        minus = TCF_clean_dict[f'{param}_Minus']['tcf']               # minus param
        derivatives[idx] = (plus - minus) / (2*delta_params[idx])               # compute derivative
               

    ############# Whitening the Data ############
    print("Whitening Data ")
    TCF_obs = TCF_noisy_dict['FID_NOISE_SMOOTHING']['tcf']                  # TCF of the observed sim (FID + noise + smoothing)
    std_obs = np.std(TCF_obs, axis=0)                                       # std of this TCF
    mask = std_obs > 0                                                      # only keep data where std is > 0
    whitened_TCF_obs = TCF_obs[:, mask] / std_obs[mask]                     # whiten the obs TCF
    rvals = rvals[mask]                                                     # whiten the rvals
    derivatives_whitened = derivatives[:, :, mask] / std_obs[mask]          # whiten the derivatives

    
    ############# Compute Data Cov Matrix ############
    print("Computing Data Cov Matrix")
    data_cov_matrix = np.cov(whitened_TCF_obs, rowvar=False)                # data cov matrix = covariance of obs TCF
    data_cov_matrix_inv = np.linalg.inv(data_cov_matrix)
    cond_num = np.log10(np.linalg.cond(data_cov_matrix))                    # compute the condition number
    print(f'log10 Condition number: {cond_num:.2f}')

    
    ############# using only a sub set of data (to reduce condition number) #############

    data_cov_matrix = data_cov_matrix[:max_idx, :max_idx]
    cond_num = np.log10(np.linalg.cond(data_cov_matrix))                    # compute the condition number
    print(f'log10 Condition number of sub set of data [0:{max_idx}]: {cond_num:.2f}')
    
    data_cov_matrix_inv = np.linalg.inv(data_cov_matrix)
    derivatives_whitened_subset = derivatives_whitened[:, :, :max_idx]
    rvals = rvals[:max_idx]
    

    ############# Convergence testing + Compute FM ############
    print("Convergence Testing ")
    nbins_kept = whitened_TCF_obs.shape[1]
    
    samples = np.arange(5, n_realisations + 5, 10)                          # only compute a sample of FMs for speed
    nparams = len(sim_params)
    Fisher_matrix = np.zeros((samples.size, nparams, nparams))
    Fisher_matrix_Inv = np.zeros((samples.size, nparams, nparams))          # initialise the inv FM

    for r, sample_size in enumerate(samples):                               # compute the FM using the dervatives and thedata cov matrices
        deriv_sample = np.mean(derivatives_whitened_subset[:, :sample_size, :], axis=1)
        for i in range(nparams):
            for j in range(nparams):
                Fisher_matrix[r, i, j] = deriv_sample[i] @ (data_cov_matrix_inv @ deriv_sample[j])
        Fisher_matrix_Inv[r] = np.linalg.inv(Fisher_matrix[r])

    return {
        'derivatives': derivatives_whitened,           # derivatives
        'data_cov_matrix': data_cov_matrix,            # data covariance matrix
        'Fisher_matrix': Fisher_matrix,                # Fisher Matrix ([-1] = final FM)
        'res_cov_matrix': Fisher_matrix_Inv,           # Inverse Fisher Matrix = Result Covariance Matrix ([-1] = final inverse FM)
        'rvals': rvals                                 # r values
    }


####################################################################################################################################################
################################ Results ###########################################################################################################
####################################################################################################################################################

max_idx = 30 # Max index chosen to optimise the stability of the data covariance matrix

# --- run the fisher function ---
results = compute_tcf_fisher(TCF_clean_dict, TCF_noisy_dict, delta_params, sim_params, max_idx=max_idx)

derivatives = results['derivatives']
data_cov_matrix = results['data_cov_matrix']
Fisher_matrix = results['Fisher_matrix'][-1]
res_cov_matrix = results['res_cov_matrix'][-1]
rvals = results['rvals']


# --- sanity checks ---
print("\n=== Outputs ===")
print("rvals:", results['rvals'].shape)
print("derivatives:", derivatives.shape)
print("data cov matrix shape:", data_cov_matrix.shape)
print("FM shape:", Fisher_matrix.shape)
print("Result Covariance Matrix:", res_cov_matrix.shape)

path_to_save = f"/data/cluster/lcrascal/SIM_data/results/{noise_config}/"
os.makedirs(path_to_save, exist_ok=True) # check path exists

# Save
with open(f"{path_to_save}fisher_results_{noise_config}.pkl", "wb") as f:
    pickle.dump(results, f)

# Load back
with open(f"{path_to_save}fisher_results_{noise_config}.pkl", "rb") as f:
    results_loaded = pickle.load(f)

print(results_loaded.keys())



####################################################################################################################################################
################################ Plots #############################################################################################################
####################################################################################################################################################
path_to_save = f"/data/cluster/lcrascal/SIM_data/results/{noise_config}/"

############ 1. Corner Plot of Results #############################################################################################################

sampled_fisher_data = np.random.multivariate_normal(fid_vals, res_cov_matrix, size=1000000)
param_ranges=None
if param_ranges is None:
    param_ranges = [(None, None)] * len(fid_vals)

fig = corner.corner(
    fisher_data,
    labels=Fisher_Param_labels,
    plot_datapoints=False,
    levels=(0.68, 0.95),
    truths=fid_vals,
    plot_density=False,
    color='blue',
    fill_contours=True,
    contour_kwargs={"linewidths": 0.8}
    # range=param_ranges
)

fig.suptitle(f"Corner Plot for Fisher Forecast ({noise_config})", fontsize=16)
fig.subplots_adjust(top=0.93)  
fig.savefig(f"{path_to_save}corner_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)  

############ 2. Data Cov Matrix + Stability Checks #################################################################################################
############ 2. a. Data Cov Matrix

fig, ax = plt.subplots()

c = ax.pcolormesh(
    rvals,
    rvals,
    data_cov_matrix,
    cmap='coolwarm'
)

ax.set_title(f'Data Covariance Matrix ({noise_config})')
ax.set_xlabel('r (Mpc)')
ax.set_ylabel('r (Mpc)')
fig.colorbar(c, ax=ax, label=r'Value')

# --- save and close ---
fig.savefig(f"{path_to_save}data_cov_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

############ 2. b. Stability Check (Data Cov * Inv Data Cov) with histogram of values
I = np.log10(abs(np.matmul(np.linalg.inv(data_cov_matrix), data_cov_matrix))+1e-16)

# First plot
fig1, ax1 = plt.subplots()
c = ax1.pcolormesh(rvals,
                   rvals,
                   I)
fig1.colorbar(c, ax=ax1).set_label('log$_{10} C^{-1}C$')
ax1.set_xlabel('r (Mpc)')
ax1.set_ylabel('r (Mpc)')
ax1.set_title('Stability Check Matrix')

fig1.savefig(f"{path_to_save}stability_check_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig1)


# Second plot
fig2, ax2 = plt.subplots()
ax2.hist(I[I != 1.].reshape(-1), bins=100)
ax2.set_xlabel('log$_{10} C^{-1}C$')
ax2.set_yscale('log')
ax2.set_title('Histogram of Values in Stability Check Matrix')
ax2.set_ylabel('Counts')

fig2.savefig(f"{path_to_save}stability_check_hist.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

############ 2. c. Condition Number vs Subset of Data Cov Matrix Plot

results_for_condition_num_test = compute_tcf_fisher(TCF_clean_dict, TCF_noisy_dict, delta_params, sim_params, max_idx=100)
data_cov_matrix_for_condition_num_test = results_for_condition_num_test['data_cov_matrix']

cond_num_list = []
val_range = np.arange(10, 100, step=1)

for i in val_range:
    min_val = 0
    max_val = i

    # compute the condition number
    cond_num = np.log10(np.linalg.cond(data_cov_matrix[min_val:max_val, min_val:max_val]))        
    cond_num_list.append(cond_num)

fig, ax = plt.subplots()
ax.plot(val_range, cond_num_list)
ax.set_ylabel('log Condition Number')
ax.set_xlabel('Max r bin index')
ax.set_title(f'Comparison of Condition Numbers of Sub Sections of the Data Cov Matrix ({noise_config})')
ax.grid(True)

fig.savefig(f"{path_to_save}condition_number_test.png", dpi=300, bbox_inches="tight")
plt.close(fig)

############ 3. Convergence of the Inv FM diagonal elements ########################################################################################
Inv_FMs_list = results['res_cov_matrix']
samples = np.arange(5, n_realisations + 5, 10)

# --- Absolute convergence plot ---
fig1, ax1 = plt.subplots(figsize=(12, 4))
for i, param in enumerate(sim_params):
    if 'HII' in param:
        color = 'orange'
    if 'R_BUBBLE_MAX' in param:
        color = 'red'
    if 'Tvir' in param:
        color = 'green'
        
    y_vals = [np.diag(Inv_FMs_list[k])[i] - np.diag(Inv_FMs_list[-1])[i] 
              for k in range(samples.size)]
    ax1.plot(samples, y_vals, marker='.', label=param, color=color)

ax1.legend()
ax1.set_ylabel(r'$\sigma^2_{ii}$')
ax1.set_xlabel('Number of samples')
ax1.set_title('Convergence of the Fisher Inverse Matrix Diagonal Elements')

fig1.savefig(f"{path_to_save}inv_FM_convergence_diag.png", dpi=300, bbox_inches="tight")
plt.close(fig1)


# --- Normalised convergence plot ---
fig2, ax2 = plt.subplots(figsize=(12, 4))
for i, param in enumerate(sim_params):
    if 'HII' in param:
        color = 'orange'
    if 'R_BUBBLE_MAX' in param:
        color = 'red'
    if 'Tvir' in param:
        color = 'green'
    y_vals = [np.diag(Inv_FMs_list[k])[i] - np.diag(Inv_FMs_list[-1])[i] 
              for k in range(samples.size)]
    max_val = np.max(np.abs(y_vals))
    y_vals_normalized = np.array(y_vals) / max_val if max_val != 0 else np.zeros_like(y_vals)
    ax2.plot(samples, y_vals_normalized, marker='.', label=param, color=color)

ax2.legend()
ax2.set_ylabel(r'$\sigma^2_{ii}$ (normalised)')
ax2.set_xlabel('Number of samples')
ax2.set_title('Normalised Convergence of the Fisher Inverse Matrix Diagonal Elements')

fig2.savefig(f"{path_to_save}inv_FM_convergence_diag_normalised.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

############ 4. Plots of the Derivatives ###########################################################################################################
def plot_derivatives(derivatives, rvals, labels=None, title=None, savepath=None):
    """
    Plot mean ± 1σ over realisations for each parameter's derivative vs r.
    Save the figure to a file.
    
    """
    n_params, n_reals, n_bins = derivatives.shape
    assert rvals.shape[0] == n_bins, "rvals must have same length as the last dim of derivatives"
    if labels is None:
        labels = [f"param {i+1}" for i in range(n_params)]

    mean = np.nanmean(derivatives, axis=1)  # (n_params, n_bins)
    std  = np.nanstd(derivatives, axis=1)   # (n_params, n_bins)

    valid = np.isfinite(rvals)
    r = rvals[valid]

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2.8*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        m = mean[i, valid]
        s = std[i, valid]

        if labels[i] == 'Tvir':
            color = 'green'

        if labels[i] == 'Rmax':
            color = 'red'

        if labels[i] == 'HII':
            color = 'orange'

        ax = axes[i]
        ax.plot(r, m, lw=1.8, label=labels[i], color=color)
        ax.axhline(0, ls='--', lw=0.8)
        ax.set_ylabel("derivative")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("r")
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved plot to {savepath}")
    else:
        plt.show()

plot_derivatives(
    derivatives, rvals,
    labels=["Tvir", "Rmax", "HII"],
    title="Derivatives",
    savepath=f"{path_to_save}derivatives_all_params.png"
)


############ 5. 2D slices of SIM data (fiducial clean and with noise) ##############################################################################

def plot_2d_sim_slices(simpaths, titles, savepath=None):
    """
    Plot 2D slices from four simulation files in a 2x2 grid.

    Parameters
    ----------
    simpaths : list[str]
        List of 4 HDF5 file paths.
    titles : list[str]
        List of 4 titles, one per subplot.
    savepath : str or None
        If given, save the figure to this path. Otherwise, show interactively.
    """
    assert len(simpaths) == 4, "Need exactly 4 sim paths"
    assert len(titles) == 4, "Need exactly 4 titles"

    slices = []
    for sp in simpaths:
        with h5py.File(sp, "r") as f:
            slices.append(f['brightness_lightcone'][0, 0, :, :])
            L_Mpc_perh = f['box_length'][...]
            L_Mpc = L_Mpc_perh/0.6674
            print(L_Mpc)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for ax, slc, title in zip(axes, slices, titles):
        im = ax.imshow(slc, origin='lower', cmap='RdBu_r', extent=(0, L_Mpc, 0, L_Mpc))
        ax.set_xlabel('x [Mpc]')
        ax.set_ylabel('y [Mpc]')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7, label=r'$\delta T_b$ [mK]')

    fig.suptitle(f"Comparison of 2D Sim Slices (Noise Config: {noise_config})", fontsize=16)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved 4-panel plot to {savepath}")
    else:
        plt.show()


# Example use:
plot_2d_sim_slices(
    [simlist[0], noisy_simlist[0], noisy_simlist[1], noisy_simlist[2]],
    titles=["Clean", "Noisy", "Obs", "Noise Only Lc"],
    savepath=f"{path_to_save}2dslices_FID_sim.png"
)


############ 6. TCF plots (clean and with noise) ###################################################################################################
def plot_avg_tcf(tcf_dict, title="Dictionary", keys=None, show_errorband=False, fill_alpha=0.25, savepath=None):
    """
    Plot average TCF vs rvals for selected entries in a TCF dictionary,
    with optional ±1σ error bands.

    Parameters
    ----------
    tcf_dict : dict
        Dictionary with structure {key: {"tcf": array, "rvals": array}}.
    title : str
        Title for the plot.
    keys : list[str] or None
        List of keys to plot. If None, plot all entries.
    show_errorband : bool
        If True, plot ±1σ error bands around the mean.
    fill_alpha : float
        Transparency for error bands.
    """
    plt.figure(figsize=(8, 5))
    
    if keys is None:
        keys = list(tcf_dict.keys())
    
    for key in keys:
        print(key)
        if 'HII' in key:
            color = 'orange'
        if 'R_BUBBLE_MAX' in key:
            color = 'red'
        if 'Tvir' in key:
            color = 'green'
        if 'FID' in key:
            color = 'C0'
            linestyle = '-'
        if 'Plus' in key:
            linestyle = '--'
        if 'Minus' in key:
            linestyle = ':'
            
        if 'FID_NOISE' in key:
            color = 'blueviolet'
        if 'FID_NOISE_SMOOTHING' in key:
            color = 'limegreen'
        if 'FID_NOISE_ONLY_LC' in key:
            color = 'coral'
        
        if key in tcf_dict and "tcf" in tcf_dict[key] and "rvals" in tcf_dict[key]:
            tcf     = np.asarray(tcf_dict[key]["tcf"])   # (n_real, n_bins)
            rvals   = np.asarray(tcf_dict[key]["rvals"])
            avg_tcf = np.mean(tcf, axis=0)
            std_tcf = np.std(tcf, axis=0)

            plt.plot(rvals, avg_tcf, label=key, color=color, linestyle=linestyle)
            if show_errorband:
                plt.fill_between(rvals, avg_tcf - std_tcf, avg_tcf + std_tcf, alpha=fill_alpha)
        else:
            print(f"⚠️ Skipping {key}: not found or missing data.")
    
    plt.title(f"{title}")
    plt.xlabel("r (Mpc)")
    plt.ylabel("S(r)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

############ 6. a. All clean sims

# all clean sims
plot_avg_tcf(TCF_clean_dict, "TCFs of All Clean Sims", keys=None, show_errorband=False, savepath=f"{path_to_save}TCFs_all_clean_sims.png")


############ 6. b. Comparison between plus/minus sims

# HII_EFF_FACTOR ±
plot_avg_tcf(TCF_clean_dict, "TCF Comparison plus/minus HII", keys=["FID", "HII_EFF_FACTOR_Plus", "HII_EFF_FACTOR_Minus"], show_errorband=False, savepath=f"{path_to_save}HII_TCF_comparison.png")

# ION_Tvir_MIN ±
plot_avg_tcf(TCF_clean_dict, "TCF Comparison plus/minus Tvir", keys=["FID", "ION_Tvir_MIN_Plus", "ION_Tvir_MIN_Minus"], show_errorband=False, savepath=f"{path_to_save}Tvir_TCF_comparison.png")

# R_BUBBLE_MAX ±
plot_avg_tcf(TCF_clean_dict, "TCF Comparison plus/minus Rmax", keys=["FID", "R_BUBBLE_MAX_Plus", "R_BUBBLE_MAX_Minus"], show_errorband=False, savepath=f"{path_to_save}Rmax_TCF_comparison.png")


############ 6. c. Noisy sims

# Plot everything in noisy dict
plot_avg_tcf(TCF_noisy_dict, f"TCFs of Noisy sims ({noise_config})", keys=["FID_NOISE", "FID_NOISE_SMOOTHING", "FID_NOISE_ONLY_LC"], show_errorband=False, savepath=f"{path_to_save}TCFs_all_noisy_sims.png")




