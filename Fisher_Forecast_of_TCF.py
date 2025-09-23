# not sure which functions are neccessary yet


####################################################################################################################################################
################################ Global Parameters #################################################################################################
####################################################################################################################################################
 
### --- Astrophysical parameters of the simulation --- ###
params = ['ION_Tvir_MIN','R_BUBBLE_MAX','HII_EFF_FACTOR']
Fisher_Param = ['$T_{Vir}$','$R_{Max}$','$\zeta$']

nparams = len(params)

# parameter values for fiducial simulation
fid = [50000, 15, 30] # Tvir, Rmax, HII 

# parameter ± for derivatives
delta_params = [5000, 5, 5] # Tvir, Rmax, HII  


### --- simulation h5 files --- ###
# - clean sims - #
SIM_FID = '/data/cluster/lcrascal/SIM_data/SIM_FID/Lightcone_FID_400_Samples.h5'

SIM_HII_plus = '/data/cluster/lcrascal/SIM_data/SIM_HII_plus/Lightcone_HII_EFF_FACTOR_400_Samples_Plus.h5'
SIM_HII_minus = '/data/cluster/lcrascal/SIM_data/SIM_HII_minus/Lightcone_HII_EFF_FACTOR_400_Samples_Minus.h5'

SIM_Rmax_plus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_plus/Lightcone_R_BUBBLE_MAX_400_Samples_Plus.h5'  
SIM_Rmax_minus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_minus/Lightcone_R_BUBBLE_MAX_400_Samples_Minus.h5'

SIM_Tvir_plus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_plus/Lightcone_ION_Tvir_MIN_400_Samples_Plus.h5'
SIM_Tvir_minus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_minus/Lightcone_ION_Tvir_MIN_400_Samples_Minus.h5'

# - Noisy sims (AAstar 100hrs) - #
SIM_FID_Noisy = '/data/cluster/lcrascal/SIM_data/SIM_FID/AAstar_100hrs/Lightcone_FID_400_Samples_NOISE.h5'
SIM_FID_Obs = '/data/cluster/lcrascal/SIM_data/SIM_FID/AAstar_100hrs/Lightcone_FID_400_Samples_NOISE_SMOOTHING.h5'
SIM_FID_NoiseOnly = '/data/cluster/lcrascal/SIM_data/SIM_FID/AAstar_100hrs/Lightcone_FID_400_Samples_NOISE_ONLY_LC.h5'

simlist = [
    SIM_FID,           
    SIM_HII_plus,
    SIM_HII_minus,
    SIM_Rmax_plus]#,
    #SIM_Rmax_minus,
    #SIM_Tvir_plus,
    #SIM_Tvir_minus
#]

noisy_simlist = [
    SIM_FID_Noisy, 
    SIM_FID_Obs, 
    SIM_FID_Nois# parameters describing the simulations:

    
 ### --- parameters describing the simulation --- ###

with h5py.File(SIM_FID, 'r') as f:
    frequencies = f['frequencies'][...]
    redshifts = f['redshifts'][...]
    box_length = float(f['box_length'][0])  # Mpc/h
    box_dim = int(f['ngrid'][0])
    n_realisations = int(f['nrealisations'][0])
nfreq = frequencies.size
print(f'Lightcone runs from z={redshifts.min():.2f} to z = {redshifts.max():.2f}.')eOnly]


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
TCF_clean = {}
TCF_noisy = {}

# Loop clean
for f in simlist:
    key = (
        Path(f).stem
        .replace("Lightcone_", "")
        .replace("_400_Samples", "")
    )
    TCF_clean[key] = load_tcf_data(f)

# Loop noisy
for f in noisy_simlist:
    key = (
        Path(f).stem
        .replace("Lightcone_", "")
        .replace("_400_Samples", "")
    )
    TCF_noisy[key] = load_tcf_data(f)


print(f"✅ Loaded {len(TCF_clean)} clean sims, {len(TCF_noisy)} noisy sims")


# checking and plots
# def inspect_tcf_dict(tcf_dict, name="Dictionary"):
#     """
#     Print summary of a TCF dictionary.
    
#     Parameters
#     ----------
#     tcf_dict : dict
#         Dictionary with structure {key: {"tcf": array, "rvals": array}}.
#     name : str
#         Label for the dictionary in the printout.
#     """
#     print(f"\n📂 {name}: {len(tcf_dict)} entries")
#     for key, dat in tcf_dict.items():
#         tcf_shape = dat["tcf"].shape if "tcf" in dat else None
#         r_shape   = dat["rvals"].shape if "rvals" in dat else None
#         print(f"  {key:25s} → TCF {tcf_shape}, rvals {r_shape}")


# inspect_tcf_dict(TCF_clean, "Clean sims")
# inspect_tcf_dict(TCF_noisy, "Noisy sims (AAstar100)")

# def plot_avg_tcf(tcf_dict, name="Dictionary", keys=None):
#     """
#     Plot average TCF vs rvals for selected entries in a TCF dictionary.
    
#     Parameters
#     ----------
#     tcf_dict : dict
#         Dictionary with structure {key: {"tcf": array, "rvals": array}}.
#     name : str
#         Title for the plot.
#     keys : list[str] or None
#         List of keys to plot. If None, plot all entries.
#     """
#     plt.figure(figsize=(8, 5))
    
#     # default: plot all keys
#     if keys is None:
#         keys = list(tcf_dict.keys())
    
#     for key in keys:
#         if key in tcf_dict and "tcf" in tcf_dict[key] and "rvals" in tcf_dict[key]:
#             avg_tcf = np.mean(tcf_dict[key]["tcf"], axis=0)   # average over realisations
#             rvals   = tcf_dict[key]["rvals"]
#             plt.plot(rvals, avg_tcf, label=key)
#         else:
#             print(f"⚠️ Skipping {key}: not found or missing data.")
    
#     plt.title(f"{name}")
#     plt.xlabel("r")
#     plt.ylabel("S(r)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

    
# # Plot only HII_EFF_FACTOR ±
# plot_avg_tcf(TCF_clean, "FID sim", keys=["FID"])

# # Plot everything in noisy dict
# plot_avg_tcf(TCF_noisy, "Noisy sims (AAstar100)")


####################################################################################################################################################
################################ Function to Compute Fisher Forecast ###############################################################################
####################################################################################################################################################


def compute_tcf_fisher(tcf_data_dict, 
    delta_params,
    sim_params,
    n_realisations
):
    """Compute Fisher matrix from TCF simulation files.""" 

    ############ Compute derivatives ############
    print("Computing Derivatives")
    rvals_len = tcf_data_dict[f'{sim_params[0]}_Plus']['rvals'].shape[0]
    TCF_derivs = np.zeros((len(sim_params), n_realisations, rvals_len))
    for idx, param in enumerate(sim_params):
        plus = tcf_data_dict[f'{param}_Plus']['tcf_clean']
        minus = tcf_data_dict[f'{param}_Minus']['tcf_clean']
        TCF_derivs[idx] = (plus - minus) / delta_params[idx]
    rvals = tcf_data_dict[f'{sim_params[0]}_Plus']['rvals']

    ############# Whitening the Data ############
    print("Whitening Data ")
    TCF_fid = tcf_data_dict['FID']['tcf_obs'] # change to TCF_fid_obs?
    std_fid = np.std(TCF_fid, axis=0) # change to std_fid_obs ?
    mask = std_fid > 0
    white_fid = TCF_fid[:, mask] / std_fid[mask]
    rvals = rvals[mask]
    TCF_derivs_white = TCF_derivs[..., mask] / std_fid[mask]

    ############# Compute Data Cov Matrix ############
    print("Computing Data Cov Matrix")
    data_cov_matrix = np.cov(white_fid, rowvar=False)
    cond_num = np.log10(np.linalg.cond(data_cov_matrix))
    print(f'log10 Condition number: {cond_num:.2f}')

    ############# Convergence testing + Compute FM ############
    print("Convergence Testing ")
    samples = np.arange(5, n_realisations + 5, 10)
    nparams = len(sim_params)
    TCF_Fisher = np.zeros((samples.size, nparams, nparams))
    TCF_Fisher_Inv = np.zeros((samples.size, nparams, nparams))

    for r, sample_size in enumerate(samples):
        deriv_sample = np.mean(TCF_derivs_white[:, :sample_size, :], axis=1)
        for i in range(nparams):
            for j in range(nparams):
                TCF_Fisher[r, i, j] = np.dot(deriv_sample[i], np.dot(np.linalg.inv(data_cov_matrix), deriv_sample[j]))
        TCF_Fisher_Inv[r] = np.linalg.inv(TCF_Fisher[r])

    return {
        'TCF_derivs': TCF_derivs,             # derivatives
        'data_cov_matrix': data_cov_matrix,   # data covariance matrix
        'TCF_Fisher': TCF_Fisher,             # Fisher Matrix ([-1] = final FM)
        'TCF_Fisher_Inv': TCF_Fisher_Inv,     # Inverse Fisher Matrix = Result Covariance Matrix ([-1] = final inverse FM)
        'rvals': rvals                        # r values
    }




