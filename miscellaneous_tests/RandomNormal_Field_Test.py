"""
=======================================================================================

This script constructs a *Gaussian-noise mock 21cm lightcone* by taking an existing
noise-only 21cmFAST lightcone (generated with tools21cm, e.g., AA*100 hrs) and
replacing every brightness-temperature slice with a Gaussian random field that has
the *same slice-by-slice mean and variance*.

──────────────────────────────────────────────────────────────────────────────
PURPOSE
──────────────────────────────────────────────────────────────────────────────
The goal is to isolate and understand the behaviour of the Triangle Correlation
Function (TCF) in the presence of noise alone.

tools21cm generates SKA-like thermal noise that has:
    • baseline-dependent variance,
    • correlated structure,
    • non-Gaussian features from the uv-coverage,
    • redshift-dependent effects.

By replacing this with controlled Gaussian noise fields that preserve only the
slice mean and variance, we create a simplified “null” noise model. By comparing the
TCF measured from:
    (a) tools21cm noise-only lightcones, and
    (b) Gaussian-noise-only lightcones,
    
we can determine whether the unusual noise-independence of the TCF arises from:
    • tools21cm’s uv-sampling noise structure,  
      or
    • is still present for fields with generic Gaussian noise with the same 
      amplitude.

This makes the script a crucial diagnostic tool for understanding why the TCF
appears insensitive to noise level.

──────────────────────────────────────────────────────────────────────────────
SCRIPT STRUCTURE
──────────────────────────────────────────────────────────────────────────────

1. **Configuration**
   - Defines source 21cmFAST simulation file (with SKA-like AA*100 noise).
   - Defines destination path for the mock Gaussian dataset.

2. **Copy H5 File**
   - Copies the original HDF5 file to a new location so the mock dataset can be built
     without modifying the original simulation.

3. **Remove Old TCF Datasets**
   - Deletes any `TCF_*` datasets from the copied file to prevent conflicts or accidental reuse.

4. **Replace Brightness-Lightcone Field**
   - Reads the original field: shape = (n_realisations, n_redshifts, nx, ny).
   - Creates a new dataset: `brightness_lightcone_RN`.
   - For every slice (realisation i, redshift z):
        • computes slice mean and standard deviation,
        • generates a Gaussian random field with the same μ and σ,
        • writes it into the new dataset.
   - The resulting field has identical pixel statistics to the original, but no morphology.

5. **Output**
   - Produces a completely synthetic Gaussian field lightcone that mirrors the noise
     and brightness distribution of the original simulation.
   - This mock dataset is later used to compute TCFs and assess whether TCF features
     originate from physical morphology or random fluctuations.

======================================================================================

"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import re, subprocess
import pandas as pd
from datetime import datetime
import tqdm
from matplotlib import colors
import pickle as pkl
import corner
import shutil

import sys
from pathlib import Path
# Add the parent directory (where TCF_Class.py lives) to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tools21cm as t2c

import TCF_Class
print("AASTAR 100")

# ================================================================
# CREATE MOCK RANDOM NORMAL FIELD
# ================================================================

################ CONFIGURATION ################

source_path = Path("/data/cluster/lcrascal/SIM_data/SIM_FID/AAstar_100hrs/Lightcone_FID_400_Samples_NOISE.h5")
dest_path   = Path("/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_RandomNormal/Lightcone_Mock_RN_AAstar100.h5")

dest_path.parent.mkdir(parents=True, exist_ok=True)

################ COPY ORIGINAL FILE ################

print(f"Copying H5 file to {dest_path} ...")
shutil.copy2(source_path, dest_path)
print("File copied successfully.\n")

################ OPEN COPIED FILE AND REMOVE TCF DATASETS ################

with h5py.File(dest_path, "r+") as f:
    print("Existing datasets before cleanup:")
    for name in f.keys():
        print(" -", name)
    
    # List of TCF datasets to delete
    tcf_keys = [k for k in f.keys() if "TCF" in k]
    print("\nDeleting TCF-related datasets:", tcf_keys)
    
    for k in tcf_keys:
        del f[k]
    
    print("TCF datasets deleted.\n")

################ REPLACE BRIGHTNESS FIELD WITH RANDOM GAUSSIAN FIELD ################

with h5py.File(dest_path, "r+") as f:
    data = f["brightness_lightcone"]
    shape = data.shape  # (n_real, n_z, nx, ny)
    n_real, n_z, nx, ny = shape
    print(f"Brightness field shape: {shape}")

    # Create new dataset (overwrite safely)
    #del f["brightness_lightcone"]
    new_data = f.create_dataset("brightness_lightcone_RN", shape=shape, dtype=np.float32)

    print("Generating Gaussian random fields per slice...\n")
    for i in range(n_real):
        for z in range(n_z):
            print("i:", i, "z:", z)
            slice_data = data[i, z, :, :]  # original slice
            mu, sigma = slice_data.mean(), slice_data.std()
            new_slice = np.random.normal(loc=mu, scale=sigma, size=(nx, ny))
            new_data[i, z, :, :] = new_slice

            if i == 0 and z % 20 == 0:
                print(f"  Realisation {i}, z={z:3d} → mean={mu:.3f}, std={sigma:.3f}")

    print("\n All slices replaced with Gaussian random fields.")

print("\n Mock Gaussian H5 created at:", dest_path)


# ================================================================
# PLOT COMPARISON BETWEEN ORIGNAL AND GAUSSIAN NOISE FIELDS
# ================================================================

############ CONFIG — paths ############

int_hrs     = 100
noise_label = "AAstar"

RN_path       = Path(f"/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_RandomNormal/Lightcone_Mock_RN_{noise_label}{int_hrs}.h5")
truefield_path = Path(f"/data/cluster/lcrascal/SIM_data/SIM_FID/{noise_label}_{int_hrs}hrs/Lightcone_FID_400_Samples_NOISE.h5")

print(f"Comparing fields:\n  RN file:   {RN_path}\n  TRUE file: {truefield_path}\n")


############ HELPERS ############

def load_slice(path, r_idx=0, z_idx=0, dset_key="brightness_lightcone"):
    """Load a 2D slice (r_idx, z_idx) from a 3D/4D dataset inside an HDF5 file."""
    with h5py.File(path, "r") as f:
        ds = f[dset_key]

        if ds.ndim == 4:
            arr = ds[r_idx, z_idx]
        elif ds.ndim == 3:
            arr = ds[z_idx]
        else:
            raise ValueError(f"{dset_key} has unexpected shape {ds.shape}")

        # Try to read pixel length from file, otherwise infer from slice
        if "ngrid" in f:
            Lpix = int(np.array(f["ngrid"][()]).squeeze())
        else:
            Lpix = arr.shape[-1]

    return np.asarray(arr), Lpix


def shared_bins(a, b, bins=60):
    """Return bin edges spanning both arrays."""
    vmin = min(a.min(), b.min())
    vmax = max(a.max(), b.max())
    return np.linspace(vmin, vmax, bins + 1)


############ LOAD DATA############

RN_slice, RN_L = load_slice(RN_path, z_idx=0, dset_key="brightness_lightcone_RN")
TF_slice, TF_L = load_slice(truefield_path, z_idx=0, dset_key="brightness_lightcone")

# Statistics
RN_mean, RN_std = RN_slice.mean(), RN_slice.std()
TF_mean, TF_std = TF_slice.mean(), TF_slice.std()

print(f"[RN]   μ={RN_mean:.4f}, σ={RN_std:.4f}")
print(f"[TRUE] μ={TF_mean:.4f}, σ={TF_std:.4f}")

# Shared color scale
vmin = min(RN_slice.min(), TF_slice.min())
vmax = max(RN_slice.max(), TF_slice.max())


############ FIGURE 1 — SIDE-BY-SIDE IMAGES ############

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Random Normal field
im0 = axes[0].imshow(RN_slice, origin="lower", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Gaussian RN field\nμ={RN_mean:.2f}, σ={RN_std:.2f}")
axes[0].set_xlabel("pix"); axes[0].set_ylabel("pix")

# tools21cm noise field
im1 = axes[1].imshow(TF_slice, origin="lower", vmin=vmin, vmax=vmax)
axes[1].set_title(f"{noise_label}{int_hrs} tools21cm noise field\nμ={TF_mean:.2f}, σ={TF_std:.2f}")
axes[1].set_xlabel("pix"); axes[1].set_ylabel("pix")

# Shared colorbar
cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.90)
cbar.set_label(r"$\delta T_b$  [mK]")

plt.show()


############ FIGURE 2 — HISTOGRAM COMPARISON ############

bins = shared_bins(RN_slice, TF_slice, bins=60)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

# --- Gaussian RN histogram ---
ax = axes[0]
ax.hist(RN_slice.ravel(), bins=bins, alpha=0.8, edgecolor="black")
ax.axvline(RN_mean, ls="--", color="red", label=f"μ = {RN_mean:.2f}")
ax.axvline(RN_mean + RN_std, ls=":", color="orange", label=f"±1σ")
ax.axvline(RN_mean - RN_std, ls=":", color="orange")
ax.set_title("Gaussian RN pixel distribution")
ax.set_xlabel(r"$\delta T_b$  [mK]")
ax.set_ylabel("Count")
ax.legend()

# --- TRUE noise histogram ---
ax = axes[1]
ax.hist(TF_slice.ravel(), bins=bins, alpha=0.8, edgecolor="black")
ax.axvline(TF_mean, ls="--", color="red", label=f"μ = {TF_mean:.2f}")
ax.axvline(TF_mean + TF_std, ls=":", color="orange", label=f"±1σ")
ax.axvline(TF_mean - TF_std, ls=":", color="orange")
ax.set_title(f"{noise_label}{int_hrs} tools21cm noise distribution")
ax.set_xlabel(r"$\delta T_b$  [mK]")
ax.set_ylabel("Count")
ax.legend()

plt.show()


# ================================================================
# COMPUTE TCF OF THE RANDOM NORMAL SIM
# ================================================================
# THIS WONT WORK YET! 

# # --- file list --- #
# RN_mock_h5_AAstar100 = Path('/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_RandomNormal/Lightcone_Mock_RN_AAstar100.h5')
# simlist = [RN_mock_h5_AAstar100]

# print("SIMLIST:", simlist)

# # TCF parameters
# tcf_code_dir = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files"

# # Call run_all 
# run_all(
#     simlist=simlist,
#     tcf_code_dir=tcf_code_dir,   
#     z_indices=0,
#     nthreads=5,
#     nbins=100,
#     rmin=3,
#     rmax=100,
#     overwrite_h5=True,
#     overwrite_txt=False,
#     continue_on_error=True,
#     obs_time=1000, # XXXXXX
#     total_int_time=6.0, 
#     int_time=10.0, 
#     declination=-30.0, 
#     subarray_type="AAstar", # XXXXXX
#     save_uvmap="/data/cluster/lcrascal/uvmaps/uvmap_AAstar_1000hrs.h5",  # XXXXXX
#     njobs=1, 
#     checkpoint=8, 
#     bmax_km=2.0,
#     include_clean=True # XXXXXX
# )

# ================================================================
# PLOTS
# ================================================================