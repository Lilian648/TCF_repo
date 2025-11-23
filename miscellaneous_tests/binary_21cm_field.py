"""
====================================================================================================
Script: Compute Binary-Field TCFs from 21cmFAST Mock Simulation
Author: Liliane (PhD — EoR / TCF Pipeline)

DESCRIPTION
-----------
This script takes 21cmFAST-like mock simulation fields (stored as 2D text slices), converts them 
into binary ionisation-like fields, computes the Triangle Correlation Function (TCF) for each 
binary realisation using the external C++ TCF code (SC.h / SC_2d.o), and finally plots the ensemble 
of TCFs (all realisations + mean + ±1σ band) for a quick visual “Binary Field Morphology Test”.

The script is structured in three major phases:

----------------------------------------------------------------------------------------------------
PHASE 1 — Convert original 21cm slices into binary fields
----------------------------------------------------------------------------------------------------
• Input: A folder containing many `realisation_X.txt` files (brightness temperature slices).
• For each file:
    - Load the 2D field.
    - Convert it to a binary field:
          value != min(field) → 1
          value == min(field) → min(field)  (usually 0)
    - De-mean the resulting binary field.
    - Save it back to the same directory as `Binary_realisation_X.txt`.

Purpose:
Binary fields mimic ionisation masks (0/1 fields) and allow comparison with analytic and mock-bubble
theories. De-meaning ensures the TCF code receives a mean-zero field (required for correct behaviour).

----------------------------------------------------------------------------------------------------
PHASE 2 — Compute the TCF for each binary field
----------------------------------------------------------------------------------------------------
• Load simulation metadata from the original HDF5 simulation:
    - Box size in Mpc/h → converted to Mpc.
    - Grid dimension (DIM).
• Configure global TCF parameters:
    - nbins, rmin, rmax, nthreads, L (Mpc), DIM (pixels).
• Create a `Compute_TCF` class instance from `TCF_Class`, which:
    - Updates SC.h with global parameters (L, N, nbins, etc.).
    - For each field, rewrites the filename inside SC.h.
    - Compiles and runs `SC_2d.o` via `make`.
    - Reads the resulting spherical correlations into a pandas DataFrame.

• Loop over all `Binary_realisation_X.txt` files:
    - Compute the TCF.
    - Store each result in a dictionary keyed by the field name.

• Save all TCF results for the binary fields into a single pickle file:
      TCF_results_Binary_field.pkl
  (mapping filename → DataFrame with columns: r, Re_s(r), Im_s(r), N_modes)

----------------------------------------------------------------------------------------------------
PHASE 3 — Plot the Binary Field TCF Ensemble (Binary Field Morphology Test)
----------------------------------------------------------------------------------------------------
• Load `TCF_results_Binary_field.pkl`.
• Stack Re[s(r)] for all binary realisations.
• Compute:
    - The mean TCF across realisations.
    - The standard deviation at each r (for an error band).
• Create a diagnostic plot showing:
    - Thin grey line for each individual realisation.
    - Thick coloured line for the mean Re[s(r)].
    - A shaded ±1σ band around the mean.

• The plot is titled:
      "Binary Field Morphology Test — TCF Ensemble (Mean ± 1σ)"
  and can be saved as:
      plots_TCF_binary_field/TCF_ensemble_Binary_field.png
  (saving is controlled by commenting/uncommenting the `plt.savefig(...)` line).

Purpose:
This final plot provides a quick visual check of the morphology encoded in the binary fields as seen 
by the TCF: how consistent the realisations are, the typical amplitude and shape of Re[s(r)], and the 
spread across the ensemble.


====================================================================================================
"""



import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import time
import re, subprocess
import pandas as pd
from datetime import datetime
import tqdm
from matplotlib import colors
import pickle as pkl
import corner

import sys
from pathlib import Path
# Add the parent directory (where TCF_Class.py lives) to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tools21cm as t2c

import TCF_Class
print("here we go")

####################################################################################################################################################
################################ Convert 21cmFAST fields to BINARY fields ##########################################################################
####################################################################################################################################################

def convert_21cm_to_binary_field(input_dir):
    print("converting 21cm fields to binary fields")

    pattern = re.compile(r"realisation_\d+\.txt$")   # exact match, ends with .txt
    
    for index, file_path in enumerate(sorted(input_dir.glob("realisation_*.txt"))):
        if not pattern.search(file_path.name):
            continue
 
        print(index)
        # Load the original field
        field = np.loadtxt(file_path)
        
        # Convert to binary (non-zero → 1)
        min_val = np.min(field)
        binary_field = np.where(field != min_val, 1, min_val)

        # de-mean field again
        mean_binary = np.mean(binary_field)
        binary_field = binary_field - mean_binary
        
        # Define new filename: add 'Binary_' prefix
        output_name = f"Binary_{file_path.name}"
        output_path = input_dir / output_name
        
        # Save binary field
        np.savetxt(output_path, binary_field, fmt="%.8e")
        
        print(f" Saved: {output_path.name}")
    
    print("\nAll binary fields created successfully in:", input_dir)


# --- Folder containing all your realisation text files ---
input_dir = Path("/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_Binary/Mock_SIM_Binary_txtfiles/Lightcone_zidx0")

convert_21cm_to_binary_field(input_dir)

####################################################################################################################################################
################################ Compute TCF of BINARY fields ######################################################################################
####################################################################################################################################################

# ----- Obtain Field Parameters ---- #
sim_path = Path("/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_Binary/Lightcone_MOCK_Binary.h5")
with h5py.File(sim_path, "r") as f:
    print("========= extracting sim parameters ========")
    L_Mpcperh = f['box_length'][...]                    # Mpc/h
    L = L_Mpcperh/0.6774                                # Mpc
    DIM = np.shape(f['brightness_lightcone'][...])[2]   # pixels

print("field length Moc/h", L_Mpcperh)
print("field length Mpc", L)
print("field dimensions pixels", DIM)

    
# --- Paths & params --- #
tcf_code_dir = Path("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files")
input_dir    = Path("/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_Binary/Mock_SIM_Binary_txtfiles/Lightcone_zidx0")

# Set these to your actual box size and DIM
L   = L      # Mpc
DIM = DIM    # pixels

nthreads = 5
nbins    = 100
rmin     = 3
rmax     = 100

# --- Build list of ONLY the desired files: Binary_realisation_X.txt --- #
# (sorted numerically by X if present)
pattern = re.compile(r"^Binary_realisation_(\d+)\.txt$")
file_list = sorted(
    (p for p in input_dir.glob("Binary_realisation_*.txt") if pattern.match(p.name)),
    key=lambda p: int(pattern.match(p.name).group(1))
)

print(f"Found {len(file_list)} files matching Binary_realisation_*.txt")


# --- Create TCF class instance (updates SC.h once with global params) --- #
tcf = TCF_Class.Compute_TCF(
    tcf_code_dir=tcf_code_dir,
    L=L,
    DIM=DIM,
    nthreads=nthreads,
    nbins=nbins,
    rmin=rmin,
    rmax=rmax
)

# --- Loop through selected files and compute TCF --- #
results = {}

for file_path in file_list:
    print(f"\n Computing TCF for {file_path.name}")
    try:
        df = tcf.compute_TCF_of_single_field(file_path)
        results[file_path.stem] = df
        print(f" Done — TCF computed for {file_path.name}")
    except Exception as e:
        print(f" Failed on {file_path.name}: {e}")

print("\nAll selected files processed!")

# --- Save a pickle with all DataFrames (optional) ---
out_pkl = Path("/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_Binary/TCF_results_Binary_field.pkl")

with open(out_pkl, "wb") as f:
    pkl.dump(results, f)

print(f" Saved all TCF results to: {out_pkl}")




####################################################################################################
########################## Plot TCFs of Binary Fields (Mean + ±1σ + Realisations) ##################
####################################################################################################

print("\n=============== Plotting Binary Field TCF Ensemble ===============")

# --- Path to the saved pickle file ---
pkl_path = out_pkl

# --- Load results dictionary from pickle ---
with open(pkl_path, "rb") as f:
    results_dict = pkl.load(f)

print(f"Loaded {len(results_dict)} TCF realisations from:\n  {pkl_path}")

# --- Sort dictionary entries for reproducibility ---
items = sorted(results_dict.items())

# --- Extract r grid (from the first TCF) ---
first_name, first_df = items[0]
r = first_df["r"].values

# --- Stack Re[s(r)] values ---
Re_all = []

for name, df in items:
    if not np.allclose(df["r"].values, r):
        raise ValueError(f"❌ r-grid mismatch in file: {name}")
    Re_all.append(df["Re_s_r"].values)

Re_all = np.vstack(Re_all)        # shape = (n_realisations, n_r)
mean_Re = Re_all.mean(axis=0)     # mean across slices
std_Re  = Re_all.std(axis=0)      # standard deviation


# --- Plotting ---
plt.figure(figsize=(7, 5))

# 1) Plot all realisations (light grey)
for arr in Re_all:
    plt.plot(r, arr, color="gray", alpha=0.25, linewidth=0.8)

# 2) Plot mean
plt.plot(r, mean_Re, color="C0", linewidth=2, label="Mean Re[s(r)]")

# 3) Plot ±1σ error band
plt.fill_between(
    r,
    mean_Re - std_Re,
    mean_Re + std_Re,
    color="C0",
    alpha=0.3,
    label="±1σ"
)

plt.xlabel("r (Mpc)")
plt.ylabel("Re[s(r)]")
plt.title("Binary Field Morphology Test\nTCF Ensemble (Mean ± 1σ)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# --- Prepare output directory ---
plots_dir = pkl_path.parent / "plots_TCF_binary_field"
plots_dir.mkdir(parents=True, exist_ok=True)
save_path = plots_dir / "TCF_ensemble_Binary_field.png"

#plt.savefig(save_path, dpi=200, bbox_inches="tight") # uncomment to save
plt.close()

#print(f"✅ Saved TCF plot to:\n  {save_path}\n")
