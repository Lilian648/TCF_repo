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
        np.savetxt(output_path, binary_field, fmt="%d")
        
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


