"""
This is the complete (or what will be the complete) pipeline for computing a TCF of the LoReLi sims
(computing TCF with TCFpy code, not c++)

1. input list of sims
2. add SKA noise (optional) to each sim
3. extract 2D slices 
4. compute TCF and save

+ checks 
"""

from TCFpy import tcf, get_bispec

import sys
print("Which env am i in?", sys.executable)

import os
import re
import time
import copy as cp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from pathlib import Path
import re, subprocess
import math
import matplotlib.patheffects as pe
import pickle as pkl
import time
import h5py

import tools21cm as t2c


################################################################################################################################################
#################################### Functions #################################################################################################
################################################################################################################################################

def extract_LoReLi_slices_every_dMpc(
    cube_3d,
    output_dir,
    box_size_mpc=296.0,
    delta_mpc=10.0,
    axis="z",
    demean=True
):
    """
    Extract 2D slices from a 3D LoReLi cube at regular physical intervals
    (e.g. every 10 Mpc) and save them as .txt files.

    Parameters
    ----------
    cube_3d : np.ndarray
        3D array of shape (Nx, Ny, Nz), e.g. cube_z6 = sim.T21cm[37].
    output_dir : str or Path
        Directory where the .txt slices will be saved.
    box_size_mpc : float
        Physical size of the simulation box in Mpc (assumed cubic).
    delta_mpc : float
        Desired spacing between slices in Mpc (e.g. 10 Mpc).
    axis : {"x", "y", "z", 0, 1, 2}
        Axis along which to take slices. Can be int or "x"/"y"/"z".
    demean : bool
        If True, subtract mean from each 2D slice before saving.

    Returns
    -------
    saved_files : list of str
        List of full file paths to the saved .txt slices.
    """

    cube_3d = np.asarray(cube_3d)
    if cube_3d.ndim != 3:
        raise ValueError(f"cube_3d must be 3D, got shape {cube_3d.shape}")

    # Map axis if given as string
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis.lower()
        if axis not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', 'z', 0, 1, or 2, got {axis}")
        axis = axis_map[axis]

    N = cube_3d.shape[axis]
    cell_size_mpc = box_size_mpc / N

    # Closest integer step in cells to target delta_mpc
    step_cells = int(round(delta_mpc / cell_size_mpc))
    step_cells = max(step_cells, 1)  # just in case

    print(f"Box size: {box_size_mpc} Mpc")
    print(f"N cells along axis {axis}: {N}")
    print(f"Cell size: {cell_size_mpc:.4f} Mpc")
    print(f"Requested spacing: {delta_mpc} Mpc")
    print(f"Using step of {step_cells} cells (~{step_cells*cell_size_mpc:.3f} Mpc)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Indices along chosen axis
    slice_indices = np.arange(0, N, step_cells, dtype=int)

    for idx in slice_indices:
        # Extract the 2D slice depending on axis
        if axis == 0:
            slice_2d = cube_3d[idx, :, :]
        elif axis == 1:
            slice_2d = cube_3d[:, idx, :]
        elif axis == 2:
            slice_2d = cube_3d[:, :, idx]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        if demean:
            mu = slice_2d.mean()
            slice_2d = slice_2d - mu

        # Physical position of this slice from the "origin"
        r_mpc = idx * cell_size_mpc

        r_mpc_int = int(round(r_mpc))
        fname = output_dir / f"slice_axis{axis}_idx{idx}_r{r_mpc_int}Mpc.txt"
        np.savetxt(fname, slice_2d, fmt="%.8e")
        saved_files.append(str(fname))

        print(f"  ✔ Saved slice at idx={idx}, r≈{r_mpc:.2f} Mpc -> {fname}")

    print("\n All slices extracted.")
    return saved_files


def pyTCF_of_2Dslice(field2d, L, rvals, outfile):
    """
    Compute the TCF of a single 2D field.

    Parameters
    ----------
    field2d : 2D array
        Real-space field (e.g. one z-slice of a lightcone)
    L : float
        Physical size of the box (Mpc)
    rvals : array
        Scales at which to compute the TCF
    outfile : str or None, optional
        If provided, path to a text file where the results are saved.

    Returns
    -------
    nmodes : nmodes : ndarray of int, shape (Nr,)
        Number of Fourier-space triangle configurations contributing to
        each r-bin.
    sr_vals : array
        TCF evaluated at each rval
    """
    tstart = time.time()
    print("tstart:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tstart)))

    ndim = 2

    print("computing bispectrum started")
    # 1) Compute phase-only bispectrum
    Bk = get_bispec(field2d, L).real # real par only
    print("computing bispectrum finsihed")

    print("computing tcf started")
    # 2) Compute TCF for each r
    sr_vals = np.zeros(len(rvals))
    nmodes = np.zeros(len(rvals))
    for i, r in enumerate(rvals):
        print(f" computing TCF for rval={i+1}/{len(rvals)}")
        nmodes[i], sr_vals[i] = tcf(r, Bk, L, ndim)

    print("computing tcf ended")


    print("saving")
    # Optional saving to text file
    if outfile is not None:
        header = (
            f"dim {ndim}\n"
            f"nmodes   s(r)   r"
        )

        data = np.column_stack((nmodes, sr_vals, rvals))
        np.savetxt(outfile, data, header=header, comments="# ")

    tend = time.time()
    print(f"time taken {tstart-tend:.1f}s")

    return nmodes, sr_vals


################################################################################################################################################
#################################### Main: Run All #############################################################################################
# currently incomplete - will put into main() function soon
################################################################################################################################################

def TCFpipeline_single_sim(sim_name, rvals, z_target=6, out_dir="tests/sn10038_txtfiles/zidx_37/", delta_mpc=10.0, overwrite_tcf=False):

    """
    CHANGES TO BE MADE:
        - want to define the output directory within the function so that it has the correct info in the folder names


    
    GOAL: Run the LoReLi -> slices -> TCF pipeline for one simulation.

    Parameters
    ----------
    sim_name : str
        LoReLi simulation identifier (e.g. "10038").
    rvals : array_like
        1D array of r values (Mpc) at which to compute the TCF.
    out_dir : str
        Directory where slice txt files and TCF result txt files will be saved.
    delta_mpc : float
        Physical spacing (Mpc) between extracted slices.
    z_target : float
        Target redshift; the cube closest to this redshift is used.
    overwrite_tcf : bool
        If True, recompute TCF results even if the output file already exists.

    Outputs
    -------
    - Slice files:       out_dir/slice_axis{axis}_idx{...}_r{...}Mpc.txt
    - TCF result files:  same name with suffix _TCFresult.txt
    - Summary plot:      out_dir/TCF_mean_z{z_target}.png
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rvals = np.asarray(rvals, dtype=float)

    print("CHECKING GRAPHS")


    # ----------------------------
    # 1. Load sim
    # ----------------------------
    # base_dir = '/data/cluster/emc-brid/Datasets/LoReLi' # where Lisa keeps the LoReLi info/sims

    # sim = Cat(sim_name,redshift_range='all', skip_early=False, path_spectra='spectra', path_sim='/data/cluster/emc-brid/Datasets/LoReLi/simcubes', 
    #           base_dir=base_dir, load_params=False, load_spectra=False, just_Pee=True, reinitialise_spectra=False, save_spectra=False, 
    #           load_density_cubes=False, load_xion_cubes=False, load_T21cm_cubes=True, verbose=True, debug=False)

    # # sim params
    # L = sim.box_size # Mpc (check units?)
    
    # ----------------------------
    # 2. Choose redshift cube 
    # (eg, choosing cube closest to z=6)
    # ---------------------------- 
    # z_idx = np.abs(sim.z - z_target).argmin()
    # z_used = float(sim.z[z_idx])
    # cube = sim.T21cm[z_idx]  # expected shape (N,N,N)

    #################################################################################
    ######################## TESTING VERSION (smaller cube) #########################
    #################################################################################
    cube = np.load(sim_name) # a 3d cube array (NOT the loreli sim object)
    L = 296/4

    # ----------------------------
    # 3. Extract slices for all axes
    # (all three axes - will all save to save folder with indication of axis in filename)
    # ---------------------------- 
    # saved_files = []
    # axes = ["x", "y", "z"]
    # for ax in axes:
    #     saved_files += extract_LoReLi_slices_every_dMpc(cube_3d=cube, output_dir=out_dir, box_size_mpc=L, delta_mpc=delta_mpc, 
    #                                                     axis=ax, demean=True)


    # # ----------------------------
    # # 4. Compute TCF for each slice file
    # # ----------------------------
    # tcf_files = []
    # tstart = time.time()
    # for index, slice_file in enumerate(saved_files):
        
    #     print(f"Realisation {index+1}/{len(saved_files)}")
    #     slice_path = Path(slice_file)

    #     # Output naming: same file name + _TCFresult before extension
    #     out_path = slice_path.with_name(slice_path.stem + "_TCFresult.txt")

    #     if out_path.exists() and not overwrite_tcf:
    #         print(f"  ↪ Skipping existing TCF: {out_path}")
    #         tcf_files.append(str(out_path))
    #         continue

    #     field2d = np.loadtxt(slice_path)

    #     nmodes, sr = pyTCF_of_2Dslice(field2d, L, rvals, str(out_path))
    #     tcf_files.append(str(out_path))
        
    # tend = time.time()
    # print(f"total time taken for all realisations = {tstart-tend:.1f}s") 

    # ----------------------------
    # 5) Load all TCF result files and make mean + 1σ band plot
    # ----------------------------
    # If you prefer, you can use `tcf_files` directly; glob is handy if you rerun later.
    tcf_files = sorted(out_dir.glob("*_TCFresult.txt"))

    if len(tcf_files) == 0:
        raise RuntimeError(f"No *_TCFresult.txt files found in {out_dir}")

    sr_stack = []
    for f in tcf_files:
        data = np.loadtxt(f)
        # file columns: nmodes, s(r), r  (as you save them)
        s = data[:, 1]
        r = data[:, 2]
        sr_stack.append(s)

    sr_stack = np.vstack(sr_stack)              # shape (Nslices, Nr)
    sr_mean = sr_stack.mean(axis=0)
    sr_std = sr_stack.std(axis=0, ddof=1) if sr_stack.shape[0] > 1 else np.zeros_like(sr_mean)

    # plot
    plt.figure()
    plt.plot(rvals, sr_mean, label=f"mean over {sr_stack.shape[0]} slices")
    plt.fill_between(rvals, sr_mean - sr_std, sr_mean + sr_std, alpha=0.3, label="±1σ")
    plt.xlabel("r [Mpc]")
    plt.ylabel("s(r)")
    #plt.title(f"LoReLi {sim_name}: TCF mean (z≈{sim.z[z_used]:.2f})") NON-TEST VERSION
    plt.title(f"LoReLi mock: TCF mean (z≈6)")  ## XXXXX
    plt.legend()
    plt.tight_layout()

    z_used = 6  ## XXXXX
    plot_path = out_dir / f"TCF_plt_z{str(z_used).replace('.','p')}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\n Done. Saved mean plot: {plot_path}")
    z_idx = 37 ## XXXXX
    return {
        "z_idx": z_idx,
        #"z_used": float(sim.z[z_idx]),
        #"slice_files": saved_files,
        "tcf_files": [str(p) for p in tcf_files],
    }

# sim_name = "mock_LoReLi_sim_N64.npy"
# rvals = np.linspace(2, 50, 49)
# results = TCFpipeline_single_sim(sim_name, rvals, z_target=6, out_dir="tests/sn10038_txtfiles/zidx_37/", delta_mpc=10.0, overwrite_tcf=True)





################################################################################################################################################
#################################### Adding Noise + Smoothing to Clean Slice ###################################################################
################################################################################################################################################



def make_observed_2Dslice(
    clean_xy,
    redshift,
    box_length_Mpc,
    noise_params,
    uvmap_filename,
    bmax_km,
):
    """
    Create an 'observed' 2D slice: clean + noise, then instrument smoothing.

    Returns
    -------
    noisy_xy : (N,N) clean+noise
    obs_xy   : (N,N) smoothed version
    """
    N = clean_xy.shape[0]

    sim_params = {"redshift": float(redshift), "box_length_Mpc": float(box_length_Mpc), "box_dim": int(N)}

    # --- make 3-channel noise so smooth_lightcone is happy ---
    dz = 1e-3
    zs = np.array([redshift - dz, redshift, redshift + dz], dtype=float)

    # generate 3-channel noise (x,y,zchan)
    noiseonly_xyz = t2c.noise_lightcone(
        ncells=N,
        zs=zs,
        obs_time=noise_params["obs_time"],
        total_int_time=noise_params["total_int_time"],
        int_time=noise_params["int_time"],
        declination=noise_params["declination"],
        subarray_type=noise_params["subarray_type"],
        boxsize=box_length_Mpc,
        verbose=bool(noise_params.get("verbose", False)),
        save_uvmap=str(uvmap_filename),
        n_jobs=int(noise_params.get("njobs", 1)),
        checkpoint=noise_params.get("checkpoint", False),
    ).astype(np.float32)

    noiseonly_xy = noiseonly_xyz[:, :, 1]

    # embed clean slice into 3 channels (repeat same map)
    clean_xyz = np.repeat(clean_xy[:, :, None], 3, axis=2).astype(np.float32)

    noisy_xyz = clean_xyz + noiseonly_xyz
    noisy_xy = noisy_xyz[:, :, 1]  # take the central channel as your target-z result

    obs_xyz, zs_used = t2c.smooth_lightcone(
        lightcone=noisy_xyz,
        z_array=zs,
        box_size_mpc=box_length_Mpc,
        max_baseline=bmax_km,
    )
    obs_xyz = obs_xyz.astype(np.float32)

    obs_xy = obs_xyz[:, :, 1]  # again take central channel

    return noiseonly_xy, noisy_xy, obs_xy


# testing


clean_xy = np.load("tests/mock_LoReLi_sim_N64.npy")[:, :, 0]

obs_time = 1000.                      # total observation hours
total_int_time = 6.                   # hours per day
int_time = 10.                        # seconds
declination = -30.0                   # declination of the field in degrees
subarray_type = "AA4"
njobs = 1
checpoints = 16
bmax_km = 2. #* units.km # km needed for smoothibg

uvmap_filename = "tests/uvmap_mock.h5"

redshift = 6.0
box_length_Mpc = 296.0/4.0  # 1/4 of the full sim size
box_dim = 64

noise_params = {
    "obs_time": 1000.0,         # total observation hours
    "total_int_time": 6.0,      # hours per day
    "int_time": 10.0,           # seconds
    "declination": -30.0,       # degrees
    "subarray_type": "AA4",
    "verbose": True,
    "njobs": 1,
    "checkpoint": 16,
}

noiseonly_xy, noisy_xy, obs_xy = make_observed_2Dslice(
    clean_xy,
    redshift,
    box_length_Mpc,
    noise_params,
    uvmap_filename,
    bmax_km)