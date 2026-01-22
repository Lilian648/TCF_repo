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


def compute_tcf_for_file_list(slice_files_list, L, rvals, overwrite=False):
    """
    Compute TCF for each 2D slice and save *_TCFresult.txt
    in the SAME directory as the slice file.

    Returns
    -------
    tcf_files : list[Path]
        Paths to the saved TCF result files.
    """
    t_overall_start = time.time()
    
    tcf_files = []

    for idx, slice_file in enumerate(slice_files_list):
        slice_path = Path(slice_file)

        print(f"➤ {idx+1}/{len(slice_files_list)}  {slice_path.name}")
        tstart = time.time()

        # same directory, same stem
        out_path = slice_path.with_name(slice_path.stem + "_TCFresult.txt")

        if out_path.exists() and not overwrite:
            print(f"  ↪ Skipping existing TCF")
            tcf_files.append(out_path)
            continue

        field2d = np.loadtxt(slice_path)
        nmodes, sr = pyTCF_of_2Dslice(field2d, L, rvals, str(out_path))

        tcf_files.append(out_path)

        tend = time.time()
        print(f"  ✓ time: {tend - tstart:.1f}s")

    t_overall_end = time.time()
    print(f"TCFs of entire list computed in time: {t_overall_end - t_overall_start:.1f}s")

    return tcf_files



################################################################################################################################################
#################################### Adding Noise + Smoothing to Clean Slice ###################################################################
################################################################################################################################################




def add_noise_smoothing(clean_xy, uvmap_filename, sim_params, noise_params, depth_mhz):
    """
    Create a single 2D thermal-noise slice at one redshift using tools21cm.noise_map.

    Parameters
    ----------
    uvmap_filename : str or Path
        Cache file for UV maps.
    sim_params : dict
        Must contain: 
        - redshift (float), 
        - box_length_Mpc (float), 
        - box_dim (int)
    noise_params : dict
        Must contain: 
        - obs_time, 
        - total_int_time, 
        - int_time,
        - declination,
        - subarray_type,
        - bmax_km
        Optional: "
        - uv_weighting,
        - verbose,
        - njobs,
        - checkpoint
    depth_mhz : float
        Frequency channel width (MHz) used to set the thermal noise amplitude.

    Returns
    -------
    noise_xy : ndarray, shape (N,N)
        2D noise realisation in mK (same convention as noise_lightcone output slices).
    """
    # unpack parameters
    z = float(sim_params["redshift"])
    N = int(sim_params["box_dim"])
    boxsize = float(sim_params["box_length_Mpc"])

    obs_time = float(noise_params["obs_time"])
    total_int_time = float(noise_params["total_int_time"])
    int_time = float(noise_params["int_time"])
    declination = float(noise_params["declination"])
    subarray_type = noise_params["subarray_type"]
    bmax_km = float(noise_params["bmax_km"])

    uv_weighting = noise_params.get("uv_weighting", "natural")
    verbose = bool(noise_params.get("verbose", False))
    njobs = int(noise_params.get("njobs", 1))
    checkpoint = noise_params.get("checkpoint", 16)

    uvpath = Path(uvmap_filename)
    uvpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'Reusing' if uvpath.exists() else 'Will save'} UV map at: {uvpath}")

    # 1. Build/load the uv map for this redshift
    uvs = t2c.get_uv_map_lightcone(
        N, 
        np.array([z], dtype=float),
        subarray_type=subarray_type,
        total_int_time=total_int_time,
        int_time=int_time,
        boxsize=boxsize,
        declination=declination,
        save_uvmap=str(uvpath),
        n_jobs=njobs,
        verbose=verbose,
        checkpoint=checkpoint,
    )
    N_ant = uvs.get("Nant") or uvs.get("N_ant")  # version differences
    uv_map = uvs[f"{z:.3f}"]

    # 2. Generate 2D noise (Jy)
    noise2d_jy = t2c.noise_map(
        N, z, depth_mhz,
        obs_time=obs_time,
        subarray_type=subarray_type,
        boxsize=boxsize,
        total_int_time=total_int_time,
        int_time=int_time,
        declination=declination,
        uv_map=uv_map,
        N_ant=N_ant,
        uv_weighting=uv_weighting,
        verbose=False,
    )

    # Convert to Kelvin/mK like noise_lightcone does
    noise_xy = t2c.jansky_2_kelvin(noise2d_jy, z, boxsize=boxsize)

    # 3. Add nosie to clean sim
    noisy_xy = clean_xy + noise_xy

    # 4. Smoothing
    dtheta = (1.0 + z) * 21e-5 / bmax_km  # radians-ish small-angle factor used by tools21cm
    ang_res_mpc = dtheta * t2c.cm.z_to_cdist(z)   # comoving Mpc

    # convert to sigma in pixels
    fwhm = ang_res_mpc * N / boxsize

    # apply Gaussian smoothing
    obs_xy = t2c.smooth_gauss(noisy_xy, fwhm=fwhm)

    return noise_xy, noisy_xy, obs_xy


# testing


# obs_time = 1000.                      # total observation hours
# total_int_time = 6.                   # hours per day
# int_time = 10.                        # seconds
# declination = -30.0                   # declination of the field in degrees
# subarray_type = "AA4"
# bmax_km = 2. #* units.km # km needed for smoothibg

# verbose = True
# uvmap_filename = "tests/uvmap_mock.h5"
# njobs = 1
# checkpoint = 16

# sim_params = {
#     "redshift": 6.0,
#     "box_length_Mpc": 296.0/4.0,  # 1/4 of the full sim size
#     "box_dim": 64,
# }
# print(type(sim_params["redshift"]))

# depth_mhz = 0.07 # no idea if this is correct or not

# noise_params = {
#     "obs_time": 1000.0,         # total observation hours
#     "total_int_time": 6.0,      # hours per day
#     "int_time": 10.0,           # seconds
#     "declination": -30.0,       # degrees
#     "subarray_type": "AA4",
#     "verbose": True,
#     "njobs": 1,
#     "checkpoint": 16,
#     "bmax_km": bmax_km
# }

# clean_xy = np.load("tests/mock_LoReLi_sim_N64.npy")[:, :, 0]
# noise_xy, noisy_xy, obs_xy = add_noise_smoothing(clean_xy, uvmap_filename, sim_params, noise_params, depth_mhz)

# np.savetxt("tests/mock_noise_slice.txt", noise_xy)
# np.savetxt("tests/mock_noisy_slice.txt", noisy_xy)
# np.savetxt("tests/mock_obs_slice.txt", obs_xy)


################################################################################################################################################
#################################### Main: Run All #############################################################################################
# currently incomplete - need to add addition of noise/smoothing section 
################################################################################################################################################

def TCFpipeline_single_sim(sim_name, rvals, z_target, noise_params, uvmap_filename, delta_mpc=10.0, overwrite_tcf=False):

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

    it_begins = time.time()
    print(f" %%%%%% IT BEGINS AT TIME: {it_begins}")
    
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

    z_idx = 38
    
    # define base directory
    base_dir = Path(f"tests/sn{sim_name}/zidx{z_idx}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # create sibling folders for results
    clean_dir = base_dir / "clean_slices"
    noise_dir = base_dir / "noise_slices"
    obs_dir   = base_dir / "obs_slices"
    
    clean_dir.mkdir(exist_ok=True)
    noise_dir.mkdir(exist_ok=True)
    obs_dir.mkdir(exist_ok=True)


    
    rvals = np.asarray(rvals, dtype=float)

    #################################################################################
    ######################## TESTING VERSION (smaller cube) #########################
    #################################################################################
    print(sim_name)
    # for testing i am just putting a specific filename in here!!!!
    cube = np.load("tests/mock_LoReLi_sim_N64.npy") # a 3d cube array 
    L = 296/4
    z_idx = 38
    z_used = 6
    N = cube.shape[0] # box dimensions (NxNxN)

    # ----------------------------
    # 3. Extract 2D slices from all axes
    # (all three axes - will all save to save folder with indication of axis in filename)
    # ---------------------------- 
    clean_files = []
    axes = ["x", "y", "z"]
    for ax in axes:
        print(f"Extracting 2D slices along axis {ax}")
        clean_files += extract_LoReLi_slices_every_dMpc(cube_3d=cube, output_dir=clean_dir, box_size_mpc=L, delta_mpc=delta_mpc, 
                                                        axis=ax, demean=True)


    # ----------------------------
    # 4. Add noise + smoothing
    # ----------------------------
    
    # Choose a bandwidth assumption for one 2D “channel map”.
    # For now keep it fixed + documented (you can upgrade this later).
    depth_mhz = float(noise_params.get("depth_mhz", 0.2))  # MHz, e.g. 0.2
    
    sim_params = {
        "redshift": float(z_used),
        "box_length_Mpc": float(L),
        "box_dim": int(N),
    }
    
    noise_files = []
    obs_files = []
    
    for index, slice_path in enumerate(clean_files):

        print(f"Adding noise + smoothing to {index+1}/{len(clean_files)}")
        
        slice_path = Path(slice_path)
        filename_root = slice_path.stem
    
        clean_xy = np.loadtxt(slice_path)
        noise_xy, noisy_xy, obs_xy = add_noise_smoothing(clean_xy, uvmap_filename, sim_params, noise_params, depth_mhz)
    
        noise_out = noise_dir / f"{filename_root}_NOISE.txt"
        obs_out   = obs_dir   / f"{filename_root}_OBS.txt"
    
        np.savetxt(noise_out, noise_xy)
        np.savetxt(obs_out,   obs_xy)
    
        noise_files.append(noise_out)
        obs_files.append(obs_out)
    
    

    # ----------------------------
    # 5. Compute TCF for each slice file
    # ----------------------------

    # clean slices
    tcf_clean_files = compute_tcf_for_file_list(clean_files, L, rvals, overwrite=overwrite_tcf)

    # noise only slices
    tcf_noise_files = compute_tcf_for_file_list(noise_files, L, rvals, overwrite=overwrite_tcf)

    # observed slices
    tcf_obs_files = compute_tcf_for_file_list(obs_files, L, rvals, overwrite=overwrite_tcf)

    it_ends = time.time()
    print(f" %%%%%% IT ENDS AT TIME: {it_ends}, taking {it_ends - it_begins}s total")
    
    return {
        "z_idx": z_idx,
        "z_used": float(sim.z[z_idx]),
        "clean_files": saved_files,
        "tcf_files": [str(p) for p in tcf_files],
    }
        


################## TESTING ##################

# ----------------------------
# Define test parameters
# ----------------------------
sim_name = "10038"

# small r range for speed
rvals = np.linspace(2, 50, 49)

noise_params = {
    "obs_time": 1000.0,        # hours
    "total_int_time": 6.0,     # hours
    "int_time": 10.0,          # seconds
    "declination": -30.0,
    "subarray_type": "AA4",
    "bmax_km": 2.0,
    "depth_mhz": 1,            # MHz
    "verbose": True,
    "njobs": 1,
}

uvmap_filename = "tests/uvmap_mock.h5"

# ----------------------------
# Run pipeline
# ----------------------------
print("Running TCF pipeline test...")



result = TCFpipeline_single_sim(
    sim_name=sim_name,
    rvals=rvals,
    z_target=6.0,
    noise_params=noise_params,
    uvmap_filename=uvmap_filename,
    delta_mpc=10.0,          
    overwrite_tcf=True
)

print("\nPipeline returned:")
for k, v in result.items():
    print(f"{k}: {v}")





################################################################################################################################################
####################################  ###################################################################
################################################################################################################################################



