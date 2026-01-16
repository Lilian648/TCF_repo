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
        print(f" loop {i+1}/{len(rvals)}")
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


##### load sim #####
print("loading sim")

sn = '10038'
base_dir = '/data/cluster/emc-brid/Datasets/LoReLi' # where Lisa keeps the LoReLi info/sims

sim = Cat(sn,
    redshift_range='all', #[sim.z.min(), sim.z.max()]
    skip_early=False,
    path_spectra='spectra',
    path_sim='/data/cluster/emc-brid/Datasets/LoReLi/simcubes',
    base_dir=base_dir,
    load_params=False,
    load_spectra=False,
    just_Pee=True,
    reinitialise_spectra=False,
    save_spectra=False,
    load_density_cubes=False,
    load_xion_cubes=False,
    load_T21cm_cubes=True,
    verbose=True,
    debug=False)

##### extract all 2D slices along z axis #####

print("extracting z slices")

# choose which redshift cube to use
cube_z6_idx = np.abs(sim.z - 6).argmin()
print(cube_z6_idx)


cube_z6 = sim.T21cm[37]  # shape (256, 256, 256)

out_dir = "tests/sn10038_txtfiles/zidx_37/"  
saved_files = extract_LoReLi_slices_every_dMpc(
    cube_3d=cube_z6,
    output_dir=out_dir,
    box_size_mpc=296.0,
    delta_mpc=10.0,
    axis="z",      # or "x" / "y" if you prefer
    demean=True
)


##### compute TCF of single 2D slice #####

print("computing TCF")

field2d = np.loadtxt("tests/sn10038_txtfiles/zidx_37/slice_axis2_idx0_r0Mpc_SMALL.txt")
L = sim.box_size
rvals = np.linspace(2, 50, 49)
outfile = "tests/sn10038_txtfiles/zidx_37/slice_axis2_idx0_r0Mpc_TCFresult.txt"

nmodes, sr = pyTCF_of_2Dslice(field2d, L, rvals, outfile)

##### plot TCF and real field image #####

plt.plot(rvals, sr)
plt.title("TCF of 2Dslice, cut to size 64x64")
plt.savefig("tests/sn10038_txtfiles/zidx_37/slice_axis2_idx0_r0Mpc_TCFresult_Plot.png")
plt.show()

data = np.clip(field2d, np.min(field2d), 50)
plt.imshow(data)
plt.colorbar(extend="max")
plt.title("real field image, cut to size 64x64")
plt.savefig("tests/sn10038_txtfiles/zidx_37/slice_axis2_idx0_r0Mpc_FieldImage.png")
plt.show()
