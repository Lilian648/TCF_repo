import subprocess
import pandas as pd
import os
import re
import scipy.stats as stats
import glob, pickle
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import Random_bubbles as RB
import Random_bubbles1_Extended as RB_extended


###################################################################################################################################################
############################## Function to create a bubble field (limited parameters) ##############################################################
###################################################################################################################################################

def Create_Bubble_Field_Func(DIM, ff, radius, NDIM, nooverlap, periodic):
    """
    Generates a model field and returns the filename and a data cube
    cube.box = 2/3D data array
    """

    # create field
    cube = RandomBubbles(DIM, ff, radius, NDIM, nooverlap=False, periodic=True)

    # save field to txt file
    RandomBubbles.write_ionisation_field(cube) 

    # get filename
    filename = RandomBubbles.print_filechain(cube)
    
    return filename, cube


# Example usage

DIM = 200
ff = 0.05
radius = 5
NDIM = 2
nooverlap = False
periodic = False


filename, cube = Create_Bubble_Field_Func(DIM, ff, radius, NDIM, nooverlap, periodic)

data = np.loadtxt(filename + '.txt')

plt.imshow(cube.box)
plt.imshow(data) # these two imshows are equivalent



###################################################################################################################################################
############################## Function to Create a bubble Field (with all possible parameters) ###################################################
###################################################################################################################################################

def Create_Bubble_Field_Func_Extended(DIM, fillingfraction, radius, sigma, NDIM, nooverlap, periodic, 
                     radius_distribution, gaussian_profile, verbose=False, save=True, show_plot=False):
    """
    Generates a model field and returns the filename (without extension) and data cube (which contains 2/3D data array and field info

    eg:
    2d data array = cube.box
    list of bubble radii = cube.bubble_radii
    """

    # Combine radius and sigma into params list
    params = [radius, sigma]
    
    # Create and save the cube using RandomBubbles' built-in save function
    cube = RB_extended.RandomBubbles(
        DIM=DIM,
        fillingfraction=fillingfraction,
        params=params,
        NDIM=NDIM,
        nooverlap=nooverlap,
        periodic=periodic,
        radius_distribution=radius_distribution,
        gaussian_profile=gaussian_profile,
        verbose=verbose,
        save=save
    )

    # Call summary with show_plot=False to suppress plotting
    cube.summary(show_plot=show_plot)
    
    # Construct the filename to include all parameters
    filename = "Field_DIM%i_FF%.2f_R%i_S%.2f_N%i_nooverlap%s_periodic%s_RD%i_GP%s" % (
        cube.DIM,
        cube.fillingfraction,
        cube.mean_radius,         # Assuming radius is stored as mean_radius
        cube.sigma_radius,
        cube.NDIM,
        str(cube.nooverlap),      # Convert booleans to strings for filename clarity
        str(cube.periodic),
        cube.distribution,
        str(cube.gaussian_profile)
    )
    
    return filename, cube

# expamle usage

DIM = 200
ff = 0.05
radius = 5
ff = 0.01
sigma = 2
NDIM = 2
nooverlap = False
periodic = False
radius_distribution = 1
gaussian_profile = False

filename, cube = Create_Bubble_Field_Func_Extended(DIM, ff, radius, sigma, NDIM, nooverlap, periodic, 
                     radius_distribution, gaussian_profile, verbose=False, save=True, show_plot=False)
print(cube.bubble_radii)

data = np.loadtxt(filename + '.txt')

plt.imshow(cube.box)



###################################################################################################################################################
############################## Creat Bubble Field with Thermal Noise ##############################################################################
###################################################################################################################################################

def create_bubble_field_with_thermal_noise(
    DIM, ff, radius, sigma, NDIM, nooverlap, periodic, radius_distribution, gaussian_profile,
    T_sys, bandwidth, integration_time, num_antennas, save=True, save_dir="."):
    """
    Create a clean RandomBubbles-Extended field, add thermal noise, and (optionally) save.

    Returns
    -------
    filename_stub : str
    field_noisy : np.ndarray
    cube : object         # the RB_extended cube with .box set to the noisy field
    """
    # --- generate the clean field  ---
    # NOTE: requires Create_Bubble_Field_Func_Extended to accept `save=False`
    _, cube = Create_Bubble_Field_Func_Extended(
        DIM, ff, radius, sigma, NDIM, nooverlap, periodic,
        radius_distribution, gaussian_profile,
        verbose=False, save=False, show_plot=False
    )
    field = cube.box

    # --- sanity checks for noise params ---
    if None in (T_sys, bandwidth, integration_time, num_antennas):
        raise ValueError(
            "Provide T_sys, bandwidth, integration_time and num_antennas.")


    # --- noise draw ---
    sigma_noise = T_sys / np.sqrt(bandwidth * integration_time * num_antennas)
    thermal_noise = np.random.normal(loc=0, scale=sigma_noise, size=data.shape)
    field_noisy = (field + thermal_noise)

    # keep cube consistent for downstream users
    if hasattr(cube, "box"):
        cube.box = field_noisy

    # --- filename + save ---
    filename_stub = (
        f"Field_NOISY"
        f"_FF{ff:.3f}_R{radius}_S{sigma}"
        f"_N{NDIM}_DIM{DIM}"
        f"_noov{int(bool(nooverlap))}_per{int(bool(periodic))}"
        f"_RD{radius_distribution}_GP{int(bool(gaussian_profile))}"
    )
    if save:
        np.savetxt(os.path.join(save_dir, filename_stub + ".txt"), field_noisy)

    return filename_stub, field_noisy, cube


# Example Usage

DIM = 200
ff = 0.05
radius = 5
ff = 0.01
sigma = 2
NDIM = 2
nooverlap = False
periodic = False
radius_distribution = 1
gaussian_profile = False
T_sys = 100
bandwidth = 100
integration_time = 600
num_antennas = 128


filename_stub, field_noisy, cube = create_bubble_field_with_thermal_noise(DIM, ff, radius, sigma, NDIM, nooverlap, periodic, radius_distribution, gaussian_profile,
                                       T_sys, bandwidth, integration_time, num_antennas, save=True, save_dir=".")

print(filename_stub )
print(np.shape(field_noisy))
plt.imshow(cube.box)