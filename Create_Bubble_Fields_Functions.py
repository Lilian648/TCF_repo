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
import PS_Functions 


###################################################################################################################################################
############################## Function to create a bubble field (limited parameters) ##############################################################
###################################################################################################################################################

def Create_Bubble_Field_Func(DIM, ff, radius, NDIM, nooverlap, periodic):
    """
    Generates a model field and returns the filename and a data cube
    cube.box = 2/3D data array
    """

    # create field
    cube = RB.RandomBubbles(DIM, ff, radius, NDIM, nooverlap=False, periodic=True)

    # save field to txt file
    RB.RandomBubbles.write_ionisation_field(cube) 

    # get filename
    filename = RB.RandomBubbles.print_filechain(cube)
    
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

print("XX 1 XX")

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


print("XX 2 XX")
###################################################################################################################################################
############################## Add Thermal Noise to Bubble Field  #################################################################################
###################################################################################################################################################

def add_thermal_noise(field, T_sys, bandwidth, integration_time, num_antennas, save=True, save_dir="."):
    """
    Add thermal noise to a given data field, and (optionally) save.

    field = input field (2D)
    other params = noise params

    Returns
    -------
    filename_stub : str
    field_noisy : np.ndarray
    cube : object         # the RB_extended cube with .box set to the noisy field
    """

    # --- sanity checks for noise params ---
    if None in (T_sys, bandwidth, integration_time, num_antennas):
        raise ValueError(
            "Provide T_sys, bandwidth, integration_time and num_antennas.")


    # --- noise draw ---
    sigma_noise = T_sys / np.sqrt(bandwidth * integration_time * num_antennas)
    thermal_noise = np.random.normal(loc=0, scale=sigma_noise, size=field.shape)
    field_noisy = (field + thermal_noise)


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

    return filename_stub, field_noisy



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

T_sys = 1000
bandwidth = 100
integration_time = 600
num_antennas = 128

# first create a single bubble field
filename, cube = Create_Bubble_Field_Func_Extended(DIM, ff, radius, sigma, NDIM, nooverlap, periodic, 
                     radius_distribution, gaussian_profile, verbose=False, save=True, show_plot=False)

field = cube.box

# add thermal noise
filename_stub, field_noisy = add_thermal_noise(field,
                                       T_sys, bandwidth, integration_time, num_antennas, save=True, save_dir=".")

print(filename_stub)
print(np.shape(field_noisy))
plt.imshow(field_noisy)



###################################################################################################################################################
############################## Apply Resolution to a Bubble Field #################################################################################
###################################################################################################################################################


    
def apply_telescope_resolution(field, L, fwhm, NDIM, return_PS_checks=False):
    """
    Convolve a 2D/3D field with a Gaussian PSF in Fourier space.
    (currently only works for 2D because PS_3D does not exist)

    Parameters
    ----------
    field : ndarray          # shape (DIM, DIM[, DIM])
    L : float                # physical box length (same units as fwhm, e.g. Mpc)
    fwhm : float             # beam FWHM in physical units
    NDIM : int               # 2 or 3
    dtype : np.dtype or None # cast output to this (e.g. np.float32)
    return_ps : bool         # if True, compute and return blurred PS via ps2d/ps3d_func
    ps2d_func, ps3d_func : callables like PS_2D/PS_3D if return_ps=True

    Returns
    -------
    If return_PS_checks is False:
        blurred_field
    If return_PS_checks is True:
        blurred_field, g_fourier_2D, k, PS_blurred
        (Note: for NDIM=3 the returned "g_fourier_2D" is actually the ND Gaussian kernel.)
    """

    print(return_PS_checks)

    # --- 1. parameters ---
    DIM = np.shape(field)[0]
    print("DIM", DIM)
    print("L", L)

    pixel_unit = L / DIM # units of a single pixel
    kernel_sigma_Mpc = fwhm / np.sqrt(8 * np.log(2))   # convert FWHM -> sigma of the kernel in Mpc

    # Frequency grids (cycles/length), convert to angular k (1/length * 2π)
    freq = 2*np.pi*np.fft.fftfreq(DIM, d=pixel_unit)

    # --- 2. Compute Power Spectrum --- #
    if NDIM == 2:
        k, PS = PS_Functions.PS_2D(field, L, DIM)  # Compute 2D power spectrum
    if NDIM == 3:
        
        k, PS = PS_Functions.PS_3D(field, L, DIM)  # Compute 2D power spectrum
        # AS A NOTE THE PS_3D DOES NOT YET EXIST 

    # --- 3. Create Gaussian Kernel --- # 
    if NDIM == 2:
        
        # 3.a. Compute Fourier-space grid (Mpc⁻¹)
        kx = 2 * np.pi * np.fft.fftfreq(DIM, d=pixel_unit)  
        ky = 2 * np.pi * np.fft.fftfreq(DIM, d=pixel_unit)  
        KX, KY = np.meshgrid(kx, ky)

        # 3.b. Create Gaussian kernel in Fourier space (2D)
        g_fourier_2D = np.exp(-0.5 * (KX**2 + KY**2) * kernel_sigma_Mpc**2)

        # 3.c. Create Gaussian kernel in Fourier space (1D, radial)
        g_fourier_1D = np.exp(-0.5 * k**2 * kernel_sigma_Mpc**2) 

    
    if NDIM == 3:

        # 3.a. Compute Fourier-space grid (Mpc⁻¹) 
        kx = 2 * np.pi * np.fft.fftfreq(DIM, d=pixel_unit)  
        ky = 2 * np.pi * np.fft.fftfreq(DIM, d=pixel_unit)
        kz = 2 * np.pi * np.fft.fftfreq(DIM, d=pixel_unit)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    
        # 3.b. Create Gaussian kernel in Fourier space (2D)
        g_fourier_2D = np.exp(-0.5 * (KX**2 + KY**2 + KZ**2) * kernel_sigma_Mpc**2)

        # 3.c. Create Gaussian kernel in Fourier space (1D, radial)
        g_fourier_1D = np.exp(-0.5 * k**2 * kernel_sigma_Mpc**2)

    # --- 4. Convolution with field --- #
    # 4.a. Apply Gaussian kernel to the field in Fourier space
    field_FT = np.fft.fftn(field)  # Compute Fourier transform of field
    blurred_field_FT = field_FT * g_fourier_2D  # Multiply by Gaussian kernel
    blurred_field_2D = np.fft.ifftn(blurred_field_FT).real  # Inverse FFT back to real space

    # 4.b. Apply Gaussian kernel to the power spectrum
    PS_blurred = PS * g_fourier_1D**2 

    if return_PS_checks == False:
        return blurred_field_2D, None

    if return_PS_checks == True:
        return blurred_field_2D, (g_fourier_2D, k, PS_blurred)


# example usage

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

L = 200
fwhm = 50
return_PS_checks = True

# create field
filename, cube = Create_Bubble_Field_Func_Extended(DIM, ff, radius, sigma, NDIM, nooverlap, periodic, 
                     radius_distribution, gaussian_profile, verbose=False, save=True, show_plot=False)

field = cube.box

# apply resolution
field_with_resolution, checks = apply_telescope_resolution(field, L, fwhm, NDIM, return_PS_checks)
print(np.shape(field_with_resolution))

# plot field with resolution effect
plt.imshow(field_with_resolution)
plt.show()

# --- checks ---#
if checks is not None:
    # unpack the diagnostics
    g_fourier_2D, k, PS_blurred = checks
    
    print(np.shape(k), np.shape(PS_blurred), np.shape(g_fourier_2D))

    # plot PS
    plt.loglog(k, PS_blurred)
    plt.show()

    # plot gaussian kernel in Fourier Space
    plt.imshow(np.fft.fftshift(g_fourier_2D))
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.show()



###################################################################################################################################################
############################## Add Thermal Noise and/or Apply Resolution to a Bubble Field ########################################################
###################################################################################################################################################




def apply_observation_effects(field, *,
                              thermal=None,               # dict or None
                              resolution=None,            # dict or None
                              order=("thermal","resolution"),
                              return_resolution_checks=False):
    """
    Apply zero or more observation effects to `field` in the specified `order`.
    - `thermal`: dict of kwargs for add_thermal_noise (or None)
    - `resolution`: dict of kwargs for apply_telescope_resolution (or None), e.g. {'L':..., 'fwhm':..., 'NDIM':2}
    - `order`: tuple containing any subset of {"thermal","resolution"}
    - `return_resolution_checks`: if True, forward the PS checks out

    Returns
    -------
    field_out, extras
      extras is None if no resolution checks were requested,
      otherwise {"resolution_checks": (g_fourier_nd, k, PS_blurred)}.
    """
    out = np.asarray(field)

    extras = None

    for step in order:
        if step == "thermal" and thermal:
            out = add_thermal_noise(out, **thermal)

        if step == "resolution" and resolution:
            # ensure the flag is passed through
            res_kwargs = dict(resolution)
            res_kwargs.setdefault("return_PS_checks", return_resolution_checks)

            out, checks = apply_telescope_resolution(out, **res_kwargs)
            if return_resolution_checks:
                extras = {"resolution_checks": checks}

    return out, extras

# example usage

# example usage

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

# create field
filename, cube = Create_Bubble_Field_Func_Extended(DIM, ff, radius, sigma, NDIM, nooverlap, periodic, 
                     radius_distribution, gaussian_profile, verbose=False, save=True, show_plot=False)

field = cube.box

##### thermal noise only #####

thermal_cfg = dict(
    T_sys=3000.0,             # K
    bandwidth=100,            # Hz
    integration_time=3600,    # s
    num_antennas=128          
)

field_noisy, extras = apply_observation_effects(
    field,
    thermal=thermal_cfg,
    resolution=None
)

# plot noisy field
plt.imshow(field_noisy)
plt.title("Bubble Field with Thermal Noise")
plt.show()



##### resolution effects only #####

L = 200
fwhm = 50
return_PS_checks = False

res_cfg = dict(
    L=200.0,   # Mpc (same units as fwhm)
    fwhm=5.0,  # Mpc
    NDIM=2     # or 3, must match field.ndim
)

field_res, checks = apply_observation_effects(
    field,
    thermal=None,
    resolution=res_cfg,
    return_resolution_checks=return_PS_checks
)


plt.imshow(field_res)
plt.title("Bubble Field with Resolution Effects")
plt.show()

if checks is not None:
    g_fourier_2D, k, PS_blurred = checks["resolution_checks"]
    
    # plot PS
    plt.loglog(k, PS_blurred)
    plt.title("Power Spectrum")
    plt.xlabel('k (radially averaged)')
    plt.ylabel('P(k)')
    plt.show()

    # plot gaussian kernel in Fourier Space
    plt.imshow(np.fft.fftshift(g_fourier_2D))
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title("Gaussian Kernel in Fourier Space")
    plt.show()




##### both thermal noise and resolution  #####

thermal_cfg = dict(T_sys=3000.0, bandwidth=100, integration_time=3600.0, num_antennas=128)
res_cfg     = dict(L=200.0, fwhm=5.0, NDIM=2)

field_obs, extras = apply_observation_effects(
    field,
    thermal=thermal_cfg,
    resolution=res_cfg,
    order=("thermal", "resolution"),  # add noise, then smooth signal+noise
    return_resolution_checks=True
)

plt.imshow(field_obs)
plt.title("Bubble Field with Thermal Noise AND Resolution Effects")
plt.show()

# Optional diagnostics:
if extras is not None:
    gk, k, PS_blurred = extras["resolution_checks"]

