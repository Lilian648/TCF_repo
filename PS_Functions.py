import subprocess
import pandas as pd
import os
import re
import scipy.stats as stats
import glob, pickle
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def PS_2D(data, length, npix):
    """
    Computes the radially averaged 2D power spectrum of a given 2D field.

    This function performs the following steps:
        1. Computes the 2D Fourier Transform (FT) of the input field.
        2. Normalizes the FT to account for the grid size.
        3. Computes the power spectrum as the squared magnitude of the FT.
        4. Computes the radial wavenumber (k) values.
        5. Bins the power spectrum radially in k-space using logarithmic binning.

    Parameters:
        data (numpy.ndarray): 2D array representing the input field.
        length (float): Physical length of the domain (along a single axis) in physical units (eg Mpc).
        npix (int): Number of pixels in the data array (along a single axis).
            unit of a single pixel = (length/npix)^2

    Returns:
        k_bins_centres_filtered (numpy.ndarray): Central values of the k-space bins in (physcial units)^-1
        PS_radial_filtered (numpy.ndarray): Averaged power spectrum values in each k-bin in (physcial units)
    """

    ###### Compute Power Spectrum ######

    # Compute the 2D Fourier Transform
    FT_data = np.fft.fftn(data)

    # Normalize the Fourier transform for the finite grid
    FT_data *= (length / npix) ** 2  

    # Compute the power spectrum (squared Fourier coefficients)
    PS = np.abs(FT_data) ** 2 * (2 * np.pi / length) ** 2  

    ###### Compute k-space Values ######

    # Compute 1D k-space frequencies (kx, ky) and scale to physical wavenumbers
    kx = np.fft.fftfreq(npix, d=length / npix) * 2 * np.pi
    ky = np.fft.fftfreq(npix, d=length / npix) * 2 * np.pi

    # Compute 2D radial k-grid (sqrt(kx^2 + ky^2) for each point)
    kx_2D, ky_2D = np.meshgrid(kx, ky, indexing="ij")
    kr_grid = np.sqrt(kx_2D**2 + ky_2D**2)  

    ###### Radial Binning ######

    # Flatten arrays for binning
    PS = PS.flatten()
    kr_grid = kr_grid.flatten()

    # Define bin edges for k-space (logarithmically spaced for better resolution at small k)
    kmin = 2 * np.pi / length  # Smallest k value (Nyquist condition factor 2 removed)
    kmax = np.pi * npix / length  # Nyquist frequency
    k_bins = np.logspace(np.log10(kmin), np.log10(kmax), npix)  

    # Compute bin centers
    k_bins_centres = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Perform radial binning of the power spectrum
    PS_radial, _, _ = stats.binned_statistic(kr_grid, PS, statistic="mean", bins=k_bins)

    # Remove NaN values from the binned results
    valid_indices = ~np.isnan(PS_radial)
    k_bins_centres_filtered = k_bins_centres[valid_indices]
    PS_radial_filtered = PS_radial[valid_indices]

    return k_bins_centres_filtered, PS_radial_filtered


# ## Power Spectrum with Error Band

# In[12]:


def power_spect_with_errorband(Create_Field_Func, physical_length,
        DIM, fillingfraction, radius, NDIM, nooverlap, periodic, num_realizations, confidence=2):
    """
    Generates multiple realizations of a toy model, calculates the mean power spectrum,
    and computes a specified confidence band in logarithmic space.
    
    Args:
        physical_length (int): size of the field in physical units, eg 25 Mpc^2 (in 3D, the field is a cube, Mpc^3)
            when plotting the field you will need to pass the physical_length as the max of the extent values for
            the imshow plot (in both x,y directions)
            
        Parameters to pass to RandomBubbles:    
            DIM (int): Dimension of the box.
            fillingfraction (float): Fraction of the box to fill.
            radius (float): Radius of the bubbles.
            NDIM (int): Number of dimensions.
            nooverlap (bool): If True, bubbles do not overlap.
            periodic (bool): If True, periodic boundary conditions.

        Errorband parameters: 
            num_realizations (int): Number of realizations to generate.
            confidence (float): Number of sigma for the confidence level (e.g., 5 for 5-sigma).
        
    Returns:
        k_values (array): k values for the power spectrum.
        PS_mean (array): Mean power spectrum values.
        upper_bound (array): Upper bound of the confidence band.
        lower_bound (array): Lower bound of the confidence band.
    """
    PS_list = []
    k_list = []

    # Generate realizations of the toy model
    for i in range(num_realizations):

        print(f"PARAMETERS: physical_length={physical_length}, DIM={DIM}, fillingfraction={fillingfraction}, radius={radius}, NDIM={NDIM}, nooverlap={nooverlap}, periodic={periodic}, num_realizations={num_realizations}, confidence={confidence}")

        print(f"=========== Running realisation {i+1}/{num_realizations} ===========")

        file = Create_Field_Func(DIM, fillingfraction, radius, NDIM, nooverlap, periodic)

        data = np.loadtxt(file + ".txt")
        
        if NDIM == 2:
            k_values, PS_values = PS_2D(data, physical_length, DIM)
        else:
            raise ValueError("NDIM must be either 2! (have not yet added 3D)")
        
        # Add a small value to avoid log(0) and take the log of the PS values
        PS_list.append(np.log10(PS_values))
        k_list.append(k_values)
    
    

    # Calculate the mean and standard deviation in log space
    PS_mean_log = np.mean(PS_list, axis=0)
    PS_std_log = np.std(PS_list, axis=0)

    # Convert the mean and bounds back to linear space with specified confidence level
    PS_mean = 10 ** PS_mean_log
    upper_bound = 10 ** (PS_mean_log + confidence * PS_std_log)
    lower_bound = 10 ** (PS_mean_log - confidence * PS_std_log)

    return k_values, PS_mean, upper_bound, lower_bound