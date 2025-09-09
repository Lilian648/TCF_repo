import subprocess
import pandas as pd
import os
import re
import scipy.stats as stats
import glob, pickle
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from Random_bubbles import *


print("hi")


### Function to create a bubble field 

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

