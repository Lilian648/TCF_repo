#ifndef _SC_H_
#define _SC_H_
#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <string>
#include <omp.h>
#include <fftw3.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;

static const double pi = 3.141592653589793;
static const double s3 = sqrt(3.0);

static const int nthreads = 5; //number of threads for parallelisation
static const int N = 256; //sampling number (pix dimensions of box)
static const double L = 295.24653085326247; //length of the realspace box in Mpc
static const string filename_box = "/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/Mock_SIM_2/Lightcone_MOCK_2_txtfiles/Lightcone_zidx0/realisation_2"; // don't need the .txt at end of filename  

static const int nbins = 100;//number of bins for correlation scales
static const double rmin = 3.0;
static const double rmax = 100.0; //min and max value of correlation scales probed in Mpc 

// #define pi (double) 3.141592653589793
// #define s3 (double) sqrt(3.0)

// #define nthreads (int) 20 //number of threads for parallelisation
// #define N (int) 512 //sampling number
// #define L (double) 400 //length of the realspace box
// #define NDIM (int) 2 //dimension of the box
// #define filename_box (str) "Field_20binary_bubbles_nooverlap=False_radius=20_xhII0.100_N512_2D"

// #define nbins (int) 100 //number of bins for correlation scales
// #define rmin (double) 1.
// #define rmax (double) 30.  //min and max value of correlation scales probed 

#endif /* SC_H */