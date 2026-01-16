"""
This script shows an example of how to use the tcf code.
"""

import numpy as np
import matplotlib.pyplot as plt
from tcf import tcf, get_bispec
from tqdm import tqdm
import time

if __name__ == '__main__':

	L = 200.  # physical size of the box
	n = 100  # number of pixels along on dimension
	rlin = np.linspace(2, 10., 50)
	ndim = 2
	noise = np.random.normal(0, 1, size=(n,n))  # GRF

	# Compute the TCF of a Gaussian random noise
	print('\nTCF for noise (regular computation)')
	t2 = time.time()
	Bk_noise = get_bispec(noise, L)
	snoise = np.array([tcf(r, Bk_noise.real, L, ndim)[1] for r in tqdm(rlin)])
	t3 = time.time()

	# plot the results
	plt.figure()
	plt.axhline(0, color='k', ls=':')
	plt.plot(rlin, snoise, label=rf'$t={t3-t2:.1f}$s')
	plt.xlabel(r'$r$ [Mpc]')
	plt.ylabel(r'TCF $s(r)$')
	plt.legend()
	plt.tight_layout()

