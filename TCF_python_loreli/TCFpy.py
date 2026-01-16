from scipy.special import j0
import numpy as np

def window_function(x, ndim=2):
	if ndim==2:
		return j0(x) 
	elif ndim==3:
		return np.sinc(x)
	else:
		raise ValueError('Only work in 2 or 3 dimensions.')

# def tcf(r, Bk, L, ndim):
# 	"""
# 	Computes TCF for a given r and bispectrum.

# 	Parameters
# 	----------
# 		r: float
# 			Value of the scale (triangle side length) the TCF is computed at.
# 		Bk: array of floats
# 			Real part of the bispectrum of the field considered.
# 			Shape is (n, n, n, n) for n the number of pixel along one dim
# 			of the field (n, n).
# 		L: float
# 			Side length in Mpc of the field.
# 		ndim: int
# 			Number of dimensions of your input field (2 or 3).

# 	Returns
# 	-------
# 		nmodes: int
# 			Number of modes contributing to the r-bin.
# 		s: float
# 			Value of the TCF at r.

# 	"""
#     print("X starting")
# 	n = Bk.shape[0]
# 	# define k-axis in Fourier space
# 	kcoord = np.fft.fftfreq(n, d=L/n/2./np.pi)
# 	# find kcoordinates in ndim and corresponding k-norm
#     print("X1")
# 	if ndim==2:
#         print("X2")
# 		kx, ky = np.meshgrid(kcoord, kcoord)
# 		qx, qy = np.meshgrid(kcoord, kcoord)
# 		knorm = np.sqrt(kcoord[:, None]**2 + kcoord**2)
# 	elif ndim==3:
# 		kx, ky, kz = np.meshgrid(kcoord, kcoord, kcoord)
# 		qx, qy, qz = np.meshgrid(kcoord, kcoord, kcoord)
# 		knorm = np.sqrt(kcoord[:, None, None]**2 + kcoord[None, :, None]**2 + kcoord[None, None, :]**2)
# 	qnorm = np.copy(knorm)
# 	# definition of the p-vector (eq.10)
#     print("X3")
# 	px = kx[..., None, None] + 0.5*qx + np.sqrt(3.)/2.*qy
# 	py = ky[..., None, None] - np.sqrt(3)/2.*qx + 0.5*qy
# 	if ndim==2:
#         print("X4")
# 		pnorm = np.sqrt(px**2 + py**2)
# 	elif ndim==3:
# 		pz = kz + qz
# 		pnorm = np.sqrt(px**2 + py**2 + pz**2)
# 	# window function selecting the appropriate triangles
#     print("X5")
# 	w = window_function(pnorm * r, ndim)
# 	# keep only modes below confusion scale (see paper)
# 	mask = (knorm <= np.pi/r)
# 	return Bk[mask, mask].size, (r/L)**(3.*ndim/2.) * np.sum(w[mask][:, mask] * Bk[mask][:, mask])

def tcf(r, Bk, L, ndim):
    """
    Computes TCF for a given r and bispectrum.
    """

    n = Bk.shape[0]
    # define k-axis in Fourier space
    kcoord = np.fft.fftfreq(n, d=L/n/2./np.pi)

    # find kcoordinates in ndim and corresponding k-norm
    if ndim == 2:
        kx, ky = np.meshgrid(kcoord, kcoord)
        qx, qy = np.meshgrid(kcoord, kcoord)
        knorm = np.sqrt(kcoord[:, None]**2 + kcoord**2)
    elif ndim == 3:
        kx, ky, kz = np.meshgrid(kcoord, kcoord, kcoord)
        qx, qy, qz = np.meshgrid(kcoord, kcoord, kcoord)
        knorm = np.sqrt(
            kcoord[:, None, None]**2
            + kcoord[None, :, None]**2
            + kcoord[None, None, :]**2
        )
    else:
        raise ValueError("ndim must be 2 or 3")

    qnorm = np.copy(knorm)

    # definition of the p-vector (eq.10)
    px = kx[..., None, None] + 0.5*qx + np.sqrt(3.)/2.*qy
    py = ky[..., None, None] - np.sqrt(3.)/2.*qx + 0.5*qy

    if ndim == 2:
        pnorm = np.sqrt(px**2 + py**2)
    else:  # ndim == 3
        pz = kz + qz
        pnorm = np.sqrt(px**2 + py**2 + pz**2)

    # window function selecting the appropriate triangles
    w = window_function(pnorm * r, ndim)

    # keep only modes below confusion scale (see paper)
    mask = (knorm <= np.pi/r)

    return (
        Bk[mask, mask].size,
        (r/L)**(3.*ndim/2.) * np.sum(w[mask][:, mask] * Bk[mask][:, mask])
    )


# def get_bispec(field, L):
# 	"""
# 	Computes the normalised bispectrum of an input field.

# 	Parameters
# 	----------
# 		field: array of floats
# 			(Real) field to compute the bispectrum of.
# 			Shape is (n, n) for a two-dimensional field.
# 		L: float
# 			Side length in Mpc of the field.

# 	Returns
# 	-------
# 		Bk: array of complex numbers
# 			Real part of the bispectrum of the field considered.
# 			Shape is (n, n, n, n) for n the number of pixel along one dim
# 			of the field, for a two-dimensional field.

# 	"""
#     print("Y1")
# 	ndim = field.ndim
# 	n = field.shape[0]
# 	# take fourier transform of the input field
#     print("Y2")
# 	field_k = np.fft.fftn(field)
# 	m = np.abs(field_k)<2.*np.pi/L
# 	# normalise by the absolute value of the FT (eq. 2)
# 	# to isolate phase information
#     print("Y3")
# 	epsilon_k = np.divide(field_k, np.abs(field_k), where=~m)
# 	# remove random phase terms
# 	epsilon_k[m] = 0.
# 	# find coordinates of k+q vector (eq. 7b)
# 	if ndim==2:
#         print("Y4")
# 		ikx, iky = np.indices((n, n))
# 	elif ndim==3:
# 		ikx, iky, ikz = np.indices((n, n, n))
#     print("Y5")
# 	isx = np.mod(ikx + ikx[..., None, None], n)
# 	isy = np.mod(iky + iky[..., None, None], n)
# 	# output triple product
#     print("Y6")
# 	return epsilon_k * epsilon_k[..., None, None] * np.conj(epsilon_k[isx, isy])



def get_bispec(field, L):
    """
    Computes the normalised bispectrum of an input field.

    Parameters
    ----------
    field : array_like
        (Real) field to compute the bispectrum of.
        Shape is (n, n) for a two-dimensional field.
    L : float
        Side length in Mpc of the field.

    Returns
    -------
    Bk : ndarray of complex
        Phase-only bispectrum of the field.
        For a 2D field, shape is (n, n, n, n) corresponding to B(k, q).
    """
    ndim = field.ndim
    n = field.shape[0]

    # take fourier transform of the input field
    field_k = np.fft.fftn(field)

    m = np.abs(field_k) < 2.0 * np.pi / L

    # normalise by the absolute value of the FT (eq. 2)
    # to isolate phase information
    epsilon_k = np.divide(field_k, np.abs(field_k), where=~m)

    # remove random phase terms
    epsilon_k[m] = 0.0

    # find coordinates of k+q vector (eq. 7b)
    if ndim == 2:
        ikx, iky = np.indices((n, n))
    elif ndim == 3:
        ikx, iky, ikz = np.indices((n, n, n))
    else:
        raise ValueError("Only works in 2 or 3 dimensions.")

    print("Y5")
    isx = np.mod(ikx + ikx[..., None, None], n)
    isy = np.mod(iky + iky[..., None, None], n)

    # output triple product
    print("Y6")
    Bk = epsilon_k * epsilon_k[..., None, None] * np.conj(epsilon_k[isx, isy])
    
    return Bk
