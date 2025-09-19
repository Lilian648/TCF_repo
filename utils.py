"""
utils.py
---------
A collection of helper functions for quick debugging, plotting,
and checking results in data analysis pipelines.
"""

import os
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis, norm
from contextlib import contextmanager

# checking things


# ----------------------
# 🔎 Data Inspection
# ----------------------

def summarize_array(arr, name="array"):
    """Print shape, dtype, min, max, mean, std, NaN count."""
    arr = np.asarray(arr)
    print(f"Summary for {name}:")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={np.nanmin(arr):.3g}, max={np.nanmax(arr):.3g}")
    print(f"  mean={np.nanmean(arr):.3g}, std={np.nanstd(arr):.3g}")
    print(f"  NaNs={np.isnan(arr).sum()} / {arr.size}")


def check_h5_datasets(h5_path):
    """List datasets in an HDF5 file with their shapes/dtypes."""
    with h5py.File(h5_path, "r") as f:
        print(f"Contents of {h5_path}:")
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_attrs)


# ----------------------
# 📊 Plotting Helpers
# ----------------------

def plot_slice2d(arr2d, extent=None, cmap="RdBu_r", title=None, cbar_label=None):
    """Quick imshow plot of a 2D slice with colorbar."""
    im = plt.imshow(arr2d, extent=extent, origin="lower", cmap=cmap)
    plt.colorbar(im, label=cbar_label or "Value")
    if title:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_hist_with_stats(arr, bins=50, title=None):
    """Histogram with Gaussian fit and basic stats."""
    arr = np.ravel(arr)
    mean, std = np.mean(arr), np.std(arr)
    sk, ku = skew(arr), kurtosis(arr)
    
    plt.hist(arr, bins=bins, density=True, alpha=0.6, color="steelblue", edgecolor="k")
    
    # Gaussian fit overlay
    x = np.linspace(mean - 4*std, mean + 4*std, 200)
    plt.plot(x, norm.pdf(x, mean, std), "r--", label="Gaussian fit")
    
    plt.axvline(mean, color="k", linestyle="--", lw=1)
    plt.title(title or "Histogram")
    plt.xlabel("Value"); plt.ylabel("Density")
    plt.legend()
    plt.show()
    
    print(f"Stats for {title or 'array'}:")
    print(f"  mean={mean:.3f}, std={std:.3f}, skew={sk:.3f}, kurtosis={ku:.3f}")
    return {"mean": mean, "std": std, "skew": sk, "kurtosis": ku}


# ----------------------
# 📈 Stats Helpers
# ----------------------

def describe_distribution(arr):
    """Return mean, std, skew, kurtosis, and quantiles of an array."""
    arr = np.ravel(arr)
    stats = {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "skew": skew(arr),
        "kurtosis": kurtosis(arr),
        "q25": np.percentile(arr, 25),
        "median": np.median(arr),
        "q75": np.percentile(arr, 75),
        "iqr": np.percentile(arr, 75) - np.percentile(arr, 25),
    }
    return stats


# ----------------------
# 🗂 File Helpers
# ----------------------

def list_files(folder, pattern="*.h5"):
    """List files matching a pattern inside a folder."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    for f in files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"{f.name:40s} {size_mb:6.2f} MB")
    return files


def ensure_dir(path):
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------
# 🧪 Debugging Tools
# ----------------------

@contextmanager
def timeit_context(name="block"):
    """Time a block of code with `with timeit_context(...):`."""
    start = time.time()
    yield
    print(f"⏱ {name} took {time.time()-start:.2f}s")


def print_memory_usage(arr, name="array"):
    """Rough estimate of array memory in MB."""
    arr = np.asarray(arr)
    size_mb = arr.nbytes / (1024*1024)
    print(f"{name}: {size_mb:.2f} MB in memory")




# ----------------------------
# Specific TCF Project Tools
# ----------------------------


# --- 1. Testing the Add Noise to sims function --- #

### get data ###

# --- parameters --- #
# 1.  AAstar, 1000hrs 
obs_time = 1000.                       # total observation hours
total_int_time = 6.                   # hours per day
int_time = 10.                        # seconds
declination = -30.0                   # declination of the field in degrees
subarray_type = "AA4"
bmax_km = 2. #* units.km # km

verbose = False
save_uvmap = "/data/cluster/lcrascal/uvmaps/uvmap_AA4_1000hrs.h5"
njobs = 1
checkpoint = 16


# --- test inputs --- #
mock_clean = '/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1.h5'


out_paths = add_noise_and_smooth_all_realisations(
    clean_h5_file=mock_clean,
    obs_time=obs_time,            # fake parameters just for test
    total_int_time=total_int_time,
    int_time=int_time,
    declination=declination,
    subarray_type=subarray_type,
    verbose=verbose,
    save_uvmap=save_uvmap,
    njobs=njobs,
    checkpoint=checkpoint,
    bmax_km=bmax_km,
)

print("\nOutputs created:")
for k, v in out_paths.items():
    print(f"  {k}: {v}")

# --- verify outputs ---
for key in ["noise_only_h5", "noisy_h5", "observed_h5"]:
    path = Path(out_paths[key])
    print(f"\n File Name: {path.name}")
    with h5py.File(path, "r") as f:
        for dsname in ["brightness_lightcone", "redshifts", "frequencies", "ngrid"]:
            assert dsname in f, f"{dsname} missing from {path.name}"
        arr = f["brightness_lightcone"]
        print(f" shape={arr.shape}, dtype={arr.dtype}")
        # quick sample statistics for the first realisation
        sample = arr[0]
        print("  sample stats:",
              f"min={np.nanmin(sample):.3g}, max={np.nanmax(sample):.3g}, mean={np.nanmean(sample):.3g}")


### plot data ###


import numpy as np
import matplotlib.pyplot as plt

print(out_paths)
with h5py.File('/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1.h5', "r") as f:
    slice2D_clean = f['brightness_lightcone'][0, 0, :, :]

with h5py.File('/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1_NOISE_ONLY_LC.h5', "r") as f:
    slice2D_noiseonly = f['brightness_lightcone'][0, 0, :, :]

with h5py.File('/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1_NOISE.h5', "r") as f:
    slice2D_noisy = f['brightness_lightcone'][0, 0, :, :]
    
with h5py.File('/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1_NOISE_SMOOTHING.h5', "r") as f:
    slice2D_obs = f['brightness_lightcone'][0, 0, :, :]



# assume these exist and all have shape (Ny, Nx) with the same L:
# slice2D_clean, slice2D_noiseonly, slice2D_noisy, slice2D_obs
# and L is the box size in Mpc (e.g., L = 295)

L = float(295)  # ensure it's a float
extent = (0, L, 0, L)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
axs = axes.ravel()

def show(ax, img, title):
    im = ax.imshow(img, origin='lower', extent=extent, cmap='RdBu_r')
    ax.set_xlabel('x [Mpc]')
    ax.set_ylabel('y [Mpc]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r'$\delta T_b$ [mK]')
    return im

# 1) clean
show(axs[0], slice2D_clean, 'Clean 21cm Map')

# 2) noise only
show(axs[1], slice2D_noiseonly, 'Noise-only Lightcone Slice')

# 3) noisy = clean + noise
show(axs[2], slice2D_noisy, 'Noisy 21cm Map')

# 4) observed (noise + smoothing) + overlay contours from clean
show(axs[3], slice2D_obs, 'Obs 21cm Map with Clean Contours')
ny, nx = slice2D_clean.shape
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
# choose contour levels (example: 3 quantiles of the clean map)
levels = [1e-4]
axs[3].contour(x, y, slice2D_clean, levels=levels, colors='black', linewidths=1.0)

plt.show()



### plot histograms and stats of data ####


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # needs scipy installed

# flatten all maps into 1D arrays for histogramming
maps = {
    "Clean": slice2D_clean.ravel(),
    "Noise-only": slice2D_noiseonly.ravel(),
    "Noisy": slice2D_noisy.ravel(),
    "Observed (Noise+Smoothing)": slice2D_obs.ravel(),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
axs = axes.ravel()

def print_stats(title, x):
    mean  = np.mean(x)
    std   = np.std(x, ddof=1)
    skew  = stats.skew(x, bias=False, nan_policy="omit")
    kurt  = stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit")  # 0 for perfect normal
    med   = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr   = q75 - q25
    xmin, xmax = np.min(x), np.max(x)
    print(f"\n[{title}]")
    print(f"  mean={mean:.4g}, std={std:.4g}, skew={skew:.4g}, kurtosis(excess)={kurt:.4g}")
    print(f"  median={med:.4g}, IQR={iqr:.4g} (Q25={q25:.4g}, Q75={q75:.4g})")
    print(f"  min={xmin:.4g}, max={xmax:.4g}")

for ax, (title, data) in zip(axs, maps.items()):
    # robust range per panel
    vmin, vmax = np.percentile(data, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 70)

    # histogram
    counts, edges, _ = ax.hist(
        data, bins=bins, color='steelblue', alpha=0.7, edgecolor='k'
    )

    # fit normal (mu, sigma)
    mu_hat, sigma_hat = stats.norm.fit(data)

    # overlay fitted normal scaled to histogram counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    pdf = stats.norm.pdf(centers, loc=mu_hat, scale=sigma_hat)   # density
    y_fit = pdf * data.size * bin_width                          # scale to counts
    ax.plot(centers, y_fit, lw=2, label=fr"Normal fit: $\mu={mu_hat:.3g}$, $\sigma={sigma_hat:.3g}$")

    # cosmetics
    ax.set_title(f"{title} Histogram")
    ax.set_xlabel(r'$\delta T_b$ [mK]')
    ax.set_ylabel("Pixel count")
    ax.legend(loc="best", frameon=False)

    # print stats to console
    print_stats(title, data)

plt.show()


