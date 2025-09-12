import os
import h5py
import numpy as np
from pathlib import Path


####################################################################################################################################################
################################ Extract z slices from h5 file #####################################################################################
####################################################################################################################################################


def extract_SIM_z_slices_to_txtfiles(h5_filepath, output_dir, z_indices=None):
    """
    Extract 2D slices at specified z-indices and save each realisation as .txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_filepath, 'r') as f:
        ds = f['brightness_lightcone']

        # Infer dims
        if ds.ndim == 4:                    # (n_realisations, z, y, x)
            n_realisations, n_freq = ds.shape[0], ds.shape[1]
        elif ds.ndim == 3:                  # (z, y, x) — single realisation
            n_realisations, n_freq = 1, ds.shape[0]
        else:
            raise ValueError(f"Unexpected ndim={ds.ndim}; expected 3 or 4.")

        # Normalize input
        if z_indices is None:
            z_indices = list(range(n_freq))
        elif isinstance(z_indices, int):
            z_indices = [z_indices]
        else:
            z_indices = list(z_indices)

        saved_dirs = []

        for z_idx in z_indices:
            if not (0 <= z_idx < n_freq):
                raise ValueError(f"z_idx {z_idx} out of bounds [0, {n_freq-1}]")

            # Create the per-z subfolder (this was the missing piece)
            output_folder = output_dir / f"Lightcone_zidx{z_idx}"
            output_folder.mkdir(parents=True, exist_ok=True)

            print(f"\nExtracting slices for z_idx={z_idx} -> {output_folder}")

            if ds.ndim == 4:
                for i in range(n_realisations):
                    slice_2d = ds[i, z_idx, :, :]
                    np.savetxt(output_folder / f"realisation_{i}.txt", slice_2d)
            else:
                slice_2d = ds[z_idx, :, :]
                np.savetxt(output_folder / "realisation_0.txt", slice_2d)

            print(f"  ✔ Saved {n_realisations} slice(s) for z_idx={z_idx}")
            saved_dirs.append(str(output_folder))

    return saved_dirs


# example usage
"""
mock_h5_filepath = 'XXX/data/cluster/lcrascal/SIM_data/h5_files/Lightcone_MOCK.h5'
output_dir = 'XXX/data/cluster/lcrascal/SIM_data/h5_files/mock_txtfiles/'

zrange = np.linspace(0, 12, 13, dtype=int)
extract_SIM_z_slices_to_txtfiles(mock_h5_filepath, output_dir, z_indices=zrange)
"""

####################################################################################################################################################
################################     #####################################################################################
####################################################################################################################################################



