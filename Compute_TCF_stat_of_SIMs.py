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
################################ Compute TCF of a single SIM #######################################################################################
####################################################################################################################################################

def compute_TCF_of_SIM_data(
    sim_filepath,
    txtfiles_folder_path,
    z_indices=0,
    *,
    tcf_code_dir,
    nthreads=5,
    nbins=100,
    rmin=0.5,
    rmax=60.0,
    overwrite=True
):
    sim_filepath = str(sim_filepath)
    txtfiles_folder_path = Path(txtfiles_folder_path)
    txtfiles_folder_path.mkdir(parents=True, exist_ok=True)

    # --- infer DIM and L from the H5 (once) ---
    with h5py.File(sim_filepath, "r") as f:
        
        ds = f["brightness_lightcone"]

        # --- shape checks --- #
        assert ds.ndim == 4, f"Expected 4D lightcone, got {ds.ndim}D with shape {ds.shape}"
        n_real, n_freq, ny, nx = ds.shape
        assert ny == nx, f"Expected square x–y plane; got {ny}×{nx}"
        
        # axis-order check: z/frequency must be axis 1
        if "frequencies" in f:
            assert f["frequencies"].shape[0] == n_freq, (f"Expected z-axis (axis=1) to match /frequencies length " f"({f['frequencies'].shape[0]}), got {n_freq}")
        if "redshifts" in f:
            assert f["redshifts"].shape[0] == n_freq, (f"Expected z-axis (axis=1) to match /redshifts length " f"({f['redshifts'].shape[0]}), got {n_freq}")
        # --------------------- #

        # get parameters
        DIM = 100# int(ds.shape[-1])                # XXXXXXXXXXXX SET AS 100 FOR SPEED TO CHECK, UNDO THIS
        # robust read of box_length whether (1,) or (1,1)
        L = float(f["box_length"][...].squeeze())

        # ensure z_indices is a list of value(s)
        if z_indices is None:
            z_indices = list(range(n_freq))
        elif isinstance(z_indices, (int, np.integer)):
            z_indices = [int(z_indices)]
        else:
            z_indices = [int(z) for z in z_indices]


    # --- extract slices to txt (per z an output folder Lightcone_zidx{z}) ---
    extract_SIM_z_slices_to_txtfiles(h5_filepath=sim_filepath, output_dir=str(txtfiles_folder_path), z_indices=z_indices)

    # --- TCF instance (reused for all slices) ---
    tcf = Compute_TCF(
        tcf_code_dir=str(tcf_code_dir),
        L=L, DIM=DIM,
        nthreads=nthreads, nbins=nbins, rmin=rmin, rmax=rmax
    )

    # precompile filename pattern for data files only
    pat_data = re.compile(r"^realisation_(\d+)\.txt$")

    # --- loop over requested z ---
    with h5py.File(sim_filepath, "r+") as f:
        for i, z in enumerate(z_indices, start=1):
            print(f" ---> Processing z_idx {z} ({i}/{len(z_indices)}) -----")
            
            folder = txtfiles_folder_path / f"Lightcone_zidx{z}"
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected folder not found: {folder}")

            # strictly match raw data files, ignore TCF outputs
            data_files = sorted(
                [fn for fn in os.listdir(folder) if pat_data.fullmatch(fn)],
                key=lambda s: int(pat_data.fullmatch(s).group(1))
            )
            n_realisations = len(data_files)
            if n_realisations == 0:
                raise RuntimeError(f"No realisation_*.txt files found in {folder}")

            # set up arrays for TCF data
            TCF_Sr_vals = np.zeros((n_realisations, nbins), dtype=np.float32)
            r_values = None

            for idx, fname in enumerate(data_files):
                print(f"  --->  Realisation {idx}/{n_realisations} -----")
                
                fpath = folder / fname
                df = tcf.compute_TCF_of_single_field(str(fpath))  # returns ["r","Re_s_r","Im_s_r","N_modes"]
                if r_values is None:
                    r_values = df["r"].to_numpy(dtype=np.float32)
                    # basic sanity: expected bins
                    if r_values.size != nbins:
                        raise ValueError(f"TCF nbins mismatch: got {r_values.size}, expected {nbins}")

                TCF_Sr_vals[idx, :] = df["Re_s_r"].to_numpy(dtype=np.float32)

            # write datasets for this z (optionally overwrite)
            ds_TCF = f"TCF_zidx{z}"
            ds_r   = f"{ds_TCF}_rvals"

            for name in (ds_TCF, ds_r):
                if name in f:
                    if overwrite:
                        del f[name]
                    else:
                        raise ValueError(f"Dataset '{name}' exists. Use overwrite=True to replace it.")

            f.create_dataset(ds_TCF, data=TCF_Sr_vals, dtype=np.float32)
            f.create_dataset(ds_r, data=r_values, dtype=np.float32)

            # (optional) also save mean & 1σ error bars of the mean
            # C = np.cov(TCF_Sr_vals, rowvar=False, ddof=1)
            # err_mean = np.sqrt(np.diag(C) / n_realisations).astype(np.float32)
            # f.create_dataset(ds_TCF+"_mean", data=TCF_Sr_vals.mean(axis=0).astype(np.float32))
            # f.create_dataset(ds_TCF+"_err",  data=err_mean)

    print("✔ Finished computing TCFs and writing to H5.")

# example usage
# --- paths ---
# mock_h5 = Path("XX/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/Lightcone_MOCK.h5")
# txt_out = Path("XX/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/mock_txtfiles")


# # --- run the function on z_idx=0 only (fast check) ---
# compute_TCF_of_SIM_data(
#     sim_filepath=mock_h5,
#     txtfiles_folder_path=txt_out,
#     z_indices=0,   # just test first slice
#     tcf_code_dir="/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files",
#     nthreads=2,    # keep it small for testing
#     nbins=100,      # fewer bins = faster
#     rmin=0.5,
#     rmax=60.0,
#     overwrite=True
# )

# # --- check results in the mock H5 file ---
# with h5py.File(mock_h5, "r") as f:
#     # print the datasets created
#     for name in f.keys():
#         if "TCF_zidx0" in name:
#             print(f"{name}: shape={f[name].shape}")


####################################################################################################################################################
################################  #######################################################################################
####################################################################################################################################################
