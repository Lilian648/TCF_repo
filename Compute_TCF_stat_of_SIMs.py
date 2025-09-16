import os
import h5py
import numpy as np
from pathlib import Path

import TCF_Class

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


def compute_TCF_of_single_SIM_all_realisations(
    sim_filepath,
    z_indices=0,
    *,
    tcf_code_dir,
    nthreads=5,
    nbins=100,
    rmin=3,
    rmax=60.0,
    overwrite_h5=True,        # overwrite H5 datasets
    overwrite_txt=False,      # re-extract .txt even if already present
    continue_on_error=False   # skip bad realisations instead of raising
):

    h = 0.6774
    
    sim_filepath = Path(sim_filepath).resolve()
    sim_dir = sim_filepath.parent
    sim_stem = sim_filepath.stem

    # Create txt folder for this sim
    txt_root = sim_dir / f"{sim_stem}_txtfiles"
    txt_root.mkdir(parents=True, exist_ok=True)
    print(f"📂 TXT files will be stored in: {txt_root}")


    # --- read sim metadata once --- #
    with h5py.File(sim_filepath, "r") as f:
        ds = f["brightness_lightcone"]
        assert ds.ndim == 4, f"Expected 4D lightcone, got {ds.ndim}D with shape {ds.shape}"
        n_real, n_freq, ny, nx = ds.shape
        assert ny == nx, f"Expected square x–y plane; got {ny}×{nx}"
        if "frequencies" in f:
            assert f["frequencies"].shape[0] == n_freq
        if "redshifts" in f:
            assert f["redshifts"].shape[0] == n_freq
        L_Mpc_perh = float(f["box_length"][...].squeeze()) #
        L = L_Mpc_perh/h # Mpc
        DIM = int(nx)
        print("L:", L)
        print("DIM:", DIM)

    # normalize z_indices 
    if z_indices is None:                            # case for all z slices in sim
        z_indices = list(range(n_freq))
    elif isinstance(z_indices, (int, np.integer)):   # case for only a single z slice
        z_indices = [int(z_indices)]
    else:                                            # case for a subset of z slices in sim
        z_indices = [int(z) for z in z_indices]

    # --- extract slices to txtfiles --- #
    # Only extract if overwrite_txt=True or slice folders missing
    need_extract = overwrite_txt or any(
        not (txt_root / f"Lightcone_zidx{z}").exists() for z in z_indices
    )
    if need_extract:
        print("📂 Extracting slices to .txt files...")
        extract_SIM_z_slices_to_txtfiles(
            h5_filepath=str(sim_filepath), output_dir=str(txt_root), z_indices=z_indices
        )
    else:
        print("📂 Using already-extracted .txt slices (overwrite_txt=False)")


    # --- TCF Class instance --- #
    tcf = TCF_Class.Compute_TCF(
        tcf_code_dir=str(tcf_code_dir),
        L=L, DIM=DIM,
        nthreads=nthreads, nbins=nbins, rmin=rmin, rmax=rmax
    )

    pat_data = re.compile(r"^realisation_(\d+)\.txt$")  # raw slices only

    t0_all = time.time()
    with h5py.File(sim_filepath, "r+") as f_out:
        for i, z in enumerate(z_indices, start=1):
            t0 = time.time()
            folder = txt_root / f"Lightcone_zidx{z}"
            print(f"▶ z_idx {z} ({i}/{len(z_indices)}) → {folder}")

            # --- Get all sim slices txt files --- #
            data_files = []
            
            # Loop through all items in the folder
            for entry in os.scandir(folder):
                if entry.is_file():  # make sure it's a plain file, not a directory
                    filename = entry.name
                    match = pat_data.fullmatch(filename)  # check if filename matches regex "realisation_<id>.txt"
                    if match:
                        # add to list of data files
                        data_files.append((int(match.group(1)), filename))
            
            # Sort the list of tuples by the integer ID
            data_files.sort(key=lambda pair: pair[0])
            
            # Extract just the filenames, in the correct numeric order
            names_sorted = [filename for (_, filename) in data_files]
            
            # Count how many realisation files we found
            n_realisations = len(names_sorted)
            if n_realisations == 0:
                raise RuntimeError(f"No realisation_*.txt files in {folder}")

            print(f" Found {n_realisations} realisations")

            # Pre-create datasets and stream rows
            ds_TCF = f"TCF_zidx{z}"
            ds_r   = f"{ds_TCF}_rvals"
            if ds_TCF in f_out:
                if overwrite_h5:
                    print(f"   Overwriting existing dataset {ds_TCF}")
                    del f_out[ds_TCF]
                    if ds_r in f_out:   # delete r dataset too if it’s there
                        del f_out[ds_r]
                else:
                    raise ValueError(f"Dataset '{ds_TCF}' (and r-values) exist. Use overwrite_h5=True.")

            dset_TCF = f_out.create_dataset(ds_TCF, shape=(n_realisations, nbins), dtype=np.float32)
            r_vals_written = False

            # loop realisations
            for j, fname in enumerate(names_sorted, start=1):
                fpath = folder / fname
                started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"      → Starting Realisation {j}/{n_realisations}: {fname}  (start: {started})")
                
                try:
                    df = tcf.compute_TCF_of_single_field(str(fpath))
                except Exception as e: # just incase something fails
                    msg = f"   ✗ Realisation {j}/{n_realisations} failed: {e}"
                    if continue_on_error:
                        print(msg)
                        continue
                    raise

                if not r_vals_written: # add r values to the h5 file
                    r_values = df["r"].to_numpy(dtype=np.float32)
                    if r_values.size != nbins:
                        raise ValueError(f"TCF nbins mismatch: got {r_values.size}, expected {nbins}")
                    f_out.create_dataset(ds_r, data=r_values, dtype=np.float32)
                    r_vals_written = True

                dset_TCF[j-1, :] = df["Re_s_r"].to_numpy(dtype=np.float32)
                if j % max(1, n_realisations//10) == 0 or j == n_realisations:
                    print(f"   →  Completed Realisation {j}/{n_realisations}")

            print(f"   ✓ z_idx {z} done in {time.time()-t0:.1f}s")

    print(f"✔ All done in {time.time()-t0_all:.1f}s")

# example usage

# --- paths ---
# mock_h5 = Path("/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/Mock_SIM_1/Lightcone_MOCK_1.h5")
# tcf_code_dir = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files"


# # --- run the pipeline ---
# compute_TCF_of_single_SIM_all_realisations(
#     sim_filepath=mock_h5,
#     z_indices=0,
#     tcf_code_dir=tcf_code_dir,
#     nthreads=5,
#     nbins=100,
#     rmin=3,
#     rmax=100.0,
#     overwrite_h5=True,
#     overwrite_txt=False,      # don’t re-extract if already there
#     continue_on_error=False
# )

# check results
# z_to_test = [0]
# nbins_test = 100
# # --- verify outputs exist and have correct shapes ---
# with h5py.File(mock_h5, "r") as f:
#     # discover n_realisations from the source dataset
#     n_real = f["brightness_lightcone"].shape[0]

#     for z in z_to_test:
#         ds_TCF = f"TCF_zidx{z}"
#         ds_r   = f"{ds_TCF}_rvals"
#         assert ds_TCF in f, f"Missing dataset {ds_TCF}"
#         assert ds_r   in f, f"Missing dataset {ds_r}"

#         T = f[ds_TCF][...]     # shape: (n_realisations, nbins)
#         r = f[ds_r][...]       # shape: (nbins,)
#         print(f"{ds_TCF}: shape={T.shape}, {ds_r}: shape={r.shape}")

#         # shape checks
#         assert T.shape[0] == n_real, f"Row count mismatch ({T.shape[0]} vs {n_real})"
#         assert T.shape[1] == nbins_test, f"nbins mismatch ({T.shape[1]} vs {nbins_test})"
#         assert r.shape[0] == nbins_test, "r grid length mismatch"

#         # quick content sanity (not all zeros / NaNs)
#         print("  sample row stats: min={:.3g} max={:.3g}".format(np.nanmin(T[0]), np.nanmax(T[0])))
#         print("  r range: [{:.3g}, {:.3g}]".format(float(r.min()), float(r.max())))


# for t in T:
#     plt.plot(r,t, color='gray')

# plt.plot(r, np.mean(T, axis=0), color='crimson')
# #plt.ylim(-0.2, 0.2)
# plt.xlim(0, 100)
# plt.show()

# with h5py.File(mock_h5, "r") as f:
#     slice2D = f['brightness_lightcone'][0, 0, :, :]

# L = 295
# plt.imshow(slice2D, extent=(0, L, 0, L))
# plt.xlabel('x Mpc')
# plt.ylabel('y Mpc')
# plt.show()
        

####################################################################################################################################################
################################ Compute TCF of all SIMs ###########################################################################################
####################################################################################################################################################


def run_all(
    simlist,
    *,
    tcf_code_dir,             # path to your TCF code dir
    z_indices=0,              # None = all; int or list ok
    nthreads=5,
    nbins=100,
    rmin=3,
    rmax=100,
    overwrite_h5=True,
    overwrite_txt=True,
    continue_on_error=False
):
    simlist = [Path(p).resolve() for p in simlist]

    t0_all = time.time()
    print(f"🚀 Starting TCF run for {len(simlist)} sims\n")

    for s_idx, sim_path in enumerate(simlist, start=1):
        sim_name = sim_path.stem  # e.g. "Lightcone_FID_..."

        # the per-sim function will create: sim_path.parent / f"{sim_name}_txtfiles"
        expected_txt = sim_path.parent / f"{sim_name}_txtfiles"

        print(f"\n========== [{s_idx}/{len(simlist)}] {sim_name} ==========")
        print(f"H5:   {sim_path}")
        print(f"TXT:  {expected_txt}  (auto-created if needed)")

        t0 = time.time()
        try:
            compute_TCF_of_single_SIM_all_realisations(
                sim_filepath=sim_path,
                z_indices=z_indices,
                tcf_code_dir=tcf_code_dir,
                nthreads=nthreads,
                nbins=nbins,
                rmin=rmin,
                rmax=rmax,
                overwrite_h5=overwrite_h5,
                overwrite_txt=overwrite_txt,
                continue_on_error=continue_on_error,
            )
        except Exception as e:
            print(f"✗ FAILED on {sim_name}: {e}")
            if not continue_on_error:
                raise
            continue

        # (optional) quick sanity: list newly written datasets
        try:
            with h5py.File(sim_path, "r") as f:
                keys = sorted(k for k in f.keys() if str(k).startswith("TCF_zidx"))
                preview = ", ".join(keys[:4]) + (" ..." if len(keys) > 4 else "")
                if keys:
                    print(f"   ✓ Wrote datasets: {preview}")
                else:
                    print("   (no TCF_zidx* datasets found)")
        except Exception:
            pass

        print(f"⏱  {sim_name} done in {time.time()-t0:.1f}s")

    print(f"\n✔ All sims done in {time.time()-t0_all:.1f}s")



# example usage

# # configure once
# tcf_code_dir = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files"

# # your list of H5 files
# mock_h5_1 = Path("/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/Mock_SIM_1/Lightcone_MOCK_1.h5")
# mock_h5_2 = Path("/data/cluster/lcrascal/SIM_data/h5_files/mock_tests/Mock_SIM_2/Lightcone_MOCK_2.h5")
# simlist = [mock_h5_1, mock_h5_2]

# # run (fast test config: 2 z-slices, fewer bins/threads)
# run_all(
#     simlist,
#     tcf_code_dir=tcf_code_dir,
#     z_indices=0,   # None for all
#     nthreads=5,
#     nbins=100,
#     rmin=3,
#     rmax=100,
#     overwrite_h5=True,
#     overwrite_txt=True,
#     continue_on_error=False,
# )

# same checks as compute_TCF_of_single_SIM_all_realisations function  

####################################################################################################################################################
################################  ###########################################################################################
####################################################################################################################################################
