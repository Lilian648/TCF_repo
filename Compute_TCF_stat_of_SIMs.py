import os
import h5py
import numpy as np
from pathlib import Path
import re, time
from datetime import datetime
import tools21cm as t2c


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

# mock_h5_filepath = 'XXX/data/cluster/lcrascal/SIM_data/h5_files/Lightcone_MOCK.h5'
# output_dir = 'XXX/data/cluster/lcrascal/SIM_data/h5_files/mock_txtfiles/'

# zrange = np.linspace(0, 12, 13, dtype=int)
# extract_SIM_z_slices_to_txtfiles(mock_h5_filepath, output_dir, z_indices=zrange)


####################################################################################################################################################
################################ Add Noise + Smoothing to a single SIM #############################################################################
####################################################################################################################################################

def add_noise_and_smooth_all_realisations(
    clean_h5_file,
    *,
    obs_time,
    total_int_time,
    int_time,
    declination,
    subarray_type,
    verbose,
    save_uvmap,
    njobs,
    checkpoint,
    bmax_km
):
    """
    From a clean lightcone H5 (n_real, z, x, y), produce:
      <stem>_NOISE_ONLY_LC.h5          (noise-only)
      <stem>_NOISE.h5                  (clean + noise)
      <stem>_NOISE_SMOOTHING.h5        (subtract LOS-mean(clean) + noise, then smooth)
    All saved in canonical (n_real, z, x, y). Each output also stores 'redshifts_used'
    which matches any flip performed internally by smooth_lightcone.

    Notes:
    - t2c.noise_lightcone returns (x,y,z)
    - t2c.smooth_lightcone expects (x,y,z) and returns (x,y,z) with a possible axis-2 flip
      if input redshifts are decreasing. It returns the redshifts it used as the 2nd value.
    """

    # ---- helpers to convert per-realisation views ----
    def zxy_to_xyz(a):  # (z,x,y) -> (x,y,z)
        return np.moveaxis(a, 0, 2)

    def xyz_to_zxy(a):  # (x,y,z) -> (z,x,y)
        return np.moveaxis(a, 2, 0)

    clean_h5_file = Path(clean_h5_file).resolve()
    stem    = clean_h5_file.stem

    # Build noise tag (e.g. "AAstar_1000hrs")
    noise_tag = f"{subarray_type}_{int(obs_time)}hrs"

    # Create the output directory: parent / noise_tag
    out_dir = Path(clean_h5_file).parent / noise_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # target H5 paths
    noiseonly_out = out_dir / f"{stem}_NOISE_ONLY_LC.h5"
    noisy_out     = out_dir / f"{stem}_NOISE.h5"
    obs_out       = out_dir / f"{stem}_NOISE_SMOOTHING.h5"

    print("Starting SKA noise + smoothing for all realisations")
    print(f"Input:  {clean_h5_file.name}")
    print(f"Will create output: {noiseonly_out.name}, {noisy_out.name}, {obs_out.name}")

    # ---- read metadata once ----
    with h5py.File(clean_h5_file, 'r') as f:
        redshifts   = f['redshifts'][...]
        frequencies = f['frequencies'][...]
        box_length_Mpc_perh  = float(f['box_length'][...].squeeze()) # Mpc per h
        h = 0.6774
        box_length_Mpc = box_length_Mpc_perh/h # Mpc
        print("box_length_Mpc_perh", box_length_Mpc_perh)
        print("box_length_Mpc", box_length_Mpc)
        box_dim     = int(np.array(f['ngrid'][...]).squeeze()) # number of pixels
        ds          = f['brightness_lightcone']      # (n_real, z, x, y)
        n_real, n_z, ny, nx = ds.shape
        assert ny == nx == box_dim, f"Shape mismatch: {ds.shape} vs box_dim={box_dim}"
        print(f"Header: n_real={n_real}, n_z={n_z}, DIM={box_dim}")

    # ---- UV map path ----
    uvpath = Path(save_uvmap)
    uvpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'Reusing' if uvpath.exists() else 'Will save'} UV map at: {uvpath}")

    # ---- small factory to build output files ----
    def create_out(path):
        fout = h5py.File(path, 'w')
        fout.create_dataset("redshifts", data=redshifts)
        fout.create_dataset("frequencies", data=frequencies)
        fout.create_dataset("box_length_Mpc_perh", data=np.array([box_length_Mpc_perh], dtype=np.float64))
        fout.create_dataset("box_length_Mpc", data=np.array([box_length_Mpc], dtype=np.float64))
        fout.create_dataset("ngrid", data=np.array([box_dim], dtype=np.int64))
        fout.create_dataset("nrealisations", data=np.array([n_real], dtype=np.int64))
        dset = fout.create_dataset(
            "brightness_lightcone",
            shape=(n_real, n_z, box_dim, box_dim),
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        dset.attrs["axis_order"] = "n_real, z, x, y"
        # we'll store the actually used redshifts (may be flipped by t2c) here:
        rz = fout.create_dataset("redshifts_used", shape=(n_z,), dtype=redshifts.dtype)
        return fout, dset, rz

    f_noise, d_noise, rz_noise = create_out(noiseonly_out)
    f_noisy, d_noisy, rz_noisy = create_out(noisy_out)
    f_obs,   d_obs,   rz_obs   = create_out(obs_out)

    try:
        # set redshifts_used defaults to input; may be overwritten after a call to smooth_lightcone
        rz_noise[...] = redshifts
        rz_noisy[...] = redshifts
        rz_obs  [...] = redshifts

        with h5py.File(clean_h5_file, 'r') as f:
            ds = f['brightness_lightcone']  # (n_real, z, x, y)

            for i in range(n_real):
                # if i % 10 == 0:
                print(f" ➤ Realisation {i+1}/{n_real}")

                # (z,x,y) for a single realisation
                clean_zxy = ds[i, ...]
                # convert to (x,y,z) for t2c
                clean_xyz = zxy_to_xyz(clean_zxy)  # (x,y,z)

                # --- noise (x,y,z) ---
                noise_xyz = t2c.noise_lightcone(
                    ncells=box_dim,
                    zs=redshifts,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type=subarray_type,
                    boxsize=box_length_Mpc, # Must be in Mpc
                    verbose=verbose,
                    save_uvmap=str(uvpath),
                    n_jobs=njobs,
                    checkpoint=checkpoint,
                )

                # basic shape guard (common mistake: zxy instead of xyz)
                if noise_xyz.shape != (box_dim, box_dim, n_z):
                    if noise_xyz.shape == (n_z, box_dim, box_dim):
                        noise_xyz = zxy_to_xyz(noise_xyz)
                    else:
                        raise ValueError(f"Unexpected noise shape {noise_xyz.shape}; expected (x,y,z)=({box_dim},{box_dim},{n_z})")

                # write noise-only as (z,x,y)
                d_noise[i, ...] = xyz_to_zxy(noise_xyz)

                # --- noisy = clean + noise (still xyz) ---
                noisy_xyz = (clean_xyz + noise_xyz).astype(np.float32)
                d_noisy[i, ...] = xyz_to_zxy(noisy_xyz)

                # --- observed: subtract LOS-mean(clean), add noise, smooth ---
                # smooth_lightcone expects (x,y,z), treats axis=2 as z,
                # returns (x,y,z) and redshifts_used (possibly flipped)
                obs_xyz, rz_used = t2c.smooth_lightcone(
                    lightcone = noise_xyz + t2c.subtract_mean_signal(clean_xyz, los_axis=2),
                    z_array=redshifts,
                    box_size_mpc=box_length_Mpc,
                    max_baseline=bmax_km,
                )
                obs_xyz = obs_xyz.astype(np.float32)

                # check correct redshift used
                if (rz_used.shape != redshifts.shape) or (not np.array_equal(rz_used, redshifts)):
                    raise RuntimeError(
                        "smooth_lightcone returned a different redshift grid than provided."
                        "This usually means the function flipped the LOS because "
                        "your input redshifts were decreasing. Ensure input redshifts are strictly "
                        "increasing and try again."
                    )

                # sanity on shape
                if obs_xyz.shape != (box_dim, box_dim, n_z):
                    if obs_xyz.shape == (n_z, box_dim, box_dim):
                        obs_xyz = zxy_to_xyz(obs_xyz)
                    else:
                        raise ValueError(f"Unexpected smoothed shape {obs_xyz.shape}")

                # write observed back as (z,x,y), record the z grid actually used
                d_obs[i, ...] = xyz_to_zxy(obs_xyz)
                rz_obs[...]   = redshifts  # same for all realisations for a given input; OK to overwrite

                # check things are working sensibly
                assert np.allclose(noisy_xyz - clean_xyz, noise_xyz, atol=1e-5), \
                    "Noisy - Clean != Noise-only (something is inconsistent!)"


    finally:
        f_noise.close()
        f_noisy.close()
        f_obs.close()

    print("All realisations processed and saved.")
    return {
        "noise_only_h5": str(noiseonly_out),
        "noisy_h5": str(noisy_out),
        "observed_h5": str(obs_out)
    }


# example usage

# # --- parameters --- #
# obs_time = 1000.                       # total observation hours
# total_int_time = 6.                   # hours per day
# int_time = 10.                        # seconds
# declination = -30.0                   # declination of the field in degrees
# subarray_type = "AA4"
# bmax_km = 2. #* units.km # km

# verbose = False
# save_uvmap = "/data/cluster/lcrascal/uvmaps/uvmap_AA4_1000hrs.h5"
# njobs = 1
# checkpoint = 16


# # --- test inputs --- #
# mock_clean = '/data/cluster/lcrascal/SIM_data/mock_tests/Mock_SIM_1/Lightcone_MOCK_1.h5'


# out_paths = add_noise_and_smooth_all_realisations(
#     clean_h5_file=mock_clean,
#     obs_time=obs_time,            # fake parameters just for test
#     total_int_time=total_int_time,
#     int_time=int_time,
#     declination=declination,
#     subarray_type=subarray_type,
#     verbose=verbose,
#     save_uvmap=save_uvmap,
#     njobs=njobs,
#     checkpoint=checkpoint,
#     bmax_km=bmax_km,
# )

# print("\nOutputs created:")
# for k, v in out_paths.items():
#     print(f"  {k}: {v}")

# # --- verify outputs ---
# for key in ["noise_only_h5", "noisy_h5", "observed_h5"]:
#     path = Path(out_paths[key])
#     print(f"\n File Name: {path.name}")
#     with h5py.File(path, "r") as f:
#         for dsname in ["brightness_lightcone", "redshifts", "frequencies", "ngrid"]:
#             assert dsname in f, f"{dsname} missing from {path.name}"
#         arr = f["brightness_lightcone"]
#         print(f" shape={arr.shape}, dtype={arr.dtype}")
#         # quick sample statistics for the first realisation
#         sample = arr[0]
#         print("  sample stats:",
#               f"min={np.nanmin(sample):.3g}, max={np.nanmax(sample):.3g}, mean={np.nanmean(sample):.3g}")


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


import time, h5py
from pathlib import Path

def run_all(
    simlist,
    *,
    tcf_code_dir,             # path to TCF code dir
    z_indices=0,              # None = all; int or list ok
    nthreads=5,
    nbins=100,
    rmin=3,
    rmax=100,
    overwrite_h5=True,
    overwrite_txt=True,
    continue_on_error=False,
    # extra args needed for noise/smoothing
    obs_time=1000,
    total_int_time=6.0,
    int_time=10.0,
    declination=-30.0,
    subarray_type="AAstar",
    save_uvmap,
    njobs=4,
    checkpoint=16,
    bmax_km=2.0,
    # function specific parameters
    include_clean=True       # compute the TCF of the clean sim?
):
    """
    Loop over sims, run TCF. If 'FID' in filename, also generate noise/smoothing
    variants and run TCF on those.

    Notes:
    - By default (include_clean=True), also runs TCF on the original clean
      FID H5 (left at its parent location). Set `include_clean=False` for
      subsequent runs of other noise configurations to avoid recomputing
      the clean TCF and re-extracting TXT slices.
      
    """
    simlist = [Path(p).resolve() for p in simlist]

    t0_all = time.time()
    print(f" Starting TCF run for {len(simlist)} sims\n")

    for s_idx, sim_path in enumerate(simlist, start=1):
        sim_name = sim_path.stem
        print(f"\n========== [{s_idx}/{len(simlist)}] {sim_name} ==========")
        print(f"H5: {sim_path}")

        try:
            if "FID" in sim_name:
                print(" Detected FID sim → adding noise/smoothing variants")

                # generate noise variants
                outfiles = add_noise_and_smooth_all_realisations(
                    clean_h5_file=sim_path,
                    obs_time=obs_time,
                    total_int_time=total_int_time,
                    int_time=int_time,
                    declination=declination,
                    subarray_type=subarray_type,
                    verbose=True,
                    save_uvmap=save_uvmap,
                    njobs=njobs,
                    checkpoint=checkpoint,
                    bmax_km=bmax_km,
                )

                # collect all four sims (clean + 3 variants)
                sims_to_run = {}
                if include_clean:
                    sims_to_run["Clean"] = sim_path
                else:
                    print("  → Skipping Clean (include_clean=False)")

                sims_to_run.update({
                    "Noise-only": Path(outfiles["noise_only_h5"]),
                    "Noisy":      Path(outfiles["noisy_h5"]),
                    "Observed":   Path(outfiles["observed_h5"]),
                })

            else:
                # non-FID sims → just the original
                sims_to_run = {"Base": sim_path}

            # now run TCF for each chosen sim
            for tag, h5file in sims_to_run.items():
                print(f"\n  → Running TCF for {tag} version ({h5file})")

                try:
                    compute_TCF_of_single_SIM_all_realisations(
                        sim_filepath=h5file,
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
                    print(f"   ✗ FAILED on {tag} version: {e}")
                    if not continue_on_error:
                        raise
                    continue

        except Exception as e:
            print(f"✗ FAILED on {sim_name}: {e}")
            if not continue_on_error:
                raise
            continue

    print(f"\n✔ All sims done in {time.time()-t0_all:.1f}s")





# example usage

###############################################
########## RUNNING FOR AASTAR 100HRS ##########
###############################################
print(" %%%%%%%% RUNNING  FOR AASTAR 100HRS %%%%%%%% ")

# configure once
tcf_code_dir = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files"

# your list of H5 files
SIM_FID = '/data/cluster/lcrascal/SIM_data/SIM_FID/Lightcone_FID_400_Samples.h5'

SIM_HII_plus = '/data/cluster/lcrascal/SIM_data/SIM_HII_plus/Lightcone_HII_EFF_FACTOR_400_Samples_Plus.h5'
SIM_HII_minus = '/data/cluster/lcrascal/SIM_data/SIM_HII_minus/Lightcone_HII_EFF_FACTOR_400_Samples_Minus.h5'

SIM_Rmax_plus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_plus/Lightcone_R_BUBBLE_MAX_400_Samples_Plus.h5'  
SIM_Rmax_minus = '/data/cluster/lcrascal/SIM_data/SIM_Rmax_minus/Lightcone_R_BUBBLE_MAX_400_Samples_Minus.h5'

SIM_Tvir_plus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_plus/Lightcone_ION_Tvir_MIN_400_Samples_Plus.h5'
SIM_Tvir_minus = '/data/cluster/lcrascal/SIM_data/SIM_Tvir_minus/Lightcone_ION_Tvir_MIN_400_Samples_Minus.h5'  


# --- file list --- #
simlist = [
    SIM_FID,           
    SIM_HII_plus,
    SIM_HII_minus,
    SIM_Rmax_plus,
    SIM_Rmax_minus,
    SIM_Tvir_plus,
    SIM_Tvir_minus
    
]

# Call run_all with minimal args
run_all(
    simlist=simlist,
    tcf_code_dir=tcf_code_dir,   
    z_indices=0,
    nthreads=5,
    nbins=100,
    rmin=3,
    rmax=100,
    overwrite_h5=True,
    overwrite_txt=True,
    continue_on_error=True,
    obs_time=100, # XXXXXX
    total_int_time=6.0, 
    int_time=10.0, 
    declination=-30.0, 
    subarray_type="AAstar", # XXXXXX
    save_uvmap="/data/cluster/lcrascal/uvmaps/uvmap_AAstar_100hrs.h5",  # XXXXXX
    njobs=1, 
    checkpoint=8, 
    bmax_km=2.0,
    include_clean=True # XXXXXX
)


####################################################################################################################################################
################################  ###########################################################################################
####################################################################################################################################################
