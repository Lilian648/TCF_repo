TCF Pipeline — README (txt)

A small, HPC-friendly pipeline to:
1) generate SKA-like noise/smoothed lightcones for FID simulations,
2) extract per-redshift 2D slices to .txt, and
3) compute the Triangle Correlation Function (TCF) for every realisation and selected z slices, saving results back into the source HDF5.

--------------------------------------------------------------------
Contents
--------------------------------------------------------------------
- Overview
- Input/Output data model
- Folder layout
- Install & requirements
- Functions & parameters
- Typical workflows
- Re-run / idempotence behaviour
- Performance tips (HPC)
- Troubleshooting
- Quick FAQ

--------------------------------------------------------------------
Overview
--------------------------------------------------------------------
The pipeline is built around three functions:

1. add_noise_and_smooth_all_realisations(...)
   - For FID sims only.
   - Writes three new H5 lightcones into a noise-config subfolder:
       <stem>_NOISE_ONLY_LC.h5
       <stem>_NOISE.h5
       <stem>_NOISE_SMOOTHING.h5 (subtract LOS mean, add noise, then smooth)
   - Subfolder name is the noise tag: "<subarray_type>_<obs_time>hrs", e.g. AAstar_100hrs.

2. extract_SIM_z_slices_to_txtfiles(h5_filepath, output_dir, z_indices)
   - Extracts 2D slices for each realisation at chosen redshift indices and saves them to txt per-z subfolders:
       <sim>_txtfiles/Lightcone_zidx<z>/realisation_<i>.txt

3. compute_TCF_of_single_SIM_all_realisations(...)
   - Runs your TCF code for all (or selected) z slices and all realisations.
   - Stores TCF back into the source H5 as:
       TCF_zidx<z>           -> shape (n_realisations, nbins)
       TCF_zidx<z>_rvals     -> shape (nbins,)

run_all(...) is a convenience orchestrator that:
- Detects FID sims (filename contains "FID"), generates noise variants, and runs TCF.
- Runs TCF directly for non-FID sims.
- Optional include_clean flag lets you skip recomputing “Clean” on subsequent noise passes.

--------------------------------------------------------------------
Input/Output data model
--------------------------------------------------------------------
Expected input HDF5 (lightcone)
Required datasets:
- /brightness_lightcone : (n_real, n_z, DIM, DIM) — axis order (n_real, z, x, y)
- /redshifts            : (n_z,) — MUST be strictly increasing
- /frequencies          : (n_z,)
- /box_length           : scalar (Mpc/h)
- /ngrid                : scalar (DIM)

Generated HDF5 (noise outputs)
Each output H5 contains:
- /brightness_lightcone : (n_real, n_z, DIM, DIM) (float32, gzip level 4)
- /redshifts, /frequencies, /box_length, /ngrid
- /nrealisations
- /redshifts_used       : (n_z,) (currently identical to input if z is increasing)

TCF datasets written back to source H5
For each requested z:
- TCF_zidx<z>           : (n_real, nbins) — TCF per realisation
- TCF_zidx<z>_rvals     : (nbins,) — the r grid used by TCF

--------------------------------------------------------------------
Folder layout
--------------------------------------------------------------------
For a FID lightcone at:
  .../SIM_FID_folder/Lightcone_FID.h5
Running with subarray_type="AAstar", obs_time=100 creates:
  .../SIM_FID_folder/
    AAstar_100hrs/
      Lightcone_MOCK_FID_NOISE_ONLY_LC.h5
      Lightcone_MOCK_FID_NOISE.h5
      Lightcone_MOCK_FID_NOISE_SMOOTHING.h5

TCF and TXT live next to the H5 you run on. For example, when you run TCF on:
  .../SIM_FID_folder/AAstar_100hrs/Lightcone_FID_NOISE.h5
the extracted slices and TCF will be placed under:
  .../SIM_FID_folder/AAstar_100hrs/Lightcone_FID_NOISE_txtfiles/
    Lightcone_zidx0/
      realisation_0.txt
      ...

For Clean runs, TCF/TXT live next to the clean H5 (no extra tag subfolder).

--------------------------------------------------------------------
Install & requirements
--------------------------------------------------------------------
Python packages:
- numpy
- h5py
- tools21cm (imported as t2c)
- Your local TCF_Class module (must provide Compute_TCF.compute_TCF_of_single_field(path)
  returning a DataFrame with columns ["r", "Re_s_r"])

Tip (HPC): use a virtualenv/conda env and load any cluster modules (HDF5, FFTW, etc.) required by your TCF code.

--------------------------------------------------------------------
Functions & parameters
--------------------------------------------------------------------
add_noise_and_smooth_all_realisations(clean_h5_file, *, obs_time, total_int_time, int_time,
                                      declination, subarray_type, verbose, save_uvmap, njobs,
                                      checkpoint, bmax_km)
- Purpose: Build three noise/smoothed variants for a FID sim and save them under <subarray>_<hours>.
- Key args:
  - obs_time (hours) -> used in noise tag and noise level.
  - subarray_type ∈ { "AAstar", "AA4", ... }
  - save_uvmap -> Path to write/read UV map. Make this unique per config (e.g., include noise tag) to avoid clashes.
  - bmax_km -> max baseline (km) for smoothing.
- Returns: dict of paths: {"noise_only_h5", "noisy_h5", "observed_h5"}

extract_SIM_z_slices_to_txtfiles(h5_filepath, output_dir, z_indices=None)
- Purpose: materialise (x, y) slices per realisation as .txt for selected z indices.
- Output: <output_dir>/Lightcone_zidx<z>/realisation_<i>.txt
- Selecting z indices
	- z_indices=None           -> extract all z indices
	- z_indices=[list of ints] -> extract the z indices in this list only
	- z_indices=int            -> extract only the single z index

compute_TCF_of_single_SIM_all_realisations(sim_filepath, ..., overwrite_h5=True,
                                           overwrite_txt=False, continue_on_error=False)
- Purpose: run TCF on all/some z slices and realisations; write datasets back into sim_filepath.
- Flags:
  - overwrite_txt=False -> reuse existing txt slices if present.
  - overwrite_h5=True   -> overwrite existing TCF_zidx<z> datasets (set to False to protect/reuse).
  - continue_on_error   -> if True, skip failed realisations rather than aborting the z-loop.

run_all(simlist, *, include_clean=True, ...)
- Purpose: iterate over input H5s. For FID sims: create noise variants and run TCF. For non-FID: just run TCF.
- Key flag:
  - include_clean=True on your first overall run (to compute Clean once).
  - Set include_clean=False for subsequent noise configurations to avoid recomputing Clean.

--------------------------------------------------------------------
Typical workflows
--------------------------------------------------------------------
1) First pass — compute Clean for everyone, and one noise config
  run_all(
    simlist=[... all sims incl. FID ...],
    subarray_type="AAstar",
    obs_time=100,                         # AAstar_100hrs
    save_uvmap="/data/.../uvmap_AAstar_100hrs.h5",
    include_clean=True,                   # compute Clean now
    overwrite_txt=False,                  # reuse slices on re-runs
    overwrite_h5=False,                   # skip existing TCF datasets on re-runs
  )

2) Second pass — another noise config, don’t touch Clean
  run_all(
    simlist=[... only FID sims ...],
    subarray_type="AAstar",
    obs_time=1000,                        # AAstar_1000hrs
    save_uvmap="/data/.../uvmap_AAstar_1000hrs.h5",
    include_clean=False,                  # skip Clean
    overwrite_txt=False,
    overwrite_h5=False,
  )

3) Third pass — e.g., AA4 1000h
  run_all(
    simlist=[... only FID sims ...],
    subarray_type="AA4",
    obs_time=1000,                        # AA4_1000hrs
    save_uvmap="/data/.../uvmap_AA4_1000hrs.h5",
    include_clean=False,
    overwrite_txt=False,
    overwrite_h5=False,
  )

--------------------------------------------------------------------
Re-run / idempotence behaviour
--------------------------------------------------------------------
- TXT extraction
  Controlled by overwrite_txt in compute_TCF_*. 
  True  -> always regenerate.
  False -> reuse existing per-z folders; no re-extraction.

- TCF datasets in H5
  Controlled by overwrite_h5.
  True  -> delete and recreate TCF_zidx<z> (and its _rvals).
  False -> currently raises if a dataset already exists.
           If you prefer skip instead of raise, change the else: block to:
             print(f"   Skipping z_idx {z}: '{ds_TCF}' exists (overwrite_h5=False).")
             continue

- Clean FID
  Only compute once (first run). On later noise runs, set include_clean=False.

--------------------------------------------------------------------
Performance tips (HPC)
--------------------------------------------------------------------
- HDF5 dataset creation uses gzip level 4; good compression–speed tradeoff.
  If write speed is slow and you have space, reduce compression or adjust chunking, e.g.:
    chunks=(1, n_z, DIM, DIM)   # efficient row-wise appends per realisation
- Avoid re-extracting TXT (overwrite_txt=False).
- Ensure redshifts are strictly increasing to avoid costly reprocessing or errors.
- Use distinct save_uvmap per noise config or include the noise tag in its filename.

--------------------------------------------------------------------
Troubleshooting
--------------------------------------------------------------------

- KeyError: 'brightness_lightcone' or missing metadata
  Your input H5 must contain: brightness_lightcone, redshifts, frequencies, box_length, ngrid.

- smooth_lightcone returned a different redshift grid
  Ensure redshifts is strictly increasing before running.

- save_uvmap path errors
  Pass a concrete path string; make it unique per config:
    /data/.../uvmap_<subarray>_<hours>.h5

- tools21cm argument mismatch (n_jobs vs njobs)
  The code calls n_jobs=.... Verify your installed tools21cm version expects that name.

- Re-runs re-extract TXT unexpectedly
  TXT path depends on the H5 path. If you move or symlink the H5, TXT will be placed next to the new path.
  Keep the Clean H5 at its original location to reuse TXT.

--------------------------------------------------------------------
Quick FAQ
--------------------------------------------------------------------
Q: Why write TCF back into the same H5?
A: Keeps all derived products co-located with the source, simplifying bookkeeping.

Q: Can I choose specific z slices?
A: Yes—pass z_indices as an int, list, or None (all).

Q: Where do TXT slices go?
A: "<sim_dir>/<sim_stem>_txtfiles/Lightcone_zidx<z>/", next to the H5 you run on.

Q: How do I avoid recomputing Clean for FID on later runs?
A: Set include_clean=False in run_all when you’re only adding new noise configurations.
