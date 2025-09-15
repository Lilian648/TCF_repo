compute_TCF_of_SIM_data Pipeline Overview

Inputs:

- sim_filepath → HDF5 simulation file with lightcone data (brightness_lightcone).

- txtfiles_folder_path → directory where intermediate .txt slice files are saved.

- z_indices → one, many, or all redshift/frequency slices to process.

- TCF code parameters (tcf_code_dir, nbins, rmin, rmax, …).

Step 1. Validate simulation

- Opens the HDF5 file.

- Checks that the lightcone is 4D and ordered (n_realisations, n_freq, Ny, Nx).

- Confirms Ny == Nx and matches frequency/redshift datasets.

- Extracts grid size (DIM) and physical box length (L).

Step 2. Extract slices to .txt

- For each requested z_idx, creates a subfolder Lightcone_zidx{z}.

- Saves one .txt file per realisation → realisation_0.txt, realisation_1.txt, …

Step 3. Compute TCF for each slice

- Reuses a Compute_TCF instance to run the compiled C++ code.

- Ignores any previously generated correlation output files.

- Runs TCF on each realisation_*.txt, saving results in memory.

Step 4. Save results back into the HDF5

- Writes one dataset per slice:

- TCF_zidx{z} → array of shape (n_realisations, nbins) with Re[TCF].

- TCF_zidx{z}_rvals → corresponding r-bin values.

- Optionally overwrites existing datasets if overwrite=True.

- (Optional commented block: also saves mean TCF and 1σ error bars).

Output:

- Updated HDF5 file containing raw TCF results for all requested redshift slices.

- Intermediate .txt files (per slice per realisation) remain on disk for reproducibility.

