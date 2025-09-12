
import os
import re
import subprocess
import pandas as pd


# # TCF Class for an Input Field with Unknown Paramaters

# In[26]:

class Compute_TCF():
    
    def __init__(self, tcf_code_dir, L, DIM, nthreads=5, nbins=100, rmin=0, rmax=30):

        # files
        self.tcf_code_dir = tcf_code_dir  # directory of the files (SC.h etc) that will compute the TCF

        # Data Parameters
        self.L = L # real space dimensions of the data field (eg 100Mpc)
        self.DIM = DIM # physical dimensions of the data field (eg 250 pixels)

        # TCF parameters
        self.nthreads = nthreads
        self.nbins = nbins
        self.rmin = rmin
        self.rmax = rmax

    def Update_Header_File(self, input_field_filename_no_ext, L):
        """
        Updates SC.h with the correct parameters for a specific field.
        """
        
        header_path = os.path.join(self.tcf_code_dir, "SC.h")
        
        with open(header_path, 'r') as f:
            content = f.read()

        content = re.sub(r'static const int nthreads = \d+;', 
                         f'static const int nthreads = {self.nthreads};', content)

        content = re.sub(r'static const string filename_box = ".*";', 
                         f'static const string filename_box = "{input_field_filename_no_ext}";', content)

        content = re.sub(r'static const int nbins = \d+;', 
                         f'static const int nbins = {self.nbins};', content)

        content = re.sub(r'static const double rmin = [\d\.]+;', 
                         f'static const double rmin = {float(self.rmin)};', content)

        content = re.sub(r'static const double rmax = [\d\.]+;', 
                         f'static const double rmax = {float(self.rmax)};', content)

        content = re.sub(r'static const double L = [\d\.]+;', 
                         f'static const double L = {float(self.L)};', content)

        content = re.sub(r'static const int N = [\d\.]+;', 
                         f'static const int N = {float(self.DIM)};', content)

        with open(header_path, 'w') as f:
            f.write(content)


    def compute_TCF_of_single_Field(self, field_path):
        """
        Compute the TCF of a single input field file.
        Returns a DataFrame with r, Re_s_r, Im_s_r, N_modes.
        """
        field_path = Path(field_path).resolve()
        field_root = field_path.with_suffix("")        # Path, no .txt
        field_root_str = str(field_root)               # str for writing into SC.h
    
        # Step 1: write txtfile name into SC.h
        self.Update_Header_File(field_root_str, self.L)
        print(f"Updating SC.h with absolute path: {field_root_str}")
    
        # Step 2: run TCF code
        subprocess.run(["make"], check=True, cwd=self.tcf_code_dir)
        subprocess.run(["./SC_2d.o"], check=True, cwd=self.tcf_code_dir)
    
        # Step 3: output lives next to input
        output_path = field_root.parent / f"{field_root.name}_L{int(self.L)}_spherical_correlations.txt"
        
        if not output_path.exists():
            raise FileNotFoundError(
                f"Expected TCF output at {output_path} but it was not created.\n"
                "Check that SC.h uses the absolute path WITHOUT '.txt' and that the code ran successfully."
            )

        # Step 5: load and read data
        df = pd.read_csv(output_path, sep=r"\s+", header=None, skiprows=2, engine="python")
        df.columns = ["r", "Re_s_r", "Im_s_r", "N_modes"]
        return df

    
    



# example run

# # --- file paths --- #
# tcf_code_dir = Path("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files")          # folder with Makefile, SC.h, SC_2d.o target
# output_dir   = Path("./tcf_test_outputs")
# sim_h5_field_path   = Path('/data/cluster/lcrascal/SIM_data/h5_files/Lightcone_MOCK.h5')             # SIM field path, h5 file
# sim_slice_txtfile_path = Path('/data/cluster/lcrascal/SIM_data/h5_files/mock_txtfiles/Lightcone_zidx0/realisation_0.txt')  # single slice to test on

# # --- get dimensions of field --- # Ideally this would go furhter up inthe code to avoid having to do it for each slice, since will be consistent over all simulations
# with h5py.File(sim_h5_field_path, "r") as f:
#     DIM = int(f["brightness_lightcone"].shape[-1])                          # number of pixels in SIM (DIMxDIM)
#     L = (float(f["box_length"][...].squeeze()) / 0.6774)          # Mpc, physical size of box (LxL)

# # ---  TCF params --- #
# nthreads = 5
# nbins    = 100
# rmin     = 0.0
# rmax     = 60.0

# # --- run --- #
# tcf = Compute_TCF(
#     tcf_code_dir=str(tcf_code_dir),
#     output_dir=str(output_dir),
#     L=L, DIM=DIM,
#     nthreads=nthreads, nbins=nbins, rmin=rmin, rmax=rmax
# )

# df = tcf.compute_TCF_of_single_Field(str(sim_slice_txtfile_path))

# # --- quick sanity checks ---
# print("TCF dataframe (head):")
# print(df.head())

# # confirm output file exists where we expect
# outname = f"{sim_slice_txtfile_path.stem}_L{int(L)}_spherical_correlations.txt"
# outpath = Path(sim_slice_txtfile_path).parent / outname
# print("\nOutput file:", outpath)
# print("Exists? ->", outpath.exists())

# # simple content checks
# assert list(df.columns) == ["r", "Re_s_r", "Im_s_r", "N_modes"], "Unexpected column names"
# assert len(df) > 0, "Empty TCF output"
# assert df["r"].is_monotonic_increasing, "r not monotonic increasing?"
# assert (df["N_modes"] >= 0).all(), "Negative N_modes?"

