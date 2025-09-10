
import os
import re
import subprocess
import pandas as pd


# # TCF Class for an Input Field with Unknown Paramaters

# In[26]:


class Compute_TCF():
    
    def __init__(self, tcf_code_dir, output_dir, nthreads=5, nbins=30, rmin=0, rmax=30):

        # files
        self.tcf_code_dir = tcf_code_dir  # directory of the files (SC.h etc) that will compute the TCF
        self.output_dir = output_dir      # directory to store output TCF data
        os.makedirs(self.output_dir, exist_ok=True) # makes output directory if it doesn't already exist

        # TCF parameters
        self.nthreads = nthreads
        self.nbins = nbins
        self.rmin = rmin
        self.rmax = rmax

    def Update_Header_File(self, input_field_filename_no_ext, L):
        """Updates SC.h with the correct parameters for a specific field."""
        
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
                         f'static const double L = {float(L)};', content)

        with open(header_path, 'w') as f:
            f.write(content)

    def compute_TCF_of_single_Field(self, field_path, L):
        """
        Compute the TCF of a single input field file.
        Returns a DataFrame with r, Re_s_r, Im_s_r, N_modes.
        """
        input_field_filename_no_ext = os.path.splitext(os.path.basename(field_path))[0]
        
        # Step 1: Copy the input field into the TCF code folder
        field_target_path = os.path.join(self.tcf_code_dir, os.path.basename(field_path))
        subprocess.run(["cp", field_path, field_target_path], check=True)

        # Step 2: Update SC.h
        self.Update_Header_File(input_field_filename_no_ext, L)
        print(f"Updating SC.h with filename: {input_field_filename_no_ext}")

        # Step 3: Compile and run from the TCF folder
        subprocess.run(["make"], check=True, cwd=self.tcf_code_dir)
        subprocess.run(["./SC_2d.o"], check=True, cwd=self.tcf_code_dir)

        # Step 4: Read the output
        output_filename = f"{input_field_filename_no_ext}_L{L}_spherical_correlations.txt"
        output_path = os.path.join(self.tcf_code_dir, output_filename)

        # Step 5: ove output to output_dir
        final_output_path = os.path.join(self.output_dir, output_filename)
        subprocess.run(["mv", output_path, final_output_path], check=True)

        # Read the file
        data_df = pd.read_csv(final_output_path, sep=r'\s+', header=None, skiprows=2, engine='python')
        data_df.columns = ["r", "Re_s_r", "Im_s_r", "N_modes"]


        # Step 5: Clean up the copied field file
        try:
            os.remove(field_target_path)
            print(f"Deleted temporary field file: {field_target_path}")
        except Exception as e:
            print(f"Warning: could not delete file {field_target_path}: {e}")


        return data_df




    


# In[27]:


# example run
# tcf_instance = Compute_TCF(
#     tcf_code_dir="./TCF_required_files_and_functions",
#     output_dir="./output_TCF_files"
# )

# field_path = "./bubble_field_run_0008_of_1500-Copy1.txt"
# L = 200

# df = tcf_instance.compute_TCF_of_single_Field(field_path, L)

