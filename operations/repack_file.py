import numpy as np
import warnings
import sys
import os
import shutil
import subprocess

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.parsing import parse_key_value_args
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, parameter_GUI, log



def repack_hdf5(input_path, output_path):

        try:
            result = subprocess.run(
                ["h5repack", input_path, output_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Repack successful under filename {output_path}:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Repack failed:")
            print(e.stderr)

def main():
    if "--batch" in sys.argv:  # Batch mode

        src_filename = sys.argv[2]
        dst_filename = sys.argv[3]

        repack_hdf5(src_filename, dst_filename)

        os.remove(src_filename)

    else:  # GUI mode
        raise RuntimeError("Please run this script specifically in batch mode!")

if __name__ == "__main__":
    main()
    