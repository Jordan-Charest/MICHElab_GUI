import numpy as np
import warnings
import sys
import os
import shutil

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.parsing import parse_key_value_args
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, parameter_GUI, log



def copy_file(src, dst):

    shutil.copyfile(src, dst)
    print(f"File {src} copied to {dst}")

def main():
    if "--batch" in sys.argv:  # Batch mode

        src_filename = sys.argv[2]
        dst_filename = sys.argv[3]

        copy_file(src_filename, dst_filename)

    else:  # GUI mode
        raise RuntimeError("Please run this script specifically in batch mode!")

if __name__ == "__main__":
    main()
    