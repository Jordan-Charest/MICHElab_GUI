import numpy as np

import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.parsing import parse_key_value_args
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, parameter_GUI, log

# TODO: Implement method for 1d data
# TODO: Implement frame selection method in addition to frame average

def normalize(filename, dataset_paths, low=0, high=1, verbosity=5):

    for dataset_path in dataset_paths:

        log(f"Normalizing dataset {dataset_path} between {low} and {high}.", level=1, verbosity=verbosity)
        
        data, attributes = get_data_from_dataset(filename, dataset_path)

        def normalize_array(arr, low, high):
            norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * (high-low)
            return norm_arr + low

        if data.ndim == 1:
            data = normalize_array(data, low, high)
        elif data.ndim == 3:
            for row in range(data.shape[1]):
                for col in range(data.shape[2]):
                    data[:,row,col] = normalize_array(data[:,row,col], low, high)


        save_data_to_dataset(filename, dataset_path, data, attributes=attributes)

    log(f"\nNormalization finished.", level=1, verbosity=verbosity)

def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        normalize(filename, datasets, low=args["low"], high=args["high"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:
            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            default_params = [("Minimal value", 0),
                            ("Maximal value", 1)]
            
            parameters = parameter_GUI(default_params, box_text="Normalization Parameters")

            normalize(hdf5_file, datasets,
                        low=parameters["Minimal value"],
                        high=parameters["Maximal value"])

if __name__ == "__main__":
    main()
    