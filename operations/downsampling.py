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

def downsample(filename, dataset_paths, original_fps=12, target_fps=3, verbosity=5):

    for dataset_path in dataset_paths:

        log(f"Downsampling dataset {dataset_path} from {original_fps} to {target_fps} FPS.", level=1, verbosity=verbosity)
        
        data, attributes = get_data_from_dataset(filename, dataset_path)

        if target_fps > original_fps:
            raise ValueError("Target fps cannot be higher than original fps.")
        
        N = round(original_fps/target_fps)

        if data.ndim == 3:
            num_frames, height, width = data.shape
        elif data.ndim == 1:
            num_frames = data.shape[0]
        else:
            raise ValueError("The dataset for downsampling must be 1d or 3d.")

        num_binned_frames = num_frames // N  # Number of frames after binning

        if data.ndim == 3:
            # Reshape to (new_frames, N, height, width) and compute nanmean along axis 1 (binning axis)
            binned_data = np.nanmean(data[:num_binned_frames * N].reshape(num_binned_frames, N, height, width), axis=1)
        elif data.ndim == 1:
            binned_data = np.nanmean(data[:num_binned_frames * N].reshape(num_binned_frames, N), axis=1)

        attributes["original_fps"] = original_fps
        attributes["fps"] = target_fps

        save_data_to_dataset(filename, dataset_path, binned_data, attributes=attributes)

    log(f"\nDownsampling finished.", level=1, verbosity=verbosity)

def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        downsample(filename, datasets, original_fps=args["original_fps"], target_fps=args["target_fps"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:
            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            default_params = [("Original FPS", 12),
                            ("Target FPS", 3)]
            
            parameters = parameter_GUI(default_params, box_text="Downsampling Parameters")

            downsample(hdf5_file, datasets,
                        original_fps=parameters["Original FPS"],
                        target_fps=parameters["Target FPS"])

if __name__ == "__main__":
    main()
    