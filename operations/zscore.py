import os
from scipy.stats import zscore
import sys

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, log

# warnings.filterwarnings('ignore')

def zscoring(filename, dataset_paths, verbosity=5):

    for dataset_path in dataset_paths:

        log(f"z-scoring dataset {dataset_path}.", level=1, verbosity=verbosity)
        
        data, attributes = get_data_from_dataset(filename, dataset_path)

        if data.ndim not in [1, 3]:
            raise ValueError("Cannot zscore datasets that are not in 1 or 3 dimensions.")
        
        if data.ndim == 1:
            data = zscore(data)
        elif data.ndim == 3:
            for row in range(data.shape[1]):
                for col in range(data.shape[2]):
                    data[:,row,col] = zscore(data[:,row,col])

        attributes["zscored"] = True

        save_data_to_dataset(filename, dataset_path, data, attributes=attributes)

    log(f"\nZ-scoring finished.", level=1, verbosity=verbosity)

def main():
    if "--batch" in sys.argv: # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')

        zscoring(filename, datasets)

    else: # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:
            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            zscoring(hdf5_file, datasets)
        
if __name__ == "__main__":
    main()