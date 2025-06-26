import numpy as np
from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, parameter_GUI, log
import warnings
import sys
from utils.parsing import parse_key_value_args

warnings.filterwarnings('ignore')

# TODO: Implement method for 1d data
# TODO: Implement frame selection method in addition to frame average

def trim_time(filename, dataset_paths, start=0, stop=-1, verbosity=5):

    for dataset_path in dataset_paths:

        log(f"Trimming dataset {dataset_path}. Start at {start}, end at {stop}.", level=1, verbosity=verbosity)
        
        data, attributes = get_data_from_dataset(filename, dataset_path)

        if data.ndim == 3:
            trimmed_data = data[start:stop,:,:]
        elif data.ndim == 1:
            trimmed_data = data[start:stop]
        else:
            raise ValueError("The dataset for trimming must be 1d or 3d.")

        attributes["time_window"] = (start, stop)

        save_data_to_dataset(filename, dataset_path, trimmed_data, attributes=attributes)

    log(f"\nTime trimming finished.", level=1, verbosity=verbosity)

def main():
    if "--batch" in sys.argv:  # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')
        args = parse_key_value_args(sys.argv[4:])

        trim_time(filename, datasets, start=args["start"], stop=args["stop"])

    else:  # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:
            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            default_params = [("Start (index, incl.)", 0),
                              ("Stop (index, excl.)", -1)]
            
            parameters = parameter_GUI(default_params, box_text="Time trimming parameters")

            trim_time(hdf5_file, datasets,
                        start=parameters["Start (index, incl.)"],
                        stop=parameters["Stop (index, excl.)"])

if __name__ == "__main__":
    main()
    