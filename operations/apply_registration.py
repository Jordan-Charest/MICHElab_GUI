import os
from scipy.stats import zscore
import sys
import numpy as np

# Necessary for running as a subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hdf5 import get_data_from_dataset, save_data_to_dataset, log
from utils.parsing import parse_key_value_args

# warnings.filterwarnings('ignore')

def remove_region_zero(regions_list):

    regions_list = [str(int(i)) for i in regions_list]

    while regions_list[0] == "0":
        del regions_list[0]

    return regions_list

def restrict_visible_regions(regions_labels, visible_regions):

    # Formatting input data
    regions_labels = np.asarray(regions_labels)
    visible_regions = [str(int(i)) for i in visible_regions] # Got to be this way because HDF5 stores regions as strings

    mask = np.isin(np.asarray(regions_labels)[:,0], visible_regions)

    return np.array(regions_labels[mask,:]).tolist() # Gotta return it in list form so it is handled correctly by h5py when saving as an attribute

def signals_to_regions(data, atlas, labels_list):
    """
    Takes an array of 3D data and averages every (non-nan) point that belong to the same region, 
    according to the provided atlas. Returns a one dimensional array per frame (dim 0 = time, dim 1 = regions)."""

    regions_labels = labels_list.tolist()

    flattened_atlas = atlas.flatten()[~np.isnan(atlas.flatten())]
    visible_regions = sorted(list(set((flattened_atlas))))
    visible_regions = remove_region_zero(visible_regions) # Remove region 0, which doesn't correspond to any region

    final_labels = restrict_visible_regions(regions_labels, visible_regions) # Only keeps the regions that are visible
    # TODO: the line above is a bit clunky. Ideally the label/region pairs would be stored as a dictionary, but HDF5 cannot handle storing dictionaries

    if data.ndim == 3:
        registered_data = np.zeros((data.shape[0], 1, len(visible_regions))) # As many data points as there are regions in the atlas
        # Note: it is important to keep the array 3-dimensional even if the second axis only has size 1, because it ensures the registered data is compatible with all other operations working with 3d arrays

        for t in range(data.shape[0]): # For each time step

            data_frame = data[t,:,:]

            for i, region_label in enumerate(visible_regions): # For each region index

                region_label = int(region_label)
                registered_data[t, 0, i] = np.nanmean(data_frame[atlas == region_label])

    elif data.ndim == 2:
        registered_data = np.zeros(len(visible_regions))
        for i, region_label in enumerate(visible_regions): # For each region index
        
                region_label = int(region_label)
                registered_data[i] = np.nanmean(data[atlas == region_label])

    # Remove nan points
    keep_values = ~np.isnan(registered_data[0,0,:])
    keep_indices = np.where(keep_values)[0]
    registered_data = registered_data[:,:,keep_indices]
    final_labels = np.asarray(final_labels)[keep_indices,:].tolist()

    return registered_data, final_labels

def apply_registration(filename, dataset_paths, verbosity=5):

    # Retrieve atlas from specified file
    atlas, attr = get_data_from_dataset(filename, "registration/atlas") # TODO: I don't really like that the path to the dataset is hard-coded. See if we can change that
    original_region_labels = attr["region_labels"]

    for dataset_path in dataset_paths:

        log(f"Applying registration to dataset {dataset_path}.", level=1, verbosity=verbosity)

        registered_data_path = "registration/" + dataset_path.split('/', 1)[1] # Likewise, don't really like that it's hard-coded like that.

        data, attrs = get_data_from_dataset(filename, dataset_path) # TODO: transfer attributes to new data or not?

        if data.ndim not in [2, 3]:
            raise ValueError("Cannot apply registration to data that is not in 2 or 3 dimensions.")
        
        registered_data, labels = signals_to_regions(data, atlas, original_region_labels)

        attrs["original_data"] = dataset_path
        attrs["region_labels"] = labels

        save_data_to_dataset(filename, registered_data_path, registered_data, attributes=attrs)

    log(f"Finished applying registration.", level=1, verbosity=verbosity)

def main():
    if "--batch" in sys.argv: # Batch mode

        filename = sys.argv[2]
        datasets = sys.argv[3].split(',')

        apply_registration(filename, datasets)

    else: # GUI mode
        if len(sys.argv) < 3:
            log("Usage: python script.py <HDF5 file> <dataset1> [<dataset2> ...]", verbosity=5)
        else:

            hdf5_file = str(sys.argv[1])
            datasets = [str(x) for x in sys.argv[2:]]

            apply_registration(hdf5_file, datasets)
        
if __name__ == "__main__":
    main()