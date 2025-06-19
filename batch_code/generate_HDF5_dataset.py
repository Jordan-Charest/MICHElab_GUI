import numpy as np
import tifffile
from utils.hdf5 import create_hdf5, add_data_to_hdf5, print_hdf5_structure
import os

"""From .tif and .npy files containing signals such as GCaMP, HbT, pupillometry and others, generates a HDF5 data file.

This file's use is very similar to the HDF5 file generation tab of the dataset, but it is faster if you need to generate multiple datasets quickly.
"""

###################### PARAMETERS ######################

### import / export parameters
output_file_id = "CAN_v2" # String identifier appended to the filename

### Dataset parameters
mouse_num = 397 # mouse number
fps_num = 12    # Default fps number; can be modified on a per-dataset basis below
dataset_filename = f"D:/mouse_data/formatted/M{mouse_num}/formatted/M{mouse_num}_{output_file_id}.h5"   # Filename of the output dataset
overwrite = True                                # Overwrite the dataset if it already exists under the same name. WARNING: be careful with this option!

### Other parameters
time_window = None # Format (start, end). Restricts the datasets to the specified period, in time step indices.

### Data import
import_root_path = f"D:/mouse_data/formatted/M{mouse_num}/" # Set root path for batch importation.

filepaths_to_import = [] # Initialize list for dataset filepath names

# The filepaths are added below. Modify as needed for your own import needs.

# Comment out this part to leave out dHbT, GCaMP and green avg IOI signals
filepaths_to_import += [os.path.join(import_root_path, f"dHbT/M{mouse_num}_dHbT.tif"),
                        os.path.join(import_root_path, f"GCaMP/M{mouse_num}_GCaMP.tif"),
                        os.path.join(import_root_path, f"avg/M{mouse_num}_avg.tif")]

# Comment out this part to leave out HbO / HbR signals
# filepaths_to_import += [os.path.join(import_root_path, f"dHbT/M{mouse_num}_dHbO.tif"),
#                        os.path.join(import_root_path, f"dHbT/M{mouse_num}_dHbR.tif")]

# Comment out this part to leave out pupillo / face motion signals
filepaths_to_import += [os.path.join(import_root_path, f"face/M{mouse_num}_face_motion_{fps_num}fps_lagged.npy"),
                        os.path.join(import_root_path, f"pupillo/M{mouse_num}_pupillo_{fps_num}fps_lagged.npy")]

n_data = len(filepaths_to_import) # TODO: Probably can be removed and replaced directly in the code

keys = ["HbT", "GCaMP", "avg", "face_motion", "pupillo"]    # Strings for the data to import, same order as in filepaths_to_import
dimensions = [3, 3, 2, 1, 1]  # Number of dimensions for the signals, same order as above
fps = [fps_num, fps_num, 0, fps_num, fps_num]  # fps for each signal, same order as above. 2d datasets have fps = 0 (such as avg, for instance)

debug = False       # Print stuff for debugging purposes

###################### FUNCTIONS ######################

# HDF5 handling functions are in utils.hdf5

def read_data(filename):
    """Read data in tiff or npy files
    """
        
    if filename[-4:] == ".tif":
        return tifffile.imread(filename)
    
    elif filename[-4:] == ".npy":
        return np.load(filename)
    
    raise ValueError("Could not recognize file extension.")

def restrict_time(array, start, stop):
    """Includes start and excludes stop"""

    dimension = array.ndim

    if dimension == 3:
        return array[start:stop,:,:]
    
    elif dimension == 2: # No time dimension
        return array
    
    elif dimension == 1:
        return array[start:stop]
    
    else:
        raise ValueError("Invalid dimension for restrict_time.")


###################### MAIN SCRIPT ######################

# Create dataset file if it doesn't exist
# WARNING: setting overwrite to True will delete the whole dataset. Use it with caution!
create_hdf5(dataset_filename, overwrite=overwrite)

# Add data and attributes to HDF5 file
for i in range(n_data):

    data = np.squeeze(read_data(filepaths_to_import[i]))

    if debug:
        print(keys[i])
        print(f"Data shape: {data.shape}")

    attributes = {"fps":fps[i], "filepath":filepaths_to_import[i]}

    if time_window is not None:
        data = restrict_time(data, time_window[0], time_window[1])
        attributes["time_window"] = time_window

    add_data_to_hdf5(dataset_filename, f"{keys[i]}", data, f"data/{dimensions[i]}d", attributes=attributes)

# Resulting file structure verification
print_hdf5_structure(dataset_filename)