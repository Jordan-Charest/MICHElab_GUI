import copy
import scipy as sp
import tifffile
from toolbox_jocha.hdf5 import *
from toolbox_jocha.mouse_data import *
import os

from funcs import return_raw_data_root, return_datafile_contents, return_avg_data, return_filename, read_data

"""From .tif and .npy files containing signals such as GCaMP, HbT, pupillometry and others, generates a HDF5 data file.
"""

###################### PARAMETERS ######################
# Set the following parameters before launching script

mice_num = ["39-12", "42-12", "44-12", "45-12", "46-12", "251-6", "254-6"]
output_file_id = "RAW"
fps_num = 3
time_window = None # Restrict time window
overwrite = True    # Overwrite the dataset if it already exists under the same name
# WARNING: be careful with this option!

########## SET IMPORT DATA PARAMETERS ##########

keys = ["dHbT", "GCaMP", "green_avg", "dHbO", "dHbR", "face_motion", "face_motion_lagged"]    # Strings for the data to import, same order as in filepaths_to_import
dimensions = [3, 3, 2, 3, 3, 1, 1]  # Number of dimensions for the signals, same order as above
fps = [fps_num, fps_num, 0, fps_num, fps_num, fps_num, fps_num]    # fps for each signal, same order as above
start_index_1d = 0  # Starting index for 1d dataset; set to None for automatic gradient-based start.

debug = False       # Print stuff for debugging purposes


### Start main loop ###

for mouse_num in mice_num:

    print(f"Processing mouse M{mouse_num}.")

    ### Set data to import and attributes
    import_root_path = return_raw_data_root(mouse_num)

    ROI_filepaths = [os.path.join(import_root_path, f"dHbT.tif"),
                    os.path.join(import_root_path, f"GCaMP.tif"),
                    os.path.join(import_root_path, "green_avg.npy")]

    filepaths_to_import = ROI_filepaths

    # Comment out this part to leave out HbO / HbR signals
    filepaths_to_import += [os.path.join(import_root_path, f"dHbO.tif"),
                        os.path.join(import_root_path, f"dHbR.tif")]

    # Comment out this part to leave out pupillo / face motion signals
    filepaths_to_import += [os.path.join(import_root_path, f"../pupillo_face/M{mouse_num}_face_motion_sliced.npy"),
                            os.path.join(import_root_path, f"../pupillo_face/M{mouse_num}_face_motion_sliced_lagged.npy")]

    # os.path.join(import_root_path, f"pupillo/M{mouse_num}_pupillo_{fps_num}fps_lagged.npy")


    ###################### MAIN SCRIPT ######################

    dataset_filename = return_filename(mouse_num, output_file_id)

    # Create dataset file if it doesn't exist
    # WARNING: setting overwrite to True will delete the whole dataset. Use it with caution!
    create_hdf5(dataset_filename, overwrite=overwrite)

    # Add data and attributes to HDF5 file
    for i in range(len(filepaths_to_import)):

        data = np.squeeze(read_data(filepaths_to_import[i]))

        if debug:
            print(keys[i])
            print(f"Data shape: {data.shape}")

        attributes = {"fps":fps[i], "filepath":filepaths_to_import[i]}

        if time_window is not None:
            data = restrict_time(data, time_window[0], time_window[1])
            attributes["time_window"] = time_window

        add_data_to_hdf5(dataset_filename, f"{keys[i]}", data, f"data/{dimensions[i]}d", attributes=attributes)

    # Save attributes to data group
    attr = return_datafile_contents(mouse_num)
    save_dict_as_h5_attributes(dataset_filename, "data", attr)

    # Resulting file structure verification
    print_hdf5_structure(dataset_filename)