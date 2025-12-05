import copy
import scipy as sp
import tifffile
from toolbox_jocha.hdf5 import *
from toolbox_jocha.mouse_data import *
import os
import pickle
import sys

from funcs import return_raw_data_root, return_datafile_contents, return_avg_data, return_filename, read_data, read_regions_file, get_transformed_crop, lag_signal

"""From .tif and .npy files containing signals such as GCaMP, HbT, pupillometry and others, generates a HDF5 data file.
"""

###################### PARAMETERS ######################
# Set the following parameters before launching script

mice_num = sys.argv[1].split(",")
output_file_id = "RAW"
fps_num = 3
time_window = None # Restrict time window
overwrite = True    # Overwrite the dataset if it already exists under the same name
# WARNING: be careful with this option!

########## ATLAS PARAMETERS ##########

include_atlas = True # When set to True, will try to load the associated atlas under "registration/atlas"
atlas_path = f"D:/mouse_data/new_data/atlas/outline_mask_reduced.npy"
labels_path = f"D:/mouse_data/new_data/atlas/outline_regions_reduced.txt"
params_filename = "atlas_params.pkl"

########## SET IMPORT DATA PARAMETERS ##########

keys = ["dHbT", "GCaMP", "green_avg", "dHbO", "dHbR", "face_motion", "pupillo"]    # Strings for the data to import, same order as in filepaths_to_import
bin_atlas = 2 # Size with which to bin the atlas (often the raw data on which it is computed is 2 times larger than the data used for analysis; change as needed). None if no binning.
dimensions = [3, 3, 2, 3, 3, 1, 1]  # Number of dimensions for the signals, same order as above
fps = [fps_num, fps_num, 0, fps_num, fps_num, fps_num, fps_num]    # fps for each signal, same order as above
start_index_1d = 0  # Starting index for 1d dataset; set to None for automatic gradient-based start.


lag_signals = ["face_motion", "pupillo"] # Also add those signals as a lagged version
lag_wrt = "GCaMP" # Reference signal for the lagging
lag_range = (-6, 6) # Maximum lag, in frames
abs_r_lag = True # Whether or not to consider the absolute correlation when computing lag

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
                            os.path.join(import_root_path, f"../pupillo_face/RS_M{mouse_num.split('-')[0]}_video_trimmed_proc.npy")]

    # os.path.join(import_root_path, f"pupillo/M{mouse_num}_pupillo_{fps_num}fps_lagged.npy")


    ###################### MAIN SCRIPT ######################

    dataset_filename = return_filename(mouse_num, output_file_id)

    # Create dataset file if it doesn't exist
    # WARNING: setting overwrite to True will delete the whole dataset. Use it with caution!
    create_hdf5(dataset_filename, overwrite=overwrite)

    # Add data and attributes to HDF5 file
    for i in range(len(filepaths_to_import)):  

        data = np.squeeze(read_data(filepaths_to_import[i]))

        if keys[i] == "dHbT":
            data_shape = data.shape # Store this for the atlas later

        if debug:
            print(keys[i])
            print(f"Data shape: {data.shape}")

        attributes = {"fps":fps[i], "filepath":filepaths_to_import[i]}

        if time_window is not None:
            data = restrict_time(data, time_window[0], time_window[1])
            attributes["time_window"] = time_window

        add_data_to_hdf5(dataset_filename, f"{keys[i]}", data, f"data/{dimensions[i]}d", attributes=attributes)

        if keys[i] in lag_signals: # Also add the lagged version of this signal
            reference_signal, _ = get_data_from_dataset(dataset_filename, f"data/3d/{lag_wrt}") # Reference signal for the lag
            reference_signal = np.nanmean(reference_signal, axis=(1,2))

            lagged_signal, frames = lag_signal(data, reference_signal, lag_range=lag_range, abs_r=abs_r_lag)

            attributes.update
            add_data_to_hdf5(dataset_filename, f"{keys[i]}_lagged", lagged_signal, f"data/{dimensions[i]}d", attributes=attributes | {"lagged_wrt": lag_wrt, "frames_lagged": frames})

    if include_atlas:
        try:

            params_path = os.path.join(import_root_path, f"../{params_filename}")
            params = read_data(params_path)

            raw_atlas = read_data(atlas_path)

            transformed_atlas = get_transformed_crop(raw_atlas, params)
            atlas = np.asarray(transformed_atlas, np.float16)

            if bin_atlas is not None:
                height = atlas.shape[0]
                width = atlas.shape[1]

                atlas = atlas[::bin_atlas,::bin_atlas]

                print(f"Atlas shape: {atlas.shape}")
                print(f"Data shape: {data_shape}")

                if atlas.shape != (data_shape[1], data_shape[2]):
                    atlas = atlas[:data_shape[1], :data_shape[2]]

            add_data_to_hdf5(dataset_filename, "atlas", atlas, f"registration", attributes={"region_labels": read_regions_file(labels_path)})

        except Exception as e:
            print(e)
            raise FileNotFoundError("Error while trying to load atlas. Make sure the atlas has already been computed.")

    # Save attributes to data group
    attr = return_datafile_contents(mouse_num)
    save_dict_as_h5_attributes(dataset_filename, "data", attr)

    # Resulting file structure verification
    print_hdf5_structure(dataset_filename)