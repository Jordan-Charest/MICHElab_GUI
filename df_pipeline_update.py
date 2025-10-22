import numpy as np
import pandas as pd
import scipy as sp
import os
from toolbox_jocha.hdf5 import add_data_to_hdf5, get_data_from_dataset, create_hdf5, save_dict_to_hdf5, get_attributes, add_attributes_to_dataset
from toolbox_jocha.connectivity import bin_3d_matrix
from toolbox_jocha.parsing import smart_cast
from toolbox_jocha.correlation import r_coeff_2mats
from toolbox_jocha.ets import split_into_bins, format_array
import shutil
# import bct
from numba import njit
import networkx as nx
from time import time
import sys

time_counter = time()
verbosity = 2

def log(message, level=1, print_flag=True, track_time=True):

    global time_counter, verbosity

    elapsed_time = time() - time_counter
    time_counter = time()

    if print_flag and verbosity >= level:
        if track_time:
            print(f"Elapsed time: {elapsed_time:.2f} seconds.\n")
        print(message)

# 0 - Define functions

def file_exists(filepath): # This function is unnecessary, os.path.exists can be used directly. I just always forget what the name of the function is.
    return os.path.exists(filepath)

def return_mouse_filename(mouse_num, id):
    # return f"../data/M{mouse_num}/M{mouse_num}_{id}.h5"
    return f"E:/Backup_SSD_Jordan/TO_CC/M{mouse_num}_{id}.h5"

def return_dfc_filename(mouse_num, id, signal_str):
    return f"E:/Backup_SSD_Jordan/TO_CC/M{mouse_num}_{id}_{signal_str}_dfc.h5"

def create_dataframe(mice_num, id, update_df=None):

    dataframe_dict = {"id": [], "number": []}

    for mouse_num in mice_num:

        attr = get_attributes(return_mouse_filename(mouse_num, id), "data")

        # index.append(f"M{mouse_num}_{output_file_id}")
        dataframe_dict["number"].append(mouse_num.split("-")[0])
        dataframe_dict["id"].append(f"M{mouse_num}_{id}")

        for key, value in attr.items():

            try:
                dataframe_dict[key].append(smart_cast(value))
            except KeyError:
                dataframe_dict[key] = []
                dataframe_dict[key].append(smart_cast(value))


    df = pd.DataFrame(dataframe_dict).set_index("id")

    if update_df is not None: # That means we're updating an existing dataframe instead of creating a new one
        df_to_update = return_dataframe(update_df)
        
        df_to_update.update(df)          # update existing rows
        df = pd.concat([df_to_update, df[~df.index.isin(df_to_update.index)]])  # add new rows

        

    return df

def return_dataframe(filename):
    return pd.read_csv(filename).set_index("id")

def copy_dataframe(input_file, output_file, update=None):
    if not os.path.exists(output_file):
        shutil.copyfile(input_file, output_file)

    else: # File already exists; try to update it instead

        if update is not None:
            mice_num, id = update
            df = create_dataframe(mice_num, id, update_df=output_file)
            df.to_csv(output_file)

def update_dataframe(df, data, columns, indexes):
    if data.ndim == 1:
        data = np.array([data]).T  # Ensure 2D

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    for i, idx in enumerate(indexes):
        if idx in df.index:
            # Update existing row using Series and df.update
            for col_idx, col in enumerate(columns):
                df.at[idx, col] = data[i, col_idx]
        else:
            # Create a new row dictionary
            new_row = {col: data[i, col_idx] for col_idx, col in enumerate(columns)}
            df.loc[idx] = new_row  # This adds a new row

    return df

def flat_to_symmetric(flat, N):
    """Convert a flattened upper triangle vector to a full symmetric matrix."""
    mat = np.zeros((N, N))
    inds = np.triu_indices(N)
    mat[inds] = flat
    mat[(inds[1], inds[0])] = flat  # Reflect upper triangle to lower
    return mat

def compute_modularity(fc, n_elems):

    N = int((np.sqrt(8*n_elems+1)-1)/2)
    sym_FC = np.array(flat_to_symmetric(fc, N))

    G = nx.from_numpy_array(sym_FC)
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    Q = nx.algorithms.community.modularity(G, communities)
    return Q

def shuffle_non_nan(arr):
    
    # Initialize generator
    rng = np.random.default_rng()

    # Mask where NaNs are
    nan_mask = np.isnan(arr)

    # Extract the non-NaN values
    values = arr[~nan_mask]

    # Shuffle them without replacement
    rng.shuffle(values)

    # Create the output array and put the shuffled values back
    sample = np.zeros_like(arr)
    sample[~nan_mask] = values
    sample[nan_mask] = np.nan

    return sample

@njit
def compute_dfc_flat(signals):
    T, N = signals.shape
    # print(f"{T}, {N}")
    triu_len = (N * (N + 1)) // 2  # Number of upper triangle elements (including diagonal)
    dFC_flat = np.zeros((T, triu_len))

    for t in range(T):
        idx = 0
        for i in range(N):
            for j in range(i, N):
                dFC_flat[t, idx] = signals[t, i] * signals[t, j]
                idx += 1

    return dFC_flat

# 1 - Set parameters

# Path parameters
def mouse_directory(mouse_num):
    return f"E:/Backup_SSD_Jordan/TO_CC"

dataframe_directory = "E:/Backup_SSD_Jordan/TO_CC/dataframes/"

# General params
mice_num = list(sys.argv[1].split(","))
# data_ids = ["mvmt", "no_mvmt", "trim_mvmt"] # Typically mvmt, no_mvmt, trim_mvmt
data_ids = ["TEST"]
file_id = "v2" # File identifier. Files are formatted as "M{mouse_num}_{file_id}_{data_id}.h5", for example "M308_v1_mvmt.h5".

# Null model params
# null_models = ["null_normal", "null_shuffle"] # Typically null_normal, null_shuffle
null_models = ["null_shuffle"]
apply_shuffle_to = "TEST"
normal_params = (0, 1) # loc, scale

# dFC params
compute_dfc_signals = ["GCaMP", "dHbT"]
x_binning = 10
y_binning = 10
overwrite = True

# Segmenting params
n_segments = 2
segmenting_str = "GCaMP"

# Neurovasc coupling params
squared_r = False # Whether to compute Pearson R2 (True) or R (False)
compute_lag = True # Whether to consider optimal lag (within a window) for neurovasc coupling
max_shift_seconds = 3 # Max shift to consider for lag window, in seconds
convert_to_s = True # Convert the obtained lag from indices to seconds
lag_sign = None # Force a sign for the lag values. Either None, "positive" or "negative" (which both include 0)

# Other
n_mice = len(mice_num)


# 2 - Generate null models

log("===== Step 1 - Null models =====\n")

for mouse_num in mice_num: # For each mouse

    log(f"Processing mouse M{mouse_num}")

    data, attrs = get_data_from_dataset(return_mouse_filename(mouse_num, f"{file_id}_{data_ids[0]}"), "data/3d/GCaMP")
    registered_data, registered_attrs = get_data_from_dataset(return_mouse_filename(mouse_num, f"{file_id}_{data_ids[0]}"), "registration/3d/GCaMP")

    for null_model in null_models: # For each null_model that has to be computed

        null_model_filename = os.path.join(mouse_directory(mouse_num), f"M{mouse_num}_{file_id}_{null_model}.h5") # Saved in the same place as the other data files for this mouse

        if file_exists(null_model_filename): # Check if the file already exists
            log(f"Null model '{null_model}' already exists. Skipping.\n")
            continue # If it does, skip to next iteration

        if null_model == "null_normal": # Generate a null model with a normal distribution

            # Whole data
            random_noise = np.random.normal(normal_params[0], normal_params[1], data.shape) # Generate random noise of the same shape as mouse data
            nan_mask = np.where(np.isnan(data[0,:,:]), np.nan, 1) # Compute mask of nan values in data
            masked_random_noise = random_noise * nan_mask # Apply nan mask to random noise
            random_data = sp.stats.zscore(masked_random_noise) # zscore resulting data (along axis 0, time, by default)

            # Registered data
            random_noise = np.random.normal(normal_params[0], normal_params[1], registered_data.shape) # Generate random noise of the same shape as mouse data
            random_registered_data = sp.stats.zscore(random_noise) # zscore resulting data (along axis 0, time, by default)

            # Copy original file
            original_filename = return_mouse_filename(mouse_num, f"{file_id}_{data_ids[0]}")
            destination_filename = return_mouse_filename(mouse_num, f"{file_id}_{null_model}")
            shutil.copyfile(original_filename, destination_filename)

            # Replace existing GCaMP data with null model
            # TODO: should dHbT be replaced as well?
            add_data_to_hdf5(destination_filename, "GCaMP", random_data, "data/3d", attributes={"null_model": "null_normal", "loc": normal_params[0], "scale": normal_params[1]}, overwrite=True)
            add_data_to_hdf5(destination_filename, "GCaMP", random_registered_data, "registration/3d", attributes={"null_model": "null_normal", "loc": normal_params[0], "scale": normal_params[1]}, overwrite=True)

        if null_model == "null_shuffle": # Generate a null model by shuffling existing data values (from the data associated with apply_shuffle_to)
            pass # Compute and save

            shuffled_data = sp.stats.zscore(shuffle_non_nan(data)) # Shuffle and zscore data
            shuffled_registered_data = sp.stats.zscore(shuffle_non_nan(registered_data))

            # Copy original file
            original_filename = return_mouse_filename(mouse_num, f"{file_id}_{data_ids[0]}")
            destination_filename = return_mouse_filename(mouse_num, f"{file_id}_{null_model}")
            shutil.copyfile(original_filename, destination_filename)

            # Replace existing GCaMP data with null model
            # TODO: should dHbT be replaced as well?
            add_data_to_hdf5(destination_filename, "GCaMP", shuffled_data, "data/3d", attributes={"null_model": "null_shuffle", "obtained_from": f"M{mouse_num}_{file_id}_{data_ids[0]}"}, overwrite=True)
            add_data_to_hdf5(destination_filename, "GCaMP", shuffled_registered_data, "registration/3d", attributes={"null_model": "null_shuffle", "obtained_from": f"M{mouse_num}_{file_id}_{data_ids[0]}"}, overwrite=True)

        log(f"Null model '{null_model}' succesfully saved.\n")

    try:
        del data, registered_data, nan_mask, random_noise, masked_random_noise, random_data, shuffled_data, shuffled_registered_data
    except:
        pass

try:
    del null_model_filename # TODO: complete with other variables
except:
    pass

data_ids = data_ids + null_models # We want to consider both

# 3 - dFC files

log("\n===== Step 2 - dFC files =====\n")

for mouse_num in mice_num: # For each mouse
    log(f"Processing mouse M{mouse_num}")

    for data_id in data_ids: # For each datafile we want to consider (INCLUDING null models)
        for signal_str in compute_dfc_signals: # For each signal we want to obtain the dFC of

            dfc_filename = return_dfc_filename(mouse_num, f"{file_id}_{data_id}", signal_str)

            if file_exists(dfc_filename):  # Check if the file already exists
                log(f"dFC file for {data_id} {signal_str} already exists. Skipping.\n")
                continue # If it does, skip to next iteration

            output_filename = dfc_filename
            create_hdf5(filepath=output_filename, overwrite=overwrite)

            for registration_str in ["whole", "registered"]:
                registration_path = {"whole": "data/3d", "registered": "registration/3d"}[registration_str]
                log(f"Processing {registration_str} ({registration_path})")

                # Compute dFC and save data

                input_filename = return_mouse_filename(mouse_num, f"{file_id}_{data_id}")

                data, attr = get_data_from_dataset(input_filename, f"{registration_path}/{signal_str}")

                if registration_str != "registered": # Bin data spatially
                    binned_data = bin_3d_matrix(data, (y_binning, x_binning))
                else: binned_data = data # Do not bin data

                flattened_data, elements_mask = format_array(binned_data, return_mask=True)

                dfc = compute_dfc_flat(flattened_data)

                cts = np.zeros(dfc.shape[0])
                for i in range(dfc.shape[0]):
                    cts[i] = np.sqrt(np.nansum(np.square(dfc[i,:])))

                if registration_str == "whole":
                    attributes_to_save = {"binning_size": (x_binning, y_binning), "window_shape": binned_data.shape, "elements_mask": elements_mask}
                elif registration_str == "registered":
                    attributes_to_save = attr

                add_data_to_hdf5(output_filename, "dfc", dfc, registration_str, attributes=attributes_to_save, overwrite=True)
                add_data_to_hdf5(output_filename, "cts", cts, registration_str, attributes=None, overwrite=True)

            log(f"dFC file for {data_id} {signal_str} succesfully saved.\n")

            del attributes_to_save, dfc, cts, data, attr, binned_data, flattened_data, elements_mask

del dfc_filename

# 4 - Compute segments
# Using the computed dFC files, compute the segments. mats and indices. There will be one (two with registration) of each for each mouse and data_id combination
segment_mats = {"whole": [[] for i in data_ids], "registered": [[] for i in data_ids]}
segment_indices = {"whole": [[] for i in data_ids], "registered": [[] for i in data_ids]}

log("\n===== Step 3 - Segments =====\n")

for i, data_id in enumerate(data_ids):
    log(f"\nProcessing id {data_id}")

    for mouse_num in mice_num:
        log(f"Processing mouse M{mouse_num}")

        for registration_str in ["whole", "registered"]:
            registration_path = {"whole": "data/3d", "registered": "registration/3d"}[registration_str]
            log(f"Processing {registration_str} ({registration_path})")
        
            cts, _ = get_data_from_dataset(return_dfc_filename(mouse_num, f"{file_id}_{data_id}", segmenting_str), f"{registration_str}/cts")

            mats, indices = split_into_bins(cts, n_segments)

            segment_mats[registration_str][i].append(mats)
            segment_indices[registration_str][i].append(indices)


# 5 - Create and fill dataframes

log("\n===== Step 4 - Dataframes =====\n")

for data_id_index, data_id in enumerate(data_ids): # For each model we want to consider (INCLUDING null models)

    log(f"\nProcessing id {data_id}")

    base_df = f"{file_id}_{data_id}_df.csv"
    base_df_filename = os.path.join(dataframe_directory, base_df)

    if not file_exists(base_df_filename): # If the base dataframe doesn't exist, create it
        df = create_dataframe(mice_num=mice_num, id=f"{file_id}_{data_id}")
        df.to_csv(base_df_filename)
    else: # If it already exists, update it with mouse data
        df = create_dataframe(mice_num=mice_num, id=f"{file_id}_{data_id}", update_df=base_df_filename)
        df.to_csv(base_df_filename)

    df = return_dataframe(base_df_filename)

    # Copy dataframes if they do not exist already
    for name in ["nvc", "modularity", "funcsim", "funcrep"]:
        for registration_str in ["whole", "registered"]:
            copy_dataframe(base_df_filename, os.path.join(dataframe_directory, f"{name}_{registration_str}_"+base_df), update=(mice_num, f"{file_id}_{data_id}"))

    indexes = [f"M{mouse_num}_{file_id}_{data_id}" for mouse_num in mice_num]

    for registration_str in ["whole", "registered"]:
        registration_path = {"whole": "data/3d", "registered": "registration/3d"}[registration_str]

        log(f"Processing {registration_str}")

        ### NEUROVASCULAR COUPLING (nvc)

        log(f"Processing neurovascular coupling and lag.")

        df = return_dataframe(os.path.join(dataframe_directory, f"nvc_{registration_str}_"+base_df)) # Load appropriate dataframe

        neurovascular_coupling = np.zeros((n_mice, n_segments))
        neurcoup_whole = np.zeros(n_mice)
        lag = np.zeros((n_mice, n_segments))
        lag_whole = np.zeros(n_mice)

        for i, mouse_num in enumerate(mice_num):

            GCaMP_signal, _ = get_data_from_dataset(return_mouse_filename(mouse_num, f"{file_id}_{data_id}"), f"{registration_path}/GCaMP")
            HbT_signal, _ = get_data_from_dataset(return_mouse_filename(mouse_num, f"{file_id}_{data_id}"), f"{registration_path}/dHbT")

            fps = _["fps"]
            max_shift = max_shift_seconds * fps

            lag_mat, correlation_mat = r_coeff_2mats(GCaMP_signal, HbT_signal, max_shift=max_shift, lag=compute_lag, convert_to_s=convert_to_s, fps=fps, squared=squared_r, lag_sign=lag_sign)
            neurcoup_whole[i] = np.nanmean(correlation_mat)
            lag_whole[i] = np.nanmean(lag_mat)

            for j, indices in enumerate(segment_indices[registration_str][data_id_index][i]): # The j-th segment of mouse i

                sliced_GCaMP_signal = GCaMP_signal[indices,:,:]
                sliced_HbT_signal = HbT_signal[indices,:,:]

                lag_mat, correlation_mat = r_coeff_2mats(sliced_GCaMP_signal, sliced_HbT_signal, max_shift=max_shift, lag=compute_lag, convert_to_s=convert_to_s, fps=fps, squared=squared_r, lag_sign=lag_sign)

                neurovascular_coupling[i,j] = np.nanmean(correlation_mat)
                lag[i,j] = np.nanmean(lag_mat)

        nvc_columns = ["nvc_whole"] + [f"nvc_segment_{i}/{n_segments}" for i in range(n_segments)]
        df = update_dataframe(df, np.column_stack((neurcoup_whole, neurovascular_coupling)), nvc_columns, indexes)

        lag_columns = ["lag_whole"] + [f"lag_segment_{i}/{n_segments}" for i in range(n_segments)]
        df = update_dataframe(df, np.column_stack((lag_whole, lag)), lag_columns, indexes)

        df.to_csv(os.path.join(dataframe_directory, f"nvc_{registration_str}_"+base_df)) # save modified dataframe

        del neurovascular_coupling, neurcoup_whole, lag, lag_whole, GCaMP_signal, HbT_signal, lag_mat, correlation_mat, df, sliced_HbT_signal, sliced_GCaMP_signal

        ### FUNCTIONAL SIMILARITY (funcsim), FUNCTIONAL REPRESENTATIVITY (funcrep) AND MODULARITY (modularity)

        log(f"Processing functional similarity, functional representativity and modularity.")

        GCaMP_FC_whole = [np.nan for i in mice_num] # A
        dHbT_FC_whole = [np.nan for i in mice_num] # B

        funcsim_whole = np.zeros(n_mice) # C
        functional_similarity = np.zeros((n_mice, n_segments)) # D

        GCaMP_functional_representativity = np.zeros((n_mice, n_segments)) # E
        dHbT_functional_representativity = np.zeros((n_mice, n_segments)) # F

        GCaMP_mod_whole = np.zeros(n_mice) # G
        dHbT_mod_whole = np.zeros(n_mice) # H
        GCaMP_modularity = np.zeros((n_mice, n_segments)) # I
        dHbT_modularity = np.zeros((n_mice, n_segments)) # J

        for i, mouse_num in enumerate(mice_num):

            GCaMP_dfc, _ = get_data_from_dataset(return_dfc_filename(mouse_num, f"{file_id}_{data_id}", "GCaMP"), f"{registration_str}/dfc")
            dHbT_dfc, _ = get_data_from_dataset(return_dfc_filename(mouse_num, f"{file_id}_{data_id}", "dHbT"), f"{registration_str}/dfc")

            GCaMP_FC_whole[i] = np.mean(GCaMP_dfc, axis=0) # A
            dHbT_FC_whole[i] = np.mean(dHbT_dfc, axis=0) # B

            print(f"GCaMP_dfc shape: {GCaMP_dfc.shape}")
            print(f"dHbT_dfc shape: {dHbT_dfc.shape}")
            print(f"GCaMP_FC_whole[{i}] shape: {GCaMP_FC_whole[i].shape}")
            print(f"dHbT_FC_whole[{i}] shape: {dHbT_FC_whole[i].shape}")

            funcsim_whole[i] = np.corrcoef(GCaMP_FC_whole[i], dHbT_FC_whole[i])[0,1] # C


            GCaMP_mod_whole[i] = compute_modularity(GCaMP_FC_whole[i], GCaMP_dfc.shape[1]) # G
            dHbT_mod_whole[i] = compute_modularity(dHbT_FC_whole[i], dHbT_dfc.shape[1]) # H

            for j, indices in enumerate(segment_indices[registration_str][data_id_index][i]): # The j-th segment of mouse i

                sliced_GCaMP_dfc = GCaMP_dfc[indices,:]
                sliced_dHbT_dfc = dHbT_dfc[indices,:]

                GCaMP_FC = np.mean(sliced_GCaMP_dfc, axis=0)
                dHbT_FC = np.mean(sliced_dHbT_dfc, axis=0)

                functional_similarity[i,j] = np.corrcoef(GCaMP_FC, dHbT_FC)[0,1] # D

                GCaMP_functional_representativity[i,j] = np.corrcoef(GCaMP_FC, GCaMP_FC_whole[i])[0,1] # E
                dHbT_functional_representativity[i,j] = np.corrcoef(dHbT_FC, dHbT_FC_whole[i])[0,1] # F

                GCaMP_modularity[i,j] = compute_modularity(GCaMP_FC, GCaMP_dfc.shape[1]) # I
                dHbT_modularity[i,j] = compute_modularity(dHbT_FC, dHbT_dfc.shape[1]) # J

                del sliced_GCaMP_dfc, sliced_dHbT_dfc

            del GCaMP_dfc, dHbT_dfc

        funcrep_df = return_dataframe(os.path.join(dataframe_directory, f"funcrep_{registration_str}_"+base_df))
        modularity_df = return_dataframe(os.path.join(dataframe_directory, f"modularity_{registration_str}_"+base_df))
        funcsim_df = return_dataframe(os.path.join(dataframe_directory, f"funcsim_{registration_str}_"+base_df))

        # We're not saving A and B

        funcsim_columns = ["funcsim_whole"] + [f"funcsim_segment_{i}/{n_segments}" for i in range(n_segments)]
        funcsim_df = update_dataframe(funcsim_df, np.column_stack((funcsim_whole, functional_similarity)), funcsim_columns, indexes) # C and D
        funcsim_df.to_csv(os.path.join(dataframe_directory, f"funcsim_{registration_str}_"+base_df))

        funcrep_columns = [f"GCaMP_funcrep_segment_{i}/{n_segments}" for i in range(n_segments)] + [f"dHbT_funcrep_segment_{i}/{n_segments}" for i in range(n_segments)]
        funcrep_df = update_dataframe(funcrep_df, np.column_stack((GCaMP_functional_representativity, dHbT_functional_representativity)), funcrep_columns, indexes) # E and F
        funcrep_df.to_csv(os.path.join(dataframe_directory, f"funcrep_{registration_str}_"+base_df))

        modularity_columns = ["GCaMP_modularity_whole"] + [f"GCaMP_modularity_segment_{i}/{n_segments}" for i in range(n_segments)] + ["dHbT_modularity_whole"] + [f"dHbT_modularity_segment_{i}/{n_segments}" for i in range(n_segments)]
        modularity_df = update_dataframe(modularity_df, np.column_stack((GCaMP_mod_whole, GCaMP_modularity, dHbT_mod_whole, dHbT_modularity)), modularity_columns, indexes) # G, H, I and J
        modularity_df.to_csv(os.path.join(dataframe_directory, f"modularity_{registration_str}_"+base_df))

        del GCaMP_functional_representativity, dHbT_functional_representativity, GCaMP_modularity, dHbT_modularity, functional_similarity, GCaMP_FC_whole, dHbT_FC_whole, GCaMP_mod_whole, dHbT_mod_whole
 
log(f"Done!")