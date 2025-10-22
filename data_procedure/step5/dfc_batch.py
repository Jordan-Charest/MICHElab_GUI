from toolbox_jocha.hdf5 import get_data_from_dataset, create_hdf5, save_dict_to_hdf5, add_attributes_to_dataset
from toolbox_jocha.connectivity import bin_3d_matrix
from toolbox_jocha.ets import format_array
from numba import njit
import numpy as np

mice_num = ["410-10","412-8","412-10","415-6","415-8"]

# mice_num = ["316-8","316-10","316-12","322-6","322-8","322-10","322-12","353-6","353-8",
#             "353-10","361-6","365-6","367-6","374-6","374-8","374-10","387-6","387-10",
#             "396-6","397-6","410-6","410-8","410-10","412-8","412-10","415-6","415-8"]

signals_str = ["GCaMP", "dHbT"]

file_id = "v1_mvmt"

x, y = 10, 10

def return_filenames(mouse_num, signal_str, filename_str):
    input_filename = f"D:/mouse_data/new_data/M{mouse_num}/formatted/M{mouse_num}_{filename_str}.h5"
    output_filename = f"D:/mouse_data/new_data/M{mouse_num}/formatted/M{mouse_num}_{filename_str}_{signal_str}_dfc.h5"
    return input_filename, output_filename


##################

# def return_filenames(mouse_num, signal_str, filename_str):
#     input_filename = f"D:/mouse_data/new_data/M396-1/formatted/M{mouse_num}.h5"
#     output_filename = f"D:/mouse_data/new_data/M396-1/formatted/M{mouse_num}_{signal_str}_dfc.h5"
#     return input_filename, output_filename

##################



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

for mouse_num in mice_num:
    for signal_str in signals_str:

        print(f"Processing mouse M{mouse_num}, {signal_str} signal.")

        input_filename, output_filename = return_filenames(mouse_num, signal_str, file_id)

        print("Loading data")

        data, attr = get_data_from_dataset(input_filename, f"data/3d/{signal_str}")

        print("Data loaded")

        binned_data = bin_3d_matrix(data, (y, x))

        flattened_data, elements_mask = format_array(binned_data, return_mask=True)

        dfc = compute_dfc_flat(flattened_data)

        cts = np.zeros(dfc.shape[0])
        for i in range(dfc.shape[0]):
            cts[i] = np.sqrt(np.nansum(np.square(dfc[i,:])))

        data_to_save = {"dfc": dfc, "cts": cts}
        attributes_to_save = {"binning_size": (x, y), "window_shape": binned_data.shape, "elements_mask": elements_mask}

        create_hdf5(filepath=output_filename, overwrite=True)

        save_dict_to_hdf5(output_filename, data_to_save)
        add_attributes_to_dataset(output_filename, "dfc", attributes_to_save)

        print(f"Data saved to {output_filename}.\n")

        del data_to_save, attributes_to_save, dfc, cts