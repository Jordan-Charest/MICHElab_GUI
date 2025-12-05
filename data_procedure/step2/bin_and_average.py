import os
import numpy as np
import tifffile

from funcs import bin_3d_matrix, bin_2d_matrix, read_data, return_data_dir

# mice_num = ["233-12", "308-6", "308-8", "308-10", "308-12", "308-14", "316-6", "316-8", "316-10", "316-12",
#             "322-6", "322-8", "322-10", "322-12", "353-6", "353-8", "353-10", "361-6", "365-6", "367-6",
#             "374-6", "374-8", "374-10", "387-6", "387-10", "396-6", "397-6", "410-6", "410-8", "410-10",
#             "412-8", "412-10", "415-6", "415-8"]

mice_num = ["308-16", "316-14", "316-16", "322-14", "353-12", "387-12", "410-12", "410-14", "412-12"]
filenames = ["dHbO.tif", "dHbR.tif", "dHbT.tif", "GCaMP.tif", "rawdata_green.tif"]
bin_size = 2 # Shouldn't change, normally

for mouse_num in mice_num:

    print(f"\nProcessing mouse M{mouse_num}.")

    for filename in filenames:

        print(f"Processing {filename}.")

        file = os.path.join(return_data_dir(mouse_num), filename)

        if filename == "rawdata_green.tif": # Need to average it before saving, and save it under another name

            data = read_data(file)

            data = np.mean(data, axis=0)
            save_name = "green_avg.npy"

        else: # Other file

            data = read_data(file)

            save_name = filename

        # Spatially bin
        if data.ndim == 3:
            binned_data = bin_3d_matrix(data, (bin_size, bin_size))
        elif data.ndim == 2:
            binned_data = bin_2d_matrix(data, (bin_size, bin_size))

        # Save
        if save_name[-4:] == ".npy":
            np.save(os.path.join(return_data_dir(mouse_num), save_name), binned_data)
        elif save_name[-4:] == ".tif":
            tifffile.imwrite(os.path.join(return_data_dir(mouse_num), save_name), binned_data)
        else:
            raise ValueError("File extension not recognized.")