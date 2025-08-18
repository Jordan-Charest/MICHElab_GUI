import os
import numpy as np

from funcs import bin_3d_matrix, read_data, return_green_rawdata_path

# mice_num = ["233-12", "308-6", "308-8", "308-10", "308-12", "308-14", "316-6", "316-8", "316-10", "316-12",
#             "322-6", "322-8", "322-10", "322-12", "353-6", "353-8", "353-10", "361-6", "365-6", "367-6",
#             "374-6", "374-8", "374-10", "387-6", "387-10", "396-6", "397-6", "410-6", "410-8", "410-10",
#             "412-8", "412-10", "415-6", "415-8"]

mice_num = ["39-12", "42-12", "44-12", "45-12", "46-12", "251-6", "254-6"]
bin_size = 2 # Shouldn't change, normally

for mouse_num in mice_num:

    print(f"\nProcessing mouse M{mouse_num}.")

    root_path, filename = return_green_rawdata_path(mouse_num)

    # Load data
    data = read_data(os.path.join(root_path, filename))

    # Spatially bin green rawdata
    binned_data = bin_3d_matrix(data, (bin_size, bin_size))

    # Take the time average
    avg_data = np.mean(binned_data, axis=0)

    # Save
    np.save(os.path.join(root_path, "green_avg.npy"), avg_data)