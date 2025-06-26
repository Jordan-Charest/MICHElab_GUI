import numpy as np
from toolbox_jocha.ets import *
import numba as nb


def full_FC(data):

    # Compute edge time series (ETS) of data
    ets = compute_ets(data)

    # Compute full FC
    FC = np.mean(ets, axis=0)

    return FC

def bin_3d_matrix(mat, bin_size, nanmean=True):

    if isinstance(mat, np.ndarray):
        mat = [mat]

    binned_mats = []

    for matrix in mat:

        if matrix.ndim < 3:
            binned_mats.append(matrix)
            continue

        time_dim, rows, cols = matrix.shape
        binned_rows, binned_cols = rows // bin_size[0], cols // bin_size[1]
        binned_mat = np.zeros((time_dim, binned_rows, binned_cols))
        
        for t in range(time_dim):
            for i in range(binned_rows):
                for j in range(binned_cols):
                    if nanmean:
                        binned_mat[t, i, j] = np.nanmean(matrix[t, i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])
                    else:
                        binned_mat[t, i, j] = np.mean(matrix[t, i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])
            
        binned_mats.append(binned_mat)
    
    return binned_mats[0] if len(binned_mats) == 1 else binned_mats

