import numpy as np
import tifffile
from numba import njit

def return_data_dir(mouse_num):
    return f"E:/Backup_SSD_Jordan/M{mouse_num}/raw_data"

def return_green_rawdata_path(mouse_num):
    root = f"D:/mouse_data/new_data/M{mouse_num}/raw_data"
    file = f"rawdata_green.tif"
    return root, file

@njit
def bin_2d_matrix(mat, bin_size, nanmean=True): # TODO: Probably combine it with the 3d version

    rows, cols = mat.shape
    binned_rows, binned_cols = rows // bin_size[0], cols // bin_size[1]
    binned_mat = np.zeros((binned_rows, binned_cols))

    for i in range(binned_rows):
        for j in range(binned_cols):
            if nanmean:
                binned_mat[i, j] = np.nanmean(mat[i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])
            else:
                binned_mat[i, j] = np.mean(mat[i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])

    return binned_mat

@njit
def bin_3d_matrix(mat, bin_size, nanmean=True):

    time_dim, rows, cols = mat.shape
    binned_rows, binned_cols = rows // bin_size[0], cols // bin_size[1]
    binned_mat = np.zeros((time_dim, binned_rows, binned_cols))
    
    for t in range(time_dim):
        for i in range(binned_rows):
            for j in range(binned_cols):
                if nanmean:
                    binned_mat[t, i, j] = np.nanmean(mat[t, i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])
                else:
                    binned_mat[t, i, j] = np.mean(mat[t, i*bin_size[0]:(i+1)*bin_size[0], j*bin_size[1]:(j+1)*bin_size[1]])
        
    return binned_mat

def read_data(filename):
    """Read data in tiff or npy files
    """
        
    if filename[-4:] == ".tif":
        return tifffile.imread(filename)
    
    elif filename[-4:] == ".npy":
        return np.load(filename)
    
    raise ValueError("Could not recognize file extension.")