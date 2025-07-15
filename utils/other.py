from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.animation as animation

# Taken from https://www.geeksforgeeks.org/recursively-merge-dictionaries-in-python/
def merge_dictionaries(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            dict1[key] = merge_dictionaries(dict1[key], value)
        else:
            # Merge non-dictionary values
            dict1[key] = value
    return dict1


def ascending_array(shape):
    size = np.prod(shape)  # total number of elements
    return np.arange(1, size + 1).reshape(shape)


def average_3d_to_1d(signal, time_axis=0):

    axis = [0, 1, 2]
    axis.remove(time_axis)

    signal_1d = np.nanmean(signal, axis=tuple(axis))

    return signal_1d

def plot_first_frame(data, cmap="viridis", title="", savepath=None, noshow=False):
    
    fig, ax = plt.subplots(1, 1)
    if data.ndim == 3:
        im = ax.imshow(data[0,:,:], cmap=cmap)
    elif data.ndim == 2:
        im = ax.imshow(data[:,:], cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    plt.title(title)

    if savepath is not None:
        plt.savefig(savepath)
    if not noshow: plt.show()


def center_mean(signal, strictly_positive=True, center_loc=1):

    if strictly_positive:

        min_value = np.min(signal)
        signal -= min_value # min value is now 0

    mean = np.nanmean(signal)
    signal = signal * (center_loc/mean) # Mean is now centered at center_loc

    return signal

def cast_str_to_float_and_int(val_str):
    """Casts a string corresponding to a number to a float or an int depending on the presence of a decimal point.
    """

    try:
        if "." in val_str:
            val = float(val_str)
        else:
            val = int(val_str)

    except:
        raise ValueError("Invalid value {val} for casting to float or int.")
    
    return val

def flat_to_symmetric(flat, N):
    """Convert a flattened upper triangle vector to a full symmetric matrix."""
    mat = np.zeros((N, N))
    inds = np.triu_indices(N)
    mat[inds] = flat
    mat[(inds[1], inds[0])] = flat  # Reflect upper triangle to lower
    return mat