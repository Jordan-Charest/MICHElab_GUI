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




def center_mean(signal, strictly_positive=True, center_loc=1):

    if strictly_positive:

        min_value = np.min(signal)
        signal -= min_value # min value is now 0

    mean = np.nanmean(signal)
    signal = signal * (center_loc/mean) # Mean is now centered at center_loc

    return signal
