from scipy.ndimage import gaussian_filter1d, minimum_filter1d
import scipy.stats as stats
import numpy as np

def baseline_minfilter(signal, window=300, sigma1=5, sigma2=100, debug=False):
    signal_flatstart = np.copy(signal)
    signal_flatstart[0] = signal[1]
    smooth = gaussian_filter1d(signal_flatstart, sigma1)
    mins = minimum_filter1d(smooth, window)
    baseline = gaussian_filter1d(mins, sigma2)
    if debug:
        debug_out = np.asarray([smooth, mins, baseline])
        return debug_out
    else:
        return baseline


def compute_dff_using_minfilter(timeseries, window=200, sigma1=0.1, sigma2=50, offset=0):

    timeseries += offset

    if len(timeseries.shape) == 1:
        baseline = baseline_minfilter(timeseries, window=window, sigma1=sigma1, sigma2=sigma2)
        dff = (timeseries - baseline) / baseline
    else:
        dff = np.zeros(timeseries.shape)
        for i in range(timeseries.shape[0]):
            if np.any(timeseries[i]):
                baseline = baseline_minfilter(timeseries[i], window=window, sigma1=sigma1, sigma2=sigma2)
                dff[i] = (timeseries[i] - baseline) / baseline
    return dff