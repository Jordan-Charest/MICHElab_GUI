import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ConstantInputWarning
import numba as nb

def r_coeff(x, y, squared=False):
    """ Return R^2 where x and y are array-like."""
    r_value = np.corrcoef(x, y)[0, 1]
    
    if not squared: return r_value
    elif squared: return r_value**2

# TODO: maybe use np.correlate instead

def crosscorrelation_squared(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    return np.correlate(x, y, mode='full')[len(y)-maxlag-1:len(y)+maxlag]

def max_r_coeff(signal1, signal2, max_shift=100, squared=False, lag_sign=None):
    # Step 1: Compute the cross-correlation without subtracting the mean (since signals are z-scored)
    corr = crosscorrelation_squared(signal1, signal2, max_shift)

    if lag_sign is None:
        lags = np.arange(-max_shift, max_shift+1, 1)
    elif lag_sign.lower() == "negative":
        lags = np.arange(-max_shift, 1, 1)
    elif lag_sign.lower() == "positive":
        lags = np.arange(0, max_shift+1, 1)

    # Step 2: Find the lag with the maximum cross-correlation
    max_corr_index = np.argmax(np.abs(corr))  # max absolute cross-correlation
    max_lag = lags[max_corr_index]
    
    # Shift one signal by the best lag
    # TODO: maybe not roll?
    shifted_signal2 = np.roll(signal2, max_lag)

    # Step 3: Compute Pearson correlation and R²
    r_squared = r_coeff(signal1, shifted_signal2, squared=squared)

    return max_lag, r_squared

def r_coeff_2mats(signal1, signal2, max_shift=100, lag=False, print_flag=False, convert_to_s=False, fps=None, squared=False, lag_sign=None):
    
    R2_mat = np.zeros((signal1.shape[1], signal1.shape[2]))

    if lag: lag_mat = np.zeros((signal1.shape[1], signal1.shape[2]))

    for row in range(R2_mat.shape[0]):
        for col in range(R2_mat.shape[1]):

            if np.any(np.isnan(signal1[:,row,col])):
                R2_mat[row, col] = np.nan
                if lag: lag_mat[row, col] = np.nan
                continue

            if lag:
                lag_mat[row, col], R2_mat[row, col] = max_r_coeff(signal1[:,row,col], signal2[:,row,col], max_shift=max_shift, squared=squared, lag_sign=lag_sign)
            else:
                R2_mat[row, col] = r_coeff(signal1[:,row,col], signal2[:,row,col], squared=squared)

    if lag and convert_to_s:
        lag_mat = lag_mat / fps

    if not lag: return R2_mat
    elif lag: return lag_mat, R2_mat

###### OLDER FUNCTIONS
# See if some of those can't be removed / combined with the above

def correlation_matrix(signal1, signal2):
    """Takes two 3D signals (time x height x width) and computes the correlation matrix between them.

    Args:
        signal1 (array): first signal to correlate
        signal2 (array): second signal to correlate

    Returns:
        array: 2D array of correlation between signal1 and signal2
    """

    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    
    nan_loc1 = np.isnan(signal1[0,:,:])
    nan_loc2 = np.isnan(signal2[0,:,:])

    signal1 = np.nan_to_num(signal1, nan=0)
    signal2 = np.nan_to_num(signal2, nan=0)
    
    mat_shape = (signal1.shape[1], signal1.shape[2])

    correl_mat = np.zeros(mat_shape)

    for row in range(mat_shape[0]):
        for col in range(mat_shape[1]):

            signal1_list = signal1[:,row,col]
            signal2_list = signal2[:,row,col]

            correl_coeff = sp.stats.pearsonr(signal1_list, signal2_list).statistic

            correl_mat[row, col] = correl_coeff

    correl_mat[nan_loc1] = np.nan
    correl_mat[nan_loc2] = np.nan

    return correl_mat

def correlation_histogram(correl_mat, display_norm=False, density=True):
    """Plots and returns a histogram of the correlation values, given a correlation matrix.

    Args:
        correl_mat (array): 2D array of correlations, as returned by correlation_matrix for instance.
        display_norm (bool, optional): Display the normal curve on the plot. Defaults to False.
        density (bool, optional): Value of the density parameter passed to the ax.hist function. Defaults to True.

    Returns:
        fig (figure): Histogram figure object.
        mu (float): Mean value of the normal distribution fitted on the data.
    """


    fig, ax = plt.subplots()


    correl_mat_flat = np.ravel(correl_mat)
    ax.hist(correl_mat_flat, bins=50, density=density)


    dist = np.ravel(correl_mat)[~np.isnan(np.ravel(correl_mat))]
    mu, std = sp.stats.norm.fit(dist)

    if display_norm:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = sp.stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
    
    
    ax.plot([], [], '', label=f"Mean value: {mu:.2f}")
    ax.plot([], [], '', label=f"Std dev: {std:.2f}")
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of Correlation Matrix')
    ax.legend()
    
    # Return the figure
    return fig, mu

def correlation_threshold(signal1, signal2, threshold=0.5, correl_mat=None):
    """Removes elements in signals that correspond to a correlation below the specified threshold.

    Args:
        signal1 (array): 3D array corresponding to the first signal.
        signal2 (array): 3D array corresponding to the second signal.
        threshold (float, optional): Correlation threshold below which elements will be changed to np.nan. Defaults to 0.5.
        correl_mat (_type_, optional): Correlation matrix between signal1 and signal2. Defaults to None, will be computed if equal to None.

    Returns:
        correl_mask (array): 2D boolean mask of correlation thresholds.
        masked_signal1 (array): signal1 with the elements associated to correlation lower than the threshold set to np.nan.
        masked_signal2 (array): signal2 with the elements associated to correlation lower than the threshold set to np.nan.
    """

    if correl_mat is None:
        correl_mat = correlation_matrix(signal1, signal2)

    correl_mask = correl_mat >= threshold

    correl_mask_3d = np.repeat(correl_mask[np.newaxis, :, :], signal1.shape[0], axis=0)

    masked_signal1 = np.copy(signal1)
    masked_signal2 = np.copy(signal2)

    masked_signal1[~correl_mask_3d] = np.nan
    masked_signal2[~correl_mask_3d] = np.nan

    return correl_mask, masked_signal1, masked_signal2


@nb.njit(parallel=True)
def compute_correlation_map(data, ts):
    """
    Similar to the function compute_seedbased in Processing_GUI; computes the correlation map between
    a single timeseries and a whole window.

    Parameters:
        data (numpy.ndarray): 3D array of shape (Time, Height, Width).
        ts (numpy.ndarray): The time series to correlate.

    Returns:
        numpy.ndarray: 2D correlation map of shape (Height, Width).
    """
    T, H, W = data.shape
    corr_map = np.full((H, W), np.nan)  # Initialize with NaNs

    # Extract seed time series
    seed_ts = ts

    # Compute mean of seed time series, ignoring NaNs
    valid_seed = ~np.isnan(seed_ts)
    if np.sum(valid_seed) == 0:
        return corr_map  # If seed is all NaNs, return all NaNs

    mean_seed = np.nanmean(seed_ts)

    # Precompute global mean and std for all pixels
    mean_data = np.full((H, W), np.nan)
    std_data = np.full((H, W), np.nan)
    numerator = np.zeros((H, W))
    
    # Compute mean and std for each pixel in parallel
    for row in nb.prange(H):
        for col in range(W):
            pixel_ts = data[:, row, col]
            valid_mask = ~np.isnan(pixel_ts)
            if np.sum(valid_mask) == 0:
                continue  # Leave as NaN if all values are NaN
            
            mean_px = np.mean(pixel_ts[valid_mask])
            std_px = np.sqrt(np.sum((pixel_ts[valid_mask] - mean_px) ** 2))
            mean_data[row, col] = mean_px
            std_data[row, col] = std_px

            # Compute numerator for covariance
            numerator[row, col] = np.sum((pixel_ts[valid_mask] - mean_px) * (seed_ts[valid_mask] - mean_seed))

    # Compute std of seed
    std_seed = np.sqrt(np.nansum((seed_ts - mean_seed) ** 2))
    
    # Compute correlation while avoiding division by zero
    for row in nb.prange(H):
        for col in range(W):
            if std_seed > 0 and std_data[row, col] > 0:
                corr_map[row, col] = numerator[row, col] / (std_seed * std_data[row, col])

    return corr_map