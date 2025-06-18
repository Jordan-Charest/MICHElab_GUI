import numpy as np
from scipy.stats import zscore
from numba import njit

def zscore_3d_matrix(mat):
    for row in range(mat.shape[1]):
        for col in range(mat.shape[2]):
            mat[:,row,col] = zscore(mat[:,row,col])

    return mat

def flatten_sym_matrix(mat, exclude_diagonal=False, skip_nan=True):
    # print(mat.shape)
    indices = np.triu_indices(mat.shape[1], k=int(exclude_diagonal))
    # print(indices)
    flattened_mat = np.array([mat[i][indices] for i in range(mat.shape[0])])

    if skip_nan:
        flattened_mat = np.array([row[~np.isnan(row)] for row in flattened_mat])

    return flattened_mat

def compute_ets_old(mat):
    # Dimension 0 should be time, and dimension 1 should be features (nodes)
    return mat[:,:,np.newaxis] @ mat[:,np.newaxis,:]

# NOTE: the above is an old version of the function. It worked well, but allocated way too much memory for large arrays

def compute_ets(mat):
    
    result = np.empty((mat.shape[0], mat.shape[1], mat.shape[1]), dtype=mat.dtype)
    for i in range(result.shape[0]):
        result[i] = np.outer(mat[i], mat[i])

    return result

def compute_rss(data):
    # Dimension 0 is time, so RSS will be applied on rows

    return np.sqrt(np.sum(np.power(data, 2), axis=1))

##############################################################################################################

@njit
def compute_cofluctuation_time_series_numba(data, exclude_diagonal=True, skip_nan=True):
    T, N = data.shape
    num_elements = (N * (N - 1)) // 2 if exclude_diagonal else (N * (N + 1)) // 2
    cts = np.zeros(T, dtype=np.float64)  # Store results with higher precision

    # Compute upper-triangular indices manually
    triu_i = np.empty(num_elements, dtype=np.int32)
    triu_j = np.empty(num_elements, dtype=np.int32)
    
    index = 0
    for i in range(N):
        for j in range(i + int(exclude_diagonal), N):
            triu_i[index] = i
            triu_j[index] = j
            index += 1

    for t in range(T):
        # Compute upper-triangle elements of outer product directly
        flattened_cofluct = np.empty(num_elements, dtype=np.float64)  # Use float64
        
        for k in range(num_elements):
            i, j = triu_i[k], triu_j[k]
            flattened_cofluct[k] = data[t, i] * data[t, j]
        
        if skip_nan:
            # Remove NaNs using in-place filtering
            valid_count = 0
            for k in range(num_elements):
                if not np.isnan(flattened_cofluct[k]):
                    flattened_cofluct[valid_count] = flattened_cofluct[k]
                    valid_count += 1
            flattened_cofluct = flattened_cofluct[:valid_count]
        
        # Compute RSS (without dtype in nansum)
        cts[t] = np.sqrt(np.nansum(flattened_cofluct ** 2))  

    return cts

@njit
def compute_FC_from_signals(signals, indices):
    size = signals.shape[1]
    FC_mat = np.zeros((size, size))  # FC is a node x node matrix

    for row in range(size):
        for col in range(row, size):  # Includes diagonal
            ets = signals[:, row] * signals[:, col]  # Element-wise multiplication
            FC_mat[row, col] = np.mean(ets[indices])  # Compute mean over selected indices

            if row != col:  # Make symmetric without modifying the diagonal
                FC_mat[col, row] = FC_mat[row, col]

    return FC_mat  # Already symmetric

# Wrapper function to validate indices before calling Numba function
def compute_FC_with_check(signals, indices=None):
    if indices is None:
        indices = np.arange(signals.shape[0])  # Default: all time steps

    # Ensure indices are within valid range
    if np.any(indices >= signals.shape[0]) or np.any(indices < 0):
        raise IndexError(f"Invalid indices: {indices}. Allowed range: 0 to {signals.shape[0] - 1}")

    return compute_FC_from_signals(signals, indices)

#####################################################################################################################


def split_into_bins(mat, n_bins):

    if mat.ndim > 1:
        mat = np.squeeze(mat)
        if mat.ndim > 1:
            raise ValueError("Input array of split_into_bins is not of the right shape.")
    
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(mat)[::-1]
    

    # Split the sorted indices into n_bins subarrays
    # Such that split_indices[0] contains the indices associated with the highest elements of mat
    split_indices = np.array_split(sorted_indices, n_bins)
    reorder_indices = [np.argsort(split_indices[i]) for i in range(len(split_indices))]

    # Use these indices to get the elements
    mats = [mat[indices] for indices in split_indices]
    indices = split_indices

    # Reorder
    mats = [mats[i][reorder_indices[i]] for i in range(len(mats))]
    indices = [indices[i][reorder_indices[i]] for i in range(len(indices))]

    return mats, indices

def filter_bins_and_indices(bins, indices, min_consecutive, print_flag=False):
    """
    Filters the indices and corresponding bins such that only those indices
    that are part of a consecutive run of at least min_consecutive elements are kept.

    Parameters:
        bins (array-like): An array (or list) of floats.
        indices (array-like): An array (or list) of integer indices. Both arrays must have the same length.
        min_consecutive (int): Minimum number of consecutive integers required for a run.

    Returns:
        filtered_bins (np.ndarray): The bins array filtered to keep only the desired elements.
        filtered_indices (np.ndarray): The indices array filtered to keep only the desired elements.
    """
    # Convert to numpy arrays (if they aren't already)
    bins = np.array(bins)
    indices = np.array(indices)
    
    if len(indices) == 0:
        return bins, indices

    # List to keep track of positions (i.e. indices into the original arrays) to retain.
    keep_positions = []
    
    # Start a new run at the first position.
    current_run = [0]
    
    # Loop through indices and group them into runs.
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            # Continue the current run.
            current_run.append(i)
        else:
            # End of the current run: only keep it if it meets the min_consecutive length.
            if len(current_run) >= min_consecutive:
                keep_positions.extend(current_run)
            # Start a new run beginning with the current index.
            current_run = [i]
    
    # Check the final run after the loop.
    if len(current_run) >= min_consecutive:
        keep_positions.extend(current_run)
    
    # Use the collected positions to filter both bins and indices.
    filtered_bins = bins[keep_positions]
    filtered_indices = indices[keep_positions]

    if print_flag:
        print(f"Length of bin changed from {bins.shape} to {filtered_bins.shape}.")
    
    return filtered_bins, filtered_indices

def format_array(array, return_mask=False, zscore_signal=False):
    """Format a 3D array prior to computing FC.
    """

    array = array.reshape(array.shape[0], -1) # Reshapes to a time x nodes matrix
    mask = ~np.isnan(array[0,:])
    array = np.array([row[~np.isnan(row)] for row in array])

    if zscore_signal:
        for i in range(array.shape[1]):
            array[:,i] = zscore(array[:,i])

    if return_mask: return array, mask
    return array


def FC_components(data, n_components=5, override_indices=None, exclude_diagonal=False):
    
    # TODO: see what happens when the number of time points is not perfectly divisible by the number of components

    # Compute edge time series (ETS) of data
    ets = compute_ets(data)

    if override_indices is None:

        # Flatten the ETS into two dimensions
        flattened_ets = flatten_sym_matrix(ets, exclude_diagonal=exclude_diagonal, skip_nan=True)

        # Perform root-sum-square (RSS) on flattened ETS
        rss_ets = compute_rss(flattened_ets)

        # Split RSS data into a number of bins equal to the desired number of components
        bins, indices = split_into_bins(rss_ets, n_components)
    
    else: indices = override_indices

    # Compute and return the FC components using the bins
    FC_mats = [np.mean(ets[indices[i]], axis=0) for i in range(n_components)]

    return FC_mats

def full_FC(data):

    # Compute edge time series (ETS) of data
    ets = compute_ets(data)

    # Compute full FC
    FC = np.mean(ets, axis=0)

    return FC