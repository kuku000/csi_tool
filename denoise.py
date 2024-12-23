import numpy as np
from scipy.signal import medfilt

def hampel_filter(data, window_size=5, threshold=3):
    """
    Hampel filter to remove outliers in data.
    
    :param data: Input data (shape: [packets, subcarriers]).
    :param window_size: Size of the window to compute median (should be odd).
    :param threshold: Threshold to identify outliers.
    :return: Data with outliers replaced by the median.
    """
    # Number of packets and subcarriers
    n_packets, n_subcarriers = data.shape
    
    # Output data after Hampel filtering
    filtered_data = data.copy()
    
    # Iterate over each subcarrier
    for i in range(n_subcarriers):
        # Iterate over each packet
        for j in range(n_packets):
            # Define the window for the Hampel filter
            window_start = max(0, j - window_size // 2)
            window_end = min(n_packets, j + window_size // 2 + 1)
            window = data[window_start:window_end, i]
            
            # Compute the median and the MAD (Median Absolute Deviation)
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            # Identify outliers: if the value is more than `threshold * mad` away from the median, it's an outlier
            if np.abs(data[j, i] - median) > threshold * mad:
                filtered_data[j, i] = median
    
    return filtered_data


def moving_median_filter(data, window_size=5):
    """
    Apply moving median filter to each subcarrier in the data.

    Parameters:
    - data: 2D numpy array (packets x subcarriers)
    - window_size: int, size of the median filter window (must be odd)

    Returns:
    - filtered_data: 2D numpy array, filtered data
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1  # Make it odd if it is even

    n_packets, n_subcarriers = data.shape
    filtered_data = data.copy()

    # Apply median filter for each subcarrier
    for i in range(n_subcarriers):
        filtered_data[:, i] = medfilt(data[:, i], kernel_size=window_size)

    return filtered_data



        

