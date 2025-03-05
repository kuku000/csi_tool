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

def amp_correct(amp_matrix):
    #for n in range(num_packets):
        # 平滑處理
        #theta_n = gaussian_filter(phase_matrix[n, :], sigma=1)
        #theta_n = phase_matrix[n,:]
        # 最小二平方估計一條由 SFO 和 CFO 造成的線性偏差
        #a = np.sum(I * (theta_n - np.mean(theta_n))) / np.sum(I**2)
        #b = np.median(theta_n)
        
        # 校正相位
        #corrected_phase[n, :] = theta_n - a * I - b
        
        # 防止過度偏移，調整均值
        #corrected_phase[n, :] -= np.mean(corrected_phase[n, :])

        pass


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


def correct_cfo_sfo(phase_matrix):
    """
    校正 CSI 相位的 CFO 和 SFO。
    :param phase_matrix: numpy array, 每列表示一組子載波的相位 (shape: [num_packets, num_subcarriers])
    :param subcarrier_indices: numpy array, 子載波的索引 (shape: [num_subcarriers])
    :return: numpy array, 校正後的 CSI 相位 (shape: [num_packets, num_subcarriers])
    """
    num_packets, num_subcarriers = phase_matrix.shape
    corrected_phase = np.zeros_like(phase_matrix)
    I = []
    I = np.arange(-32, 32)  # 子载波索引
    for n in range(num_packets):
        # 單個封包的相位
        theta_n = phase_matrix[n, :]
        
        # 計算 SFO (a) 和 CFO (b)
        #I = subcarrier_indices
        a = (theta_n[-1] - theta_n[0]) / (I[-1] - I[0])  # SFO 估計
        b = np.mean(theta_n)  # CFO 估計
        print()

        # 校正相位
        corrected_phase[n, :] = theta_n - a * I - b

    return corrected_phase



from scipy.ndimage import gaussian_filter

def correct_cfo_sfo2(phase_matrix):
    """
    改進版的 CSI 相位校正函數。
    :param phase_matrix: numpy array, 每列表示一組子載波的相位 (shape: [num_packets, num_subcarriers])
    :return: numpy array, 校正後的 CSI 相位 (shape: [num_packets, num_subcarriers])
    """
    num_packets, num_subcarriers = phase_matrix.shape
    corrected_phase = np.zeros_like(phase_matrix)
    I = np.arange(-117, 117)  # 子載波索引（可根據實際數據調整範圍）

    for n in range(num_packets):
        # 平滑處理
        theta_n = gaussian_filter(phase_matrix[n, :], sigma= 0.1)
        #theta_n = phase_matrix[n,:]
        # 最小二平方估計一條由 SFO 和 CFO 造成的線性偏差
        a = np.sum(I * (theta_n - np.mean(theta_n))) / np.sum(I**2)
        b = np.median(theta_n)
        
        # 校正相位
        corrected_phase[n, :] = theta_n - a * I - b
        
        # 防止過度偏移，調整均值
        corrected_phase[n, :] -= np.mean(corrected_phase[n, :])

    return corrected_phase

def preprocess_csi_for_fingerprint(csi_matrix):
    """
    Preprocess CSI data for fingerprinting by Standardizing subcarriers.

    Parameters:
        csi_matrix (numpy.ndarray): CSI data (packets x subcarriers).

    Returns:
        numpy.ndarray: Normalized CSI data.
    """
    # Standardize each subcarrier
    mean = np.mean(csi_matrix, axis=1)
    std = np.std(csi_matrix, axis=1)
    std[std == 0] = 1  # Prevent division by zero
    return (csi_matrix - mean) / std

def preprocess_csi_for_fingerprint2(csi_matrix):
    """
    Preprocess CSI data for fingerprinting by standardizing subcarriers per packet.

    Parameters:
        csi_matrix (numpy.ndarray): CSI data (packets x subcarriers).

    Returns:
        numpy.ndarray: Standardized CSI data with per-packet normalization.
    """
    # Standardize each packet's subcarriers independently
    mean_per_packet = np.mean(csi_matrix, axis=1, keepdims=True)  # Compute mean per packet
    std_per_packet = np.std(csi_matrix, axis=1, keepdims=True)    # Compute std per packet
    std_per_packet[std_per_packet == 0] = 1  # Prevent division by zero
    return (csi_matrix - mean_per_packet) / std_per_packet






