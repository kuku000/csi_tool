import numpy as np

import matplotlib.pyplot as plt
from reader import Csi_Reader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import openpyxl
import csi_tool
csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\localization_phone\1123_phone\reference_point_21.pcap", "original")
#csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test1hz.pcap", "original")
csi_matrix = np.fft.fftshift(csi_matrix, axes=1)
print(np.abs(csi_matrix[3]))

csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
vaild_csi, no_subcarriers = csi_tool.remove_null_and_pilot(csi_matrix, no_frames, no_subcarriers)

# 檢測封包是否包含連續低能量子載波
def has_continuous_low_energy(packet, energy_threshold, min_continuous_length):
    # 計算每個子載波的振幅
    energy = np.abs(packet)
    
    # 找出低於能量閾值的子載波
    low_energy_mask = energy < energy_threshold

    # 計算連續低能量段的長度
    continuous_lengths = np.diff(np.where(np.concatenate(([0], low_energy_mask, [0])))[0]) - 1
    max_continuous_length = max(continuous_lengths, default=0)

    # 判斷是否超過最小連續長度
    return max_continuous_length >= min_continuous_length

# 設置參數
energy_threshold = 20  # 子載波能量的閾值
min_continuous_length = 2  # 最小連續低能量段的長度

# 假設 csi_matrix 的形狀是 (num_packets, num_subcarriers)
filtered_csi = np.array([
    packet for packet in vaild_csi if not has_continuous_low_energy(packet, energy_threshold, min_continuous_length)
])

print(f"原始封包數量: {csi_matrix.shape[0]}")
print(f"過濾後封包數量: {filtered_csi.shape[0]}")
print(np.abs(filtered_csi[1]))
csi_tool.csi_plot(filtered_csi,filtered_csi.shape[0],no_subcarriers)
