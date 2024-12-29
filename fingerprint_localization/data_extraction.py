import os
import sys
sys.path.append(r"c:\Users\keng-tse\Desktop\csi_tool")

import numpy as np
from csi_tool import csi_preprocessor_amp_phase, csi_preprocessor_amp_cfr_cir  
from reader import Csi_Reader

input_folder = r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\localization_phone\1223_phone\5G\20Mhz"  # 替換為你的 PCAP 資料夾路徑
output_folder = r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\localization_phone\1223_phone\5G\20Mhz\csv\CIR"  # 替換為你的輸出資料夾路徑
os.makedirs(output_folder, exist_ok=True)
pcap_files = [f for f in os.listdir(input_folder) if f.endswith('.pcap')]
for pcap_file in pcap_files:
    input_path = os.path.join(input_folder, pcap_file)
    output_path = os.path.join(output_folder, f"{os.path.splitext(pcap_file)[0]}.xlsx")

    csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(input_path, "original")
    csi_matrix = np.fft.fftshift(csi_matrix, axes=1)
    print(f"Processing file: {pcap_file}")
    processed_csi = csi_preprocessor_amp_cfr_cir(csi_matrix, no_frames, no_subcarriers, False, True, True, output_path)
    #processed_csi = csi_preprocessor_amp_phase(csi_matrix, no_frames, no_subcarriers, False, True, True, output_path)
    print(f"File saved: {output_path}")