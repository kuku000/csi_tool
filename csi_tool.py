import matplotlib.pyplot as plt
from reader import Csi_Reader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import openpyxl
#csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\localization_phone\1123_phone\reference_point_13.pcap", "original")
#csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test1hz.pcap", "original")
#csi_matrix = np.fft.fftshift(csi_matrix, axes=1)
#print(csi_matrix.shape)
# print(no_frames)
#print(csi_matrix[499])

def convert_cfr_to_cir(csi_matrix):
    # 對每一幀應用 IFFT 得到 cir
    cir_matrix = np.fft.ifft(csi_matrix, axis=1)
    return cir_matrix

def zero_out_subcarriers(csi_matrix, no_frames, no_subcarriers):
    """
    將指定的子載波（例如 NULL、PILOT 和 GUARD 子載波）設為零。

    Args:
        csi_matrix (numpy.ndarray): CFR (頻域) 矩陣，形狀為 (no_frames, no_subcarriers)。
        no_frames (int): 幀數。
        no_subcarriers (int): 子載波數量。

    Returns:
        numpy.ndarray: 子載波被設為零的 CSI 矩陣。
    """
    if no_subcarriers == 64:
        bandwidth = "20MHz"
    elif no_subcarriers == 128:
        bandwidth = "40MHz"
    elif no_subcarriers == 256:
        bandwidth = "80MHz"
    else:
        raise ValueError("Unsupported subcarrier count. Supported values: 64, 128, 256.")

    # 獲取需要設為零的子載波索引
    null_subcarriers, pilot_subcarriers, _ = get_subcarrier_exclusions(bandwidth)
    zero_indices = set(null_subcarriers + pilot_subcarriers)

    # 建立零矩陣的副本
    modified_csi = csi_matrix.copy()

    # 將對應的子載波設為零
    for frame_idx in range(no_frames):
        for sub_idx in zero_indices:
            modified_csi[frame_idx, sub_idx] = 0.0

    return modified_csi


def csi_plot(csi_matrix, no_frames, no_subcarriers, type = "amp", to_db = False):
    ##有誤
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    if type == 'phase':
        csi_matrix_real = np.unwrap(csi_matrix)
    else:
        csi_matrix_real = abs(csi_matrix)
        if to_db == True:
            csi_matrix_real = csi_energy_in_db(csi_matrix_real)
    print(csi_matrix[0])
    print(csi_matrix.shape)
    plt.figure()
    for i in range(no_frames):
        plt.plot(csi_matrix_real[i])
        plt.show()
        
    

def csi_excel(csi_matrix, no_frames, no_subcarriers, path):
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    print(csi_matrix[0])
    df = pd.DataFrame(csi_matrix)
    df.to_excel(path, index=False, header=False, float_format="%.10f")
    print(f"CSI data saved to {path}")


def remove_null_and_pilot(csi_matrix, no_frames, no_subcarriers):
    if no_subcarriers == 64:
        bandwidth = "20MHz"
    elif no_subcarriers == 128:
        bandwidth = "40MHz"
    elif no_subcarriers == 256:
        bandwidth = "80MHz"

    null, pilot, count = get_subcarrier_exclusions(bandwidth)
    excluded_subcarriers = set(null + pilot)
    vaild_csi = []
    valid_subcarriers = [j for j in range(no_subcarriers) if j not in excluded_subcarriers]
    for i in range(no_frames):
         csi_vaild = csi_matrix[i][valid_subcarriers]
         vaild_csi.append(csi_vaild)

    return np.array(vaild_csi), count

def get_subcarrier_exclusions(bandwidth):
    # Dictionary to store null and pilot subcarriers for each bandwidth
    #都還不確定是否正確
    subcarrier_exclusions = {
        '20MHz': {
            #可能會包含802.11a/g的封包
            'null': [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63],
            'pilot': [11, 25, 39, 53],
            'count': 48
        },
        '40MHz': {
            'null': [x+64 for x in [-64, -63, -62, -61, -60, -59, -1, 63,  62,  61,  60,  59,  1,  0]],
            'pilot': [75, 89, 117, 53, 39, 11],
            'count': 108
        },
        '80MHz': {
            'null': [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255],
            'pilot': [25, 53, 89, 117, 139, 167, 203, 231],
            'count':234
        }
    }
    #Return the null and pilot subcarriers for the given bandwidth
    if bandwidth in subcarrier_exclusions:
        return subcarrier_exclusions[bandwidth]['null'], subcarrier_exclusions[bandwidth]['pilot'], subcarrier_exclusions[bandwidth]['count']
    else:
        raise ValueError("Unsupported bandwidth. Choose from '20MHz', '40MHz', or '80MHz'.")
    
def csi_energy_in_db(csi_matrix):
    csi_energy = np.abs(csi_matrix)**2
    energy_db = 10 * np.log10(csi_energy)
    energy_db = np.nan_to_num(energy_db, nan=0.0)
    return energy_db




def csi_preprocessor_amp(csi_matrix, no_frames, no_subcarriers, to_db=False, remove_sub=True, save_as_xlsx=True, path=""):
    # Reshape the CSI matrix to (no_frames, no_subcarriers)
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    
    # Remove null and pilot subcarriers if needed
    if remove_sub:
        csi_matrix, no_subcarriers_process = remove_null_and_pilot(csi_matrix, no_frames, no_subcarriers)
    else:
        no_subcarriers_process = no_subcarriers  # If not removing subcarriers, use original count

    # Calculate the magnitude (振幅) of the complex CSI matrix
    csi_matrix_real = np.abs(csi_matrix) 
    print("---------------")
    print(math.sqrt(csi_matrix.real[0][0]**2 +csi_matrix.imag[0][0]**2))
    
    # Convert to dB if needed
    if to_db:
        csi_matrix_real = csi_energy_in_db(csi_matrix_real)

    # Remove rows where all values are zero
    csi_matrix_real = csi_matrix_real[~(csi_matrix_real == 0).all(axis=1)]
    no_frames = csi_matrix_real.shape[0]  
    
    if save_as_xlsx:
        try:
            csi_excel(csi_matrix_real, no_frames, no_subcarriers_process, path)
        except Exception as e:
            raise ValueError("Saving error") from e

    return csi_matrix_real


def csi_preprocessor_phase(csi_matrix, no_frames, no_subcarriers, remove_sub=True, save_as_xlsx=True, path=""):
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    
    if remove_sub:
        csi_matrix, no_subcarriers_process = remove_null_and_pilot(csi_matrix, no_frames, no_subcarriers)
    else:
        no_subcarriers_process = no_subcarriers  # If not removing subcarriers, use original count

    print( csi_matrix.shape)

    csi_matrix_phase = np.angle(csi_matrix)

    csi_matrix_phase = csi_matrix_phase[~(csi_matrix_phase == 0).all(axis=1)]
    no_frames = csi_matrix_phase.shape[0]  # Update no_frames after removal

    csi_matrix_phase = np.unwrap(csi_matrix_phase)


    
    if save_as_xlsx:
        try:
            csi_excel(csi_matrix_phase, no_frames, no_subcarriers_process, path)
        except Exception as e:
            raise ValueError("Saving error") from e

    return csi_matrix_phase


def csi_preprocessor_amp_phase(csi_matrix, no_frames, no_subcarriers, to_db = False, remove_sub=True, save_as_xlsx=True, path=""):
    #Reshape CSI matrix for processing
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    
    #Split into amplitude and phase
    amplitude = np.abs(csi_matrix)
    phase = np.angle(csi_matrix)
    phase = np.unwrap(phase)

    #Remove NULL and PILOT subcarriers if required
    if remove_sub:
        amplitude, no_subcarriers_process_amp = remove_null_and_pilot(amplitude, no_frames, no_subcarriers)
        phase, no_subcarriers_process_phase = remove_null_and_pilot(phase, no_frames, no_subcarriers)
    else:
        no_subcarriers_process_amp = no_subcarriers
        no_subcarriers_process_phase = no_subcarriers

    #Convert amplitude to dB if needed
    if to_db:
        amplitude = csi_energy_in_db(amplitude)
    
    #Remove rows where all amplitude values are zero
    amp_copy = amplitude.copy() 
    amplitude = amplitude[~(amplitude == 0).all(axis=1)]
    phase = phase[~(amp_copy == 0).all(axis=1)]  # Phase should match the amplitude rows
    no_frames = amplitude.shape[0]

    #Concatenate amplitude and phase horizontally
    csi_matrix_combined = np.hstack((amplitude, phase))
    print(csi_matrix_combined.shape)

    #Save to Excel if required
    if save_as_xlsx:
        try:
            csi_excel(csi_matrix_combined, no_frames, no_subcarriers_process_amp + no_subcarriers_process_phase, path)
        except Exception as e:
            raise ValueError("Saving error") from e

    return csi_matrix_combined
    
def csi_preprocessor_amp_cfr_cir(csi_matrix, no_frames, no_subcarriers, to_db = False, remove_sub=True, save_as_xlsx=True, path=""):
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    zeroed_csi_matrix = zero_out_subcarriers(csi_matrix, no_frames, no_subcarriers)
    
    cfr_matrix = np.abs(csi_matrix)
    cir_matrix = convert_cfr_to_cir(zeroed_csi_matrix)
    cir_matrix = np.abs(cir_matrix)

    if remove_sub:
        cfr_matrix, no_subcarriers_process_amp = remove_null_and_pilot(cfr_matrix, no_frames, no_subcarriers)
    
    if to_db:
        cfr_matrix = csi_energy_in_db(cfr_matrix)
        cir_matrix = csi_energy_in_db(cir_matrix)

    # 找到全零封包
    all_zero_mask = (cfr_matrix == 0).all(axis=1)
    all_zero_indices = np.where(all_zero_mask)[0]

    # 打印全零封包信息
    if len(all_zero_indices) > 0:
        print(f"全零封包的索引: {all_zero_indices}")
    else:
        print("沒有全零封包。")

    # 移除全零封包
    valid_rows = ~all_zero_mask
    cfr_matrix = cfr_matrix[valid_rows]
    cir_matrix = cir_matrix[valid_rows]
    no_frames = cfr_matrix.shape[0]


    csi_matrix_combined = np.hstack((cfr_matrix, cir_matrix))
    print(csi_matrix_combined.shape)

    if save_as_xlsx:
        try:
            csi_excel(csi_matrix_combined, no_frames, no_subcarriers_process_amp + no_subcarriers, path)
        except Exception as e:
            raise ValueError("Saving error") from e

    return csi_matrix_combined

    
#csi_preprocessor_amp_phase(csi_matrix, no_frames, no_subcarriers, False, False, True, True, r"C:\Users\keng-tse\Desktop\0p.xlsx")


# 將指定子載波設為零
#zeroed_csi_matrix = zero_out_subcarriers(csi_matrix, no_frames, no_subcarriers)

# 將處理後的矩陣轉換為 CIR
#cir_matrix = convert_cfr_to_cir(zeroed_csi_matrix)

# 可視化 CIR
#csi_plot(cir_matrix, no_frames, no_subcarriers)
