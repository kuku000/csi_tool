import matplotlib.pyplot as plt
from reader import Csi_Reader
import numpy as np
import pandas as pd

#csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\pc1023\1023pc_1.pcap", "original")
csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test20241021_40MHz_no_ping.pcap", "phase")
csi_matrix = np.fft.fftshift(csi_matrix, axes=1)
print(csi_matrix.shape)
print(no_frames)


def csi_plot(csi_matrix, no_frames, no_subcarriers, type = "amp", to_db = False):
    csi_matrix = csi_matrix.reshape((no_frames, no_subcarriers))
    if type == 'phase':
        csi_matrix_real = np.unwrap(csi_matrix)
    else:
        csi_matrix_real = abs(csi_matrix.real)
        if to_db == True:
            csi_matrix_real = csi_energy_in_db(csi_matrix)
    print(csi_matrix[0])
    print(csi_matrix.shape)
    plt.figure()
    for i in range(no_frames):
        plt.plot(csi_matrix_real[i])
        plt.show()
        
    

def csi_csv(csi_matrix, no_frames, no_subcarriers, path):
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

    null, pilot = get_subcarrier_exclusions(bandwidth)
    excluded_subcarriers = set(null + pilot)
    vaild_csi = []
    valid_subcarriers = [j for j in range(no_subcarriers) if j not in excluded_subcarriers]
    for i in range(no_frames):
         csi_vaild = csi_matrix[i][valid_subcarriers]
         vaild_csi.append(csi_vaild)

    return np.array(vaild_csi)

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
    # Return the null and pilot subcarriers for the given bandwidth
    if bandwidth in subcarrier_exclusions:
        return subcarrier_exclusions[bandwidth]['null'], subcarrier_exclusions[bandwidth]['pilot']
    else:
        raise ValueError("Unsupported bandwidth. Choose from '20MHz', '40MHz', or '80MHz'.")
    

def csi_energy_in_db(csi_matrix):
    csi_energy = np.abs(csi_matrix)**2
    energy_db = 10 * np.log10(csi_energy)
    return energy_db



vaild_csi = remove_null_and_pilot(csi_matrix, no_frames, no_subcarriers)
csi_plot(vaild_csi, no_frames, 108, "phase")

#csi_plot(csi_matrix, no_frames, no_subcarriers)
#csi_csv(csi_matrix, no_frames, no_subcarriers, r"C:\Users\keng-tse\Desktop\test1-2.xlsx")
