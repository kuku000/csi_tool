import matplotlib.pyplot as plt
from reader import Csi_Reader
import numpy as np
import pandas as pd

csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test1018.pcap")
print(csi_matrix.shape)
print(no_frames)


def csi_plot(csi_matrix, no_frames, no_subcarriers):
    csi_matrix = csi_matrix.reshape((no_frames, 64))
    print(csi_matrix[0])
    plt.figure()
    for i in range(no_frames):
        plt.plot(csi_matrix[i])
        

    plt.show()

def csi_csv(csi_matrix, no_frames, no_subcarriers, path):
    csi_matrix = csi_matrix.reshape((no_frames, 64))
    df = pd.DataFrame(csi_matrix)
    df.to_csv(path, index=False, header=False)
    print(f"CSI data saved to {path}")


csi_plot(csi_matrix, no_frames, no_subcarriers)
csi_csv(csi_matrix, no_frames, no_subcarriers, r"C:\Users\keng-tse\Desktop\test1-1.csv")