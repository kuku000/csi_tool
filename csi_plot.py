import matplotlib.pyplot as plt
from reader import Csi_Reader
import numpy as np


csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test20241007.pcap")
print(csi_matrix.shape)


def csi_plot(csi_matrix, no_frames):
    csi_matrix = csi_matrix.reshape((no_frames, 256))
    plt.figure()
    for i in range(no_frames):
        plt.plot(csi_matrix[i])

    plt.show()
    
csi_plot(csi_matrix, no_frames)
csi_plot(csi_matrix, no_frames)
