from CSIKit.reader import get_reader
from CSIKit.util import csitools
import numpy as np

# my_reader = get_reader(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test2.pcap")
# print(my_reader)
# csi_data = my_reader.read_file(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test2.pcap")
# csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)
# print(csi_matrix)
# print(csi_matrix.shape)
# print(no_frames)

class Csi_Reader():
    def __init__(self):

        pass

    def read(self, path):
        try:
            my_reader = get_reader(path)
            csi_data = my_reader.read_file(path)

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            

        return csi_data

    def get_csi_matrix(self, path, csi_type = "amplitude"):
        csi_data = self.read(path)
        if csi_type == "original":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, "original")
        elif csi_type == "amplitude":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)
        elif csi_type == "phase":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, "phase")

        return csi_matrix, no_frames, no_subcarriers
        


csi_matrix, no_frames, no_subcarriers = Csi_Reader().get_csi_matrix(r"C:\Users\keng-tse\Desktop\nexmon_csi-master\utils\matlab\test2.pcap")

