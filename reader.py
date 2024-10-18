from CSIKit.reader import get_reader
from CSIKit.util import csitools
import numpy as np
class Csi_Reader():
    def __init__(self):

        pass

    def read(self, path):
        try:
            my_reader = get_reader(path)
            csi_data = my_reader.read_file(path)
            print(len(csi_data.frames))

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            

        return csi_data

    def get_csi_matrix(self, path, csi_type = "amplitude"):
        csi_data = self.read(path)
        #print(csi_data.timestamps)
        #a = csi_data.get_metadata()
        #print(a.chipset)
      
        if csi_type == "original":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, "original")
        elif csi_type == "amplitude":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)
        elif csi_type == "phase":
            csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, "phase")

        return csi_matrix, no_frames, no_subcarriers
        




