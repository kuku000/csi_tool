import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

import sys
import os

# # 自訂 Dataset
# class CSIDataset(Dataset):
#     def __init__(self, amp_data, labels):
#         self.amp_data = torch.tensor(amp_data, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.float32)  # One-hot labels

#     def __len__(self):
#         return len(self.amp_data)

#     def __getitem__(self, idx):
#         return self.amp_data[idx], self.labels[idx]



# class CSIRSSIDataset(Dataset):
#     def __init__(self, amp_data, rssi_data, labels):
#         self.amp_data = torch.tensor(amp_data, dtype=torch.float32)
#         self.rssi_data = torch.tensor(rssi_data, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.float32)  # One-hot labels

#     def __len__(self):
#         return len(self.amp_data)

#     def __getitem__(self, idx):
#         return self.amp_data[idx], self.rssi_data[idx], self.labels[idx]
    


import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import sys
import os




class CSIDataset(Dataset):
    def __init__(self, amp_data, labels, augment=False, noise_std=0.01):
        self.amp_data = torch.tensor(amp_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # One-hot labels
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.amp_data)

    def __getitem__(self, idx):
        amp = self.amp_data[idx]
        label = self.labels[idx]

        if self.augment:
            noise = torch.normal(mean=0.0, std=self.noise_std, size=amp.shape)
            amp = amp + noise

        return amp, label



    


class CSIRSSIDataset(Dataset):
    def __init__(self, amp_data, rssi_data, labels, augment=False, csi_noise_std=0.01, rssi_mask_prob=0.3):
        self.amp_data = torch.tensor(amp_data, dtype=torch.float32)
        self.rssi_data = torch.tensor(rssi_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # One-hot labels
        self.augment = augment
        self.csi_noise_std = csi_noise_std
        self.rssi_mask_prob = rssi_mask_prob

    def __len__(self):
        return len(self.amp_data)

    def __getitem__(self, idx):
        amp = self.amp_data[idx].clone()
        rssi = self.rssi_data[idx].clone()
        label = self.labels[idx]

        if self.augment:
            # --- CSI 擴增：加入高斯雜訊 ---
            noise = torch.normal(mean=0.0, std=self.csi_noise_std, size=amp.shape)
            amp += noise

            # --- RSSI 擴增：隨機遮蔽部分值 ---
            mask = torch.rand(rssi.shape) < self.rssi_mask_prob
            rssi[mask] = 0.0

        return amp, rssi, label
