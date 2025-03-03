import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

import sys
import os

# 自訂 Dataset
class CSIDataset(Dataset):
    def __init__(self, amp_data, labels):
        self.amp_data = torch.tensor(amp_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # One-hot labels

    def __len__(self):
        return len(self.amp_data)

    def __getitem__(self, idx):
        return self.amp_data[idx], self.labels[idx]
