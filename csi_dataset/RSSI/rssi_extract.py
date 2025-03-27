import pandas as pd
import numpy as np
# 讀取 CSV 檔案
def process_csv(input_file, output_file):
    # 讀取 CSV 檔案
    df = pd.read_csv(input_file)
    print(df)
    
    # 提取 Label 對應的數值格式 (X-Y)
    extracted_labels = df["Label"].astype(str)
    print(extracted_labels)
    df["coordinate"] = extracted_labels
    
    # Label 對應的座標
    label_to_coordinates = {
        "1-1": (0, 0), "1-2": (0.6, 0), "1-3": (1.2, 0), "1-4": (1.8, 0), "1-5": (2.4, 0),
        "1-6": (3.0, 0), "1-7": (3.6, 0), "1-8": (4.2, 0), "1-9": (4.8, 0), "1-10": (5.4, 0), "1-11": (6.0, 0),
        "2-1": (0, 0.6), "2-11": (6.0, 0.6),
        "3-1": (0, 1.2), "3-11": (6.0, 1.2),
        "4-1": (0, 1.8), "4-11": (6.0, 1.8),
        "5-1": (0, 2.4), "5-11": (6.0, 2.4),
        "6-1": (0, 3.0), "6-2": (0.6, 3.0), "6-3": (1.2, 3.0), "6-4": (1.8, 3.0), "6-5": (2.4, 3.0),
        "6-6": (3.0, 3.0), "6-7": (3.6, 3.0), "6-8": (4.2, 3.0), "6-9": (4.8, 3.0), "6-10": (5.4, 3.0), "6-11": (6.0, 3.0),
        "7-1": (0, 3.6), "7-11": (6.0, 3.6),
        "8-1": (0, 4.2), "8-11": (6.0, 4.2),
        "9-1": (0, 4.8), "9-11": (6.0, 4.8),
        "10-1": (0, 5.4), "10-11": (6.0, 5.4),
        "11-1": (0, 6.0), "11-2": (0.6, 6.0), "11-3": (1.2, 6.0), "11-4": (1.8, 6.0), "11-5": (2.4, 6.0),
        "11-6": (3.0, 6.0), "11-7": (3.6, 6.0), "11-8": (4.2, 6.0), "11-9": (4.8, 6.0), "11-10": (5.4, 6.0), "11-11": (6.0, 6.0)
    }
    df["label_coor"] = df["coordinate"].map(label_to_coordinates)
    coordinates = {
        1: (0, 0), 40: (0.6, 0), 39: (1.2, 0), 38: (1.8, 0), 37: (2.4, 0),
        36: (3.0, 0), 35: (3.6, 0), 34: (4.2, 0), 33: (4.8, 0), 32: (5.4, 0), 31: (6.0, 0),
        2: (0, 0.6), 3: (0, 1.2), 4: (0, 1.8), 5: (0, 2.4),
        6: (0, 3.0), 7: (0, 3.6), 8: (0, 4.2), 9: (0, 4.8), 10: (0, 5.4), 11: (0, 6.0),
        12: (0.6, 6.0), 13: (1.2, 6.0), 14: (1.8, 6.0), 15: (2.4, 6.0),
        16: (3.0, 6.0), 17: (3.6, 6.0), 18: (4.2, 6.0), 19: (4.8, 6.0),
        20: (5.4, 6.0), 21: (6.0, 6.0),
        22: (6.0, 5.4), 23: (6.0, 4.8), 24: (6.0, 4.2), 25: (6.0, 3.6),
        26: (6.0, 3.0), 27: (6.0, 2.4), 28: (6.0, 1.8), 29: (6.0, 1.2), 30: (6.0, 0.6),
        41: (3.0, 0.6), 42: (3.0, 1.2), 43: (3.0, 1.8),
        44: (3.0, 2.4), 45: (3.0, 3.0), 46: (3.0, 3.6),
        47: (3.0, 4.2), 48: (3.0, 4.8), 49: (3.0, 5.4)
    }
    coordinate_to_label2 = {value: key for key, value in coordinates.items()}
    
    df["label"] = df["label_coor"].map(coordinate_to_label2)
    
    # 確保座標對應成功
    print("[DEBUG] coordinate:")
    print(df[["label", "coordinate"]].head())
    
    # 選取所需欄位並儲存成新的 CSV 檔案
    df[["timeStemp", "AP1_Rssi", "AP2_Rssi", "AP3_Rssi", "AP4_Rssi", "label", "label_coor", "coordinate"]].to_csv(output_file, index=False)
    print(f"Processed CSV saved as: {output_file}")

# 執行處理
input_csv = r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\RSSI\timestamp_allignment_Balanced_2024_12_21_rtt_logs.csv"
base_path = r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\RSSI"

output_csv = f"{base_path}\csv\{input_csv.split("\\")[-1].replace('_rtt_logs.csv', '_rssi.csv')}"
process_csv(input_csv, output_csv)