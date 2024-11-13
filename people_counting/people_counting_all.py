import os 
import sys

import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

sys.path.append(r"c:\Users\keng-tse\Desktop\csi_tool") #這行為絕對路徑，如需使用，必須要修改為當前決路徑
import csi_tool

files_labels = {
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\0p.xlsx": 0,
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\1p.xlsx": 1,
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\2p.xlsx": 2,
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\3p.xlsx": 3,
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\4p.xlsx": 4,
    r"C:\Users\keng-tse\Desktop\csi_dataset\peoplecounting\1023_all\5p.xlsx": 5,
}

test_labels = {
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\0p.xlsx": 0,
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\1p.xlsx": 1,
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\2p.xlsx": 2,
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\3p.xlsx": 3,
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\4p.xlsx": 4,
    r"C:\Users\keng-tse\Desktop\csi_tool\csi_dataset\peoplecounting\1107_under\5p.xlsx": 5,

}

def load_data(files_labels):
    data = []
    labels = []
    
    for file, label in files_labels.items():
        df = pd.read_excel(file)
        data.append(df.values)
        labels.extend([label] * len(df))  #Label all rows with the given label
    
    data = pd.DataFrame(np.vstack(data))  #Stack data vertically

    print(data)
    labels = pd.Series(labels)
    return data, labels

def knn_classification(data, labels, n_neighbors= 3, upsample = True, min_max = True):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    if upsample == True:
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if min_max == True:
        sc_X = MinMaxScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
    print(X_train[0])    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return knn

def preprocess_data(data):
    #將 CSI 能量轉換為 dB 並過濾無限大或無效值
    data = csi_tool.csi_energy_in_db(data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  #將 NaN 和無限大值轉為 0
    return data

def pca(data, n):
    pca = PCA(n_components = n)
    pca.fit(data)






#data, labels = load_data(files_labels)
#data = preprocess_data(data)
#knn_model = knn_classification(data, labels)


# 載入數據
data_scaled, labels = load_data(files_labels)
data_scaled_under, labels_under = load_data(test_labels)
# Concatenate the data DataFrames and label Series
data_scaled_under = pd.concat([data_scaled, data_scaled_under], axis=0, ignore_index=True)
labels_under = pd.concat([labels, labels_under], axis=0, ignore_index=True)


# 數據標準化
#scaler = StandardScaler()
#data_scaled = scaler.fit_transform(data)

for i in range(1,10):
# PCA降維
    n_components = i  # 要保留的主成分數量
    pca = PCA(n_components=n_components)
    #data_pca = pca.fit_transform(data_scaled)
    data_pca_under = pca.fit_transform(data_scaled_under)
    #important_features = np.argmax(np.abs(pca.components_), axis=1)


    # KNN分類
    X_train, X_test, y_train, y_test = train_test_split(data_pca_under, labels_under, test_size=0.4, random_state=2)
    #smote = SMOTE()
    #X_train, y_train = smote.fit_resample(X_train, y_train)
    #knn = KNeighborsClassifier(n_neighbors=3)
    #knn.fit(X_train, y_train)

    #rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    #rf_model.fit(X_train, y_train)

    #X_train, y_train =  data_pca, labels
    #X_test, y_test = data_pca_under, labels_under
    xgb_model = XGBClassifier(eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # 預測
    #y_pred = knn.predict(X_test)
    #y_pred = xgb_model.predict(X_test)
    #y_pred = rf_model.predict(X_test)
    y_pred = xgb_model.predict(X_test)

    # 混淆矩陣和準確率
    print("==========", i)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    
