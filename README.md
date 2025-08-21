# csi_tool
wifi csi tool 

## folder (List only the useful parts)
- **csi_dataset**:  
  - RSSI (floder): Raw RSSI data
  - localization_phone (floder): Raw CSI data
  - rssi (folder): Organized RSSI CSI data  
- **exp_script**:  
  - mutli-sniffer (folder): Not completed sound (folder): beep (chinese)  
  - csi_fc_time.sh: shell script for data collection (fixed time collection)  
  - csi_fingerprint_collection.sh: shell script for data collection (fixed frames collection)
- **fingerprint_localization**:
  - data_extraction.py: python code used for extract csi data from frames (raw pcap file)
  - repeat (folder): the experiment result for only using rssi as input 
  - rssi_DNN.ipynb: rssi DNN code
  - rssi_KNN.ipynb: rssi KNN code
- **model**:
  - data_loader.py: use to load data
  - csidataset.py: def CSIDataset class
  - model.py: Transformer model
  - train.ipynb: model training code
  - train_mirco.ipynb: model training (mirco exp)
- **model_CNN**
  - repeat_copy (folder): the experiment result for using csi and csi+rssi as input (5G)
  - csidataset.py: def Dataset class  
  - data_loader.py: use to load data  
  - train copy.ipynb: model def and model training (CSI 5G)  
  - rssicsi_train copy.ipynb: model def and model training (CSI 5G +RSSI  
- **model_KNN (no used)**
  - csidataset.py: def Dataset class
  - data_loader.py: use to load data
  - train.ipynb: model def and model training KNN TEST
- **people_counting (no used)**
- **ping**
  - ping_control.py: python code for sending ICMP PING in Windows
- **compare_model**:
  - repeat_24_copy (folder): the experiment result (2.4G)
  - 2.4GVS5G copy.ipynb:  model def and model training (CSI 2.4G)
  - 2.4Grssi_csi copy.ipynb:  model def and model training (CSI+RSSI 2.4G)
- **csi_extractor_by_zeroby0.py**: csi extractor by _zeroby0
- **csi_tool.py (Old version code)**: some csi tool function (if just want to extract csi amp or phase, use data_extraction.py)
- **reader.py (Old version code)**
-  

  
   

