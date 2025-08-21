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
  - 

