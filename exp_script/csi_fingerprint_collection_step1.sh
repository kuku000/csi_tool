#!/bin/bash

# 定義參數
CHANNEL="157"
CHANNEL_WIDTH="80"
MAC_ADDRESS="18:ce:94:01:de:5e"
#FRAME_TYPE=0x88  
RP_COUNT=5
PER_RP_COUNT=100
SAVE_PATH="/home/pi/Desktop/data/"

# 配置路徑和設置指令
NEXMON_PATH="/nexmon"
PATCH_PATH="/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi"
INTERFACE="wlan0"
MON_INTERFACE="mon0"
CSI_PORT="5500"

# 設定環境並安裝 firmware
echo "Setting up environment..."
cd /nexmon
source setup_env.sh
cd /nexmon/patches/bcm43455c0/7_45_189/nexmon_csi
make install-firmware