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


# 生成篩選參數碼
echo "Generating filter parameter code..."
PARAM_CODE=$(/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams -c $CHANNEL/$CHANNEL_WIDTH -C 1 -N 1 -m $MAC_ADDRESS)
echo "Generated PARAM_CODE: {$PARAM_CODE}"

# 配置網卡進入監聽模式
echo "Configuring monitor mode..."

sleep 2

ifconfig wlan0 up

sleep 4

pkill wpa_supplicant

sleep 2

nexutil -Iwlan0 -s500 -b -l34 -vm+IBEQAAAQAYzpQB3l4AAAAAAAAAAAAAAAAAAAAAAAAAAA==

sleep 2

iw phy $(iw dev wlan0 info | gawk '/wiphy/ {printf "phy" $2}') interface add mon0 type monitor

sleep 2

ifconfig mon0 up

sleep 2

ifconfig wlan0 up

sleep 2

# 主循環來蒐集CSI數據
for (( i=1; i<=$RP_COUNT; i++ )); do
    play /home/pi/Desktop/data/sound/start.mp3
    sleep 5
    echo "Collecting data for reference point $i..."

    # 使用tcpdump並將文件儲存到指定的參考點檔案中
    tcpdump -i $INTERFACE -c $PER_RP_COUNT dst port $CSI_PORT -w ${SAVE_PATH}reference_point_$i.pcap &
    PID=$!

    wait $PID
    kill $PID
    echo "Data collection for reference point $i completed."
    

    # 提示音效
    play /home/pi/Desktop/data/sound/next.mp3 

    sleep 15
done

echo "All data collection completed."
