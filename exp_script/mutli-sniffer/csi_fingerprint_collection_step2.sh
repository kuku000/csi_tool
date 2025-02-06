#!/bin/bash

# 接收外部参数
PER_RP_COUNT=$2
SAVE_PATH=$3
REFERENCE_POINT=$1
INTERFACE="wlan0"
CSI_PORT="5500"

# 提示音效
echo "Collecting data for reference point $REFERENCE_POINT..."

# 使用 tcpdump 收集数据
tcpdump -i $INTERFACE -c $PER_RP_COUNT dst port $CSI_PORT -w ${SAVE_PATH}reference_point_${REFERENCE_POINT}.pcap &
PID=$!

# 等待完成
wait $PID
kill $PID

echo "Data collection for reference point $REFERENCE_POINT completed."

# 返回控制台
echo "Script execution completed. Returning control to the main console."

exit 0