from scapy.all import *
import time

target_ip = "192.168.31.1"
interval = 0.1  # 每 0.01 秒發送一次 (100Hz)
packet_count = 100  # 設定發送 100 封包
start_time = time.perf_counter()

# 發送封包直到達到指定數量
for i in range(packet_count):
    send(IP(dst=target_ip)/ICMP(), verbose=False)
    # 減少每次發送後的延遲
    elapsed_time = time.perf_counter() - start_time
    time_to_next_packet = (interval * (i + 1)) - elapsed_time
    if time_to_next_packet > 0:
        time.sleep(time_to_next_packet)

end_time = time.perf_counter()

# 計算並打印實際達到的頻率
elapsed_time = end_time - start_time
print(f"實際頻率: {packet_count / elapsed_time} Hz")



