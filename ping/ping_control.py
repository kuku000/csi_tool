from scapy.all import *
import time

target_ip = "192.168.31.1"
interval = 0.1  # 每 0.01 秒發送一次 (100Hz), interval * 0.1s
packet_count = 100  # 設定發送封包次數

start_time = time.perf_counter()
for i in range(packet_count):
    send(IP(dst=target_ip)/ICMP(), verbose=False)
    elapsed_time = time.perf_counter() - start_time
    time_to_next_packet = (interval * (i + 1)) - elapsed_time
    if time_to_next_packet > 0:
        time.sleep(time_to_next_packet)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

#綜合一次算頻率
print(f"實際頻率: {packet_count / elapsed_time} Hz")



