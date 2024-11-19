import bluetooth

target_addresses = ["B8:27:EB:0E:F9:63"]  # 兩台從設備的藍牙 MAC 地址
port = 1  # 預設 SPP 埠

for address in target_addresses:
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((address, port))
    sock.send("START\n")
    sock.close()
    print(f"同步信號已發送至 {address}")
