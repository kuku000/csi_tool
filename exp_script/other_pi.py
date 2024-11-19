import bluetooth
import os

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", 1))
server_sock.listen(1)

print("等待同步指令...")
client_sock, address = server_sock.accept()
print(f"接收到來自 {address} 的連線")

data = client_sock.recv(1024).decode("utf-8").strip()
if data == "START":
    print("接收到同步指令，啟動 Shell Script")
    os.system("bash /path/to/your_script.sh")

client_sock.close()
server_sock.close()
