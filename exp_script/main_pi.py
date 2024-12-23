# import bluetooth

# target_addresses = ["B8:27:EB:0E:F9:63"]  # 兩台從設備的藍牙 MAC 地址
# port = 1  # 預設 SPP 埠

# for address in target_addresses:
#     sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
#     sock.connect((address, port))
#     sock.send("START\n")
#     sock.close()
#     print(f"同步信號已發送至 {address}")

import asyncio
from bleak import BleakScanner, BleakClient

# async def discover_devices():
#     devices = await BleakScanner.discover()
#     for device in devices:
#         print(f"Found device: {device.name}, Address: {device.address}")
#         # 獲取設備的服務和特徵
#         async with BleakClient(device.address) as client:
#             services = await client.get_services()
#             for service in services:
#                 print(f"Service UUID: {service.uuid}")
#                 for characteristic in service.characteristics:
#                     print(f"  Characteristic UUID: {characteristic.uuid}")

# # 執行異步函數
# asyncio.run(discover_devices())


# import asyncio
# from bleak import BleakClient

# # 設置目標設備的 MAC 地址
# target_addresses = ["B8:27:EB:0E:F9:63"]

# # 發送的訊息
# message = "START\n"

# async def send_signal(address):
#     async with BleakClient(address) as client:
#         # 檢查設備是否已連接
#         if client.is_connected:
#             print(f"Connected to {address}")
#             # 假設你有一個特定的特徵（characteristic）來發送訊息，這裡使用假設的 UUID。
#             # 替換下面的 UUID 及相關設置來匹配你的設備
#             characteristic_uuid = "your-characteristic-uuid"
#             await client.write_gatt_char(characteristic_uuid, message.encode('utf-8'))
#             print(f"Signal sent to {address}")
#         else:
#             print(f"Failed to connect to {address}")

# # 遍歷目標設備並發送訊號
# async def main():
#     for address in target_addresses:
#         await send_signal(address)

# # 執行異步函數
# asyncio.run(main())

import asyncio
from bleak import BleakScanner, BleakClient

# 設置目標設備的 MAC 地址
target_addresses = ["B8:27:EB:0E:F9:63"]

async def discover_and_send_signal(address):
    async with BleakClient(address) as client:
        if client.is_connected:
            print(f"Connected to {address}")
            
            # 獲取設備的服務和特徵
            services = await client.get_services()
            for service in services:
                print(f"Service UUID: {service.uuid}")
                for characteristic in service.characteristics:
                    print(f"  Characteristic UUID: {characteristic.uuid}")

                    # 假設您知道是哪個特徵 UUID 用來發送訊息，您可以替換下面的 UUID
                    # 如果找到了您想要的 UUID，這樣可以用來發送訊息
                    if characteristic.uuid == "your-characteristic-uuid":  # 替換這裡的 UUID
                        message = "START\n"
                        await client.write_gatt_char(characteristic.uuid, message.encode('utf-8'))
                        print(f"Signal sent to {address}")
        else:
            print(f"Failed to connect to {address}")

# 執行異步函數
async def main():
    for address in target_addresses:
        await discover_and_send_signal(address)

# 執行主函數
asyncio.run(main())
