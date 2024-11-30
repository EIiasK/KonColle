import asyncio
import websockets
import base64
from PIL import Image
from io import BytesIO
import os
import time
import signal
import sys
import socket
from contextlib import closing


# 释放端口函数
def release_port(port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            print(f"端口 {port} 可用，无需释放")
        except OSError:
            print(f"端口 {port} 被占用，尝试释放")
            os.system(f"netstat -ano | findstr :{port} | findstr LISTENING > temp.txt")
            with open("temp.txt", "r") as f:
                lines = f.readlines()
            os.remove("temp.txt")
            for line in lines:
                pid = line.strip().split()[-1]
                os.system(f"taskkill /F /PID {pid}")
                print(f"已释放端口 {port}，杀死进程 {pid}")

# 捕获 Ctrl+C 退出信号，释放资源
def exit_gracefully(signal, frame):
    print("服务器正在退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)

# WebSocket 服务器处理函数
async def handle_connection(websocket, path):
    print(f"客户端已连接: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                # 接收 Base64 数据并解码为图像
                image_data = base64.b64decode(message)
                image = Image.open(BytesIO(image_data))

                # 保存为一张固定文件，重复替换
                save_path = "screenshot.png"
                image.save(save_path)
                print(f"接收到截图，并保存到 {save_path}")

            except Exception as e:
                print(f"处理图片失败: {e}")
    except Exception as e:
        print(f"连接处理失败: {e}")

# 启动 WebSocket 服务
async def main():
    port = 8765
    release_port(port)  # 释放端口
    async with websockets.serve(handle_connection, "127.0.0.1", port):
        print(f"WebSocket 服务器已启动，监听端口: {port}")
        print("等待客户端连接...")
        await asyncio.Future()  # 保持运行

if __name__ == "__main__":
    asyncio.run(main())