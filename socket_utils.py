import socket
import os
import sys
import struct
import pickle
import numpy as np

BUFFER_SIZE = 1024
# 从sock中接收一个data（以pickle形式）
def receive_data(sock):
    # 接收文件头（名字，大小）
    fileinfo_size = struct.calcsize('128sq')
    buf = sock.recv(fileinfo_size)
    data = []
    # 开始接收文件
    if buf:
        filename, filesize = struct.unpack('128sq', buf)
        fn = filename.decode().strip('\x00')
        new_file_path = os.path.join('./', fn)  # 在服务器端新建图片名（可以不用新建的，直接用原来的也行，只要客户端和服务器不是同一个系统或接收到的图片和原图片不在一个文件夹下）

        # 将文件写入到本地,保存
        recvd_size = 0
        fp = open(new_file_path, 'wb')
        # 解决粘包的一个方法（见https://blog.csdn.net/qq_33733970/article/details/77481539）
        left_data = []
        while recvd_size < filesize:
            data_recv = sock.recv(1024)
            if (filesize - len(data_recv)) < 1024:
                left_data = sock.recv(filesize - len(data_recv))
                fp.write(left_data)
            fp.write(data_recv)
            recvd_size += len(data_recv)
        fp.close()
        # 打开写入的文件，使用pickle解包
        with open(new_file_path, 'rb') as f:
            data = pickle.load(f)
    return data


# 向sock中发送一个data（以pickle形式）
def send_data(sock, data):

    # data保存为当地的一个pkl文件
    file_path = './send_data.pkl'
    with open(file_path, "wb+") as f:
        pickle.dump(data, f)
    # fhead为文件头，包括文件名字以及文件大小
    fhead = struct.pack(b'128sq', bytes(os.path.basename(file_path), encoding='utf-8'),
                            os.stat(file_path).st_size)  # 将xxx.jpg以128sq的格式打包
    sock.send(fhead)
    # 发送文件
    fp = open(file_path, 'rb')  # 打开要传输的图片
    while True:
        data = fp.read(BUFFER_SIZE)  # 读入图片数据
        if not data:
            print('{0} send over...'.format(file_path))
            break
        sock.send(data)  # 以二进制格式发送图片数据