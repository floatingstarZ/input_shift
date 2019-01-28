from socket_utils import *
from PIL import Image
import PIL as pil

def sock_client_image():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('10.134.171.252', 6666))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
        # s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
    except socket.error as msg:
        print(msg)
        print(sys.exit(1))
    while True:
        # Path = int(input('Pause:'))
        data = pil.Image.open('%d.JPEG' % 1)
        send_data(s, data)
        data = receive_data(s)
        print('recieve: ', data)
    s.close()

if __name__ == '__main__':
    sock_client_image()