from PyQt5.QtCore import pyqtSlot, QPointF, QPoint,QRectF, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, QtWidgets
import numpy as np
import cv2
from image_shift import Ui_MainWindow
from socket_utils import *
from PIL import Image
import PIL as pil

class picturezoom(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        # 连接套接字
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(('10.134.171.252', 6666))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
            # s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
        except socket.error as msg:
            print(msg)
            print(sys.exit(1))

        super(picturezoom, self).__init__(parent)
        self.setupUi(self)
        # 总共的图片个数
        self.picNum = 5
        # 结果列表
        self.result_list = []
        for i in range(7):
            self.result_list.append(['Top%d' % (i+1), 0])
        # 位移
        self.shift_x = 0
        self.shift_y = 0
        # 创建像素图元
        self.items = []
        for i in range(self.picNum):
            self.items.append(QGraphicsPixmapItem())
        # 创建场景
        self.scenes = []
        for i in range(self.picNum):
            # 由于内边距的存在，需要减去2个像素
            self.scenes.append(QGraphicsScene(QRectF(0, 0, self.viewHeight-2, self.viewWidth-2)))  # 创建场景
            self.scenes[i].addItem(self.items[i])
            self.views[i].setScene(self.scenes[i])  # 将场景添加至视图

        # 绑定图像到item上面
        self.images = [] 
        img = cv2.imread("203.JPEG")  # 读取图像
        self.input_img = cv2.resize(img, (self.viewHeight, self.viewWidth), interpolation=cv2.INTER_CUBIC)
        self.images.append(self.input_img)
        self.update_image(self.images[0], self.items[0])
        self.update_result()

    # 和服务器交互，更新结果
    def update_result(self):
        send_data(self.sock, self.input_img)
        data = receive_data(self.sock)
        print('result: ', data)
        result = data[0]
        # 生成top7列表
        for i in range(7):
            if i >= len(result):
                self.result_list[i][1] = [0, 0]
                continue
            self.result_list[i][1] = result[i]
        s = self.list2str(self.result_list)
        self.textbox.setPlainText(s)
        # 更新显示位移
        self.indexbox.setPlainText('(x, y) : (%d ,%d)' % (self.shift_x, self.shift_y))
        # 更新图片
        imgs = data[1]
        for (i, img) in enumerate(imgs):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            print(img.shape)
            img = cv2.resize(img, (self.viewHeight, self.viewWidth), interpolation=cv2.INTER_CUBIC)
            print(img.shape)
            self.update_image(img, self.items[i + 1])

    # 更新图像到item上面
    # 输入是cv读入的img，是BGR形式
    def update_image(self, img, item):
        # 转换图像通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获取图像大小
        x = img.shape[1]
        y = img.shape[0]
        print('image size: %d, %d' % (x, y))
        zoomscale = 1  # 图片放缩尺度
        # 第三个参数是用来对齐的，为什么写成这个样子未知
        frame = QImage(img, x, y, x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item.setPixmap(pix)
        # self.item.setScale(self.zoomscale)
        # 设置item在scene中的坐标系原点，默认为0,0
        start_point = QPoint(0, 0)
        item.setPos(start_point)

    def update_label(self, label_content: str, label):
        _translate = QtCore.QCoreApplication.translate
        label.setText(_translate("Dialog", label_content))

    # 根据按键来移动图像，只移动原始图像（第一张图片）
    def keyPressEvent(self, event):
        shift_x = 0
        shift_y = 0
        big_step = 7
        small_step = 1
        # 位移大的移动
        if event.key() == Qt.Key_W:
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                print("press shift + up")
                shift_y = shift_y - small_step
            else:
                print('press up')
                shift_y = shift_y - big_step
        if event.key() == Qt.Key_S:
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                print("press shift + down")
                shift_y = shift_y + small_step
            else:
                print('press down')
                shift_y = shift_y + big_step
        if event.key() == Qt.Key_D:
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                print("press shift + right")
                shift_x = shift_x + small_step
            else:
                print('press Right')
                shift_x = shift_x + big_step
        if event.key() == Qt.Key_A:
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                print("press shift + left")
                shift_x = shift_x - small_step
            else:
                print('press Left')
                shift_x = shift_x - big_step
        # 更新总体位移
        self.shift_y = self.shift_y + shift_y
        self.shift_x = self.shift_x + shift_x
        # 对输入图像进行位移，更新到item，并且将input_img更新
        self.input_img = self.translate(self.images[0], self.shift_x, self.shift_y)
        self.update_image(self.input_img, self.items[0])
        self.update_result()

    # 对image平移
    def translate(self, image, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return shifted

def main():
    import sys
    app = QApplication(sys.argv)
    piczoom = picturezoom()
    piczoom.show()
    app.exec_()


if __name__ == '__main__':
    main()
    # while True:
    #     # Path = int(input('Pause:'))
    #     data = pil.Image.open('%d.JPEG' % 1)
    #     send_data(s, data)
    #     data = receive_data(s)
    #     print('recieve: ', data)
    # s.close()
