# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_shift.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QLineEdit, QTextEdit

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1205, 718)
        self.MainWindow = MainWindow
        self.buttonBox = QtWidgets.QDialogButtonBox(MainWindow)
        self.buttonBox.setGeometry(QtCore.QRect(840, 670, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        # 设置graphicsView，简称为view
        # 统一大小
        self.viewHeight = 341
        self.viewWidth = 301
        # 设置显示窗口，一个一个添加
        self.views: list = []
        self.add_view(20, 50, self.viewHeight, self.viewWidth, name='input')
        self.add_view(440, 50, self.viewHeight, self.viewWidth, name='hidden1')
        self.add_view(850, 50, self.viewHeight, self.viewWidth, name='hidden2')
        self.add_view(20, 400, self.viewHeight, self.viewWidth, name='hidden3')
        self.add_view(440, 400, self.viewHeight, self.viewWidth, name='output')

        # 设置标签，并且用name作为其初始化内容，前7个是
        self.labels = []
        # for i in range(7):
        #     self.add_label(940, 391 + i * 31, 111, 31, name='Top%d' % (i+1))
        self.add_label(160, 10, 111, 31, name='原始图像')
        self.add_label(600, 10, 111, 31, name='隐藏层1')
        self.add_label(980, 10, 111, 31, name='隐藏层2')
        self.add_label(170, 360, 111, 31, name='隐藏层3')
        self.add_label(590, 360, 111, 31, name='输出层')
        self.add_label(980, 360, 111, 31, name='输出(Top7)')
        self.add_label(980, 550, 111, 31, name='位移坐标')
        # 设置结果的文本框(多文本框)
        self.textbox = QTextEdit(MainWindow)
        self.textbox.move(850, 400)
        self.textbox.resize(341, 150)
        l = []
        for i in range(7):
            l.append(['Top%d' % (i+1), [0, 0]])
        self.textbox.setPlainText(self.list2str(l))
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 显示位移的框
        self.indexbox = QTextEdit(MainWindow)
        self.indexbox.move(850, 580)
        self.indexbox.resize(341, 80)
        self.indexbox.setPlainText('(x, y) : (%d ,%d)' % (0, 0))
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

    # list的样式为[['name', [float, int]]]
    def list2str(self, l):
        s = ''
        for item in l:
            s = s + item[0] + '   '
            s = s + '{0:.6f}'.format(item[1][0]) + '   '
            s = s + '{:d}'.format(item[1][1]) + '\n'
        return s

    def add_view(self, x, y, h, w, index=0, name='label'):
        self.views.append(QtWidgets.QGraphicsView(self.MainWindow))
        self.views[-1].setGeometry(QtCore.QRect(x, y, h, w))
        if index != -1:
            self.views[-1].setObjectName("graphicsView_%d" % index)
        self.views[-1].setObjectName("graphicsView_%s" % name)

    def add_label(self, x, y, h, w, index=-1, name='label'):
        _translate = QtCore.QCoreApplication.translate
        self.labels.append(QtWidgets.QLabel(self.MainWindow))
        self.labels[-1].setGeometry(QtCore.QRect(x, y, h, w))
        if index != -1:
            self.labels[-1].setObjectName("Label_%d" % index)
        self.labels[-1].setObjectName("Label_%s" % name)
        # 设置标签内容，内容为name
        self.labels[-1].setText(_translate("Dialog", name))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())