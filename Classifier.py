# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Classifier.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QMainWindow

from spider import *
import sys
from PIL import Image, ImageDraw, ImageFont
import time
import threading
import tensorflow as tf
import numpy as np
from spider import *
from ResNet18 import ResNet18

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(0, 60, 801, 171))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.comboBox = QtWidgets.QComboBox(self.formLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.confirmButton = QtWidgets.QPushButton(self.formLayoutWidget)
        self.confirmButton.setObjectName("pushButton")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.confirmButton)
        self.quitButton = QtWidgets.QPushButton(self.formLayoutWidget)
        self.quitButton.setObjectName("pushButton_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.quitButton)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.comboBox_2 = QtWidgets.QComboBox(self.formLayoutWidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_2)
        self.analyseButton = QtWidgets.QPushButton(self.formLayoutWidget)
        self.analyseButton.setObjectName("pushButton_3")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.analyseButton)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 260, 631, 292))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imageTitle = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.imageTitle.setFont(font)
        self.imageTitle.setObjectName("label_3")
        self.verticalLayout.addWidget(self.imageTitle)
        self.imageBox = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageBox.sizePolicy().hasHeightForWidth())
        self.imageBox.setSizePolicy(sizePolicy)
        self.imageBox.setText("")
        # self.imageBox.setPixmap(QtGui.QPixmap("截屏2021-08-12 16.29.26 2.png"))
        self.imageBox.setScaledContents(False)
        self.imageBox.setObjectName("label_4")
        self.verticalLayout.addWidget(self.imageBox)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(250, 10, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Songti SC")
        font.setPointSize(26)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setRange(0, 100)
        self.progressBar.setGeometry(QtCore.QRect(50, 230, 700, 20))
        self.progressBar.setObjectName("label_7")
        # self.progressBar.setStyleSheet("border: 10px 0px 0px 0px")
        self.upButton = QtWidgets.QPushButton(self.centralwidget)
        self.upButton.setGeometry(QtCore.QRect(690, 460, 113, 32))
        self.upButton.setObjectName("pushButton_4")
        self.downButton = QtWidgets.QPushButton(self.centralwidget)
        self.downButton.setGeometry(QtCore.QRect(690, 500, 113, 32))
        self.downButton.setObjectName("pushButton_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.quitButton.clicked.connect(MainWindow.close)
        self.confirmButton.clicked.connect(MainWindow.repaint)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.retranslateUi(MainWindow)
        self.quitButton.clicked.connect(MainWindow.close)
        self.confirmButton.clicked.connect(self.runSpider)
        self.downButton.clicked.connect(self.nextImage)
        self.upButton.clicked.connect(self.previousImage)
        self.analyseButton.clicked.connect(self.analyse)

        self.outputFolder = None
        self.displayList = []
        self.displayCount = 0

        self.model = ResNet18([2, 2, 2, 2])
        checkpoint_save_path = "./TextDis_benchmark/ResNet18/ResNet18.ckpt"
        self.model.load_weights(checkpoint_save_path)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "请输入关键词："))
        self.label_2.setText(_translate("MainWindow", "选择搜索引擎："))
        self.comboBox.setItemText(0, _translate("MainWindow", "百度"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Unsplash"))
        self.comboBox.setItemText(2, _translate("MainWindow", "全景"))
        self.confirmButton.setText(_translate("MainWindow", "确认"))
        self.quitButton.setText(_translate("MainWindow", "退出"))
        self.label_6.setText(_translate("MainWindow", "请选择已下载图片关键词："))
        self.analyseButton.setText(_translate("MainWindow", "开始分析"))
        # self.imageTitle.setText(_translate("MainWindow", "图片预览（标题）"))
        self.label_5.setText(_translate("MainWindow", "快速文本/非文本监测系统"))
        # self.progressBar.setText(_translate("MainWindow", "进度条"))
        self.upButton.setText(_translate("MainWindow", "上一张"))
        self.downButton.setText(_translate("MainWindow", "下一张"))

        self.comboBox_2.clear()
        history = []
        for dirname, _, filenames in os.walk('./spider/'):
            keyword = dirname.split("/")[-1]
            if keyword != "":
                history.append(keyword)
        for idx, keyword in enumerate(history):
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(idx, _translate("MainWindow", keyword))
        ImgPath = "./spider/cnn/untitled-1.jpg"
        img = QtGui.QImageReader(ImgPath)
        width = img.size().width()
        height = img.size().height()
        img.setScaledSize(QtCore.QSize(int(266 / height * width), 266))
        img = QtGui.QImage(img.read())
        pix = QtGui.QPixmap(img)
        self.imageBox.setPixmap(pix)

    def nextImage(self):
        if len(self.displayList) == 0:
            return
        self.displayCount += 1
        if self.displayCount >= len(self.displayList):
            self.displayCount = 0
        ImgPath = self.displayList[self.displayCount]
        img = QtGui.QImageReader(ImgPath)
        width = img.size().width()
        height = img.size().height()
        img.setScaledSize(QtCore.QSize(int(266/height*width), 266))
        img = QtGui.QImage(img.read())
        pix = QtGui.QPixmap(img)
        self.imageBox.setPixmap(pix)
        _translate = QtCore.QCoreApplication.translate
        img_name = ImgPath.split("/")[-1]
        self.imageTitle.setText(_translate("MainWindow", "text detected in picture " + img_name if ImgPath.split("/")[-2] == "text" else "text not detected in picture " + img_name))


    def previousImage(self):
        if len(self.displayList) == 0:
            return
        self.displayCount -= 1
        if self.displayCount < 0:
            self.displayCount = len(self.displayList) - 1
        # self.label_4.setPixmap(QtGui.QPixmap(self.displayList[self.displayCount]))
        ImgPath = self.displayList[self.displayCount]
        img = QtGui.QImageReader(ImgPath)
        width = img.size().width()
        height = img.size().height()
        img.setScaledSize(QtCore.QSize(int(266 / height * width), 266))
        img = QtGui.QImage(img.read())
        pix = QtGui.QPixmap(img)
        self.imageBox.setPixmap(pix)
        _translate = QtCore.QCoreApplication.translate
        img_name = ImgPath.split("/")[-1]
        self.imageTitle.setText(_translate("MainWindow", "text detected in picture " + img_name if ImgPath.split("/")[-2] == "text" else "text not detected in picture " + img_name))

    def runSpider(self):
        self.progressBar.setRange(0, 0)
        spider = threading.Thread(target=self.Spider_Sum)
        spider.start()
        time.sleep(0.1)
        history = []
        self.comboBox_2.clear()
        for dirname, _, filenames in os.walk('./spider/'):
            keyword = dirname.split("/")[-1]
            if keyword != "":
                history.append(keyword)
        print(history)
        _translate = QtCore.QCoreApplication.translate
        for idx, keyword in enumerate(history):
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(idx, _translate("MainWindow", keyword))


    def analyse(self):
        self.displayList = []
        keyword = self.comboBox_2.currentText()
        if keyword == "":
            return
        path = "./spider/" + keyword + "/"
        self.outputFolder = path
        text_path = "./result/" + keyword + "/text/"
        nonText_path = "./result/" + keyword + "/nonText/"
        if not os.path.exists("./result/" + keyword):
            os.mkdir("./result/" + keyword)
        if not os.path.exists(text_path):
            os.mkdir(text_path)
        if not os.path.exists(nonText_path):
            os.mkdir(nonText_path)
        img_names = []
        for dirname, _, filenames in os.walk(path):
            img_names += filenames
        _translate = QtCore.QCoreApplication.translate
        total = len(img_names)
        current = 1
        for img_name in img_names:
            try:
                img_orig = Image.open(path + img_name)  # 读入图片
            except:
                continue
            img = img_orig.resize((224, 224))
            img = np.array(img.convert("L"))
            img = img[tf.newaxis, ...]
            img = np.reshape(img, (1, 224, 224, 1))
            img = img / 255.
            result = self.model.predict(img)
            print(result)
            result = tf.argmax(result, axis=1)
            # img_out = Image.new('RGB', (img_orig.size[0], img_orig.size[1] + 35), (20, 136, 173))
            # img_out.paste(img_orig, (0, 0))
            # font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
            # draw = ImageDraw.Draw(img_out)
            # img_out = img_orig
            if result:
                print("\x1b[35mtext detected")
                # draw.text((10, img_orig.size[1] + 5), "text detected", (255, 255, 255), font=font)
                # img_out.save(text_path + img_name)
                os.system("cp " + "'" + path + img_name + "'" + " " + text_path)
                self.displayList.append(text_path + img_name)
            else:
                print("\x1b[35mtext not detected")
                # draw.text((10, img_orig.size[1] + 5), "text not detected", (255, 255, 255), font=font)
                # img_out.save(nonText_path + img_name)
                os.system("cp " + "'" + path + img_name + "'" + " " + nonText_path)
                self.displayList.append(nonText_path + img_name)
            # poundCount = int(current/total*80)
            # bar = "|"+"#"*poundCount+"_"*(80-poundCount)+"|"+"{}/{}".format(current, total)
            # print(bar)
            self.progressBar.setValue(int(100*current/total))
            app.processEvents()
            current += 1
        self.displayCount = 0
        ImgPath = self.displayList[self.displayCount]
        img = QtGui.QImageReader(ImgPath)
        width = img.size().width()
        height = img.size().height()
        img.setScaledSize(QtCore.QSize(int(266 / height * width), 266))
        img = QtGui.QImage(img.read())
        pix = QtGui.QPixmap(img)
        self.imageBox.setPixmap(pix)
        img_name = ImgPath.split("/")[-1]
        self.imageTitle.setText(_translate("MainWindow", "text detected in picture " + img_name if ImgPath.split("/")[-2] == "text" else "text not detected in picture " + img_name))
        # self.imageBox.setPixmap(QtGui.QPixmap(self.displayList[self.displayCount]))

    def Spider_Sum(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/60.0.3112.113 Safari/537.36 '
        }

        save_path = "./spider/"
        if not os.path.exists(save_path[:-1]):
            os.mkdir(save_path[:-1])

        keyword = self.lineEdit.text()

        # 选择爬取的网站 UnsplashSpider/BaiduSpider/QuanjingSpider
        if self.comboBox.currentText() == '百度':
            spider = BaiduSpider(keyword=keyword, headers=headers, save_path=save_path)
        if self.comboBox.currentText() == 'Unsplash':
            spider = UnsplashSpider(keyword=keyword, headers=headers, save_path=save_path)
        if self.comboBox.currentText() == '全景':
            spider = QuanjingSpider(keyword=keyword, headers=headers, save_path=save_path)

        spider.run()
        self.progressBar.setRange(0, 100)
        # app.processEvents()
        time.sleep(0.01)
        self.progressBar.setValue(100)


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建一个QApplication
    mainWindow = QMainWindow()  # 创建一个主窗口
    ui = Ui_MainWindow()  # 创建一个类
    ui.setupUi(mainWindow)  # 在指定的窗口内添加控件
    mainWindow.show()
    sys.exit(app.exec_())