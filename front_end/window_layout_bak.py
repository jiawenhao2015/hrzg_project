# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setEnabled(True)
        mainWindow.resize(1524, 868)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainWindow.sizePolicy().hasHeightForWidth())
        mainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1380, 540, 101, 161))
        self.groupBox.setObjectName("groupBox")
        self.forwardButton = QtWidgets.QRadioButton(self.groupBox)
        self.forwardButton.setGeometry(QtCore.QRect(10, 40, 100, 20))
        self.forwardButton.setChecked(False)
        self.forwardButton.setObjectName("forwardButton")
        self.backwardButton = QtWidgets.QRadioButton(self.groupBox)
        self.backwardButton.setGeometry(QtCore.QRect(10, 130, 100, 20))
        self.backwardButton.setChecked(False)
        self.backwardButton.setObjectName("backwardButton")
        self.stopButton = QtWidgets.QRadioButton(self.groupBox)
        self.stopButton.setGeometry(QtCore.QRect(10, 80, 100, 20))
        self.stopButton.setChecked(False)
        self.stopButton.setObjectName("stopButton")
        self.head_result = QtWidgets.QLabel(self.centralwidget)
        self.head_result.setGeometry(QtCore.QRect(20, 80, 640, 360))
        self.head_result.setBaseSize(QtCore.QSize(640, 360))
        self.head_result.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.head_result.setObjectName("head_result")
        self.date = QtWidgets.QLabel(self.centralwidget)
        self.date.setGeometry(QtCore.QRect(1340, 80, 201, 31))
        self.date.setObjectName("date")
        self.tail_result = QtWidgets.QLabel(self.centralwidget)
        self.tail_result.setGeometry(QtCore.QRect(690, 80, 640, 360))
        self.tail_result.setBaseSize(QtCore.QSize(640, 360))
        self.tail_result.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tail_result.setObjectName("tail_result")
        self.head_ori = QtWidgets.QLabel(self.centralwidget)
        self.head_ori.setGeometry(QtCore.QRect(20, 470, 640, 360))
        self.head_ori.setBaseSize(QtCore.QSize(640, 360))
        self.head_ori.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.head_ori.setObjectName("head_ori")
        self.tail_ori = QtWidgets.QLabel(self.centralwidget)
        self.tail_ori.setGeometry(QtCore.QRect(690, 470, 640, 360))
        self.tail_ori.setBaseSize(QtCore.QSize(640, 360))
        self.tail_ori.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tail_ori.setObjectName("tail_ori")
        self.name = QtWidgets.QLabel(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(620, 10, 191, 41))
        self.name.setBaseSize(QtCore.QSize(640, 360))
        self.name.setFrameShape(QtWidgets.QFrame.Box)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setObjectName("name")
        self.enablesystemBox = QtWidgets.QCheckBox(self.centralwidget)
        self.enablesystemBox.setGeometry(QtCore.QRect(1370, 310, 121, 20))
        self.enablesystemBox.setObjectName("enablesystemBox")
        self.enablemanulBox = QtWidgets.QCheckBox(self.centralwidget)
        self.enablemanulBox.setGeometry(QtCore.QRect(1370, 500, 121, 20))
        self.enablemanulBox.setChecked(True)
        self.enablemanulBox.setObjectName("enablemanulBox")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1340, 160, 101, 16))
        self.label.setObjectName("label")
        self.adjust_cnt = QtWidgets.QLabel(self.centralwidget)
        self.adjust_cnt.setGeometry(QtCore.QRect(1450, 160, 60, 16))
        self.adjust_cnt.setObjectName("adjust_cnt")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1340, 210, 101, 16))
        self.label_2.setObjectName("label_2")
        self.is_ready = QtWidgets.QLabel(self.centralwidget)
        self.is_ready.setGeometry(QtCore.QRect(1450, 210, 60, 16))
        self.is_ready.setObjectName("is_ready")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1524, 24))
        self.menubar.setObjectName("menubar")
        self.menutmp = QtWidgets.QMenu(self.menubar)
        self.menutmp.setObjectName("menutmp")
        mainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menutmp.menuAction())

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "钢坯头尾定位系统"))
        self.groupBox.setTitle(_translate("mainWindow", "手动辊道控制"))
        self.forwardButton.setText(_translate("mainWindow", "前进"))
        self.backwardButton.setText(_translate("mainWindow", "后退"))
        self.stopButton.setText(_translate("mainWindow", "停止"))
        self.head_result.setText(_translate("mainWindow", "head_result"))
        self.date.setText(_translate("mainWindow", "date"))
        self.tail_result.setText(_translate("mainWindow", "tail_result"))
        self.head_ori.setText(_translate("mainWindow", "head_ori"))
        self.tail_ori.setText(_translate("mainWindow", "tail_ori"))
        self.name.setText(_translate("mainWindow", "钢坯头尾定位系统"))
        self.enablesystemBox.setText(_translate("mainWindow", "系统控制辊道"))
        self.enablemanulBox.setText(_translate("mainWindow", "手动控制辊道"))
        self.label.setText(_translate("mainWindow", "当前调整次数:"))
        self.adjust_cnt.setText(_translate("mainWindow", "0"))
        self.label_2.setText(_translate("mainWindow", "钢坯到位状态:"))
        self.is_ready.setText(_translate("mainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#fc0107;\">未到位</span></p></body></html>"))
        self.menutmp.setTitle(_translate("mainWindow", "设置"))
