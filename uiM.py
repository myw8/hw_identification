# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiM.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1507, 922)
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(20, 10, 451, 16))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(520, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 60, 591, 721))
        self.label.setObjectName("label")
        self.textEdit_2 = QtWidgets.QTextEdit(Form)
        self.textEdit_2.setGeometry(QtCore.QRect(670, 10, 421, 16))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(1110, 10, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(640, 60, 571, 721))
        self.label_2.setObjectName("label_2")
        '''self.textEdit_3 = QtWidgets.QTextEdit(Form)
        self.textEdit_3.setGeometry(QtCore.QRect(20, 810, 361, 21))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(430, 810, 151, 23))
        self.pushButton_3.setObjectName("pushButton_3")'''
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(200, 850, 131, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(400, 850, 461, 41))
        self.label_3.setObjectName("label_3")
        '''self.textEdit_4 = QtWidgets.QTextEdit(Form)
        self.textEdit_4.setGeometry(QtCore.QRect(630, 810, 371, 20))
        self.textEdit_4.setObjectName("textEdit_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(1050, 810, 141, 23))
        self.pushButton_5.setObjectName("pushButton_5")'''
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(1300, 850, 121, 51))
        self.pushButton_6.setObjectName("pushButton_6")

        self.retranslateUi(Form)

        self.pushButton.clicked.connect(self.bnt1_click)
        self.pushButton_2.clicked.connect(self.bnt2_click)
        #self.pushButton_3.clicked.connect(self.bnt3_click)
        #self.pushButton_5.clicked.connect(self.bnt5_click)
        #self.pushButton_6.clicked.connect(self.bnt6_click)
        self.pushButton_4.clicked.connect(self.bnt4_click)
        self.pushButton_6.clicked.connect(Form.close)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "笔迹鉴定(二)"))
        self.pushButton.setText(_translate("Form", "选择文件1"))
        self.label.setText(_translate("Form", "TextLabel"))
        self.pushButton_2.setText(_translate("Form", "选择文件2"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        #self.pushButton_3.setText(_translate("Form", "选择提取分类模型"))
        self.pushButton_4.setText(_translate("Form", "比较"))
        self.label_3.setText(_translate("Form", "结果"))
        #self.pushButton_5.setText(_translate("Form", "选择笔迹鉴定模型"))
        self.pushButton_6.setText(_translate("Form", "退出"))

