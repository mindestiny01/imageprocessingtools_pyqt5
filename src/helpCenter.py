# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'helpCenter.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HelpCenter(object):
    def setupUi(self, HelpCenter):
        HelpCenter.setObjectName("HelpCenter")
        HelpCenter.resize(504, 416)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../component/window/question.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        HelpCenter.setWindowIcon(icon)
        HelpCenter.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(57, 57, 57);")
        self.textEdit = QtWidgets.QTextEdit(HelpCenter)
        self.textEdit.setGeometry(QtCore.QRect(20, 40, 461, 361))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(HelpCenter)
        self.label.setGeometry(QtCore.QRect(20, 10, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Cambria Math")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(HelpCenter)
        QtCore.QMetaObject.connectSlotsByName(HelpCenter)

    def retranslateUi(self, HelpCenter):
        _translate = QtCore.QCoreApplication.translate
        HelpCenter.setWindowTitle(_translate("HelpCenter", "Help Center"))
        self.textEdit.setHtml(_translate("HelpCenter", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">FILE ACTION</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">New File =  CTRL+ N</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Save File =  CTRL + S</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Exit =  CTRL + Q</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">GEOMETRY</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">RESIZE =  ALT + R</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ZOOM IN =  SHIFT + =</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ZOOM OUT =  -</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ROTATION 90 DEGREE (CLOCKWISE) =  CTRL + UP ARROW</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ROTATION 90 DEGREE =  CTRL + DOWN ARROW</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ROTATION 45 DEGREE (CLOCKWISE) =  CTRL + K</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ROTATION 45 DEGREE =  CTRL + J</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">HELP CENTER = ALT + H</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">AUTHOR =  ALT + A</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">MARKDOWN = ALT + M</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">REFERENCES = ALT + P</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label.setText(_translate("HelpCenter", "Help Center"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    HelpCenter = QtWidgets.QWidget()
    ui = Ui_HelpCenter()
    ui.setupUi(HelpCenter)
    HelpCenter.show()
    sys.exit(app.exec_())
