import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
from main import *
from PyQt5.QtCore import *

form_secondwindow = uic.loadUiType("secondwindow.ui")[0] #두 번째창 ui
class secondwindow(QDialog,QWidget,form_secondwindow):

    def __init__(self):
        super(secondwindow,self).__init__()
        self.setupUi(self)
        self.show() # 두번째창 실행
