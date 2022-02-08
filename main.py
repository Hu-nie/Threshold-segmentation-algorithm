import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
import os
from util import *
import glob
import cv2
from tqdm import tqdm
import time

class MyApp(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.ui = uic.loadUi("./myapp.ui", self)
        self.ui.show()
        self.get_value.clicked.connect(self.button_event)

    def button_event(self): 
        global text
        text = self.lineEdit.text() # line_edit text 값 가져오기 
        self.text_label.setText(text)

    def slot_fileopen(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Directory")
        print(type(fname) , fname)
        

        path = fname
        start = time.time()
        normal = list()
        set_v = int(text)
        whole_arr = getResolution(path)

        for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
            img_arr = np.expand_dims(dicomToarray(filename), axis=0)
            whole_arr = np.concatenate((whole_arr, img_arr), axis=0)


        norm_arr, min_v, max_v = image_norm3D(whole_arr[1:])
        whole_arr_f = np.sort(whole_arr.flatten())
        p_cut = np.percentile(whole_arr_f, 99.775)
        p_norm = (p_cut - whole_arr_f.mean()) / whole_arr_f.std()


        for arr in tqdm(norm_arr):
            _ , t_otsu = cv2.threshold(arr, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
            foresion = arr *(np.where(t_otsu == 255, 1, t_otsu))
            normal = normal + (foresion.flatten()).tolist()


        normal, _, _, inter = getIntersection(set_v,normal)
        si_cut = deNormalization((inter.xy)[0][0],min_v,max_v)
        si_norm = (si_cut - whole_arr_f.mean()) / whole_arr_f.std()
        duration_time = round((time.time() - start),2)
        
        self.ui.label_3.setText(str(duration_time))
        self.ui.SI_cut.setText(str(round(si_cut,2)))
        self.ui.SI_norm.setText(str(round(si_norm,2)))
        self.ui.P_cut.setText(str(round(p_cut,2)))
        self.ui.P_norm.setText(str(round(p_norm,2)))
        



app = QtWidgets.QApplication(sys.argv)
me = MyApp()
sys.exit(app.exec())