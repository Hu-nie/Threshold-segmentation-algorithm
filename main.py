from PyQt5.QtWidgets import * 
from PyQt5 import uic 
from PyQt5.QtCore import *
from util import *
import sys
form_class = uic.loadUiType("Mainwindow.ui")[0]

class Thread1(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        # self.Analysis_Button.clicked.connect(self.run)
    def run(self): 
        
        path = self.parent.path_label.text() 
        Weight = self.parent.weight_value.text()  # weight_value text 값 가져오기
        percentile = float(self.parent.percentile_value.text())  # percentile_value text 값 가져오기
    
        start = time.time()
        normal = list()
        whole_arr = getResolution(path)
        self.parent.label.setText('load dicom....')
        

        for filename in tqdm(glob.glob(os.path.join(path, "*.dcm"))):
            img_arr = np.expand_dims(dicomToarray(filename), axis=0)
            whole_arr = np.concatenate((whole_arr, img_arr), axis=0)
        
        self.parent.label.setText('Calculating Normalization......')
        
        norm_arr, min_v, max_v = image_norm3D(whole_arr[1:])
        whole_arr_f = np.sort(whole_arr.flatten())
        p_cut = np.percentile(whole_arr_f, percentile)
        p_norm = (p_cut - whole_arr_f.mean()) / whole_arr_f.std()
        
        self.parent.label.setText('Calculating cutoff......')
        
        for arr in tqdm(norm_arr):
            _, t_otsu = cv2.threshold(arr, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            foresion = arr * (np.where(t_otsu == 255, 1, t_otsu))
            normal = normal + (foresion.flatten()).tolist() ## map 함수로 변환 가능
        
        self.parent.label.setText('Calculating Intersection......')
        normal, _, _, inter = getIntersection(int(Weight), normal)
        si_cut = deNormalization((inter.xy)[0][0], min_v, max_v)
        si_norm = (si_cut - whole_arr_f.mean()) / whole_arr_f.std()
        duration_time = round((time.time() - start), 2)

        self.parent.label.setText('Done')
        self.parent.label_3.setText(str(duration_time))
        self.parent.SI_cut.setText(str(round(si_cut, 2)))
        self.parent.SI_norm.setText(str(round(si_norm, 2)))
        self.parent.P_cut.setText(str(round(p_cut, 2)))
        self.parent.P_norm.setText(str(round(p_norm, 2)))

# class Thread2(QThread): 
#     def __init__(self, parent): 
#         super().__init__(parent) 
#         self.parent = parent

#     def slot_fileopen(self):
#         path = QFileDialog.getExistingDirectory(self, "Select Directory")
        
#         self.path_label.setText(path)
    
class MyApp(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        # self.ui = uic.loadUi("./myapp.ui", self)
        self.setupUi(self)
        self.Analysis_Button.clicked.connect(self.actionFunction1)
        self.pushButton_open.clicked.connect(self.slot_fileopen)
        

    def actionFunction1(self): 
        x = Thread1(self) 
        x.start()

    def slot_fileopen(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")

        self.path_label.setText(path)
        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    me = MyApp()
    me.show()
    sys.exit(app.exec())    
