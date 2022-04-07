from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from util import *
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

form_class = uic.loadUiType("Mainwindow.ui")[0]


class Cutoff(QThread):

    # signal1 = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):

        path = self.parent.path_label.text()
        Weight = self.parent.weight_value.text()  # weight_value text 값 가져오기
        # percentile_value text 값 가져오기
        percentile = float(self.parent.percentile_value.text())

        start = time.time()
      
        self.parent.label.setText('load dicom....')

        Union_voxel = extract_voxel_data(path)
        flatten_arr= Union_voxel.flatten()
        voxel_mean, voxel_std = meanStd(Union_voxel)
        
        self.parent.label.setText('Calculating Normalization......')

        norm_arr, voxel_min, voxle_max = voxelNorm(Union_voxel)
        

        self.parent.label.setText('Calculating Otsu......')
        cutoff, fore_arr = Otsu(norm_arr)
        
        self.parent.label.setText('Calculating Intersection......')
        fore_arr = deNorm(fore_arr, voxel_min, voxle_max)
        fore_mean, fore_sd = meanStd(fore_arr)

        self.parent.label.setText('Calculating cutoff......')
        
        newton = newtonRaphson(float(Weight), fore_sd)
        
        
        p_cut = np.percentile(Union_voxel, percentile)
        p_norm = (p_cut - voxel_mean) / voxel_std
        
        fore_cut = newton*fore_sd+fore_mean
        fore_norm = (fore_cut - voxel_mean) / voxel_std
        
        
        duration_time = round((time.time() - start), 5)

        self.parent.label.setText('Done')

        self.parent.label_3.setText(str(duration_time))
        self.parent.SI_cut.setText(str(round(fore_cut, 5)))
        self.parent.SI_norm.setText(str(round(fore_norm, 5)))
        self.parent.P_cut.setText(str(round(p_cut, 5)))
        self.parent.P_norm.setText(str(round(p_norm, 5)))
        plt.clf()
    
        ax = self.parent.fig.add_subplot(121)

        ax.hist(flatten_arr, bins=500, label='normal', color='midnightblue')
        ax.axvline(round(p_cut, 2), color='red', label='x of percentile 99.5  ={:.3f}'.format(
            round(p_cut, 2)), linestyle='dashed', linewidth=1)
        ax.axvline(round(fore_cut, 2), color='blue', label='x of new_method  ={:.3f}'.format(
            round(fore_cut, 2)), linestyle='dashed', linewidth=1)
        ax.set_xlabel('Signal intensity')
        ax.set_ylabel('n')
        ax.legend(loc='upper left')

        bx = self.parent.fig.add_subplot(122)
        bx.hist(flatten_arr, bins=300, label='log',
                log=True, color='midnightblue')
        bx.axvline(round(p_cut, 2), color='red', label='x of percentile 99.5 ={:.3f}'.format(
            round(p_cut, 2)), linestyle='dashed', linewidth=1)
        bx.axvline(round(fore_cut, 2), color='blue', label='x of new_method  ={:.3f}'.format(
            round(fore_cut, 2)), linestyle='dashed', linewidth=1)
        bx.set_xlabel('Signal intensity')
        bx.set_ylabel('log(n)')
        bx.legend(loc='upper left')

        self.parent.canvas.draw()


class MyApp(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        # self.ui = uic.loadUi("./myapp.ui", self)
        self.setupUi(self)
        self._connectSignals()
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.path_label.setText(path)

        self.fig = plt.figure(figsize=(30, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.graph_v.addWidget(self.canvas)
        self.graph_v.addWidget(self.toolbar)


    def slot_fileopen(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.path_label.setText(path)

    def Analysis(self):
        x = Cutoff(self)
        x.start()

    def _connectSignals(self):
        self.pushButton_open.clicked.connect(self.slot_fileopen)
        self.Analysis_Button.clicked.connect(self.Analysis)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    me = MyApp()
    me.show()
    sys.exit(app.exec())
