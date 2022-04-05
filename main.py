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
        self.parent.label.setText('Calculating Normalization......')

        norm_arr, min_v, max_v,_, _ = voxelNorm(Union_voxel)
        Voxel_mean, Voxel_std = meanStd(Union_voxel)

        self.parent.label.setText('Calculating Otsu......')
        normal = Otsu(norm_arr)
        
        self.parent.label.setText('Calculating Intersection......')
        normal, n_pdf, g_pdf, inter = getIntersection(int(Weight), normal)

        self.parent.label.setText('Calculating cutoff......')
        
        p_cut = np.percentile(Union_voxel, percentile)
        p_norm = (p_cut - Voxel_mean) / Voxel_std
        
        si_cut = deNorm((inter.xy)[0][0],min_v,max_v)
        si_norm = (si_cut - Voxel_mean) / Voxel_std
        print(p_cut,inter.xy[0][0],min_v,max_v)
        
        duration_time = round((time.time() - start), 5)

        self.parent.label.setText('Done')

        self.parent.label_3.setText(str(duration_time))
        self.parent.SI_cut.setText(str(round(si_cut, 5)))
        self.parent.SI_norm.setText(str(round(si_norm, 5)))
        self.parent.P_cut.setText(str(round(p_cut, 5)))
        self.parent.P_norm.setText(str(round(p_norm, 5)))
        plt.clf()
    
        ax = self.parent.fig.add_subplot(131)

        ax.hist(flatten_arr, bins=500, label='normal', color='midnightblue')
        ax.axvline(round(p_cut, 2), color='red', label='x of percentile 99.5  ={:.3f}'.format(
            round(p_cut, 2)), linestyle='dashed', linewidth=1)
        ax.axvline(round(si_cut, 2), color='blue', label='x of new_method  ={:.3f}'.format(
            round(si_cut, 2)), linestyle='dashed', linewidth=1)
        ax.set_xlabel('Signal intensity')
        ax.set_ylabel('n')
        ax.legend(loc='upper left')

        bx = self.parent.fig.add_subplot(132)
        bx.hist(flatten_arr, bins=300, label='log',
                log=True, color='midnightblue')
        bx.axvline(round(p_cut, 2), color='red', label='x of percentile 99.5 ={:.3f}'.format(
            round(p_cut, 2)), linestyle='dashed', linewidth=1)
        bx.axvline(round(si_cut, 2), color='blue', label='x of new_method  ={:.3f}'.format(
            round(si_cut, 2)), linestyle='dashed', linewidth=1)
        bx.set_xlabel('Signal intensity')
        bx.set_ylabel('log(n)')
        bx.legend(loc='upper left')

        cx = self.parent.fig.add_subplot(133)
        cx.plot(normal, g_pdf, color='black')
        cx.plot(normal, int(Weight)*n_pdf, color='blue')
        cx.plot(*inter.xy, 'ro',
                label='point at x ={:.3f}'.format((inter.xy)[0][0]))
        cx.set_xlabel('Data points')
        cx.set_ylabel('Probability Density')
        cx.legend(loc='upper right')

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
