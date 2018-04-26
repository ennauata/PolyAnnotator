import sys
import os
from functools import partial
import numpy as np

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import resources
import os
# Add internal libs
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, generateColorByText
from libs.settings import Settings
# from libs.polygon import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.toolBar import ToolBar
from libs.ustr import ustr
from libs.version import __version__
import glob

__appname__ = 'annotator'


class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # init annotation dir
        self.annotDir = './annotations'
        if not os.path.isdir(self.annotDir):
            os.mkdir(self.annotDir)
            print('create dir {} for saving annotations'.format(self.annotDir))

        # init image dir, wait to be set by user in self.loadImagefolder
        self.imageDir = None

        self.zoomWidget = ZoomWidget()
        scroll = QScrollArea()
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        # add scroll widget
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        self.canvas.resize(self.canvas.sizeHint())

        # add and populate menubar
        bar = self.menuBar()
        file = bar.addMenu("File")
        loadDir = QAction("Open Image Folder", self)
        save = QAction("Save Annotation", self)
        # load = QAction("Load Annotation", self)
        quit = QAction("Close Annotator", self)

        file.addAction(save)
        # file.addAction(load)
        file.addAction(loadDir)
        file.addAction(quit)

        # set actions
        loadDir.triggered.connect(self.loadImageFolder)
        save.triggered.connect(self.save)
        # load.triggered.connect(self.canvas.load)
        quit.triggered.connect(self.quitApp)

        self.loadedFiles = []

        # add navigation sidebar 
        self.items = QDockWidget("Annotations", self)
        self.annotList = QListWidget()
        self.updateAnnotationDock()
        self.annotList.itemActivated.connect(self.updateActivatedAnnotation)
        self.items.setWidget(self.annotList)
        self.items.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.items)

        size = settings.get(SETTING_WIN_SIZE, QSize(640, 480))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)

        self.queueEvent(self.loadImage)

    def quitApp(self, q):
        QApplication.quit()

    def loadImageFolder(self, q):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', '~/', QFileDialog.ShowDirsOnly)
        self.imageDir = dir_
        self.loadedFiles = sorted(os.listdir(dir_))
        # self.updateDock()        
        if dir_ != '':
            pathofLoadedFiles = [os.path.join(str(dir_), filename) for filename in self.loadedFiles]

            self.canvas.imagePaths = pathofLoadedFiles
            self.canvas.imageIndex = 0
            self.canvas.loadImage(self.annotDir)
        return

    def save(self):
        self.canvas.save()
        self.updateAnnotationDock()

    def updateAnnotationDock(self):
        self.annotList.clear()
        self.annotFiles = sorted(os.listdir(self.annotDir))
        self.annotPaths = [os.path.join(self.annotDir, filename) for filename in self.annotFiles]
        for filename in self.annotFiles:
            self.annotList.addItem(filename)

    def updateActivatedAnnotation(self, item):
        activated_filename = item.text()
        self.canvas.updateActivatedAnnotation(activated_filename)

    def paintCanvas(self):
        # assert not self.image.isNull(), "cannot paint null image"
        self.canvas.adjustSize()
        self.canvas.update()
        return

    def loadImage(self):
        self.canvas.loadImage(self.annotDir)
        self.paintCanvas()
        self.setWindowTitle(__appname__)
        self.canvas.setFocus(True)
        return

    # def loadActivatedImage(self, item):
    #     item_idx = self.loadedFiles.index(item.text())
    #     self.canvas.imageIndex = item_idx
    #     self.canvas.loadImage()


    def queueEvent(self, function):
        QTimer.singleShot(0, function)
        return


def get_main_app(argv=[]):
    # main loop
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    win = MainWindow()
    win.show()
    return app, win


def main(argv=[]):
    # construct app and run it
    app, _win = get_main_app(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
