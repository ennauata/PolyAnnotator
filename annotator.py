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
# Add internal libs
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, generateColorByText
from libs.settings import Settings
#from libs.polygon import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.toolBar import ToolBar
from libs.ustr import ustr
from libs.version import __version__

__appname__ = 'annotator'

class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        self.dataFolder = '../data/'

        # Create a widget for edit and diffc button
        # self.diffcButton = QCheckBox(u'difficult')
        # self.diffcButton.setChecked(False)
        # self.diffcButton.stateChanged.connect(self.btnstate)
        # self.editButton = QToolButton()
        # self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)


        self.zoomWidget = ZoomWidget()

        self.canvas = Canvas()
        #self.canvas.coordinatesChanged.connect(self.centerChanged)
        #self.canvas.reloadAnnotation.connect(self.reloadAnnotation)
        #self.canvas.zoomRequest.connect(self.zoomRequest)

        # scroll = QScrollArea()
        # scroll.setWidget(self.canvas)
        # scroll.setWidgetResizable(True)
        # self.setCentralWidget(scroll)
        self.setCentralWidget(self.canvas)
        #self.canvas.resize(self.canvas.sizeHint())

        action = partial(newAction, self)

        nextU = action('&NextU', self.moveToNextUnannotated,
                      'n', 'nextU', u'Move to next unannotated example')
        next = action('&Next', self.moveToNext,
                      'Ctrl+n', 'next', u'Move to next example')


        # Store actions for further handling.
        self.actions = struct(nextU=nextU, next=next)


        self.scenePaths = os.listdir(self.dataFolder)
        self.sceneIndex = -1
        self.moveToNextUnannotated()

        size = settings.get(SETTING_WIN_SIZE, QSize(640, 480))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)

        self.queueEvent(self.loadImage)


    def paintCanvas(self):
        #assert not self.image.isNull(), "cannot paint null image"
        self.canvas.adjustSize()
        self.canvas.update()
        return

    def moveToNextUnannotated(self):
        for sceneIndex, scenePath in enumerate(self.scenePaths[self.sceneIndex + 1:]):
            if not os.path.exists(self.dataFolder + '/' + scenePath + '/corners/'):
                self.scenePath = self.dataFolder + '/' + scenePath
                self.sceneIndex = self.sceneIndex + 1 + sceneIndex
                break
            continue
        return

    def moveToNext(self):
        self.sceneIndex = min(self.sceneIndex + 1, len(self.scenePaths))
        return

    def loadImage(self):
        self.canvas.loadScene(self.scenePath)
        self.paintCanvas()
        self.setWindowTitle(__appname__ + ' ' + self.scenePath)
        self.canvas.setFocus(True)
        return


    def queueEvent(self, function):
        QTimer.singleShot(0, function)
        return


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile
    win = MainWindow()
    win.show()
    return app, win


def main(argv=[]):
    '''construct main app and run it'''
    app, _win = get_main_app(argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
