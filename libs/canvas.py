try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.lib import distance
import numpy as np
import cv2
from PIL import Image
import requests
import urllib
import sys
import glob
from scene import Scene
import os
import glob

sys.path.append('../code/')
Image.MAX_IMAGE_PIXELS = 1000000000

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

# class Canvas(QGLWidget):


class Canvas(QWidget):
    newCorner = pyqtSignal()
    cornerMoved = pyqtSignal()
    drawing = pyqtSignal(bool)

    CREATE, EDIT = list(range(2))
    image = None
    coordinatesChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.

        self.prevPoint = QPointF()
        self._painter = QPainter()
        self.sceneDir = './scenes'

        if not os.path.exists(self.sceneDir):
            os.mkdir(self.sceneDir)

        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        self.width = 400
        self.height = 400

        self.layout_width = 800
        self.layout_height = 800

        self.color_width = 640
        self.color_height = 640

        self.offsetX = 10
        self.offsetY = 10

        self.currentLabel = 0
        self.hiding = False
        self.resize(self.width, self.height)
        self.imageIndex = -1
        self.mode = 'layout'
        print('Layout Mode')
        self.ctrlPressed = False
        self.shiftPressed = False
        self.imagePaths = []
        self.images = []
        self.patchImages = []
        return

    def mousePressEvent(self, ev):

        if ev.button() == Qt.LeftButton and self.image is not None:
            point = self.transformPos(ev.pos())
            if self.mode == 'layout':
                self.scene.addLayoutCorner(point, self.layout_width, self.layout_height, selectPlane=self.ctrlPressed)
            elif self.mode == 'move':
                self.scene.selectCorner(point)
            self.prevPoint = point
            self.repaint()

        if Qt.RightButton & ev.buttons():
            pos = self.transformPos(ev.pos(), moving=True)
            self.prevPoint = QPointF(*pos)
        return

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""

        if (Qt.LeftButton & ev.buttons()):
            if self.mode == 'move':
                point = self.transformPos(ev.pos())
                self.scene.moveCorner(point)
                self.repaint()

        elif Qt.RightButton & ev.buttons():
            pos = self.transformPos(ev.pos(), moving=True)
            screen_size = np.array([int(self.frameGeometry().width()), int(self.frameGeometry().height())])
            deltas = (pos - np.array([self.prevPoint.x(), self.prevPoint.y()]))/np.array(screen_size)   
            deltas = QPointF(*deltas)
            self.center[0] += -deltas.x()
            self.center[1] += -deltas.y()
            self.changeCoordinates()
            self.prevPoint = QPointF(*pos)
            self.repaint()
        return

    def wheelEvent(self, ev):
        # qt_version = 4 if hasattr(ev, "delta") else 5
        # if qt_version == 4:
        #     if ev.orientation() == Qt.Vertical:
        #         v_delta = ev.delta()
        #         h_delta = 0
        #     else:
        #         h_delta = ev.delta()
        #         v_delta = 0
        # else:
        #     delta = ev.angleDelta()
        #     h_delta = delta.x()
        #     v_delta = delta.y()
        #     pass
        # self.scene.adjustHeight(v_delta, self.ctrlPressed)
        # self.repaint()
        return

    def handleDrawing(self, pos):
        self.update()

    def selectCornerPoint(self, point):
        """Select the first corner created which contains this point."""
        self.deSelectCorner()
        for corner in reversed(self.corners):
            if corner.selectCorner(point, self.epsilon):
                self.selectCorner(corner)
                #self.calculateOffsets(corner, point)
                break
            continue
        return

    # def paintEvent(self, event):
    #     if self.image is not None:
    #         if (self.imageIndex == -1 or not self.image) and self.mode != 'layout':
    #             return super(Canvas, self).paintEvent(event)
    #         p = self._painter
    #         p.begin(self)
    #         p.setRenderHint(QPainter.Antialiasing)
    #         p.setRenderHint(QPainter.HighQualityAntialiasing)
    #         p.setRenderHint(QPainter.SmoothPixmapTransform)

    #         if self.image is not None:
    #             p.drawPixmap(self.offsetX, self.offsetY, self.image)

    #         if not self.hiding:
    #             if self.mode == 'layout':
    #                 self.scene.paintLayout(p, self.layout_width, self.layout_height, self.offsetX, self.offsetY)
    #             else:
    #                 self.scene.paintLayout(p, self.layout_width, self.layout_height, self.offsetX, self.offsetY)

    #         p.end()
    #     return

    def paintEvent(self, event):
        if len(self.patchImages) == 0:
            return super(Canvas, self).paintEvent(event)
        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        margin_x = 0.1 * self.frameGeometry().width()
        margin_y = 0.2 * self.frameGeometry().height()
        p.drawPixmap(margin_x, margin_y, self.patchImages[0])
        p.drawPixmap(self.width + margin_x + 20, margin_y, self.patchImages[1])

        # if self.image is not None:
        #     p.drawPixmap(self.offsetX, self.offsetY, self.image)

        if not self.hiding:
            if self.mode == 'layout':
                self.scene.paintLayout(p, self.layout_width, self.layout_height, self.offsetX, self.offsetY)
            else:
                self.scene.paintLayout(p, self.layout_width, self.layout_height, self.offsetX, self.offsetY)
            p.end()
        return

    def transformPos(self, point, moving=False):
        """Convert from widget-logical coordinates to painter-logical coordinates."""

        return np.array([float(point.x() - self.offsetX), float(point.y() - self.offsetY)])


    def closeEnough(self, p1, p2):
        return distance(p1 - p2) < self.epsilon


    def keyPressEvent(self, ev):
        key = ev.key()
        if (ev.modifiers() & Qt.ControlModifier):
            self.ctrlPressed = True
            if self.hiding:
                self.repaint()
                pass
        else:
            self.ctrlPressed = False
            pass
        if (ev.modifiers() & Qt.ShiftModifier):
            self.shiftPressed = True
            if self.hiding:
                self.repaint()
                pass
        else:
            self.shiftPressed = False
            pass

        if key == Qt.Key_D:
            self.scene.disconnectGraph()
        if key == Qt.Key_Delete:
            if self.mode == 'move':
                self.scene.removeCorner()
                self.repaint()
        if key == Qt.Key_Z:
            if self.ctrlPressed:
                self.scene.removeLast()
                self.repaint()
                pass
        elif key == Qt.Key_R:
            if self.ctrlPressed:
                self.scene.reset('init')
                self.repaint()
                pass
        elif key == Qt.Key_H:
            #and Qt.ControlModifier == int(ev.modifiers()):
            self.hiding = not self.hiding
            self.repaint()
        elif key == Qt.Key_A:
            print(self.mode)
            #if self.mode != 'layout':
            if self.mode == 'layout':
                self.mode = 'move'
                print('Move Mode')
            else:
                self.mode = 'layout'
                print('Layout Mode')
            print(self.mode)
        elif key == Qt.Key_S:
            if self.ctrlPressed:
                print('save')
                self.scene.save()
        elif key == Qt.Key_Right:
            self.moveToNextImage()
        elif key == Qt.Key_Left:
            self.moveToPreviousImage()
        elif key == Qt.Key_Down:
            self.moveToNextImage(5)
        elif key == Qt.Key_Up:
            self.moveToPreviousImage(5)
        return

    def keyReleaseEvent(self, ev):
        if self.hiding and self.ctrlPressed:
            self.repaint()
            pass
        self.ctrlPressed = False
        self.shiftPressed = False
        return

    def setCurrentLabel(self, label):
        self.currentLabel = label
        return

    def loadCorners(self, corners):
        self.corners = list(corners)
        self.current = None
        self.currentGroup = currentGroup
        self.repaint()

    def onPoint(self, pos):
        for corner in self.corners:
            if corner.nearestVertex(pos, self.epsilon) is not None:
                return True
            continue
        return False

    def moveToNextImage(self, delta=1):
        self.imageIndex = min(self.imageIndex + delta, len(self.imagePaths) - 1)
        self.loadImage()
        return

    def moveToPreviousImage(self, delta=1):
        self.imageIndex = max(self.imageIndex - delta, 0)
        self.loadImage()
        return

    def loadImage(self):
        for imagePath in self.imagePaths:
            self.images.append(cv2.imread(imagePath))

        if len(self.images) > 0:
            scenePath = self._getScenePath(self.imagePaths[0])
            self.scene = Scene(scenePath)
            self.imageSize = np.array([image.shape[:2] for image in self.images])
            self.imageSize = np.stack([self.imageSize[:, 1], self.imageSize[:, 0]], axis=1)
            self.initCoordinatesRandomly()
            self.repaint()
        return

    def _getScenePath(self, imPath):
        filename = os.path.basename(imPath)
        filename_no_ext, _ = os.path.splitext(filename)
        scene_name = filename_no_ext + '_scene.npy'
        scene_path = os.path.join(self.sceneDir, scene_name)
        return scene_path

    def save(self):
        self.scene.save()

    def load(self):
        self.scene.load()
        self.repaint()

    def removeLastPoint(self):
        self.corners = self.corners[:-1]
        self.repaint()
        return

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.image:
            return self.image.size()
        return super(Canvas, self).minimumSizeHint()

    def setCenter(self, center):
        self.center = np.array(center)
        self.changeCoordinates()
        return

    def initCoordinatesRandomly(self):
        self.center = np.array([0.5, 0.5])
        self.changeCoordinates()
        return

    def changeCoordinates(self):

        self.cropPatches()
        self.coordinatesChanged.emit()
        return

    def cropPatches(self, background = False):

        self.patchImages = []
        for imageIndex in xrange(2):

            # prevent overflow
            mins = np.array([float(self.width)/(2*(self.imageSize[imageIndex][0]-1)), float(self.height)/(2*(self.imageSize[imageIndex][1]-1))])
            maxs = np.array([1.0-float(self.width)/(2*(self.imageSize[imageIndex][0]-1)), 1.0-float(self.height)/(2*(self.imageSize[imageIndex][1]-1))])

            self.center = np.maximum(self.center, mins)
            self.center = np.minimum(self.center, maxs)

            # get coordinates for cropping
            lt = [int(self.center[0] * (self.imageSize[imageIndex][0]-1) - self.width/2), int(self.center[1] * (self.imageSize[imageIndex][1]-1) - self.height/2)]
            rb = [int(self.center[0] * (self.imageSize[imageIndex][0]-1) + self.width/2), int(self.center[1] * (self.imageSize[imageIndex][1]-1) + self.height/2)]

            patch = self.images[imageIndex][lt[1]: rb[1], lt[0]: rb[0]]
            cropped_im =  cv2.resize(patch, (self.width, self.height))
            self.patchImages.append(QPixmap(QImage(cropped_im, self.width, self.height, self.width * 3, QImage.Format_RGB888)))
        return

