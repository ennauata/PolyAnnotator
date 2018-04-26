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

        self.width = 600
        self.height = 600

        self.layout_width = 800
        self.layout_height = 800

        self.color_width = 640
        self.color_height = 640

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
        self.scale = 1.0
        self.margin_x = 0.25 * self.frameGeometry().width()
        self.margin_y = 0.25 * self.frameGeometry().height()
        return

    def mousePressEvent(self, ev):

        if ev.button() == Qt.LeftButton and self.images is not None:

            # disable click ouside of region
            raw_point = self.transformPos(ev.pos())
            if  self.margin_x < raw_point[0] < self.margin_x+self.width:
                if self.margin_y < raw_point[1] < self.margin_y+self.height: 
                    
                    boxCenter = np.array([self.margin_x+self.width/2.0, self.margin_y+self.height/2.0])
                    distFromCenter = (raw_point - boxCenter)*self.scale
                    point = self.center * self.imageSize[0] + distFromCenter

                    if self.mode == 'layout':
                        self.scene.addLayoutCorner(point)
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
                raw_point = self.transformPos(ev.pos())
                if  self.margin_x < raw_point[0] < self.margin_x+self.width:
                    if self.margin_y < raw_point[1] < self.margin_y+self.height: 
                        
                        boxCenter = np.array([self.margin_x+self.width/2.0, self.margin_y+self.height/2.0])
                        distFromCenter = (raw_point - boxCenter)*self.scale
                        point = self.center * self.imageSize[0] + distFromCenter
                        self.scene.moveCorner(point)
                        self.repaint()

        elif Qt.RightButton & ev.buttons():
            pos = self.transformPos(ev.pos(), moving=True)
            screen_size = np.array([int(self.frameGeometry().width()), int(self.frameGeometry().height())])
            deltas = (pos - np.array([self.prevPoint.x(), self.prevPoint.y()]))/np.array(screen_size)   
            deltas = QPointF(*deltas)


            # decrease speed if zoomed in
            if self.scale < 1:
                self.center[0] += -deltas.x()*self.scale**2
                self.center[1] += -deltas.y()*self.scale**2
            else:
                self.center[0] += -deltas.x()
                self.center[1] += -deltas.y()

            self.changeCoordinates()
            self.prevPoint = QPointF(*pos)
            self.repaint()
        return

    def wheelEvent(self, ev):
        if ev.orientation() == Qt.Vertical and self.ctrlPressed:
            if ev.delta() >= 0:
                self.scale -= 0.1
            else:
                self.scale += 0.1

            self.scale = min(self.scale, 10)
            self.scale = max(self.scale, 0.1)
            self.changeCoordinates()
            self.repaint()
        return

    def handleDrawing(self, pos):
        self.update()

    def selectCornerPoint(self, point):
        """Select the first corner created which contains this point."""
        self.deSelectCorner()
        for corner in reversed(self.corners):
            if corner.selectCorner(point, self.epsilon):
                self.selectCorner(corner)
                break
            continue
        return
    def paintEvent(self, event):
        if len(self.patchImages) == 0:
            return super(Canvas, self).paintEvent(event)
        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(self.margin_x, self.margin_y, self.patchImages[0])
        p.drawPixmap(self.width + self.margin_x + 20, self.margin_y, self.patchImages[1])

        if not self.hiding:
            if self.mode == 'layout':
                self.scene.paintLayout(p, self.width, self.height, self.center*self.imageSize[0], self.scale, self.margin_x, self.margin_y)
            else:
                self.scene.paintLayout(p, self.width, self.height, self.center*self.imageSize[0], self.scale, self.margin_x, self.margin_y)
            p.end()
        return

    def transformPos(self, point, moving=False):
        """Convert from widget-logical coordinates to painter-logical coordinates."""

        return np.array([float(point.x()), float(point.y())])


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
            self.moveToNextImage()
        elif key == Qt.Key_Up:
            self.moveToPreviousImage()
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

    def moveToNextImage(self):
        self.imagePaths = self.imagePaths[1:] + [self.imagePaths[0]]
        self.images = self.images[1:] + [self.images[0]]
        self.cropPatches()
        self.repaint()
        return

    def moveToPreviousImage(self):
        self.imagePaths = [self.imagePaths[-1]] + self.imagePaths[:-1]
        self.images = [self.images[-1]] + self.images[:-1]
        self.cropPatches()
        self.repaint()
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
            mins = np.array([float(self.width*self.scale)/(2*(self.imageSize[imageIndex][0]-1)), float(self.height*self.scale)/(2*(self.imageSize[imageIndex][1]-1))])
            maxs = np.array([1.0-float(self.width*self.scale)/(2*(self.imageSize[imageIndex][0]-1)), 1.0-float(self.height*self.scale)/(2*(self.imageSize[imageIndex][1]-1))])

            self.center = np.maximum(self.center, mins)
            self.center = np.minimum(self.center, maxs)

            # get coordinates for cropping
            lt = [int(self.center[0] * (self.imageSize[imageIndex][0]-1) - (self.width/2)*self.scale), int(self.center[1] * (self.imageSize[imageIndex][1]-1) - (self.height/2)*self.scale)]
            rb = [int(self.center[0] * (self.imageSize[imageIndex][0]-1) + (self.width/2)*self.scale), int(self.center[1] * (self.imageSize[imageIndex][1]-1) + (self.height/2)*self.scale)]

            patch = self.images[imageIndex][lt[1]: rb[1], lt[0]: rb[0]]
            cropped_im =  cv2.resize(patch, (self.width, self.height))
            self.patchImages.append(QPixmap(QImage(cropped_im, self.width, self.height, self.width * 3, QImage.Format_RGB888)))
        return

