try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

#from PyQt4.QtOpenGL import *

#from libs.corner import Corner
#from libs.corner import Corner
from libs.lib import distance
import numpy as np
import cv2
from PIL import Image
import requests
#import StringIO
import urllib
import sys
import glob
from scene import Scene
import os

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

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.

        self.prevPoint = QPointF()
        self._painter = QPainter()
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        self.width = 1280
        self.height = 960

        self.layout_width = 1000
        self.layout_height = 1000

        self.offsetX = 10
        self.offsetY = 10

        self.currentLabel = 0
        self.hiding = False
        self.resize(self.width, self.height)
        self.imageIndex = -1
        self.mode = 'layout'
        self.ctrlPressed = False
        self.shiftPressed = False
        return

    # def creating(self):
    #     return self.mode == self.CREATE

    # def editing(self):
    #     return self.mode == self.EDIT

    # def setMode(self, value=True):
    #     self.mode = self.CREATE if value else self.EDIT
    #     if value:  # Create
    #         self.unHighlight()
    #         self.deSelectCorner()
    #     self.prevPoint = QPointF()
    #     self.repaint()


    def readDepth(self, point):
        u = point[0] / self.color_width * self.depth_width
        v = point[1] / self.color_height * self.depth_height
        return self.depth[int(round(v)), int(round(u))]

    def mousePressEvent(self, ev):
        #print(self.drawing(), pos)
        if ev.button() == Qt.LeftButton:
            point = self.transformPos(ev.pos())
            if self.mode == 'layout':
                self.scene.addLayoutCorner(point, self.layout_width, self.layout_height, selectPlane=self.ctrlPressed)
            elif self.mode == 'point':
                depth = self.readDepth(point)
                if self.scene.addCorner(point, depth, self.extrinsics_inv, self.intrinsics):
                    self.mode = 'move'
                    pass
            elif self.mode == 'move':
                self.scene.select(point)
            elif self.mode == 'add':
                depth = self.readDepth(point)
                self.scene.addPlane(point, depth, self.extrinsics_inv, self.intrinsics)
                self.mode = 'move'
                pass
            self.prevPoint = point
            pass
        #if ev.button() == Qt.RightButton:
        #pos = self.transformPos(ev.pos(), moving=True)
        #pass
        self.repaint()
        return

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""


        if (Qt.LeftButton & ev.buttons()):
            point = self.transformPos(ev.pos())
            if self.mode == 'layout':
                self.scene.moveLayoutCorner(point)
                self.repaint()
            elif self.scene.selectedCornerIndex != -1:
                if not self.shiftPressed or self.ctrlPressed:
                    self.scene.moveCorner(point, self.extrinsics_inv, self.intrinsics, self.imageIndex, axisAligned=(not self.ctrlPressed and not self.shiftPressed))
                    self.repaint()
                    pass
            elif self.scene.selectedEdgeIndex != -1 and False:
                if abs(point[1] - self.prevPoint[1]) < 10:
                    distanceScale = 1.0
                elif point[1] - self.prevPoint[1] <= -10:
                    distanceScale = 1.0 / 1.1
                else:
                    distanceScale = 1.1
                    pass
                # if self.ctrlPressed:
                #     #distanceScale = min(max(1 - (point[1] - self.prevPoint[1]) / 100, 0.8), 1.2)
                #     distanceScale = min(max(1 - (point[1] - self.prevPoint[1]) / 100, 0.8), 1.2)
                # else:
                #     distanceScale = 1.0
                #     pass
                self.scene.moveEdge(self.extrinsics_inv, self.intrinsics, self.imageIndex, distanceScale=distanceScale)
                self.repaint()
                pass
            self.prevPoint = point
            return

        return

    def mouseReleaseEvent(self, ev):
        if self.ctrlPressed and self.shiftPressed and self.scene.selectedCornerIndex != -1:
            point = self.transformPos(ev.pos())
            self.scene.moveCorner(point, self.extrinsics_inv, self.intrinsics, self.imageIndex, recording=True)
            pass
        elif self.shiftPressed and self.scene.selectedCornerIndex != -1:
            point = self.transformPos(ev.pos())
            self.scene.moveCorner(point, self.extrinsics_inv, self.intrinsics, self.imageIndex, concave=True)
            self.repaint()
            pass
        self.scene.selectedLayoutCorner = [-1, -1]
        self.scene.selectedCornerIndex = -1
        self.scene.selectedEdgeIndex = -1

        return

    def wheelEvent(self, ev):
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()
            pass
        self.scene.adjustHeight(v_delta, self.ctrlPressed)
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
                #self.calculateOffsets(corner, point)
                break
            continue
        return

    def paintEvent(self, event):
        if (self.imageIndex == -1 or not self.image) and self.mode != 'layout':
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # p.drawPixmap(self.offsetX, self.offsetY, self.image)

        # if not self.hiding:
        #     if self.mode == 'layout':
        #         self.scene.paintLayout(p, self.layout_width, self.layout_height, self.offsetX, self.offsetY)
        #     else:
        #         self.scene.updateVisiblePolygons(self.extrinsics, self.intrinsics, self.color_width, self.color_height)
        #         self.scene.paint(p, self.extrinsics, self.intrinsics, self.color_width, self.color_height, self.offsetX, self.offsetY)
        #         pass
        # elif self.mode != 'layout':
        #     if not self.ctrlPressed:
        #         self.scene.updateVisiblePolygons(self.extrinsics, self.intrinsics, self.color_width, self.color_height, hideOthers=True)
        #         self.scene.paint(p, self.extrinsics, self.intrinsics, self.color_width, self.color_height, self.offsetX, self.offsetY)
        #         pass
        #     pass

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

        if key == Qt.Key_Escape:
            #self.mode = 'moving'
            self.scene.deleteSelected()
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
            if self.mode != 'layout':
                self.mode = 'add'
                pass
        elif key == Qt.Key_Q:
            if self.mode != 'layout':
                self.mode = 'point'
                pass
        elif key == Qt.Key_S:
            if self.ctrlPressed:
                print('save')
                self.scene.save()
                pass
        elif key == Qt.Key_D:
            self.setCurrentLabel(2)
            self.setMode(False)
        elif key == Qt.Key_F:
            self.setCurrentLabel(3)
            self.setMode(False)
        elif key == Qt.Key_M:
            self.writePLYFile()
        elif key == Qt.Key_Right:
            self.moveToNextImage()
        elif key == Qt.Key_Left:
            self.moveToPreviousImage()
        elif key == Qt.Key_Down:
            self.moveToNextImage(5)
        elif key == Qt.Key_Up:
            self.moveToPreviousImage(5)
        elif key == Qt.Key_1:
            self.moveToNextImage()
            self.mode = 'move'
            self.repaint()
        elif key == Qt.Key_2:
            self.showDensityImage()
            self.mode = 'layout'
            self.repaint()
        elif key == Qt.Key_E:
            if self.ctrlPressed:
                self.scene.exportPly()
                pass
        elif key == Qt.Key_Space:
            if self.mode == 'layout':
                self.scene.layoutTo3D(self.layout_width, self.layout_height)
                self.moveToNextImage()
                self.mode = 'move'
            else:
                self.scene.finalize()
                pass
            self.repaint()
            pass
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


    def loadScene(self, scenePath):
        #imagePaths = glob.glob(scenePath + '/frames/frame-*.color.jpg')
        #print(len(imagePaths))
        self.scene = Scene(scenePath)

        # with open(scenePath + '/frames/_info.txt') as f:
        #     for line in f:
        #         line = line.strip()
        #         tokens = [token for token in line.split(' ') if token.strip() != '']
        #         if tokens[0] == "m_calibrationColorIntrinsic":
        #             self.intrinsics = np.array([float(e) for e in tokens[2:]])
        #             self.intrinsics = self.intrinsics.reshape((4, 4))
        #         elif tokens[0] == "m_colorWidth":
        #             self.color_width = int(tokens[2])
        #         elif tokens[0] == "m_colorHeight":
        #             self.color_height = int(tokens[2])
        #         elif tokens[0] == "m_depthWidth":
        #             self.depth_width = int(tokens[2])
        #         elif tokens[0] == "m_depthHeight":
        #             self.depth_height = int(tokens[2])
        #         elif tokens[0] == "m_depthShift":
        #             depth_shift = int(tokens[2])
        #         elif tokens[0] == "m_frames.size":
        #             self.numImages = int(tokens[2])
        #             pass
        #         continue
        #     pass

        self.imagePaths = []
        # for imageIndex in xrange(self.numImages):
        #     self.imagePaths.append('%s/frames/frame-%06d.color.jpg'%(scenePath, imageIndex))
        #     continue

        #if not os.path.exists(scenePath + '/annotation/room_layout.npy'):
        self.showDensityImage()
        self.mode = 'layout'
        # else:
        #     self.moveToNextImage()
        #     self.mode = 'move'
        #     pass
        self.repaint()
        return

    def showDensityImage(self):
        # image = self.scene.getDensityImage(self.layout_width, self.layout_height)
        # self.image = QPixmap(QImage(image[:, :, ::-1].reshape(-1), self.layout_width, self.layout_height, self.layout_width * 3, QImage.Format_RGB888))
        return

    def moveToNextImage(self, delta=1):
        self.imageIndex = min(self.imageIndex + delta, len(self.imagePaths) - 1)
        self.loadImage()
        return

    def moveToPreviousImage(self, delta=1):
        self.imageIndex = max(self.imageIndex - delta, 0)
        self.loadImage()
        return

    def loadImage(self):
        image = cv2.imread(self.imagePaths[self.imageIndex])
        self.image = QPixmap(QImage(image[:, :, ::-1].reshape(-1), self.color_width, self.color_height, self.color_width * 3, QImage.Format_RGB888))

        self.depth = cv2.imread(self.imagePaths[self.imageIndex].replace('color.jpg', 'depth.pgm'), -1).astype(np.float32) / 1000


        self.extrinsics_inv = []
        with open(self.imagePaths[self.imageIndex].replace('color.jpg', 'pose.txt'), 'r') as f:
            for line in f:
                self.extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        self.extrinsics_inv = np.array(self.extrinsics_inv).reshape((4, 4))
        self.extrinsics = np.linalg.inv(self.extrinsics_inv)
        self.repaint()
        return


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
