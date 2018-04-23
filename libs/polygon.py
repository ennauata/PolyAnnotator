#!/usr/bin/python
# -*- coding: utf-8 -*-


try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.lib import distance, calcAngle
import numpy as np
from copy import deepcopy

DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(0, 0, 255)

COLOR_MAP = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0)]

class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label, group, line_color=None, difficult = False, regularization_level=1):
        self.label = label
        self.group = group

        self.line_color = COLOR_MAP[self.label]
        self.points = []
        self.fill = False
        self.selected = False
        self.selectedVertex = None
        self.difficult = difficult

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (2, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color
            pass
        self.regularizationLevel = regularization_level
        self.active = True
        return


    def close(self, regularization=False):
        if regularization and self.regularizationLevel == 1 and len(self.points) >= 4:
            from sklearn.cluster import KMeans
            angles = []
            for i, p in enumerate(self.points[:-1]):
                angle = calcAngle(self.points[i + 1] - p) % np.pi
                angles.append((np.cos(angle * 2), np.sin(angle * 2)))
                continue
            angle = calcAngle(self.points[0] - self.points[-1]) % np.pi
            angles.append((np.cos(angle * 2), np.sin(angle * 2)))

            angles = np.array(angles)
            #print(angles)
            # directions = KMeans(n_clusters=2).fit_predict(angles)
            # mask_0 = directions == 0
            # mask_1 = directions == 1
            # dominantAngles = []
            # if (mask_0).sum() > (mask_1).sum():
            #     meanAngle = angles[mask_0].mean(0)
            # else:
            #     meanAngle = angles[mask_1].mean(0)
            #     directions = 1 - directions
            #     pass
            # angle = np.arctan2(meanAngle[1], meanAngle[0]) % (2 * np.pi) / 2
            # dominantAngles.append(angle)

            #dominantAngles.append((dominantAngles[0] + np.pi / 2) % np.pi)

            #print(np.sin(dominantAngles[0]) - np.cos(dominantAngles[1]))
            #dominantAngles = np.array([0, np.pi / 2])
            #directions = np.array([0, 1, 0, 1])

            directions = KMeans(n_clusters=2).fit_predict(angles)
            dominantAngles = []
            for c in xrange(2):
                meanAngle = angles[directions == c].mean(0)
                angle = np.arctan2(meanAngle[1], meanAngle[0]) % (2 * np.pi) / 2
                dominantAngles.append(angle)
                continue

            newPoints = deepcopy(self.points)
            for i, p_1 in enumerate(newPoints):
                p_2 = newPoints[(i + 1) % len(self.points)]
                point_1 = np.array([p_1.x(), p_1.y()])
                point_2 = np.array([p_2.x(), p_2.y()])
                angle = dominantAngles[directions[i]]
                direction = np.array([np.cos(angle), np.sin(angle)])
                center = (point_1 + point_2) / 2
                #length = np.abs(np.dot(point_2 - point_1, direction))
                newPoint_1 = center + np.dot(point_1 - center, direction) * direction
                newPoint_2 = center + np.dot(point_2 - center, direction) * direction
                newPoints[i] = QPointF(newPoint_1[0], newPoint_1[1])
                # print('center', center)
                # print(newPoint_1, newPoint_2)
                # print('direction', direction)
                newPoints[(i + 1) % len(self.points)] = QPointF(newPoint_2[0], newPoint_2[1])
                continue
            # print(directions, angles, dominantAngles)
            # print(self.points)
            # print(newPoints)
            self.points = newPoints
            pass


        if regularization and self.regularizationLevel == 2 and len(self.points) >= 2:
            prevPoint = self.points[0]
            if abs(self.points[-1].x() - prevPoint.x()) > abs(self.points[-1].y() - prevPoint.y()):
                self.points[-1].setY(prevPoint.y())
            else:
                self.points[-1].setX(prevPoint.x())
                pass
            pass

        self._closed = True
        return

    def reachMaxPoints(self):
        if len(self.points) >= 20:
            return True
        return False

    def addPoint(self, point):
        if self.reachMaxPoints():
            return

        if self.regularizationLevel == 2 and len(self.points) >= 1 and self.label != 2:
            prevPoint = self.points[-1]
            if abs(point.x() - prevPoint.x()) > abs(point.y() - prevPoint.y()):
                point.setY(prevPoint.y())
            else:
                point.setX(prevPoint.x())
                pass
            pass
        self.points.append(point)
        return

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def isClosed(self):
        return self._closed

    def setClose(self, closed=True):
        self._closed = closed

    def paint(self, painter, center, patchSize, width, height):
        if self.active and self.points:
            #color = self.select_line_color if self.selected else self.line_color
            color = self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            #pen.setWidth(max(1, int(round(2.0 / self.scale))))
            pen.setWidth(3)
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            positions = []
            for point in self.points:
                x = int(round((point.x() - (center[0] - patchSize / 2)) / patchSize * width))
                y = height - int(round((point.y() - (center[1] - patchSize / 2)) / patchSize * height))
                positions.append(QPoint(x, y))
                continue

            line_path.moveTo(positions[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            #self.drawVertex(vrtx_path, 0)

            for i, p in enumerate(positions):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i, center, patchSize, width, height)
            if self.isClosed():
                line_path.lineTo(positions[0])

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)
            if self.selected:
                painter.fillPath(line_path, self.select_fill_color)
                pass
            pass
        return


    def drawVertex(self, path, i, center, patchSize, width, height):
        #d = self.point_size / self.scale
        d = 6
        shape = self.point_type
        point = self.points[i]

        x = int(round((point.x() - (center[0] - patchSize / 2)) / patchSize * width))
        y = height - int(round((point.y() - (center[1] - patchSize / 2)) / patchSize * height))
        point = QPointF(x, y)

        self.vertex_fill_color = Shape.vertex_fill_color
        if self.selectedVertex is not None:
            if i == self.selectedVertex:
                size, shape = self._highlightSettings[self._highlightMode]
                d *= size
                pass
            self.vertex_fill_color = self.hvertex_fill_color
            pass

        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        if not self.active:
            return None
        for i, p in enumerate(self.points):
            #print(distance(p - point), epsilon)
            if distance(p - point) <= epsilon:
                return i
        return None

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    # def moveBy(self, offset):
    #     self.points = [p + offset for p in self.points]

    # def moveVertexBy(self, i, offset):
    #     self.points[i] = self.points[i] + offset

    def moveBy(self, offset):
        if self.selectedVertex != None:
            self.points[self.selectedVertex] += offset
        elif self.selected:
            self.points = [p + offset for p in self.points]
            pass
        return

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        shape = Shape(label=self.label, group=self.group, regularization_level=self.regularizationLevel)
        shape.points = [p for p in self.points]
        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult
        shape.selectedVertex = None
        shape.active = self.active
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
        return

    def onBoundary(self, point, epsilon):
        if not self.active:
            return None

        if len(self.points) <= 1:
            return None
        for i, p_1 in enumerate(self.points[:-1]):
            p_2 = self.points[i + 1]
            if distance(p_1 - point) + distance(p_2 - point) <= distance(p_1 - p_2) + epsilon:
                #print('on boundary')
                return i
            continue
        if self.isClosed():
            p_1 = self.points[-1]
            p_2 = self.points[0]
            if distance(p_1 - point) + distance(p_2 - point) <= distance(p_1 - p_2) + epsilon:
                #print('on boundary')
                return len(self.points) - 1
            pass
        return None

    def setActivity(self, center, patchSize):
        mins = center - patchSize / 2
        maxs = center + patchSize / 2
        self.active = False
        for point in self.points:
            if point.x() > mins[0] and point.x() < maxs[0] and point.y() > mins[1] and point.y() < maxs[1]:
                self.active = True
            continue
        return

    def selectShape(self, point, epsilon):
        if self.active == False:
            return False
        self.selectedVertex = self.nearestVertex(point, epsilon)
        if self.selectedVertex != None:
            return True

        if self.containsPoint(point) or self.onBoundary(point, epsilon) != None:
            self.selected = True
            return True
        return False

    def inShape(self, point, epsilon):
        if self.active == False:
            return False
        if self.nearestVertex(point, epsilon) != None:
            return True

        if self.containsPoint(point) or self.onBoundary(point, epsilon) != None:
            return True
        return False
