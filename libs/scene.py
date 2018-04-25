import numpy as np

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
import copy
import os
from utils import *

COLOR_MAP = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 0, 255)]


class Scene():
    def __init__(self, scenePath):
        self.planes = None #np.load(scenePath + '/annotation/planes.npy')
        self.numPlanes = 0 #self.planes.shape[0]
        self.loadPoints(scenePath)

        self.colorMap = ColorPalette(1000).getColorMap()
        self.scenePath = scenePath

        self.reset()
        return

    def reset(self, mode='load'):

        if mode == 'init' or not os.path.exists(self.scenePath + '/annotation/scene_info.npy'):
            # self.dividePlanes()
            # self.findCorners()
            self.corners = np.array([])
            self.faces = []
            self.cornersOpp = np.array([])
            self.dominantNormals = []

        else:
            self.load()
            pass
        self.layoutGraph = {}
        self.prevCorner = None
        self.isDisconnected = False
        self.edgesTracker = []
        self.selectedCorner = None
        self.isAfterDelete = False
        return

    def loadPoints(self, scenePath):
        points = []
        segmentation = []
        planeNumPoints = 0 #np.zeros(self.numPlanes)
        # with open(scenePath + '/annotation/planes.ply') as f:
        #     lineIndex = 0
        #     for line in f:
        #         if lineIndex >= 12:
        #             values = [token for token in line.strip().split(' ') if token.strip() != '']
        #             if len(values) != 6:
        #                 break
        #             points.append(np.array([float(token) for token in values[:3]]))
        #             indices = [int(value) for value in values[3:]]
        #             if indices[0] == 255 and indices[1] == 255 and indices[2] == 255:
        #                 planeIndex = -1
        #             else:
        #                 assert((indices[0] * 256 * 256 + indices[1] * 256 + indices[2]) % 100 == 0)
        #                 planeIndex = (indices[0] * 256 * 256 + indices[1] * 256 + indices[2]) / 100 - 1
        #                 planeNumPoints[planeIndex] += 1
        #                 pass
        #             segmentation.append(planeIndex)
        #             pass
        #         lineIndex += 1
        #         continue
        #     pass
        self.points = np.array(points)
        self.segmentation = np.array(segmentation)
        self.planeNumPoints = planeNumPoints
        return

    def paintLayout(self, painter, width, height, offsetX, offsetY):

        
        color = QColor(self.colorMap[0][0], self.colorMap[0][1], self.colorMap[0][2])
        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)
        d = 10            
        corner_path = QPainterPath()
        boundary_path = QPainterPath()

        # draw all points in the graph
        for pt in self.layoutGraph.keys():

        
            point = QPoint(int(round(pt[0] + offsetX)), int(round(pt[1] + offsetY)))
            if pt is not self.prevCorner:
                corner_path.addEllipse(point, d / 2.0, d / 2.0)

            # draw all neighbours
            for n_pt in self.layoutGraph[pt]:
                n_point = QPoint(int(round(n_pt[0] + offsetX)), int(round(n_pt[1] + offsetY)))
                boundary_path.moveTo(point)
                boundary_path.lineTo(n_point)

        painter.drawPath(corner_path)
        painter.drawPath(boundary_path)

        # paint previous corner with another color
        if self.prevCorner is not None:
            color = QColor(self.colorMap[1][0], self.colorMap[1][1], self.colorMap[1][2])
            pen = QPen(color)
            pen.setWidth(3)
            painter.setPen(pen)
            d = 10
            corner_path = QPainterPath()

            point = QPoint(int(round(self.prevCorner[0] + offsetX)), int(round(self.prevCorner[1] + offsetY)))
            corner_path.addEllipse(point, d / 2.0, d / 2.0)
            painter.drawPath(corner_path)
        return

    def selectCorner(self, point, epsilon=10):
        self.selectedCorner = None
        for corner in self.layoutGraph.keys():
            if np.linalg.norm(point - np.array(corner)) < epsilon:
                self.selectedCorner = corner
        return

    def moveCorner(self, point):
        # update dict
        if self.selectedCorner is not None and tuple(point) not in self.layoutGraph.keys():
            self.selectedCorner = tuple(self.selectedCorner)
            aux_nbs = list(self.layoutGraph[self.selectedCorner])
            del self.layoutGraph[self.selectedCorner]
            for node in aux_nbs:
                idx = self.layoutGraph[node].index(self.selectedCorner)
                self.layoutGraph[node][idx] = tuple(point)
            self.layoutGraph[tuple(point)] = aux_nbs

            # update prev node if needed
            if self.prevCorner == self.selectedCorner:
                self.prevCorner = tuple(point)

            # update selected corner
            self.selectedCorner = tuple(point)
        return

    def save(self):
        scene_info = {'corners': self.corners, 'cornersOpp': self.cornersOpp, 'faces': self.faces, 'dominantNormals': self.dominantNormals}
        np.save(self.scenePath + '/annotation/scene_info.npy', scene_info)
        return

    def load(self):
        scene_info = np.load(self.scenePath + '/annotation/scene_info.npy')[()]
        self.corners = scene_info['corners']
        self.cornersOpp = scene_info['cornersOpp']
        self.faces = scene_info['faces']
        self.dominantNormals = scene_info['dominantNormals']
        return

    def removeCorner(self):
        aux_nbs = list(self.layoutGraph[self.selectedCorner])
        del self.layoutGraph[self.selectedCorner]
        for node in aux_nbs:
            idx = self.layoutGraph[node].index(self.selectedCorner)
            del self.layoutGraph[node][idx]
        self.prevCorner = None
        self.edgesTracker = []
        self.isAfterDelete = True
        return

    def removeLast(self):

        # remove last edge
        to_remove = None
        if len(self.edgesTracker) > 0:
            c1 = self.edgesTracker[-1][0]
            c2 = self.edgesTracker[-1][1]

            # update annotation
            idx = self.layoutGraph[c1].index(c2)
            del self.layoutGraph[c1][idx]
            if len(self.layoutGraph[c1]) == 0:
                del self.layoutGraph[c1]

            idx = self.layoutGraph[c2].index(c1)
            del self.layoutGraph[c2][idx]
            if len(self.layoutGraph[c2]) == 0:
                del self.layoutGraph[c2]           

            # update tracker
            self.edgesTracker = self.edgesTracker[:-1]

            # update prev corner
            if len(self.edgesTracker) > 0:
                self.prevCorner = self.edgesTracker[-1][1]
            else:
                self.prevCorner = None

        # if there is only one corner
        elif self.prevCorner is not None and not self.isAfterDelete:
            del self.layoutGraph[self.prevCorner]
            self.prevCorner = None
        return


    def disconnectGraph(self):
        self.prevCorner = None
        print('Graph Disconnected')


    def addLayoutCorner(self, point, width, height, selectPlane=False, epsilon=10):

        # check if is a new corner
        isNewEdge = False
        isNewCorner = True
        currCorner = tuple(point)
        for existingCorner in self.layoutGraph.keys():
            if np.linalg.norm(point - np.array(existingCorner)) < epsilon:
                isNewCorner = False
                currCorner = existingCorner

        # graph is currently connected
        if self.prevCorner is not None:

            # closing one edge with a new node
            if isNewCorner:
                self.layoutGraph[currCorner] = [self.prevCorner]
                isNewEdge = True
            # closing one edge with an existing node
            else:
                # add only if connection does not exist
                if self.prevCorner not in self.layoutGraph[currCorner]:
                    self.layoutGraph[currCorner].append(self.prevCorner)
                    isNewEdge = True

            # add only if connection does not exist
            if currCorner not in self.layoutGraph[self.prevCorner]:
                self.layoutGraph[self.prevCorner].append(currCorner)
                isNewEdge = True
        else:
            # graph is disconnected and clicked on non existing node
            if isNewCorner:
                self.layoutGraph[currCorner] = []

        # if edge is added keep track
        if isNewEdge: 
            self.edgesTracker.append((self.prevCorner, currCorner))
            self.isAfterDelete = False
        self.prevCorner = currCorner
        return

    def moveLayoutCorner(self, point):

        return

    def adjustHeight(self, heightDelta, opposite=False):
        for cornerIndex in self.faces[self.selectedFaceIndex]:
            direction = (self.cornersOpp[cornerIndex] - self.corners[cornerIndex])
            if np.linalg.norm(direction) > 1e-4:
                direction = direction / np.linalg.norm(direction)
            else:
                direction, offset = self.fitPlane([self.corners[_] for _ in self.faces[self.selectedFaceIndex]], return_normal_offset=True)
                pass
            if not opposite:
                self.corners[cornerIndex] -= float(heightDelta) * direction / 10000
            else:
                direction *= -1
                self.cornersOpp[cornerIndex] += float(heightDelta) * direction / 10000
                pass
            continue
        return

    def exportPlyPoint(self):
        with open('test/scene.ply', 'w') as f:
            header = """ply
            format ascii 1.0
            element vertex """
            header += str(len(self.faces) * 600)
            header += """
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            """
            f.write(header)

            cornerOffset = len(self.corners)
            allCorners = np.concatenate([self.corners, self.cornersOpp], axis=0)
            allFaces = copy.deepcopy(self.faces)
            for faceIndex, face in enumerate(self.faces):
                allFaces.append([cornerIndex + cornerOffset for cornerIndex in face])
                for c in xrange(len(face)):
                    allFaces.append([face[c], face[(c + 1) % len(face)], face[(c + 1) % len(face)] + cornerOffset, face[c] + cornerOffset])
                    continue
                continue

            for faceIndex, face in enumerate(allFaces):
                color = self.colorMap[faceIndex]
                faceCorners = np.array([allCorners[cornerIndex] for cornerIndex in face])
                cooeficients = np.random.random((100, 4))
                cooeficients /= np.maximum(cooeficients.sum(1, keepdims=True), 1e-4)
                points = np.matmul(cooeficients, faceCorners)
                for point in points:
                    f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
                    #f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                    continue
                continue
            f.close()
            pass
        return

    def exportPly(self):

        cornerOffset = len(self.corners)
        allCorners = np.concatenate([self.corners, self.cornersOpp], axis=0)
        allFaces = copy.deepcopy(self.faces)
        for faceIndex, face in enumerate(self.faces):
            allFaces.append([cornerIndex + cornerOffset for cornerIndex in face])
            for c in xrange(len(face)):
                allFaces.append([face[c], face[(c + 1) % len(face)], face[(c + 1) % len(face)] + cornerOffset, face[c] + cornerOffset])
                continue
            continue

        with open('test/scene.ply', 'w') as f:
            header = """ply
format ascii 1.0
element vertex """
            header += str(len(allCorners))
            header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
            header += str(len(allFaces))
            header += """
property list uchar int vertex_index
end_header
"""
            f.write(header)

            for corner in allCorners:
                f.write(str(corner[0]) + ' ' + str(corner[1]) + ' ' + str(corner[2]) + ' 255 0 0\n')
                continue

            for faceIndex, face in enumerate(allFaces):
                #f.write('4 ' + str(face[0] + 1) + ' ' + str(face[1] + 1) + ' ' + str(face[2] + 1) + ' ' + str(face[3] + 1) + '\n')
                f.write('4 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                continue
            f.close()
            pass
        return

