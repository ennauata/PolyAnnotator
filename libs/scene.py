import numpy as np
import copy
import os
from utils import *
import glob

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

COLOR_MAP = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 0, 255)]


class Scene():
    def __init__(self, annotDir):
        self.planes = None  # np.load(scenePath + '/annotation/planes.npy')
        self.numPlanes = 0  # self.planes.shape[0]

        self.colorMap = ColorPalette(1000).getColorMap()
        self.activeAnnot = None
        self.annotDir = annotDir
        self.annotFiles = []
        self.annotPaths = []

        self.reset()
        self.loadAllGraphs()

    def reset(self):
        self.allGraph = {}  # the graph for all saved annotations
        self.layoutGraph = {}  # only the latest unsaved sample
        self.prevCorner = None
        self.isDisconnected = False
        self.edgesTracker = []
        self.selectedCorner = None
        self.isAfterDelete = False

    def resetLatest(self):
        self.layoutGraph = {}  # only the latest unsaved sample
        self.prevCorner = None
        self.isDisconnected = False
        self.edgesTracker = []
        self.selectedCorner = None
        self.isAfterDelete = False

    def loadAllGraphs(self):
        if self.annotDir is not None:
            self.annotFiles = sorted(glob.glob(os.path.join(self.annotDir, '*.npy')))
            for filePath in self.annotFiles:
                singleGraph = self.loadGraph(filePath)
                self.updateGraph(singleGraph)

    def loadGraph(self, filePath):
        sampleInfo = np.load(filePath)[()]
        graph = sampleInfo['graph']
        return graph

    def updateGraph(self, graph):
        for pt, n_pts in graph.items():
            if pt in self.allGraph:
                for n_pt in n_pts:
                    if n_pt not in self.allGraph[pt]:
                        self.allGraph[pt].append(n_pt)
            else:
                self.allGraph[pt] = n_pts

    def updateActivatedAnnotation(self, annotFilename):
    	print('activate {}'.format(annotFilename))
        
        if self.activeAnnot is not None:
            self.loadAllGraphs()

        self.activeAnnot = annotFilename

        if annotFilename in self.annotFiles:
            annotPath = os.path.join(self.annotDir, str(annotFilename))
            sampleInfo = np.load(annotPath)[()]
            self.layoutGraph = sampleInfo['graph']
            self.isDisconnected = sampleInfo['isDisconnected']
            self.edgesTracker = sampleInfo['edgesTracker']
            self.selectedCorner = sampleInfo['selectedCorner']
            self.isAfterDelete = sampleInfo['isAfterDelete']
            self.prevCorner = sampleInfo['prevCorner']

            # remove graph from all_graphs
            for pt in self.layoutGraph.keys():
                if pt in self.allGraph.keys():
                    del self.allGraph[pt]
            return self.layoutGraph.keys()[0]

        else:
            raise ValueError("Can't find the annotation file {} in {}".format(annotFilename, self.annotDir))
        return None

    def savelatestSample(self, filename):
        sample_info = {'graph': self.layoutGraph,
                       'isDisconnected': self.isDisconnected,
                       'edgesTracker': self.edgesTracker,
                       'selectedCorner': self.selectedCorner,
                       'isAfterDelete': self.isAfterDelete,
                       'prevCorner': self.prevCorner}

        if self.activeAnnot is not None:
            save_path = os.path.join(self.annotDir, str(self.activeAnnot))

        else:
            save_path = os.path.join(self.annotDir, filename)
        
        if len(self.layoutGraph.keys()) > 0:
            np.save(save_path, sample_info)
        else:
            if os.path.isfile(save_path):
                os.remove(save_path)

        self.activeAnnot = None
        self.updateGraph(self.layoutGraph)
        self.resetLatest()
        self.loadAllGraphs()

    def paintLayout(self, painter, width, height, imCenter, scale, offsetX, offsetY):
        color = [QColor(self.colorMap[2][0], self.colorMap[2][1], self.colorMap[2][2]), QColor(self.colorMap[0][0], self.colorMap[0][1], self.colorMap[0][2])]
        pens = [QPen(color[0]), QPen(color[1])]

        # apply scale
        d = 5 * (1.0 / scale)
        d = int(min(d, 10))
        d = int(max(d, 1))
        penW = 2.0 / scale
        penW = int(max(penW, 1))
        penW = int(min(penW, 3))

        pens[0].setWidth(penW)
        pens[1].setWidth(penW)
        
        corner_paths = [QPainterPath(), QPainterPath()]
        boundary_paths = [QPainterPath(), QPainterPath()]

        # compute box center
        wCenter = np.array([offsetX, offsetY]) + np.array([width / 2.0, height / 2.0])

        for k, graph in enumerate([self.allGraph, self.layoutGraph]):
            # draw all points in the graph
            for pt in graph.keys():

                distFromCenter = (pt - imCenter) / scale
                if np.abs(distFromCenter[0]) < width / 2.0 and np.abs(distFromCenter[1]) < height / 2.0:

                    new_pt = distFromCenter + wCenter
                    point = QPoint(int(new_pt[0]), int(new_pt[1]))

                    if pt is not self.prevCorner:
                        corner_paths[k].addEllipse(point, d / 2.0, d / 2.0)

                    # draw all neighbours
                    for n_pt in graph[pt]:
                        distFromCenter = (n_pt - imCenter) / scale
                        if np.abs(distFromCenter[0]) < width / 2.0 and np.abs(distFromCenter[1]) < height / 2.0:
                            new_n_pt = distFromCenter + wCenter
                            n_point = QPoint(int(new_n_pt[0]), int(new_n_pt[1]))
                            boundary_paths[k].moveTo(point)
                            boundary_paths[k].lineTo(n_point)


            painter.setPen(pens[k]) 
            painter.drawPath(corner_paths[k])
            painter.drawPath(boundary_paths[k])

        # paint previous corner with another color
        if self.prevCorner is not None:
            color = QColor(self.colorMap[1][0], self.colorMap[1][1], self.colorMap[1][2])
            pen = QPen(color)
            pen.setWidth(penW)
            painter.setPen(pen)
            corner_path = QPainterPath()

            distFromCenter = (self.prevCorner - imCenter) / scale
            if np.abs(distFromCenter[0]) < width / 2.0 and np.abs(distFromCenter[1]) < height / 2.0:
                new_n_pt = distFromCenter + wCenter
                point = QPoint(int(new_n_pt[0]), int(new_n_pt[1]))
                corner_path.addEllipse(point, d / 2.0, d / 2.0)
                painter.drawPath(corner_path)
        return

    def selectCorner(self, point, epsilon=2):
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

            # update edge tracker
            for k in range(len(self.edgesTracker)):
                n1, n2 = self.edgesTracker[k]

                if n1 == self.selectedCorner:
                    n1 = tuple(point)
                if n2 == self.selectedCorner:
                    n2 = tuple(point)
                self.edgesTracker[k] = (n1, n2)

            # update prev node if needed
            if self.prevCorner == self.selectedCorner:
                self.prevCorner = tuple(point)

            # update selected corner
            self.selectedCorner = tuple(point)
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

    def addLayoutCorner(self, point, epsilon=2):

        # check if is a new corner
        isNewEdge = False
        isNewCorner = True
        point = point.astype('int32')
        currCorner = tuple(point)

        for existingCorner in self.layoutGraph.keys():
            if np.linalg.norm(point - np.array(existingCorner)) < epsilon:
                isNewCorner = False
                currCorner = existingCorner

        # graph is currently connected
        if self.prevCorner is not None and self.prevCorner is not currCorner:

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

    def adjustHeight(self, heightDelta, opposite=False):
        for cornerIndex in self.faces[self.selectedFaceIndex]:
            direction = (self.cornersOpp[cornerIndex] - self.corners[cornerIndex])
            if np.linalg.norm(direction) > 1e-4:
                direction = direction / np.linalg.norm(direction)
            else:
                direction, offset = self.fitPlane([self.corners[_] for _ in self.faces[self.selectedFaceIndex]],
                                                  return_normal_offset=True)
                pass
            if not opposite:
                self.corners[cornerIndex] -= float(heightDelta) * direction / 10000
            else:
                direction *= -1
                self.cornersOpp[cornerIndex] += float(heightDelta) * direction / 10000
                pass
            continue
        return
