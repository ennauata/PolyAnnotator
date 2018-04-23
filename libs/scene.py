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

        self.cornerFixedImages = {}
        self.cornerFixedFlags = {}
        self.edgeFixedImages = {}
        self.dominantNormals = []
        self.UVs = []
        self.UVsOpp = []
        self.selectedCornerIndex = -1
        self.selectedEdgeIndex = -1
        self.selectedFaceIndex = -1

        if mode == 'init' or not os.path.exists(self.scenePath + '/annotation/scene_info.npy'):
            # self.dividePlanes()
            # self.findCorners()
            self.corners = np.array([])
            self.faces = []
            self.cornersOpp = np.array([])
            self.dominantNormals = []

        else:
            self.load()
            #self.addCube()
            pass
        #self.visiblePolygons = [[[0, 1] for _ in face] for face in self.faces]
        self.visiblePolygons = []
        self.layoutFaces = []
        self.layoutFace = []
        self.concaveFace = []
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

    # def dividePlanes(self):
    #     horizontalMask = np.abs(self.planes[:, 2] / np.maximum(np.linalg.norm(self.planes, axis=-1), 1e-4)) > np.cos(np.deg2rad(45))
    #     horizontalPlanes = np.concatenate([self.planes[horizontalMask], np.expand_dims(self.planeNumPoints[horizontalMask], -1)], axis=-1)
    #     verticalPlanes = np.concatenate([self.planes[np.logical_not(horizontalMask)], np.expand_dims(self.planeNumPoints[np.logical_not(horizontalMask)], axis=-1)], axis=-1)

    #     horizontalPlanes = horizontalPlanes[np.argsort(horizontalPlanes[:, 2])]
    #     numHorizontalPoints = horizontalPlanes[:, 3].sum()
    #     largeHorizontalPlaneThreshold = 0.1
    #     largeVerticalPlaneThreshold = 0.02
    #     outmostPlaneThreshold = 0.8
    #     self.floor = None
    #     floorIndex = -1

    #     # print(largeHorizontalPlaneThreshold * numHorizontalPoints)
    #     # print(self.points[:, 2].min(), self.points[:, 2].max())
    #     # print(np.argmax(self.planeNumPoints), np.max(self.planeNumPoints))
    #     # plane = self.planes[np.argmax(self.planeNumPoints)]
    #     # print(plane)
    #     # print(horizontalMask[68])
    #     # print(np.abs(plane[2] / np.maximum(np.linalg.norm(plane, axis=-1), 1e-4)))
    #     # exit(1)
    #     for planeIndex, horizontalPlane in enumerate(horizontalPlanes):
    #         if horizontalPlane[3] > largeHorizontalPlaneThreshold * numHorizontalPoints:
    #             plane = horizontalPlane[:3]
    #             offset = np.linalg.norm(plane)
    #             normal = plane / np.maximum(offset, 1e-4)
    #             sideMask = np.sum(self.points * np.expand_dims(normal, 0), axis=-1) > offset
    #             count = np.sum(sideMask)
    #             if count < (1 - outmostPlaneThreshold) * len(self.points) or count > outmostPlaneThreshold * len(self.points):
    #                 self.floor = plane
    #                 floorIndex = planeIndex
    #             else:
    #                 #print(count, (1 - outmostPlaneThreshold) * len(self.points), outmostPlaneThreshold * len(self.points))
    #                 pass
    #             break
    #         continue
    #     if floorIndex == -1:
    #         print('floor not found')
    #         self.floor = np.array([0, 0, self.points[:, 2].min()])
    #         pass

    #     # self.ceiling = None
    #     # ceilingIndex = -1
    #     # for planeIndex, horizontalPlane in enumerate(horizontalPlanes[::-1]):
    #     #     if horizontalPlane[3] > largeHorizontalPlaneThreshold * numHorizontalPoints:
    #     #         plane = horizontalPlane[:3]
    #     #         offset = np.linalg.norm(plane)
    #     #         normal = plane / np.maximum(offset, 1e-4)
    #     #         sideMask = np.sum(self.points * np.expand_dims(normal, 0), axis=-1) > offset
    #     #         count = np.sum(sideMask)
    #     #         if len(horizontalPlanes) - 1 - planeIndex != floorIndex and (count < (1 - outmostPlaneThreshold) * len(self.points) or count > outmostPlaneThreshold * len(self.points)):
    #     #             self.ceiling = plane
    #     #             ceilingIndex = len(horizontalPlanes) - 1 - planeIndex
    #     #             pass
    #     #         break
    #     #     continue

    #     # if ceilingIndex == -1:
    #     #     print('ceiling not found')
    #     #     self.ceiling = np.array([0, 0, self.points[:, 2].max()])
    #     #     pass

    #     self.horizontalPlanes = np.array([horizontalPlane for planeIndex, horizontalPlane in enumerate(horizontalPlanes) if planeIndex not in [floorIndex]])

    #     verticalAngles = np.array([np.arctan2(verticalPlane[0], verticalPlane[1]) for verticalPlane in verticalPlanes])

    #     from sklearn.cluster import KMeans
    #     kmeans = KMeans(n_clusters=2).fit(np.expand_dims(verticalAngles, -1))
    #     meanAngles = kmeans.cluster_centers_.squeeze(-1)


    #     angleDifferenceThreshold = np.deg2rad(20)
    #     groupIndices_1 = np.logical_or(np.abs(verticalAngles - meanAngles[0]) % np.pi < angleDifferenceThreshold, np.abs(verticalAngles - meanAngles[0]) % np.pi > np.pi - angleDifferenceThreshold)
    #     groupIndices_2 = np.logical_or(np.abs(verticalAngles - meanAngles[1]) % np.pi < angleDifferenceThreshold, np.abs(verticalAngles - meanAngles[1]) % np.pi > np.pi - angleDifferenceThreshold)
    #     verticalPlanes = [verticalPlanes[groupIndices_1], verticalPlanes[groupIndices_2]]
    #     verticalAngles = [verticalAngles[groupIndices_1], verticalAngles[groupIndices_2]]

    #     self.verticalPlanes = []
    #     self.walls = [[], []]

    #     for group, (angle, planes) in enumerate(zip(meanAngles, verticalPlanes)):
    #         unitVector = np.array([np.cos(angle), np.sin(angle)])
    #         offsets = (planes[:, :2] * np.expand_dims(unitVector, 0)).sum(axis=-1)
    #         planes = planes[np.argsort(offsets)]
    #         numVerticalPoints = planes[:, 3].sum()
    #         #print(planes)
    #         #print(largeVerticalPlaneThreshold * numVerticalPoints)
    #         wallIndices = []
    #         for planeIndex, plane in enumerate(planes):
    #             if plane[3] > largeVerticalPlaneThreshold * numVerticalPoints:
    #                 plane = plane[:3]
    #                 offset = np.linalg.norm(plane)
    #                 normal = plane / np.maximum(offset, 1e-4)
    #                 sideMask = np.sum(self.points * np.expand_dims(normal, 0), axis=-1) > offset
    #                 count = np.sum(sideMask)
    #                 if count < (1 - outmostPlaneThreshold) * len(self.points) or count > outmostPlaneThreshold * len(self.points):
    #                     self.walls[group].append(plane)
    #                     wallIndices.append(planeIndex)
    #                 else:
    #                     print('wall not found', group, 0, planeIndex, plane, count, outmostPlaneThreshold * len(self.points), (1 - outmostPlaneThreshold) * len(self.points))
    #                     pass
    #                 break
    #             continue
    #         for planeIndex, plane in enumerate(planes[::-1]):
    #             if plane[3] > largeVerticalPlaneThreshold * numVerticalPoints:
    #                 plane = plane[:3]
    #                 offset = np.linalg.norm(plane)
    #                 normal = plane / np.maximum(offset, 1e-4)
    #                 sideMask = np.sum(self.points * np.expand_dims(normal, 0), axis=-1) > offset
    #                 count = np.sum(sideMask)
    #                 #print(count, len(sideMask))
    #                 if len(planes) - 1 - planeIndex not in wallIndices and (count < (1 - outmostPlaneThreshold) * len(self.points) or count > outmostPlaneThreshold * len(self.points)):
    #                     self.walls[group].append(plane)
    #                     wallIndices.append(len(planes) - 1 - planeIndex)
    #                 else:
    #                     print('wall not found', group, 1, planeIndex, plane, count, outmostPlaneThreshold * len(self.points), (1 - outmostPlaneThreshold) * len(self.points))
    #                     pass
    #                 break
    #             continue

    #         self.verticalPlanes.append(np.array([plane for planeIndex, plane in enumerate(planes) if planeIndex not in wallIndices]))
    #         continue
    #     return

    # def findCorners(self):
    #     corners = []
    #     cornerIndices = []
    #     for index_1, wall_1 in enumerate(self.walls[0]):
    #         for index_2, wall_2 in enumerate(self.walls[1]):
    #             for index_3, horizontalPlane in enumerate([self.floor]):
    #                 A = np.stack([wall_1[:3], wall_2[:3], horizontalPlane[:3]], axis=0)
    #                 b = np.linalg.norm(A, axis=-1)
    #                 A /= np.maximum(np.expand_dims(b, axis=-1), 1e-4)
    #                 corners.append(np.matmul(np.linalg.inv(A), b))
    #                 cornerIndices.append([index_1, index_2, index_3])
    #                 continue
    #             continue
    #         continue
    #     self.corners = np.array(corners)
    #     self.cornerFixedImages = [{} for corner in self.corners]

    #     cornerIndices = np.array(cornerIndices)
    #     faces = []
    #     for faceType in xrange(3):
    #         for faceDirection in xrange(2):
    #             indices = (cornerIndices[:, faceType] == faceDirection).nonzero()[0]
    #             if len(indices) != 4:
    #                 continue
    #             sortedIndices = [indices[0]]
    #             indices = indices[1:]
    #             while len(sortedIndices) < 4:
    #                 for index in indices:
    #                     if index not in sortedIndices and (cornerIndices[index] == cornerIndices[sortedIndices[-1]]).sum() == 2:
    #                         sortedIndices.append(index)
    #                         break
    #                         pass
    #                     continue
    #                 continue
    #             print(sortedIndices)
    #             faces.append(sortedIndices)
    #             continue
    #         continue

    #     # for cornerIndex_1, (corner_1, indices_1) in enumerate(zip(corners, cornerIndices)):
    #     #     for cornerIndex_2, (corner_2, indices_2) in enumerate(zip(corners, cornerIndices)):
    #     #         if cornerIndex_2 <= cornerIndex_1:
    #     #             continue
    #     #         if (indices_1 == indices_2).sum() == 2:
    #     #             connections.append([cornerIndex_1, cornerIndex_2])
    #     #             pass
    #     #         continue
    #     #     continue

    #     self.faces = faces

    #     if len(self.faces) == 1:
    #         self.corners = self.corners[self.faces[0]]
    #         self.faces[0] = np.arange(len(self.faces[0]), dtype=np.int32).tolist()
    #         pass

    #     print(self.corners)
    #     print(self.faces)
    #     return


    def updateVisiblePolygons(self, extrinsics, intrinsics, width, height, hideOthers=False):
        self.visiblePolygons = []
        if len(self.corners) == 0:
            return
        cornerOffset = len(self.corners)
        #print('update', self.corners.shape, self.cornersOpp.shape)
        transformedCorners = np.matmul(extrinsics, np.concatenate([np.concatenate([self.corners, self.cornersOpp], axis=0), np.ones((self.corners.shape[0] + self.cornersOpp.shape[0], 1))], axis=-1).transpose()).transpose()

        UVs = np.matmul(intrinsics, transformedCorners.transpose()).transpose()
        U = np.round(UVs[:, 0] / np.abs(UVs[:, 2])).astype(np.int32)
        V = np.round(UVs[:, 1] / np.abs(UVs[:, 2])).astype(np.int32)
        cornerVisibleMask = np.logical_and(np.logical_and(np.logical_and(U >= 0, U < width), np.logical_and(V >= 0, V < width)), UVs[:, 2] > 0)

        transformedCorners = transformedCorners[:, :3] / transformedCorners[:, 3:4]


        cameraCenter = np.zeros(3)
        maxDepth = 10.0
        xfov = np.arctan2(width / 2, intrinsics[0, 0])
        yfov = np.arctan2(height / 2, intrinsics[1, 1])
        #print(width, height, intrinsics)
        #cameraPlanes = np.array([[cameraCenter, [maxDepth * np.tan(yfov), maxDepth * np.tan(xfov), maxDepth], [maxDepth * np.tan(yfov), -maxDepth * np.tan(xfov), maxDepth]], [cameraCenter, [maxDepth * np.tan(yfov), -maxDepth * np.tan(xfov), maxDepth], [-maxDepth * np.tan(yfov), -maxDepth * np.tan(xfov), maxDepth]], [cameraCenter, [-maxDepth * np.tan(yfov), -maxDepth * np.tan(xfov), maxDepth], [-maxDepth * np.tan(yfov), maxDepth * np.tan(xfov), maxDepth]], [cameraCenter, [-maxDepth * np.tan(yfov), maxDepth * np.tan(xfov), maxDepth], [maxDepth * np.tan(yfov), maxDepth * np.tan(xfov), maxDepth]]])
        cameraPoints = np.expand_dims(np.array([np.tan(xfov), np.tan(yfov), 1]) * maxDepth, 0) * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]])
        cameraPlanes = [[cameraCenter, cameraPoints[c], cameraPoints[(c + 1) % 4]] for c in xrange(4)]

        #print(transformedCorners.shape)
        # print(intrinsics)
        # print(self.corners[:4])
        # print(transformedCorners[:4])
        # print(cornerVisibleMask[:4])
        # print(cameraPlanes)

        allFaces = copy.deepcopy(self.faces)
        if hideOthers:
            if self.selectedFaceIndex != -1:
                allFaces = [allFaces[self.selectedFaceIndex]]
            else:
                allFaces = []
                pass
            pass
        for faceIndex, face in enumerate(self.faces):
            if hideOthers and faceIndex != self.selectedFaceIndex:
                continue
            allFaces.append([cornerIndex + cornerOffset for cornerIndex in face])
            for c in xrange(len(face)):
                allFaces.append([face[c], face[(c + 1) % len(face)], face[(c + 1) % len(face)] + cornerOffset, face[c] + cornerOffset])
                continue
            continue

        #print(transformedCorners[:4])
        for face in allFaces:
            polygon = []
            for c in xrange(len(face)):
                intersectionRatios = []
                cornerIndex_1 = face[c]
                cornerIndex_2 = face[(c + 1) % len(face)]
                if cornerVisibleMask[cornerIndex_1] and cornerVisibleMask[cornerIndex_2]:
                    polygon.append(transformedCorners[cornerIndex_1])
                    polygon.append(transformedCorners[cornerIndex_2])
                    continue

                wallLine = [transformedCorners[cornerIndex_1], transformedCorners[cornerIndex_2]]
                for cameraPlane in cameraPlanes:
                    intersection, ratio = intersectFaceLine(cameraPlane, wallLine, return_ratio=True)
                    #print(intersection, ratio)
                    if intersection:
                        intersectionRatios.append(ratio)
                        pass
                    continue

                if len(intersectionRatios) == 0:
                    continue
                if cornerVisibleMask[cornerIndex_1]:
                    minRatio = 0
                else:
                    minRatio = min(intersectionRatios)
                    pass
                if cornerVisibleMask[cornerIndex_2]:
                    maxRatio = 1
                else:
                    maxRatio = max(intersectionRatios)
                    pass
                #print(c, minRatio, maxRatio)
                if minRatio < maxRatio:
                    polygon.append(transformedCorners[cornerIndex_1] + (transformedCorners[cornerIndex_2] - transformedCorners[cornerIndex_1]) * minRatio)
                    polygon.append(transformedCorners[cornerIndex_1] + (transformedCorners[cornerIndex_2] - transformedCorners[cornerIndex_1]) * maxRatio)
                    pass
                continue

            if len(polygon) == 0:
                continue
            polygon = np.array(polygon)
            polygonUV = np.matmul(intrinsics, np.concatenate([polygon, np.ones((polygon.shape[0], 1))], axis=-1).transpose()).transpose()
            polygonUV = polygonUV[:, :2] / np.maximum(np.abs(polygonUV[:, 2:3]), 1e-4)

            self.visiblePolygons.append(polygonUV)
            #print(polygon)
            #exit(1)
            continue

        transformedCorners = np.matmul(intrinsics, np.matmul(extrinsics, np.concatenate([self.corners, np.ones((self.corners.shape[0], 1))], axis=-1).transpose())).transpose()
        self.UVs = np.round(transformedCorners[:, :3] / np.abs(transformedCorners[:, 2:3])).astype(np.int32)

        transformedCorners = np.matmul(intrinsics, np.matmul(extrinsics, np.concatenate([self.cornersOpp, np.ones((self.corners.shape[0], 1))], axis=-1).transpose())).transpose()
        self.UVsOpp = np.round(transformedCorners[:, :3] / np.abs(transformedCorners[:, 2:3])).astype(np.int32)

        return

    def paintLayout(self, painter, width, height, offsetX, offsetY):
        # color = COLOR_MAP[0]
        # pen = QPen(color)
        # pen.setWidth(3)
        # painter.setPen(pen)
        # d = 10

        for faceIndex, face in enumerate(self.layoutFaces):
            face, planeIndex = face
            color = QColor(self.colorMap[faceIndex][0], self.colorMap[faceIndex][1], self.colorMap[faceIndex][2])
            pen = QPen(color)
            pen.setWidth(3)
            painter.setPen(pen)
            d = 10

            corner_path = QPainterPath()
            boundary_path = QPainterPath()
            for _, point in enumerate(face + [face[0]]):
                point = QPoint(int(round(point[0] + offsetX)), int(round(point[1] + offsetY)))
                corner_path.addEllipse(point, d / 2.0, d / 2.0)
                if _ == 0:
                    boundary_path.moveTo(point)
                else:
                    boundary_path.lineTo(point)
                    pass
                pass
            painter.drawPath(corner_path)
            painter.drawPath(boundary_path)

            if planeIndex != -1:
                color = QColor(self.colorMap[faceIndex][0], self.colorMap[faceIndex][1], self.colorMap[faceIndex][2], 64)

                pen = QPen(color)
                painter.setPen(pen)
                point_path = QPainterPath()
                for point in self.points[self.segmentation == planeIndex]:
                    point_path.addEllipse(QPoint((point[0] - self.minXY[0]) / self.maxRange * width, (point[1] - self.minXY[1]) / self.maxRange * height), 1, 1)
                    continue
                painter.drawPath(point_path)
            continue

        color = QColor(self.colorMap[len(self.layoutFaces)][0], self.colorMap[len(self.layoutFaces)][1], self.colorMap[len(self.layoutFaces)][2])
        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)
        d = 10

        corner_path = QPainterPath()
        boundary_path = QPainterPath()
        for _, point in enumerate(self.layoutFace):
            point = QPoint(int(round(point[0] + offsetX)), int(round(point[1] + offsetY)))
            corner_path.addEllipse(point, d / 2.0, d / 2.0)
            if _ == 0:
                boundary_path.moveTo(point)
            else:
                boundary_path.lineTo(point)
                pass
            pass
        painter.drawPath(corner_path)
        painter.drawPath(boundary_path)
        return

    def paint(self, painter, extrinsics, intrinsics, width, height, offsetX, offsetY):
        if len(self.corners) == 0:
            return



        # print(self.corners)
        # print(np.stack([U, V], axis=-1))
        # print(extrinsics)
        # print(transformedCorners[:, 2])

        # behindCameraMask = transformedCorners[:, 2] < 0
        # #U[behindCameraMask] *= -1
        # #V[behindCameraMask] *= -1
        # infinity = 1000
        # for cornerIndex in behindCameraMask.nonzero()[0]:
        #     #break
        #     U[cornerIndex] = (U[cornerIndex] - intrinsics[0, 2]) * infinity + intrinsics[0, 2]
        #     V[cornerIndex] = (V[cornerIndex] - intrinsics[1, 2]) * infinity + intrinsics[1, 2]
        #     continue


        for faceIndex, polygon in enumerate(self.visiblePolygons):
            # points = []
            # for c, alphas in enumerate(polygon):
            #     if alphas[1] <= alphas[0]:
            #         continue
            #     point_1 = self.corners[face[c]]
            #     point_2 = self.corners[face[(c + 1) % len(face)]]
            #     points.append(point_1 + (point_2 - point_1) * alphas[0])
            #     points.append(point_1 + (point_2 - point_1) * alphas[1])
            #     continue
            # if len(points) == 0:
            #     continue
            # points = np.array(points)
            # transformedPoints = np.matmul(intrinsics, np.matmul(extrinsics, np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1).transpose())).transpose()
            # UVs = np.stack([np.round(transformedPoints[:, 0] / np.abs(transformedPoints[:, 2])).astype(np.int32), np.round(transformedPoints[:, 1] / np.abs(transformedPoints[:, 2])).astype(np.int32)], axis=-1)
            # UVs = np.concatenate([UVs, UVs[0:1]], axis=0)

            color = QColor(255, 255, 255)
            pen = QPen(color)
            pen.setWidth(3)
            painter.setPen(pen)

            face_path = QPainterPath()
            for _, UV in enumerate(polygon):
                point = QPoint(UV[0] + offsetX, UV[1] + offsetY)
                if _ == 0:
                    face_path.moveTo(point)
                else:
                    face_path.lineTo(point)
                    pass
                continue

            painter.drawPath(face_path)
            color = QColor(self.colorMap[faceIndex][0], self.colorMap[faceIndex][1], self.colorMap[faceIndex][2], 64)
            painter.fillPath(face_path, color)
            continue


        # pointDegrees = np.zeros(len(self.corners), dtype=np.int32)
        # for face in self.faces:
        #     np.add.at(pointDegrees, face, 1)
        #     continue

        color = COLOR_MAP[0]
        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)
        corner_path = QPainterPath()
        d = 10

        for cornerIndex, UV in enumerate(np.concatenate([self.UVs, self.UVsOpp], axis=0)):
            if UV[0] >= 0 and UV[0] < width and UV[1] >= 0 and UV[1] < height and UV[2] > 0:
                #color = COLOR_MAP[pointDegrees[cornerIndex]]
                corner_path.addEllipse(QPoint(int(round(UV[0] + offsetX)), int(round(UV[1] + offsetY))), d / 2.0, d / 2.0)
                pass
            continue
        painter.drawPath(corner_path)
        return

    def selectCorner(self, point, epsilon):
        for cornerIndex, UV in enumerate(self.UVs):
            if np.linalg.norm(point - UV[:2]) < epsilon:
                self.selectedCornerIndex = cornerIndex
                self.selectedFaceIndex = [faceIndex for faceIndex, face in enumerate(self.faces) if cornerIndex in face][0]
            continue
        return

    def select(self, point, epsilon=10):
        self.selectedCornerIndex = -1
        self.selectedEdgeIndex = -1
        self.selectedFaceIndex = -1
        self.selectCorner(point, epsilon)
        if self.selectedCornerIndex == -1:
            self.selectEdge(point, epsilon)
            pass
    # def selectEdge(self, point, epsilon=10):
    #     connections = {}
    #     for face in self.faces:
    #         for c in xrange(len(face)):
    #             connections[(min(face[c], face[(c + 1) % len(face)]), max(face[c], face[(c + 1) % len(face)]))] = True
    #             continue
    #         continue
    #     for connection in connections:
    #         UV_1 = self.UVs[connection[0]]
    #         UV_2 = self.UVs[connection[1]]
    #         normal = (UV_1 - UV_2).astype(np.float32)
    #         normal /= max(np.linalg.norm(normal), 1e-4)
    #         normal = np.array([normal[1], -normal[0]])

    #         distance = np.dot(UV_1, normal) - np.dot(point, normal)
    #         if abs(distance) < epsilon:
    #             intersectionPoint = point + distance * normal
    #             alpha = np.linalg.norm(intersectionPoint - UV_1) / max(np.linalg.norm(UV_2 - UV_1), 1e-4)
    #             return (connection, alpha)
    #         continue
    #     return ()

    def selectEdge(self, point, epsilon=10):
        for cornerIndex, (UV_1, UV_2) in enumerate(zip(self.UVs, self.UVsOpp)):
            UV_1 = UV_1[:2]
            UV_2 = UV_2[:2]
            normal = (UV_1 - UV_2).astype(np.float32)
            normal /= max(np.linalg.norm(normal), 1e-4)
            normal = np.array([normal[1], -normal[0]])

            distance = np.dot(UV_1, normal) - np.dot(point, normal)
            if abs(distance) < epsilon:
                intersectionPoint = point + distance * normal
                alpha = np.linalg.norm(intersectionPoint - UV_1) / max(np.linalg.norm(UV_2 - UV_1), 1e-4)
                self.selectedEdgeIndex = cornerIndex
            continue
        return

    def addCube(self, imageCenter):
        print('add cube')
        newFace = self.faces[-1]
        normal, offset = self.fitPlane([self.corners[_] for _ in newFace], return_normal_offset=True)
        lineLength = 10
        if len(self.faces) == 1:
            ceilingOffset = (self.points * np.expand_dims(normal, 0)).sum(axis=-1).max()
            newCorners = np.array([self.corners[cornerIndex] + (ceilingOffset - offset) * normal for cornerIndex in newFace])
            self.cornersOpp = np.array(newCorners)
        else:
            intersectionRatios = []
            for cornerIndex in newFace:
                corner = self.corners[cornerIndex]
                #print(corner, imageCenter, normal)
                #print(np.dot(corner - imageCenter, normal))
                if np.dot(corner - imageCenter, normal) < 0:
                    normal *= -1
                    pass
                cornerLine = [corner, corner + normal * lineLength]
                print('corner line', cornerLine)
                for face in self.faces[:-1]:
                    intersection, ratio = intersectFaceLine([self.corners[cornerIndex] for cornerIndex in face], cornerLine, return_ratio=True)

                    #print([self.corners[cornerIndex] for cornerIndex in face])
                    #print(intersection, ratio)
                    # print(cornerLine)
                    # exit(1)
                    # print(ratio)
                    if intersection:
                        intersectionRatios.append(ratio)
                        pass
                    intersection, ratio = intersectFaceLine([self.cornersOpp[cornerIndex] for cornerIndex in face], cornerLine, return_ratio=True)
                    if intersection:
                        intersectionRatios.append(ratio)
                        pass
                    for c in xrange(len(face)):
                        intersection, ratio = intersectFaceLine([self.corners[face[c]], self.corners[face[(c + 1) % len(face)]], self.cornersOpp[face[(c + 1) % len(face)]], self.cornersOpp[face[c]]], cornerLine, return_ratio=True)
                        if intersection:
                            intersectionRatios.append(ratio)
                            pass
                        continue
                    continue
                continue
            intersectionRatios = np.array(intersectionRatios)
            intersectionRatio = intersectionRatios[intersectionRatios.argmin()]
            newCorners = np.array([self.corners[cornerIndex] + intersectionRatio * normal * lineLength for cornerIndex in newFace])
            self.cornersOpp = np.concatenate([self.cornersOpp, newCorners], axis=0)
            pass
        # print(self.corners[4:])
        # print(self.cornersOpp[4:])
        # exit(1)
        return

    def updateCube(self, faceIndex, imageCenter):
        self.selectedFaceIndex = faceIndex

        updateFace = self.faces[faceIndex]
        normal, offset = self.fitPlane([self.corners[_] for _ in updateFace], return_normal_offset=True)
        lineLength = 10

        if faceIndex == 0:
            ceilingOffset = (self.points * np.expand_dims(normal, 0)).sum(axis=-1).max()
            #print(ceilingOffset)

            for cornerIndex in updateFace:
                self.cornersOpp[cornerIndex] = self.corners[cornerIndex] + (ceilingOffset - offset) * normal
                continue
        else:
            intersectionRatios = []
            for cornerIndex in updateFace:
                corner = self.corners[cornerIndex]
                if np.dot(corner - imageCenter, normal) < 0:
                    normal *= -1
                    pass
                cornerLine = [corner, corner + normal * lineLength]
                for _, face in enumerate(self.faces):
                    if _ == faceIndex:
                        continue
                    intersection, ratio = intersectFaceLine([self.corners[cornerIndex] for cornerIndex in face], cornerLine, return_ratio=True)
                    if intersection:
                        intersectionRatios.append(ratio)
                        pass
                    intersection, ratio = intersectFaceLine([self.cornersOpp[cornerIndex] for cornerIndex in face], cornerLine, return_ratio=True)
                    if intersection:
                        intersectionRatios.append(ratio)
                        pass
                    for c in xrange(len(face)):
                        intersection, ratio = intersectFaceLine([self.corners[face[c]], self.corners[face[(c + 1) % len(face)]], self.cornersOpp[face[(c + 1) % len(face)]], self.cornersOpp[face[c]]], cornerLine, return_ratio=True)
                        if intersection:
                            intersectionRatios.append(ratio)
                            pass
                        continue
                    continue
                continue
            intersectionRatios = np.array(intersectionRatios)
            intersectionRatio = intersectionRatios[intersectionRatios.argmin()]
            for cornerIndex in updateFace:
                self.cornersOpp[cornerIndex] = self.corners[cornerIndex] + intersectionRatio * normal * lineLength
                continue
            pass
        return

    def moveCorner(self, point, extrinsics_inv, intrinsics, imageIndex, axisAligned=True, recording=False, concave=False):
        #moveOnPlane = moveOnPlane and not self.annotatingRoomLayout

        imageCenter = extrinsics_inv[:3, 3]
        direction = np.array([(point[0] - intrinsics[0, 2]) / intrinsics[0, 0], (point[1] - intrinsics[1, 2]) / intrinsics[1, 1], 1])
        direction /= max(np.linalg.norm(direction), 1e-4)
        direction = np.matmul(extrinsics_inv[:3, :3], direction)

        imageInfo = np.concatenate([imageCenter, direction], axis=0)
        if self.selectedCornerIndex not in self.cornerFixedImages:
            self.cornerFixedImages[self.selectedCornerIndex] = {}
            pass


        if concave:
            for faceIndex, face in enumerate(self.faces):
                if self.selectedCornerIndex not in face:
                    continue
                normal, offset = self.fitPlane([self.corners[_] for _ in face], return_normal_offset=True)
                alpha = (offset - np.dot(normal, imageCenter)) / np.dot(normal, direction)
                newCorner = imageCenter + alpha * direction

                index = face.index(self.selectedCornerIndex)
                adjustedCorners = []
                adjustedCornersOpp = []
                newFace = []
                for c in xrange(len(face)):
                    if c == (index - 1) % len(face) or c == (index + 1) % len(face):
                        neighborCorner = self.corners[face[c % len(face)]]
                        edgeDirection = self.corners[self.selectedCornerIndex] - neighborCorner
                        edgeDirection = edgeDirection / np.maximum(np.linalg.norm(edgeDirection), 1e-4)
                        dotProduct = np.dot(edgeDirection, newCorner - neighborCorner)
                        if c == (index - 1) % len(face):
                            newFace.append(face[c])
                            newFace.append(len(self.corners) + len(adjustedCorners))
                        else:
                            newFace.append(len(self.corners) + len(adjustedCorners))
                            newFace.append(face[c])
                            pass
                        adjustedCorners.append(neighborCorner + dotProduct * edgeDirection)
                        adjustedCornersOpp.append(self.cornersOpp[face[c % len(face)]] + dotProduct * edgeDirection)
                    else:
                        newFace.append(face[c])
                        pass
                    continue
                self.cornersOpp[self.selectedCornerIndex] = newCorner + self.cornersOpp[self.selectedCornerIndex] - self.corners[self.selectedCornerIndex]
                self.corners[self.selectedCornerIndex] = newCorner
                self.corners = np.concatenate([self.corners, np.array(adjustedCorners)], axis=0)
                self.cornersOpp = np.concatenate([self.cornersOpp, np.array(adjustedCornersOpp)], axis=0)
                self.faces[faceIndex] = newFace
                #print(newCorner, self.corners[self.selectedCornerIndex])
                #print(newFace, np.array(adjustedCorners), np.array(adjustedCornersOpp))
                #exit(1)
                continue
            #self.updateFaces(self.selectedCornerIndex, imageCenter=imageCenter)
            return


        newCornerInfo = (1000000, None)
        for face in self.faces:
            if self.selectedCornerIndex not in face:
                continue
            normal, offset = self.fitPlane([self.corners[_] for _ in face], return_normal_offset=True)
            alpha = (offset - np.dot(normal, imageCenter)) / np.dot(normal, direction)
            newCorner = imageCenter + alpha * direction
            if axisAligned and len(self.faces) > 1:
                adjustedCorners = []
                index = face.index(self.selectedCornerIndex)
                for c in [index - 1, index + 1]:
                    neighborCorner = self.corners[face[c % len(face)]]
                    dotProducts = (self.dominantNormals * np.expand_dims(newCorner - neighborCorner, 0)).sum(-1)
                    directionIndex = np.abs(dotProducts).argmax()
                    adjustedCorners.append((c, neighborCorner + dotProducts[directionIndex] * self.dominantNormals[directionIndex]))
                    continue
                distances = np.array([np.linalg.norm(corner[1] - newCorner) for corner in adjustedCorners])
                newCorner = adjustedCorners[distances.argmin()][1]
                c = adjustedCorners[distances.argmax()][0] % len(face)
                dotProducts = (self.dominantNormals * np.expand_dims(self.corners[face[c]] - newCorner, 0)).sum(-1)
                directionIndex = np.abs(dotProducts).argmax()
                self.cornersOpp[face[c]] = newCorner + dotProducts[directionIndex] * self.dominantNormals[directionIndex] + self.cornersOpp[face[c]] - self.corners[face[c]]
                self.corners[face[c]] = newCorner + dotProducts[directionIndex] * self.dominantNormals[directionIndex]
                pass

            distance = np.linalg.norm(newCorner - self.corners[self.selectedCornerIndex])
            if distance < newCornerInfo[0]:
                newCornerInfo = (distance, newCorner)
                pass
            continue
        if newCornerInfo[0] < 1000000:
            self.cornersOpp[self.selectedCornerIndex] = newCornerInfo[1] + self.cornersOpp[self.selectedCornerIndex] - self.corners[self.selectedCornerIndex]
            self.corners[self.selectedCornerIndex] = newCornerInfo[1]
            #self.updateFaces(self.selectedCornerIndex, imageCenter=imageCenter)
            pass

        if recording:
            self.cornerFixedImages[self.selectedCornerIndex][imageIndex] = imageInfo
            pass
        return


        # cornerEdges = []
        # edgeUpdated = False
        # for face in self.faces:
        #     for c in xrange(4):
        #         edge = (min(face[c], face[(c + 1) % 4]), max(face[c], face[(c + 1) % 4]))
        #         if edge in self.edgeFixedImages:
        #             self.updateEdge(edge)
        #             edgeUpdated = True
        #             pass
        #         continue
        #     continue

        # if edgeUpdated:
        #     return



        # if len(self.cornerFixedImages[cornerIndex]) == 1:
        #     alpha = (direction * (self.corners[cornerIndex] - imageCenter)).sum()
        #     # print(extrinsics_inv)
        #     # print(imageCenter, direction)
        #     # print(cornerIndex)
        #     # print(self.corners[cornerIndex])
        #     self.corners[cornerIndex] = imageCenter + alpha * direction
        #     # print(self.corners[cornerIndex])
        #     # exit(1)
        # else:
        #     imageInfoArray = []
        #     for imageIndex, imageInfo in self.cornerFixedImages[cornerIndex].iteritems():
        #         imageInfoArray.append(imageInfo)
        #         continue
        #     A = np.zeros((len(imageInfoArray) * 3, 3 + len(imageInfoArray)))
        #     b = np.zeros(len(imageInfoArray) * 3)
        #     for imageIndex, imageInfo in enumerate(imageInfoArray):
        #         for c in xrange(3):
        #             A[imageIndex * 3 + c, c] = 1
        #             continue
        #         A[imageIndex * 3:imageIndex * 3 + 3, 3 + imageIndex] = -imageInfo[3:]
        #         b[imageIndex * 3:imageIndex * 3 + 3] = imageInfo[:3]
        #         continue
        #     #print(A)
        #     self.corners[cornerIndex] = np.linalg.lstsq(A, b, rcond=None)[0][:3]
        #     #exit(1)
        #     pass

        # self.updateFaces(cornerIndex)
        return

    # def updateCorner(self, cornerIndex):
    #     print('update corner', self.selectedCornerIndex, len(self.cornerFixedImages[cornerIndex]))

    #     if len(self.cornerFixedImages[cornerIndex]) <= 3:
    #         return

    #     imageInfoArray = []
    #     for imageIndex, imageInfo in self.cornerFixedImages[cornerIndex].iteritems():
    #         imageInfoArray.append(imageInfo)
    #         continue
    #     A = np.zeros((len(imageInfoArray) * 3, 3 + len(imageInfoArray)))
    #     b = np.zeros(len(imageInfoArray) * 3)
    #     for imageIndex, imageInfo in enumerate(imageInfoArray):
    #         for c in xrange(3):
    #             A[imageIndex * 3 + c, c] = 1
    #             continue
    #         A[imageIndex * 3:imageIndex * 3 + 3, 3 + imageIndex] = -imageInfo[3:]
    #         b[imageIndex * 3:imageIndex * 3 + 3] = imageInfo[:3]
    #         continue

    #     self.corners[cornerIndex] = np.linalg.lstsq(A, b, rcond=None)[0][:3]

    #     self.cornerFixedFlags[cornerIndex] = True
    #     self.updateFaces(cornerIndex, refitting=True)
    #     return


    def fitPlane(self, corners, return_corners=False, return_normal_offset=False, weights=None):
        if np.any(weights == None):
            weights = np.ones(len(corners))
            pass
        A = np.array(corners)
        b = np.ones(len(corners))
        A *= np.expand_dims(weights, axis=-1)
        b *= weights
        plane = np.linalg.lstsq(A, b, rcond=None)[0]
        offset = 1 / max(np.linalg.norm(plane), 1e-4)
        normal = plane * offset
        plane = normal * offset
        if not return_corners:
            if return_normal_offset:
                return normal, offset
            else:
                return plane
            pass
        newCorners = []
        for corner in corners:
            distance = offset - (corner * normal).sum()
            newCorners.append(corner + distance * normal)
            continue
        if return_normal_offset:
            return normal, offset, newCorners
        else:
            return plane, newCorners
        return

    def updateFaces(self, cornerIndex, imageCenter, refitting=False):
        cornerFaces = []
        for faceIndex, face in enumerate(self.faces):
            if cornerIndex != -1 and cornerIndex not in face:
                continue
            if refitting:
                normal, offset = self.fitPlane([self.corners[_] for _ in face], return_normal_offset=True, weights=[max(float(_ in self.cornerFixedFlags), 0.01) for _ in face])

                for index in face:
                    corner = self.corners[index]
                    distance = offset - (corner * normal).sum()
                    corner += distance * normal
                    self.corners[index] = corner
                    continue
                pass
            self.updateCube(faceIndex, imageCenter)
            continue
        return

    # def moveEdge(self, edgeInfo, point, extrinsics_inv, intrinsics, imageIndex):
    #     imageCenter = extrinsics_inv[:3, 3]
    #     direction = np.array([(point[0] - intrinsics[0, 2]) / intrinsics[0, 0], (point[1] - intrinsics[1, 2]) / intrinsics[1, 1], 1])
    #     direction /= max(np.linalg.norm(direction), 1e-4)
    #     direction = np.matmul(extrinsics_inv[:3, :3], direction)

    #     imageInfo = np.concatenate([imageCenter, direction, np.array([edgeInfo[1]])], axis=0)
    #     if edgeInfo[0] not in self.edgeFixedImages:
    #         self.edgeFixedImages[edgeInfo[0]] = {}
    #         pass
    #     self.edgeFixedImages[edgeInfo[0]][imageIndex] = imageInfo

    #     #self.updateEdge(edgeInfo[0])
    #     return

    def moveEdge(self, point, extrinsics_inv, intrinsics, imageIndex, distanceScale=1):
        cornerIndex = self.selectedEdgeIndex
        if cornerIndex not in self.cornerFixedFlags:
            return
        imageCenter = extrinsics_inv[:3, 3]
        direction = np.array([(point[0] - intrinsics[0, 2]) / intrinsics[0, 0], (point[1] - intrinsics[1, 2]) / intrinsics[1, 1], 1])
        direction /= max(np.linalg.norm(direction), 1e-4)
        direction = np.matmul(extrinsics_inv[:3, :3], direction)
        corner = self.corners[cornerIndex]
        cornerOpp = self.cornersOpp[cornerIndex]

        ortho = np.cross(corner - imageCenter, direction)
        ortho /= np.linalg.norm(ortho)
        cornerOpp = corner + np.cross(ortho, np.cross(cornerOpp - corner, ortho))
        cornerOpp = (cornerOpp - imageCenter) * distanceScale + imageCenter
        normal = cornerOpp - corner
        normal = normal / np.linalg.norm(normal)
        print(distanceScale, normal)

        # print(normal)
        # print(cornerOpp)
        # print(self.cornersOpp[cornerIndex], self.corners[cornerIndex])
        # print(imageCenter, direction)
        # exit(1)
        for faceIndex, face in enumerate(self.faces):
            if cornerIndex not in face:
                continue
            for otherCornerIndex in face:
                if otherCornerIndex == cornerIndex:
                    continue
                #normal, _ = self.fitPlane([self.corners[_] for _ in face], return_normal_offset=True)
                #print(corner + np.cross(normal, np.cross(self.corners[otherCornerIndex] - corner, normal)), self.corners[otherCornerIndex])

                self.corners[otherCornerIndex] = corner + np.cross(normal, np.cross(self.corners[otherCornerIndex] - corner, normal))
                continue
            self.updateCube(faceIndex, imageCenter)
            continue
        return


    # def updateEdge(self, edge):
    #     numFixedImages = len(self.edgeFixedImages[edge]) + len(self.cornerFixedImages[edge[0]]) + len(self.cornerFixedImages[edge[1]])
    #     A = np.zeros((numFixedImages * 3, 6 + numFixedImages))
    #     b = np.zeros(numFixedImages * 3)
    #     offset = 0
    #     for _, imageInfo in self.edgeFixedImages[edge].iteritems():
    #         for c in xrange(3):
    #             A[offset * 3 + c, c] = 1 - imageInfo[6]
    #             A[offset * 3 + c, 3 + c] = imageInfo[6]
    #             continue
    #         A[offset * 3:offset * 3 + 3, 6 + offset] = -imageInfo[3:6]
    #         b[offset * 3:offset * 3 + 3] = imageInfo[:3]
    #         offset += 1
    #         continue
    #     for direction, cornerIndex in enumerate(edge):
    #         for _, imageInfo in self.cornerFixedImages[cornerIndex].iteritems():
    #             for c in xrange(3):
    #                 A[offset * 3 + c, 3 * direction + c] = 1
    #                 continue
    #             A[offset * 3:offset * 3 + 3, 6 + offset] = -imageInfo[3:6]
    #             b[offset * 3:offset * 3 + 3] = imageInfo[:3]
    #             offset += 1
    #             continue
    #         continue
    #     if numFixedImages * 3 <= 6 + numFixedImages:
    #         A_ori = np.zeros((6, 6 + numFixedImages))
    #         b_ori = np.zeros(6)
    #         weights = np.zeros(2)
    #         for offset in xrange(len(self.edgeFixedImages[edge])):
    #             weights[0] += A[offset * 3, 0]
    #             weights[1] += A[offset * 3, 3]
    #             continue
    #         weights /= max(weights.sum(), 1e-4)
    #         for direction, cornerIndex in enumerate(edge):
    #             for c in xrange(3):
    #                 A_ori[direction * 3 + c, 3 * direction + c] = weights[direction]
    #                 continue
    #             b_ori[direction * 3:direction * 3 + 3] = self.corners[cornerIndex] * weights[direction]
    #             continue
    #         A = np.concatenate([A, A_ori], axis=0)
    #         b = np.concatenate([b, b_ori], axis=0)
    #         pass

    #     X = np.linalg.lstsq(A, b, rcond=None)[0]
    #     # print(A)
    #     # print(b)
    #     # print(self.corners[edge[0]])
    #     # print(self.corners[edge[1]])
    #     # print(X[:6])
    #     self.corners[edge[0]] = X[:3]
    #     self.corners[edge[1]] = X[3:6]
    #     return

    def unprojectPoint(self, point, depth, extrinsics_inv, intrinsics):
        cameraPoint = np.array([(point[0] - intrinsics[0, 2]) / intrinsics[0, 0] * depth, (point[1] - intrinsics[1, 2]) / intrinsics[1, 1] * depth, depth, 1])
        point3D = np.matmul(extrinsics_inv, cameraPoint)
        if point3D[3] == 0:
            return np.zeros(3)
        else:
            return point3D[:3] / point3D[3]
        return

    def addPlane(self, point, depth, extrinsics_inv, intrinsics):
        print('add plane')
        point3D = self.unprojectPoint(point, depth, extrinsics_inv, intrinsics)
        planeIndex = self.segmentation[np.argmin(np.linalg.norm(self.points - np.expand_dims(point3D, 0), axis=-1))]
        if planeIndex == -1:
            return
        plane = self.planes[planeIndex]

        planeOffset = np.linalg.norm(plane)
        planeNormal = plane / max(planeOffset, 1e-4)

        center = plane
        planePoints = self.points[self.segmentation == planeIndex]
        if len(self.faces) == 0:
            directions = [np.array([1, 0, 0]), np.array([0, 1, 0])]
        else:
            dotProducts = np.abs((self.dominantNormals * np.expand_dims(planeNormal, 0)).sum(-1))
            verticalDirectionIndex = dotProducts.argmax()
            if dotProducts.max() > np.cos(np.deg2rad(20)):
                print('axis aligned')
                planeNormal = self.dominantNormals[verticalDirectionIndex]
                center = (planePoints * np.expand_dims(planeNormal, 0)).sum(-1).mean() * planeNormal
                pass
            directions = [self.dominantNormals[c] for c in xrange(3) if c != verticalDirectionIndex]
            pass
        directions = [np.cross(np.cross(direction, planeNormal), planeNormal) for direction in directions]
        directions = [direction / max(np.linalg.norm(direction), 1e-4) for direction in directions]

        ranges = []
        for direction in directions:
            projections = ((planePoints - np.expand_dims(center, 0)) * np.expand_dims(direction, axis=0)).sum(axis=-1)
            ranges.append((projections.min(), projections.max()))
            continue
        newCorners = np.array([center + ranges[0][0] * directions[0] + ranges[1][0] * directions[1], center + ranges[0][1] * directions[0] + ranges[1][0] * directions[1], center + ranges[0][1] * directions[0] + ranges[1][1] * directions[1], center + ranges[0][0] * directions[0] + ranges[1][1] * directions[1]])

        print(newCorners)
        print(plane)
        #print(point3D)

        self.faces.append([len(self.corners) + c for c in xrange(4)])
        if len(self.corners) == 0:
            self.corners = newCorners
        else:
            self.corners = np.concatenate([self.corners, newCorners], axis=0)
            pass
        imageCenter = extrinsics_inv[:3, 3]
        self.addCube(imageCenter)
        return

    def addCorner(self, point, depth, extrinsics_inv, intrinsics, label=0, epsilon=0.1):
        point3D = self.unprojectPoint(point, depth, extrinsics_inv, intrinsics)
        if len(self.concaveFace) >= 3 and np.linalg.norm(point3D - self.concaveFace[0]) < epsilon:
            newCorners = np.array(self.concaveFace)
            self.faces.append([len(self.corners) + c for c in xrange(len(self.concaveFace))])
            self.corners = np.concatenate([self.corners, newCorners], axis=0)
            self.cornersOpp = np.concatenate([self.cornersOpp, newCorners], axis=0)
            self.concaveFace = []
            return True
        else:
            self.concaveFace.append(point3D)
            return False
        return

    def finalizeRoomLayout(self):
        assert(len(self.faces) == 1)

        floorFace = self.faces[0]
        floorCorners = [self.corners[_] for _ in floorFace]
        floorNormal, floorOffset = self.fitPlane(floorCorners, return_normal_offset=True)
        ceilingOffsets = [np.dot(corner, floorNormal) for cornerIndex, corner in enumerate(self.corners) if cornerIndex not in floorFace]
        if len(ceilingOffsets) == 0:
            ceilingOffset = (self.points * np.expand_dims(floorNormal, 0)).sum(axis=-1).max()
        else:
            ceilingOffset = sum(ceilingOffsets) / len(ceilingOffsets)
            pass
        self.corners = copy.deepcopy(floorCorners)
        self.faces.append([cornerIndex + len(floorFace) for cornerIndex in floorFace])
        wallNormals = [[], []]
        for cornerIndex, corner in enumerate(floorCorners):
            self.corners.append(corner + (ceilingOffset - floorOffset) * floorNormal)
            self.faces.append([cornerIndex, (cornerIndex + 1) % len(floorFace), (cornerIndex + 1) % len(floorFace) + len(floorFace), cornerIndex + len(floorFace)])
            wallNormals[cornerIndex % 2].append(np.cross(floorCorners[(cornerIndex + 1) % len(floorFace)] - corner, floorNormal))
            continue
        self.corners = np.array(self.corners)
        self.dominantNormals = [floorNormal]
        for normals in wallNormals:
            normalSum = np.zeros(3)
            for normal in normals:
                normalSum += normal / max(np.linalg.norm(normal), 1e-4)
                continue
            self.dominantNormals.append(normalSum / max(np.linalg.norm(normalSum), 1e-4))
            continue
        self.dominantNormals = np.array(self.dominantNormals)
        return

    def finalize(self):
        if self.selectedCornerIndex != -1:
            self.updateCorner(self.selectedCornerIndex)
            self.selectedCornerIndex = -1
        elif len(self.faces) == 1:
            self.finalizeRoomLayout()
            pass
        # if self.annotatingRoomLayout or True:
        #     self.finalizeRoomLayout()
        #     self.annotatingRoomLayout = False
        # else:
        #     self.finalizeFace()
        #     pass
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
        # print(self.corners)
        # print(self.faces)
        # exit(1)
        return

    def adjustFace(self):
        if self.selectedFaceIndex != -1:
            pass
        return

    def deleteSelected(self):
        if self.selectedFaceIndex != -1:
            face = self.faces[self.selectedFaceIndex]
            self.faces.remove(face)
            self.removeCorners()
            self.selectedFaceIndex = -1
            pass
        return

    def removeLast(self):
        self.selectedFaceIndex = len(self.faces) - 1
        self.deleteSelected()
        self.selectedFaceIndex = -1
        return

    def removeCorners(self):
        cornerIndexMap = {}
        corners = []
        cornersOpp = []
        self.cornerFixedFlags = {}
        self.cornerFixedImages = {}
        for face in self.faces:
            for cornerIndex in face:
                cornerIndexMap[cornerIndex] = len(corners)
                corners.append(self.corners[cornerIndex])
                cornersOpp.append(self.cornersOpp[cornerIndex])
                continue
            continue
        self.corners = np.array(corners)
        self.cornersOpp = np.array(cornersOpp)
        self.faces = [[cornerIndexMap[cornerIndex] for cornerIndex in face] for face in self.faces]
        return

    def getDensityImage(self, width, height):
        # coordinates = self.points[:, :2]
        # self.minXY = coordinates.min(0)
        # self.maxRange = (coordinates.max(0) - self.minXY).max()
        # padding = self.maxRange * 0.05
        # self.minXY -= padding
        # self.maxRange += padding * 2
        # imageSizes = np.expand_dims(np.array([width, height]), 0)
        # coordinates = (coordinates - np.expand_dims(self.minXY, 0)) / self.maxRange * imageSizes
        # coordinates = np.minimum(np.maximum(coordinates, 0), imageSizes - 1).astype(np.int32)
        # density = np.zeros((height * width))
        # np.add.at(density, coordinates[:, 1] * width + coordinates[:, 0], 1)
        # density = np.minimum(density.reshape((height, width)) / 2 * 255, 255).astype(np.uint8)
        # density = np.tile(np.expand_dims(density, axis=-1), [1, 1, 3])

        #self.getLayoutFaces(width, height)
        return None

    def layoutPointTo2D(self, point, width, height):
        return np.array([point[0] / width * self.maxRange + self.minXY[0], point[1] / height * self.maxRange + self.minXY[1]])

    def finalizeLayoutFace(self):
        if len(self.dominantNormals) == 0:
            anglesArray = [[], []]
            for c in xrange(len(self.layoutFace)):
                direction = self.layoutFace[(c + 1) % len(self.layoutFace)] - self.layoutFace[c]
                anglesArray[c % 2].append(np.arctan2(direction[0], -direction[1]))
                continue
            self.dominantNormals = [[0, 0, 1]]
            for angles in anglesArray:
                angle = sum(angles) / len(angles)
                self.dominantNormals.append([np.cos(angle), np.sin(angle), 0])
                continue
            self.dominantNormals = np.array(self.dominantNormals)
            pass
        for c in xrange(len(self.layoutFace)):
            direction = self.layoutFace[(c + 1) % len(self.layoutFace)] - self.layoutFace[c]
            direction3D = np.array([direction[0], direction[1], 0])
            normal = self.dominantNormals[np.abs(np.sum(self.dominantNormals * np.expand_dims(direction3D, 0), axis=-1)).argmax()]
            normal = normal[:2]
            self.layoutFace[(c + 1) % len(self.layoutFace)] = self.layoutFace[c] + np.dot(direction, normal) * normal
            continue
        return

    def addLayoutCorner(self, point, width, height, selectPlane=False, epsilon=10):
        if selectPlane and len(self.layoutFaces) > 0:
            point2D = self.layoutPointTo2D(point, width, height)
            distances = np.linalg.norm(self.points[:, :2] - np.expand_dims(point2D, axis=0), axis=-1)
            zs = self.points[:, 2].copy()
            zs *= (distances < 0.05).astype(np.float32)
            planeIndex = self.segmentation[zs.argmax()]
            self.layoutFaces[-1][1] = planeIndex
            return
        #for prevPoint in self.layoutFace:
        if len(self.layoutFace) >= 3 and np.linalg.norm(point - self.layoutFace[0]) < epsilon:
            self.finalizeLayoutFace()
            self.layoutFaces.append([self.layoutFace, -1])
            self.layoutFace = []
        else:
            self.selectedLayoutCorner = [-1, -1]
            if len(self.layoutFace) == 0:
                for faceIndex, face in enumerate(self.layoutFaces):
                    for cornerIndex, corner in enumerate(face[0]):
                        if np.linalg.norm(point - corner) < epsilon:
                            self.selectedLayoutCorner = [faceIndex, cornerIndex]
                            break
                        continue
                    continue
                pass
            if self.selectedLayoutCorner[0] == -1:
                self.layoutFace.append(point)
                pass
            pass
        return

    def moveLayoutCorner(self, point):
        if self.selectedLayoutCorner[0] == -1 or self.selectedLayoutCorner[1] == -1:
            return
        self.layoutFaces[self.selectedLayoutCorner[0]][0][self.selectedLayoutCorner[1]] = point
        self.finalizeLayoutFace()
        return

    def layoutTo3D(self, width, height):
        print('layout to 3D')
        for face in self.layoutFaces:
            if face[1] == -1:
                continue
            z = self.points[self.segmentation == face[1]].mean(0)[2]
            #print(self.points[self.segmentation == face[1]])
            newCorners = []
            for corner in face[0]:
                corner2D = self.layoutPointTo2D(corner, width, height)
                newCorners.append(np.concatenate([corner2D, np.array([z])]))
                continue
            self.faces.append([len(self.corners) + c for c in xrange(len(face[0]))])
            newCorners = np.array(newCorners)
            if len(self.corners) == 0:
                self.corners = newCorners
            else:
                self.corners = np.concatenate([self.corners, newCorners], axis=0)
                pass
            self.addCube([0, 0, 5])
            face[1] = -1
            continue

        if len(self.layoutFace) > 0:
            minZ = self.points.min(0)[2]
            maxZ = self.points.max(0)[2]
            newCorners = []
            newCornersOpp = []
            self.layoutFace += self.layoutFace[len(self.layoutFaces) - 2:0:-1]

            for corner in self.layoutFace:
                corner2D = self.layoutPointTo2D(corner, width, height)
                newCorners.append(np.concatenate([corner2D, np.array([minZ])]))
                newCornersOpp.append(np.concatenate([corner2D, np.array([maxZ])]))
                continue
            self.faces.append([len(self.corners) + c for c in xrange(len(self.layoutFace))])
            newCorners = np.array(newCorners)
            newCornersOpp = np.array(newCornersOpp)
            if len(self.corners) == 0:
                self.corners = newCorners
                self.cornersOpp = newCornersOpp
            else:
                self.corners = np.concatenate([self.corners, newCorners], axis=0)
                self.cornersOpp = np.concatenate([self.cornersOpp, newCornersOpp], axis=0)
                pass
            self.layoutFace = []
            pass
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

    def getLayoutFaces(self, width, height):
        print('3D to layout')
        # self.layoutFaces = []
        # imageSizes = np.expand_dims(np.array([width, height]), 0)
        # for face in self.faces:
        #     UVs = np.array([self.corners[cornerIndex][:2] for cornerIndex in face])
        #     UVs = (UVs - np.expand_dims(self.minXY, 0)) / self.maxRange * imageSizes
        #     #UVs = np.minimum(np.maximum(UVs, 0), imageSizes - 1).astype(np.int32)
        #     self.layoutFaces.append((UVs.tolist(), -1))
        #     continue
        return
