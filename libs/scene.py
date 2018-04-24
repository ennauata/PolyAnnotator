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
            pass
        #self.visiblePolygons = [[[0, 1] for _ in face] for face in self.faces]
        self.visiblePolygons = []
        self.layoutFaces = []
        self.layoutFace = []
        self.concaveFace = []
        self.layoutGraph = {}
        self.prevCorner = None
        self.isDisconnected = False
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
            corner_path.addEllipse(point, d / 2.0, d / 2.0)

            # draw all neighbours
            for n_pt in self.layoutGraph[pt]:
                n_point = QPoint(int(round(n_pt[0] + offsetX)), int(round(n_pt[1] + offsetY)))
                boundary_path.moveTo(point)
                boundary_path.lineTo(n_point)

        painter.drawPath(corner_path)
        painter.drawPath(boundary_path)
        return

    def paint(self, painter, extrinsics, intrinsics, width, height, offsetX, offsetY):
        if len(self.corners) == 0:
            return

        for faceIndex, polygon in enumerate(self.visiblePolygons):

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

    def moveCorner(self, point, extrinsics_inv, intrinsics, imageIndex, axisAligned=True, recording=False, concave=False):

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
                continue
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
            pass

        if recording:
            self.cornerFixedImages[self.selectedCornerIndex][imageIndex] = imageInfo
            pass
        return

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

        for faceIndex, face in enumerate(self.faces):
            if cornerIndex not in face:
                continue
            for otherCornerIndex in face:
                if otherCornerIndex == cornerIndex:
                    continue

                self.corners[otherCornerIndex] = corner + np.cross(normal, np.cross(self.corners[otherCornerIndex] - corner, normal))
                continue
            self.updateCube(faceIndex, imageCenter)
            continue
        return

    def unprojectPoint(self, point, depth, extrinsics_inv, intrinsics):
        cameraPoint = np.array([(point[0] - intrinsics[0, 2]) / intrinsics[0, 0] * depth, (point[1] - intrinsics[1, 2]) / intrinsics[1, 1] * depth, depth, 1])
        point3D = np.matmul(extrinsics_inv, cameraPoint)
        if point3D[3] == 0:
            return np.zeros(3)
        else:
            return point3D[:3] / point3D[3]
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

    def layoutPointTo2D(self, point, width, height):
        return np.array([point[0] / width * self.maxRange + self.minXY[0], point[1] / height * self.maxRange + self.minXY[1]])


    def disconnectGraph(self):
        self.prevCorner = None
        print('Graph Disconnected')


    def addLayoutCornerMod(self, point, width, height, selectPlane=False, epsilon=10):

        # check if is a new corner
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
            # closing one edge with an existing node
            else:
                # add only if connection does not exist
                if self.prevCorner not in self.layoutGraph[currCorner]:
                    self.layoutGraph[currCorner].append(self.prevCorner)
            # add only if connection does not exist
            if currCorner not in self.layoutGraph[self.prevCorner]:
                self.layoutGraph[self.prevCorner].append(currCorner)
        else:
            # graph is disconnected and clicked on non existing node
            if isNewCorner:
                self.layoutGraph[currCorner] = []

        self.prevCorner = currCorner
        return

    def addLayoutCorner(self, point, width, height, selectPlane=False, epsilon=10):

        if len(self.layoutFace) >= 3 and np.linalg.norm(point - self.layoutFace[0]) < epsilon:
            self.layoutFaces.append([self.layoutFace, -1])
            print(self.layoutFace)
            self.layoutFace = []
            print('test_1')
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
            print('test_2')
            if self.selectedLayoutCorner[0] == -1:
                self.layoutFace.append(point)
                pass
            pass
        return

    def moveLayoutCorner(self, point):
        if self.selectedLayoutCorner[0] == -1 or self.selectedLayoutCorner[1] == -1:
            return
        self.layoutFaces[self.selectedLayoutCorner[0]][0][self.selectedLayoutCorner[1]] = point
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

