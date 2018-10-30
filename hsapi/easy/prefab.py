#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from ..high import *
import os
import sys
import abc
import numpy
import cv2


def GetDefaultGraphRelPath(fileName):
    """
    获取graphs默认路径
    """
    path = os.path.relpath(__file__)
    path = os.path.dirname(path)
    return os.path.join(path, "../../examples/graphs", fileName)


def GetDefaultMiscRelPath(fileName):
    """
    获取misc默认路径
    """
    path = os.path.relpath(__file__)
    path = os.path.dirname(path)
    return os.path.join(path, "../../examples/misc", fileName)


class SSD(object):
    """
    MobileSSD抽象类
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def labels(self):
        """
        分类的标签
        """
        raise NotImplementedError

    @abc.abstractproperty
    def thresh(self):
        """
        分类的阈值
        """
        raise NotImplementedError

    def getBoundingBoxes(self, output, image_size):
        """
        从神经网络的输出的结果中解析位置信息

        - output: 神经网络输出
        - image_size: 图像大小
        """
        num = int(output[0])
        boxes = []
        for box_index in range(num):
            base_index = 7 + box_index * 7
            score = output[base_index+2]
            if numpy.isnan(score) or score <= self.thresh:
                continue
            clas = int(output[base_index + 1])-1
            score = output[base_index + 2]
            x1 = int(output[base_index + 3] * image_size[1])
            y1 = int(output[base_index + 4] * image_size[0])
            x2 = int(output[base_index + 5] * image_size[1])
            y2 = int(output[base_index + 6] * image_size[0])
            boxes.append([clas,score,x1,y1,x2,y2])
        return boxes

    def plot(self, result):
        """
        根据输出结果在图像上叠加位置信息

        - result: 输出结果
        """
        return self.overlay(result[0], result[1])

    def overlay(self, image, boxes):
        """
        在图像上叠加位置信息

        - image: 图像
        - boxes: 位置信息
        """
        image_size = image.shape[:2]

        for box in boxes:
            class_id = box[0]
            percentage = int(box[1] * 100)

            label_text = self.labels[int(class_id)] + " (" + str(percentage) + "%)"
            box_w = box[4]-box[2]
            box_h = box[5]-box[3]

            if (box_w > image_size[0] * 0.8) or (box_h > image_size[1] * 0.8):
                continue

            box_color = (255, 128, 0)
            box_thickness = 2
            cv2.rectangle(image, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), box_color, box_thickness)

            label_background_color = (255, 128, 0)
            label_text_color = (0, 255, 255)

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = box[2]
            label_top = box[3] - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = box[2] + label_size[0]
            label_bottom = box[3] + label_size[1]
            cv2.rectangle(image, (int(label_left - 1), int(label_top - 1)), (int(label_right + 1), int(label_bottom + 1)),
                          label_background_color, -1)
            cv2.putText(image, label_text, (int(label_left), int(label_bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
        return image

    def crop(self, result, square=True):
        """
        根据输出结果裁剪出识别的图像

        - result: 输出结果
        - square: 正方形
        """
        image = result[0]
        boxes = result[1]
        image_size = image.shape[:2]

        crops = []
        info = []

        for box in boxes:
            class_id = box[0]
            if (box[4] - box[2] > image_size[0]*0.8) or (box[5] - box[3] > image_size[1] * 0.8):
                continue
            if square:
                w = box[4] - box[2] + 1
                h = box[5] - box[3] + 1
                max_side = numpy.maximum(h, w)
                box[2] = box[2] + numpy.round((w - max_side) * 0.5)
                box[3] = box[3] + numpy.round((h - max_side) * 0.5)
                box[4] = box[2] + max_side - 1
                box[5] = box[3] + max_side - 1

            box[2] = numpy.maximum(0, box[2])
            box[3] = numpy.maximum(0, box[3])
            box[4] = numpy.minimum(image_size[1] - 1, box[4])
            box[5] = numpy.minimum(image_size[0] - 1, box[5])
            info.append([self.labels[int(class_id)], box[1], box[2:6]])
            crops.append(image[int(box[3]):int(box[5]),int(box[2]):int(box[4]),:])
        return crops, info


class ObjectDetector(Net, SSD):
    """
    预置模型 - 物体检测
    """
    scale = 0.007843
    mean = -1.0
    netSize = (300, 300)
    graphPath = GetDefaultGraphRelPath("graph_object_SSD")

    labels = [  "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"  ]
    thresh = 0.8

    def run(self, image=None, **kwargs):
        image, output = super().run(image, **kwargs)
        boxes = self.getBoundingBoxes(output, image.shape[:2])
        return (image, boxes)


class FaceDetector(Net, SSD):
    """
    预置模型 - 人脸检测
    """
    scale = 0.007843
    mean = -1.0
    netSize = (300, 300)
    graphPath = GetDefaultGraphRelPath("graph_face_SSD")

    labels = ["Face"]
    thresh = 0.8

    def run(self, image=None, **kwargs):
        image, output = super().run(image, **kwargs)
        boxes = self.getBoundingBoxes(output, image.shape[:2])
        return (image, boxes)


class FaceDetectorPlus(Net, SSD):
    """
    预置模型 - 人脸检测+
    """
    scale = 1
    mean = -110.5
    netSize = (320, 320)
    graphPath = GetDefaultGraphRelPath("graph_face_SSD_Plus")

    labels = ["Face"]
    thresh = 0.8

    def run(self, image=None, **kwargs):
        image, output = super().run(image, **kwargs)
        boxes = self.getBoundingBoxes(output, image.shape[:2])
        return (image, boxes)

class AgeDetector(Net):
    scale = 1
    mean = 87.768914374
    netSize = (227, 227)
    graphPath = GetDefaultGraphRelPath("age_graph")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, image=None, **kwargs):
        image, output = super().run(image, **kwargs)
        return (image,output)


class GenderDetector(Net):
    scale = 1
    mean = 94.030366627
    netSize = (227, 227)
    graphPath = GetDefaultGraphRelPath("gender_graph")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, image=None, **kwargs):
        image, output = super().run(image, **kwargs)
        return (image,output)

class Mnist(Net):
    """
    预置模型 - Mnist
    """
    scale = 0.007843
    mean = -1.0
    netSize = (28, 28)
    graphPath = GetDefaultGraphRelPath("graph_mnist")

    def run(self, image, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image, output = super().run(image, **kwargs)
        output = numpy.argmax(output)
        return (image, output)


class GoogleNet(Net):
    scale = 0.007843
    mean = -1.0
    netSize = (224, 224)
    graphPath = GetDefaultGraphRelPath("graph_googlenet")


class SceneRecorder(Net):
    """
    预置模型 - 情景记录器
    """
    scale = 0.007843
    mean = -1.0
    netSize = (224, 224)
    graphPath = GetDefaultGraphRelPath("graph_g")

    threshPerc = 0.3
    metric = "euclidean"
    saveFilename="record.dat"
    numBin = 5

    def record(self, result, key, **kwargs):
        """
        根据GoogleNet输出结果进行ANN检索，返回相似度结果

        - result: 输出的结果
        - key: 指令
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'featBin'):
            self.featDim = result[1].shape[0]
            self.init_recorder()

        if key == -1:
            if self.activated:
                return self.runANN(result[1])
            return None
        if isinstance(key, int):
            key = chr(key)

        if key.isdigit() and int(key) in range(1,self.numBin+1):
            self.msg('Record to bin: ' + key)
            if key in self.featBin:
                self.featBin[key]['feats'].append(result[1])
                self.featBinLength[int(key)-1] += 1
                self.dispBins()
        elif key is 'r' or key is 'R':
            self.compressFeatBin()
            self.buildANN()
        elif key is 's' or key is 'S':
            self.saveBinsToLocal()
        elif key is 'l' or key is 'L':
            self.loadBinsToLocal()
        elif key is 'p' or key is 'P':
            self.resetBins()
        return None

    def init_recorder(self):
        import annoy
        self.msg('Please enter 1-%d to record' % self.numBin)
        self.activated = False
        self.featBinLength = []

        if not hasattr(self, 'featBin'):
            self.featBin = {}
            self.featBin = {str(x):{} for x in range(1,self.numBin+1)}
            for n in range(1,self.numBin+1):
                self.featBin[str(n)]['feats'] = []
                self.featBinLength.append(0)
        else:
            for n in range(1,self.numBin+1):
                featLen = len(self.featBin[str(n)]['feats'])
                self.featBinLength.append(featLen)
                if (featLen > 0):
                    self.featDim = len(self.featBin[str(n)]['feats'][0])
            self.compressFeatBin()
            self.buildANN()
            self.dispBins()

    def compressFeatBin(self):
        binList = []
        for idx in range(self.numBin):
            if self.featBinLength[idx] > 0:
                binList.append(idx)
        if len(binList) > 1:
            # Use interclass distance: pick the first feature from two class and calculate a 'reference background distance'
            minDist = sys.maxsize
            for n in range(len(binList)):
                for m in range(n+1, len(binList)):
                    dist = numpy.linalg.norm(self.featBin[str(binList[n]+1)]['feats'][0] - self.featBin[str(binList[m]+1)]['feats'][0])
                    minDist = dist if (dist < minDist) else minDist

                    self.msg('Compress Feature Bins', '-')
                    self.msg_debug('Bin[%d]-Bin[%d]:%2.2f' % (binList[n]+1, binList[m]+1, dist))

            self.estiBGdist = minDist
            self.thresh = minDist * self.threshPerc
            self.msg('Estimated BG dist: %2.2f' % minDist)
            self.msg('Use %2.2f as inner-dist thresh' % self.thresh)
            if self.thresh < 0.4:
                self.msg('Warning: BG dist too close!','*')

            self.msg('Compressing','.')

            for n in range(len(binList)):
                idx = str(binList[n]+1)
                newList = [self.featBin[idx]['feats'][0]]
                for i in range(1, self.featBinLength[binList[n]]):
                    minDist = sys.maxsize
                    feat2 = self.featBin[idx]['feats'][i]
                    for feat in newList:
                        dist = numpy.linalg.norm(feat - feat2)
                        minDist = dist if (dist < minDist) else minDist
                    if minDist > self.thresh:
                        newList.append(feat2)
                self.featBin[idx]['feats'] = newList

            # Update
            for n in range(self.numBin):
                self.featBinLength[n] = len(self.featBin[str(n+1)]['feats'])
            self.dispBins()
            self.msg('Compress finished','-')
        else:
            self.msg('Please record second class')

    def buildANN(self):
        import annoy
        self.binList = []
        for idx in range(self.numBin):
            if self.featBinLength[idx] > 0:
                self.binList.append(idx)

        self.msg('Building ANN trees','-')
        for n in range(len(self.binList)):
            idx = str(self.binList[n]+1)
            self.featBin[idx]['ann'] = annoy.AnnoyIndex(self.featDim, self.metric)
            cnt = 0
            for i in range(self.featBinLength[self.binList[n]]):
                feat = self.featBin[idx]['feats'][i]
                self.featBin[idx]['ann'].add_item(cnt, feat)
                cnt += 1
            self.featBin[idx]['ann'].build(20)
            self.msg('Bin[%s] finished' % idx)
        self.msg('Building finished','-')
        self.activated = True

    def runANN(self,feat):
        self.msg('Running ANN','-')
        dists = []
        for n in range(self.numBin):
            idx = str(n+1)
            if 'ann' in self.featBin[idx]:
                [_, dist] = self.featBin[idx]['ann'].get_nns_by_vector(feat, 1, search_k=-1, include_distances=True)
                dists.append(-dist[0])
            else:
                dists.append(-sys.maxsize)
        result = self.softmax(numpy.array(dists))
        for n in range(self.numBin):
            self.msg_debug('[%d]: %2.2f' % (n+1, result[n]))

        self.msg('Probabilities','-')
        for n in range(self.numBin):
            self.msg('%s' % ('|'*int(10*result[n])))

        return result

    def saveBinsToLocal(self):
        import pickle
        with open(self.saveFilename, 'wb') as fp:
            featList = []
            for i in range(self.numBin):
                featList.append(self.featBin[str(i+1)]['feats'])
            pickle.dump(featList, fp)
        self.msg('Save complete','+')

    def loadBinsToLocal(self):
        filename = self.saveFilename
        if os.path.isfile(filename):
            import pickle
            with open(filename, 'rb') as fp:
                featList = pickle.load(fp)
            self.numBin = len(featList)
            for i in range(self.numBin):
                self.featBin[str(i+1)]['feats'] = featList[i]
            self.init_recorder()
        else:
            self.msg('Cannot find data file!')

    def resetBins(self):
        del self.featBin
        self.msg('Reset!','+')

    def dispBins(self):
        res = '-'
        for n in range(self.numBin):
            res += '[%d]-' % self.featBinLength[n]
        self.msg(res)

    def softmax(self, x):
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum()


class SketchGuess(Net):
    """
    预置模型 - 简笔画识别
    """
    scale = 0.007843
    mean = -1.0
    netSize = (28, 28)
    graphPath = GetDefaultGraphRelPath("graph_sg")
