#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from ..core import *
import abc
import cv2
import numpy


def GetDevice(index=0):
    """
    获取已连接的设备
    """
    devices = EnumerateDevices()
    if len(devices) == 0:
        raise Exception("Device not found")
    return Device(devices[index])


class Net(object):
    """
    抽象类，该类对Device类和Graph类进行了一些简单的封装
    """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def scale(self):
        """
        图像归一化参数
        """
        raise NotImplementedError

    @abc.abstractproperty
    def mean(self):
        """
        图像归一化参数
        """
        raise NotImplementedError

    @abc.abstractproperty
    def netSize(self):
        """
        神经网络输入的图像大小
        """
        raise NotImplementedError

    @abc.abstractproperty
    def graphPath(self):
        """
        graph文件路径
        """
        raise NotImplementedError

    verbose = 2

    def __init__(self, **kwargs):
        """
        实例化类对象
        """
        self.device = None
        self.deviceIdx = 0 # The index only for initializing the device, not for a reference
        self.zoom = True
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.device is None:
            self.device = GetDevice(self.deviceIdx)
        
        if not self.device.activated:
            self.device.OpenDevice()

        with open(self.graphPath, mode='rb') as f:
            graph_data = f.read()
        self.graph = self.device.AllocateGraph(graph_data, self.scale, self.mean)

    def cleanup(self):
        """
        释放神经网络资源
        """
        self.graph.DeallocateGraph()

    def quit(self):
        """
        释放资源并关闭设备
        """
        self.cleanup()
        if self.device.activated:
            self.device.CloseDevice()

    def run(self, image=None, **kwargs):
        """
        执行一次神经网络

        - image: 输入的图像，None则表示使用设备摄像头
        """
        for k, v in kwargs.items(): 
            setattr(self, k, v)

        if image is None:
            image = self.graph.GetImage(self.zoom)
        else:
            img2load = cv2.resize(image, self.netSize).astype(float)
            img2load *= self.scale
            img2load += self.mean
            self.graph.LoadTensor(img2load.astype(numpy.float16), None)

        output, _ = self.graph.GetResult()
        return (image, output)

    def msg(self, string, pad=" "):
        if self.verbose >= 1:
            print('| %s |' % string.center(30, pad))
    
    def msg_debug(self, string, pad=" "):
        if self.verbose >= 2:
            print('* %s *' % string.center(30, pad))
