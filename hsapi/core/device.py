#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from .base import *
from .graph import *
import numpy
import cv2
from ctypes import *
from enum import Enum, unique


@unique
class DeviceOption(Enum):
    """
    Device选项
    """    
    TEMP_LIM_LOWER = 1 
    TEMP_LIM_HIGHER = 2
    BACKOFF_TIME_NORMAL = 3
    BACKOFF_TIME_HIGH = 4
    BACKOFF_TIME_CRITICAL = 5
    TEMPERATURE_DEBUG = 6
    THERMAL_STATS = 1000
    OPTIMISATION_LIST = 1001
    THERMAL_THROTTLING_LEVEL = 1002


class Device:
    """
    设备类
    """
    def __init__(self, name):
        """
        实例化类对象
        """
        self.handle = c_void_p()
        self.name = name
        self.activated = False

    def OpenDevice(self):
        """
        打开设备
        """
        status = dll.hsOpenDevice(bytes(bytearray(self.name, "utf-8")), byref(self.handle))
        if status != Status.OK.value:
            raise Exception(Status(status))
        self.activated = True

    def CloseDevice(self):
        """
        关闭设备
        """
        status = dll.hsCloseDevice(self.handle)
        self.handle = c_void_p()
        if status != Status.OK.value:
            raise Exception(Status(status))
        self.activated = False

    def UpdateApp(self, fileName):
        """
        固件升级

        - fileName: 固件文件
        """
        status = dll.hsUpdateApp(self.handle, c_char_p(fileName.encode('utf-8')))
        self.handle = c_void_p()
        if status != Status.OK.value:
            raise Exception(Status(status))
        self.activated = False

    def SetDeviceOption(self, opt, data):
        """
        设置Device选项

        - opt:  参考DeviceOption
        - data: 值
        """
        if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
            data = c_float(data)
        else:
            data = c_int(data)
        status = dll.hsSetDeviceOption(self.handle, opt.value, pointer(data), sizeof(data))
        if status != Status.OK.value:
            raise Exception(Status(status))

    def GetDeviceOption(self, opt):
        """
        获取Device选项值

        - opt:  参考DeviceOption
        """
        if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
            optdata = c_float()
        elif (opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or
              opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or
              opt == DeviceOption.THERMAL_THROTTLING_LEVEL):
            optdata = c_int()
        else:
            optdata = POINTER(c_byte)()
        optsize = c_uint()
        status = dll.hsGetDeviceOption(self.handle, opt.value, byref(optdata), byref(optsize))
        if status != Status.OK.value:
            raise Exception(Status(status))
        if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
            return optdata.value
        elif (opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or
              opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or
              opt == DeviceOption.THERMAL_THROTTLING_LEVEL):
            return optdata.value
        v = create_string_buffer(optsize.value)
        memmove(v, optdata, optsize.value)
        if opt == DeviceOption.OPTIMISATION_LIST:
            l = []
            for i in range(40):
                if v.raw[i * 50] != 0:
                    ss = v.raw[i * 50:]
                    end = ss.find(b'\x00')
                    val = ss[0:end].decode()
                    if val:
                        l.append(val)
            return l
        if opt == DeviceOption.THERMAL_STATS:
            return numpy.frombuffer(v.raw, dtype=numpy.float32)
        return int.from_bytes(v.raw, byteorder='little')

    def AllocateGraph(self, graphfile, std_value=1.0, mean_value=0.0):
        """
        分配神经网络资源，返回Graph实例

        - graphfile:    graph文件路径
        - std_value:    图像归一化参数
        - mean_value:   图像归一化参数
        """
        hgraph = c_void_p()
        status = dll.hsAllocateGraph(self.handle, byref(hgraph), graphfile, len(graphfile))
        if status != Status.OK.value:
            raise Exception(Status(status))
        return Graph(hgraph, std_value, mean_value)

    def GetImage(self, zoomMode=True):
        """
        获取设备图像

        - zoomMode: 图像缩放模式 (True: 640x360, False: 1920x1080) 
        """
        image = c_void_p()
        mode = c_bool(zoomMode)
        status = dll.hsDeviceGetImage(self.handle, byref(image),mode)
        if status == Status.NO_DATA.value:
            return None, None
        if status != Status.OK.value:
            raise Exception(Status(status))

        if zoomMode == True:
            v = create_string_buffer(640*360*3)
            memmove(v, image, 640*360*3)
            image = numpy.frombuffer(v.raw, dtype=numpy.uint8).reshape(360,640,3)
            import time
            time.sleep(0.015)
        else:
            v = create_string_buffer(1920*1080*3)
            memmove(v, image, 1920*1080*3)
            image = numpy.frombuffer(v.raw, dtype=numpy.uint8)
            r = image[0:1920*1080].reshape(1080,1920)
            g = image[1920*1080:1920*1080+int(1920*1080)].reshape(1080,1920)
            b = image[1920*1080+int(1920*1080):1920*1080+int(2*1920*1080)].reshape(1080, 1920)
            image = cv2.merge(([b,g,r])).astype(numpy.uint8)

        return image


def EnumerateDevices():
    """
    枚举所有已连接的设备
    """
    name = create_string_buffer(28)
    i = 0
    devices = []
    while True:
        if dll.hsGetDeviceName(i, name, 28) != 0:
            break
        devices.append(name.value.decode("utf-8"))
        i = i + 1
    return devices
