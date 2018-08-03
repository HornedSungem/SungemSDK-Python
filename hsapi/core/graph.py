#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from .base import *
import numpy
import cv2
from ctypes import *


class Graph:
	def __init__(self, handle,std_value,mean_value):
		self.handle = handle
		self.userobjs = {}
		self.std_value = std_value
		self.mean_value = mean_value

	def SetGraphOption(self, opt, data):
		data = c_int(data)
		status = dll.hsSetGraphOption(self.handle, opt.value, pointer(data), sizeof(data))
		if status != Status.OK.value:
			raise Exception(Status(status))

	def GetGraphOption(self, opt):
		if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONT_BLOCK:
			optdata = c_int()
		else:
			optdata = POINTER(c_byte)()
		optsize = c_uint()
		status = dll.hsGetGraphOption(self.handle, opt.value, byref(optdata), byref(optsize))
		if status != Status.OK.value:
			raise Exception(Status(status))
		if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONT_BLOCK:
			return optdata.value
		v = create_string_buffer(optsize.value)
		memmove(v, optdata, optsize.value)
		if opt == GraphOption.TIME_TAKEN:
			return numpy.frombuffer(v.raw, dtype=numpy.float32)
		if opt == GraphOption.DEBUG_INFO:
			return v.raw[0:v.raw.find(0)].decode()
		return int.from_bytes(v.raw, byteorder='little')

	def DeallocateGraph(self):
		status = dll.hsDeallocateGraph(self.handle)
		self.handle = 0
		if status != Status.OK.value:
			raise Exception(Status(status))

	def LoadTensor(self, tensor, userobj):
		tensor = tensor.tostring()
		userobj = py_object(userobj)
		key = c_long(addressof(userobj))
		self.userobjs[key.value] = userobj
		status = dll.hsLoadTensor(self.handle, tensor, len(tensor), key)
		if status == Status.BUSY.value:
			return False
		if status != Status.OK.value:
			del self.userobjs[key.value]
			raise Exception(Status(status))
		return True

	def GetResult(self):
		tensor = c_void_p()
		tensorlen = c_uint()
		userobj = c_long()
		status = dll.hsGetResult(self.handle, byref(tensor), byref(tensorlen), byref(userobj))
		if status == Status.NO_DATA.value:
			return None, None
		if status != Status.OK.value:
			raise Exception(Status(status))
		v = create_string_buffer(tensorlen.value)
		memmove(v, tensor, tensorlen.value)
		tensor = numpy.frombuffer(v.raw, dtype=numpy.float16)
		retuserobj = self.userobjs[userobj.value]
		del self.userobjs[userobj.value]
		return tensor, retuserobj.value

	def GetImage(self, zoomMode):
		image = c_void_p()
		userobj = py_object([None])
		key = c_long(addressof(userobj))
		self.userobjs[key.value] = userobj
		std_value = c_float(self.std_value)
		mean_value = c_float(self.mean_value)
		mode = c_bool(zoomMode)
		status = dll.hsGetImage(self.handle, byref(image),key,std_value,mean_value,mode)
		if status == Status.NO_DATA.value:
			return None
		if status != Status.OK.value:
			del self.userobjs[key.value]
			raise Exception(Status(status))

		if zoomMode == True:
			v = create_string_buffer(640*360*3)
			memmove(v, image, 640*360*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8).reshape(360,640,3)
		else:
			v = create_string_buffer(1920*1080*3)
			memmove(v, image, 1920*1080*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8)
			r = image[0:1920*1080].reshape(1080,1920)
			g = image[1920*1080:1920*1080+int(1920*1080)].reshape(1080,1920)
			b = image[1920*1080+int(1920*1080):1920*1080+int(2*1920*1080)].reshape(1080, 1920)
			image = cv2.merge(([b,g,r])).astype(numpy.uint8)
		return image
