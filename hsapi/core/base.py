#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import os
import platform
from ctypes import *
from enum import Enum, unique


filepath = "../../SungemSDK/lib/"
filename = "libhs.so"
if platform.system() == "Linux":
    filepath += "linux/%s" % (platform.machine())
elif platform.system() == "Darwin":
    filepath += "macos"
    filename = "libhs.dylib"
else:
    raise Exception("Unsupported operating system")

dll = CDLL(os.path.join(os.path.dirname(__file__), filepath, filename))


@unique
class GlobalOption(Enum):
	LOG_LEVEL = 0


@unique
class Status(Enum):
	OK = 0
	BUSY = -1
	ERROR = -2
	OUT_OF_MEMORY = -3
	DEVICE_NOT_FOUND = -4
	INVALID_PARAMETERS = -5
	TIMEOUT = -6
	NOT_FOUND = -7
	NO_DATA = -8
	GONE = -9
	UNSUPPORTED_GRAPH_FILE = -10
	MYRIAD_ERROR = -11


@unique
class GraphOption(Enum):
	ITERATIONS = 0
	NETWORK_THROTTLE = 1
	DONT_BLOCK = 2
	TIME_TAKEN = 1000
	DEBUG_INFO = 1001


def SetGlobalOption(opt, data):
	data = c_int(data)
	status = dll.hsSetGlobalOption(opt.value, pointer(data), sizeof(data))
	if status != Status.OK.value:
		raise Exception(Status(status))


def GetGlobalOption(opt):
	if opt == GlobalOption.LOG_LEVEL:
		optsize = c_uint()
		optvalue = c_uint()
		status = dll.hsGetGlobalOption(opt.value, byref(optvalue), byref(optsize))
		if status != Status.OK.value:
			raise Exception(Status(status))
		return optvalue.value
	optsize = c_uint()
	optdata = POINTER(c_byte)()
	status = dll.hsGetDeviceOption(0, opt.value, byref(optdata), byref(optsize))
	if status != Status.OK.value:
		raise Exception(Status(status))
	v = create_string_buffer(optsize.value)
	memmove(v, optdata, optsize.value)
	return v.raw


def BootUpdateApp(fileName):
	status = dll.hsBootUpdateApp(c_char_p(fileName.encode('utf-8')))
	if status != Status.OK.value:
		raise Exception(Status(status))