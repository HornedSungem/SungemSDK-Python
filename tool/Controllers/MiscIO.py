# Copyright 2017 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import ctypes
import sys
import struct
import numpy as np
import warnings
import os
import os.path
import time
from enum import Enum
from ctypes import *
import re
import platform

sys.path.insert(0, '../hsapi')
from core import *

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


from Controllers.FileIO import *
from Controllers.DataTransforms import *

myriad_debug_size = 120
handler = None
no_conf_warning_thrown = False
device = None


def set_string_range(string, length):
    """
    Pads string with null values to a certain size.
    :param string: message to store
    :param length: max size the message can be
    :return: the message pruned/padded to length size.
    """
    return ("{:\0<" + str(length) + '.' + str(length) + "}").format(string)


def get_myriad_info(arguments, myriad_param):
    global device
    if device is None:
        devices = EnumerateDevices()
        if len(devices) == 0:
            throw_error(ErrorTable.USBError, 'No devices found')

        # Choose the first device unless manually specified
        if arguments.device_no is not None:
            print(devices[0], arguments.device_no)
            device = Device(arguments.device_no)
        else:
            device = Device(devices[0])
        try:
            device.OpenDevice()
        except:
            throw_error(ErrorTable.USBError, 'Error opening device')
    return


def run_myriad(blob, arguments, file_gen=False):
    """
    Runs our myriad elf
    :param elf: path to elf.
    :param blob: blob object.
    :return:

    Side Effects: Creates some .npy files containing versions of the myriad output before and after transformation.
    """

    global device
    global myriad_debug_size

    net = blob.network
    f = open(blob.blob_name, 'rb')
    blob_file = f.read()
    if device is None:
        devices = EnumerateDevices()
        if len(devices) == 0:
            throw_error(ErrorTable.USBError, 'No devices found')

        # Choose the first device unless manually specified
        if arguments.device_no is not None:
            device = Device(arguments.device_no)
        else:
            device = Device(devices[0])

        try:
            device.OpenDevice()
        except:
            throw_error(ErrorTable.USBError, 'Error opening device')
    net.inputTensor = net.inputTensor.astype(dtype=np.float16)
    if arguments.save_input is not None:
        net.inputTensor.tofile(arguments.save_input)
    print("USB: Transferring Data...")
    graph = device.AllocateGraph(blob_file)
    graph.SetGraphOption(
        GraphOption.ITERATIONS,
        arguments.number_of_iterations)
    graph.SetGraphOption(
        GraphOption.NETWORK_THROTTLE,
        arguments.network_level_throttling)

    sz = net.outputTensor

    for y in range(arguments.stress_full_run):
        if arguments.timer:
            import time
            ts = time.time()
        graph.LoadTensor(net.inputTensor, None)
        try:
            myriad_output, userobj = graph.GetResult()
        except Exception as e:
            if e.args[0] == dll.hsStatus.MYRIAD_ERROR:
                debugmsg = graph.GetGraphOption(GraphOption.DEBUG_INFO)
                throw_error(ErrorTable.MyriadRuntimeIssue, debugmsg)
            else:
                throw_error(ErrorTable.MyriadRuntimeIssue, e.args[0])

        if arguments.timer:
            ts2 = time.time()
            print("\033[94mTime to Execute : ", str(
                round((ts2 - ts) * 1000, 2)), " ms\033[39m")

        print("USB: Myriad Execution Finished")

    timings = graph.GetGraphOption(GraphOption.TIME_TAKEN)
    if arguments.mode in [OperationMode.temperature_profile]:
        tempBuffer = device.GetDeviceOption(DeviceOption.THERMAL_STATS)
    throttling = device.GetDeviceOption(
        DeviceOption.THERMAL_THROTTLING_LEVEL)
    if throttling == 1:
        print("*********** THERMAL THROTTLING INITIATED ***********")
    if throttling == 2:
        print("************************ WARNING ************************")
        print("*           THERMAL THROTTLING LEVEL 2 REACHED          *")
        print("*********************************************************")

    if arguments.save_output is not None:
        myriad_output.tofile(arguments.save_output)

    if arguments.mode in [OperationMode.temperature_profile]:
        net.temperature_buffer = tempBuffer

    if net.outputIsSsdDetOut:
        no_detections = int(myriad_output[0])
        myriad_output = myriad_output[7 : (no_detections + 1) * 7]
        myriad_output = myriad_output.reshape(no_detections, 7, 1)
        if arguments.parser == Parser.Caffe:
            myriad_output = yxz_to_zyx(myriad_output)
    else:
        myriad_output = myriad_output.reshape(sz)

        if arguments.parser == Parser.Caffe:
            if net.outputNeedsTransforming and len(myriad_output.shape) > 2:
                if len(myriad_output.shape) == 4:
                    myriad_output = myriad_output.reshape(myriad_output.shape[1:])
                if file_gen:
                    np.save(arguments.outputs_name + "_result.npy", myriad_output)
                myriad_output = yxz_to_zyx(myriad_output)
        elif arguments.parser == Parser.TensorFlow:
            myriad_output = myriad_output.reshape(myriad_output.shape[1:])
        else:
            throw_error(ErrorTable.ParserNotSupported, string)

    if file_gen:
        np.save(arguments.outputs_name + "_result.npy", myriad_output)

    print("USB: Myriad Connection Closing.")
    graph.DeallocateGraph()
    device.CloseDevice()
    print("USB: Myriad Connection Closed.")
    return timings, myriad_output


def run_emulation(blob):
    """
    Runs Python Verification ?
    :param blob:
    :return:
    """
    net = blob.network
    print(net.name)
    # Traverse


def parse_img(path, new_size, raw_scale=1, mean=None, channel_swap=None):
    """
    Parse an image with the Python Imaging Libary and convert to 4D numpy array

    :param path:
    :param new_size:
    :return:
    """
    import PIL
    from PIL import Image
    import skimage
    import skimage.io
    import skimage.transform

    if path == "None" or path is None:
        return np.ones(new_size)

    if path == "None" or path is None:
        print("No Image Detected, Using Array of Ones")
        return np.ones(new_size)

    if path.split(".")[-1].lower() in ["png", "jpeg", "jpg", "bmp", "gif"]:

        greyscale = True if new_size[2] == 1 else False
        data = skimage.img_as_float(
            skimage.io.imread(
                path, as_grey=greyscale)).astype(
            np.float32)

    elif path.split(".")[-1] in ["npy"]:
        im = np.load(path)

        if (len(im.shape) == 2):
            if(im.shape[0] != new_size[2] or im.shape[1] != new_size[3]):
                throw_error(ErrorTable.InvalidInputFile)
        elif (len(im.shape) == 3):
            if(im.shape[0] != new_size[2] or im.shape[1] != new_size[3]):
                throw_error(ErrorTable.InvalidInputFile)
        else:
            throw_error(ErrorTable.InvalidInputFile)
        data = np.asarray(im)

    elif path.split(".")[-1] in ["mat"]:
        print("Filetype not officially supported use at your own peril: MAT File")
        import scipy.io
        im = scipy.io.loadmat(path)
        data = np.asarray(im)

    else:
        print("Unsupported")
        throw_error(ErrorTable.InputFileUnsupported)

    if (len(data.shape) == 2):
        # Add axis for greyscale images (size 1)
        data = data[:, :, np.newaxis]

    data = skimage.transform.resize(data, new_size[2:])
    data = np.transpose(data, (2, 0, 1))
    data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))

    data *= raw_scale

    if channel_swap is not None:
        data[0] = data[0][np.argsort(channel_swap), :, :]

    if mean is not None:
        # Try loading mean from .npy file
        if re.search('[a-zA-Z]+', mean):
            try:
                mean = np.load(mean)
            except:
                throw_error(ErrorTable.InvalidNpyFile, mean)

            mean = mean.mean(1).mean(1)
            mean_arr = np.zeros(data.shape[1:])

            for x in range(mean.shape[0]):
                mean_arr[x].fill(mean[x])

            data[0] -= mean_arr

        # Else, try loading mean as tuple
        elif re.search('[,]+', mean):
            try:
                (R,G,B) = mean.split(',')
            except:
                throw_error(ErrorTable.InvalidTuple, mean)

            mean = np.asarray([float(R), float(G), float(B)])
            mean_arr = np.zeros(data.shape[1:])

            for x in range(mean.shape[0]):
                mean_arr[x].fill(mean[x])

            data[0] -= mean_arr

        # Else, load mean as single number
        elif re.search(r'\d+', mean):
            try:
                data = data - float(mean)
            except:
                throw_error(ErrorTable.InvalidMean, mean)

        # Else, invalid mean input
        else:
            throw_error(ErrorTable.InvalidMean, mean)

    return data


def predict_parser(net_desc):
    """
    Based on the filetype, we should be able to predict what parser we want.
    :param net_desc: network description path
    :return: parser enum
    """
    filetype = net_desc.split(".")[-1]
    if filetype in ["prototxt"]:
        return Parser.Caffe
    elif filetype in ["pb", "protobuf", "txt", "meta"]:
        return Parser.TensorFlow
    else:
        throw_error(ErrorTable.UnrecognizedFileType)


def parse_optimization(line, stage):
    """
    Transforms the user string "optimization" into the standardized optimiztion flag format:
    opt_stage_rX_rY_sX_sY
    :param line: user line
    :param stage: stage to obtain further meta info.
    :return: processed string.
    """

    a = "opt_"
    a += stage_as_label(stage.op) + "_"
    a += line

    b = "opt_"
    b += stage_as_label(stage.op) + "_"
    b += str(stage.radixX) + "_"
    b += str(stage.radixY) + "_"
    b += line

    c = "opt_"
    c += stage_as_label(stage.op) + "_"
    c += str(stage.radixX) + "_"
    c += str(stage.radixY) + "_"
    c += str(stage.strideX) + "_"
    c += str(stage.strideY) + "_"
    c += line

    # For deconvolution we have optimizations for MxN kernels but only for
    # stride 1 and same padding.
    d = "opt_"
    d += stage_as_label(stage.op) + "_M_N_"
    d += str(stage.strideX) + "_"
    d += str(stage.strideY) + "_"
    d += line

    return [a, b, c, d]


def debug_label(s, line):
    if line == s:
        return True
    else:
        return False


def check_generic_label(line, stage):
    s = stage_as_label(stage.op) + ":"
    s += str(stage.radixX) + "x"
    s += str(stage.radixY) + "_s"
    s += str(stage.strideX) + "_"
    s += str(stage.strideY)

    if debug_label(s, line):
        return True

    s = stage_as_label(stage.op) + ":"
    s += str(stage.radixX) + "x"
    s += str(stage.radixY) + "_s"
    s += str(stage.strideX)

    if debug_label(s, line):
        return True

    s = stage_as_label(stage.op) + ":"
    s += str(stage.radixX) + "x"
    s += str(stage.radixY)

    if debug_label(s, line):
        return True

    s = stage_as_label(stage.op)

    if debug_label(s, line):
        return True

    return False


def parseOptimizations(myriad_config, opt_controller):
    """
    Parses Optimization File.

    :return:
    """

    print(myriad_config.optimization_list)

    for opt in myriad_config.optimization_list:
        parts = opt.split("_")
        # Make sure all field are at least a default None
        parts += [None] * (7 - len(parts))

        op_name = parts[1]

        conf = {
            "radixX": parts[2],
            "radixY": parts[3],
            "strideX": parts[4],
            "strideY": parts[5],
            "name_of_opt": parts[6],
        }
        opt_controller.add_available_optimization(op_name, conf)


def readOptimisationMask(name, stage, myriad_config, args):
    """
    0 = Nothing Found
    1 = Specific Name Found
    2 = Generic Header Found
    3 = Generic Spec. Header Found
    4 = Optimization Details Found (Spec)
    5 = Optimization Details Found Gen)
    6 = End

    :param name:
    :param stage:
    :param myriad_config:
    :return:
    """

    defaultOptimisation = 0x80000000
    startDefault = defaultOptimisation

    if myriad_config.optimization_list is None or (
        args.conf_file == "optimisation.conf" and not os.path.isfile(
            args.conf_file)):
        return defaultOptimisation

    try:
        with open(args.conf_file) as f:
            found = 0
            optimisations = 0
            opt_selected = False
            shv = 0

            for line in f:
                line = line.rstrip()

                # Find "Generic"
                if line in ["generic optimisations", "generic"]:
                    found = 2
                    optimisations = 0
                    opt_selected = False

                # Find Specific Layer Label
                elif line == name:
                    found = 1
                    optimisations = defaultOptimisation
                    shv = 0
                    opt_selected = False

                # Nothing Found
                elif line == '':
                    if found == 2 or found == 3 or found == 5:
                        if shv == 0:
                            defaultOptimisation = optimisations
                        found = 0
                    elif found == 1 or found == 4:
                        print(
                            "Layer (a)",
                            name,
                            "use the optimisation mask which is: ",
                            format(
                                optimisations,
                                "#0x"))
                        return optimisations

                # Parse Specific Label Optimizations
                elif found == 1:
                    opt_lines = parse_optimization(line, stage)
                    for opt_line in opt_lines:
                        if opt_line in myriad_config.optimization_list and not opt_selected:
                            print(
                                "Spec opt found",
                                opt_line,
                                " 1<<",
                                myriad_config.optimization_list.index(opt_line))
                            if optimisations == defaultOptimisation:
                                optimisations = 0
                            defaultOptimisation = startDefault
                            optimisations = optimisations | (
                                1 << myriad_config.optimization_list.index(opt_line))
                            opt_selected = True
                            found = 4

                            if shv == 0:
                                defaultOptimisation = optimisations | defaultOptimisation
                    if len(line) >= 7 and line[0:7] == 'shaves=':
                        shv = min(int(line[7:]), args.number_of_shaves)
                        # Bits 27-30 contain shaves number (0=default)
                        optimisations = optimisations | (shv << 27)
                        found = 6

                # Parse Generic Opt
                elif found == 3:
                    opt_lines = parse_optimization(line, stage)
                    for opt_line in opt_lines:
                        if opt_line in myriad_config.optimization_list and not opt_selected:
                            print(
                                "Generic Spec opt found",
                                opt_line,
                                " 1<<",
                                myriad_config.optimization_list.index(opt_line))
                            optimisations = optimisations | (
                                1 << myriad_config.optimization_list.index(opt_line))
                            opt_selected = True
                            found = 5

                elif found == 4:
                    if len(line) >= 7 and line[0:7] == 'shaves=':
                        shv = min(int(line[7:]), args.number_of_shaves)
                        # Bits 27-30 contain shaves number (0=default)
                        optimisations = optimisations | (shv << 27)
                        found = 6

                # Parse Generic Label Optimizations
                elif found == 2:
                    if check_generic_label(line, stage):
                        shv = 0
                        found = 3

            print(found, format(defaultOptimisation, "#0x"))
            # If the final empty line is missing
            if found == 2 or found == 5:
                defaultOptimisation = optimisations
            elif found == 6:
                print(
                    "Layer (b)",
                    name,
                    "use the optimisation mask which is: ",
                    format(
                        optimisations,
                        "#0x"))
                return optimisations
    except FileNotFoundError:
        global no_conf_warning_thrown

        if not no_conf_warning_thrown:
            throw_warning(ErrorTable.OptimizationParseError)
            no_conf_warning_thrown = True
        return defaultOptimisation

    print(
        "Layer",
        name,
        "use the generic optimisations which is: ",
        format(
            defaultOptimisation,
            "#0x"))
    return defaultOptimisation
