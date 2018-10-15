#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../../')
from hsapi import ObjectDetector

WEBCAM = False # Set to True if use Webcam
net = ObjectDetector(zoom = True, verbose = 2)
if WEBCAM: video_capture = cv2.VideoCapture(0)

try:
    while True:
        if WEBCAM: _, img = video_capture.read()
        else: img = None
        result = net.run(img)
        img = net.plot(result)
        cv2.imshow("20 VOC Object Detector", img)
        cv2.waitKey(1)
finally:
    net.quit()
