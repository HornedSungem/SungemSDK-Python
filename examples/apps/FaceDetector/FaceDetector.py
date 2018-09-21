#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../../')
from hsapi import FaceDetector

WEBCAM = False # Set to True if use Webcam
net = FaceDetector(zoom = True, verbose = 2, thresh=0.55)
if WEBCAM: video_capture = cv2.VideoCapture(0)

try:
    while True:
        if WEBCAM: _, img = video_capture.read()
        else: img = None
        result = net.run(img)
        img = net.plot(result)
        cv2.imshow("Face Detector", img)
        cv2.waitKey(1)
finally:
    net.quit()
