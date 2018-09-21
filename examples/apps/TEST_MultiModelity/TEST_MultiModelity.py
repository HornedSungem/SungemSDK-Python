#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../../')
import hsapi as hs


WEBCAM = False # Set to True if use Webcam
net = hs.FaceDetector(zoom = True, verbose = 2, thresh=0.55)
net2 = hs.ObjectDetector(device=net.device, zoom = True, verbose = 2, thresh=0.55)
if WEBCAM: video_capture = cv2.VideoCapture(0)

try:
    while True:
        if WEBCAM: _, img = video_capture.read()
        else: img = None
        result = net.run(img)
        img = result[0]
        result2 = net2.run(img)

        img = net.overlay(img, result[1])
        img = net2.overlay(img, result2[1])

        cv2.imshow("Face/Obj Detector", img)
        cv2.waitKey(1)
finally:
    net.quit()
