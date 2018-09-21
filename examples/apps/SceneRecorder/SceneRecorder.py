#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import cv2, sys
sys.path.append('../../../')
from hsapi import SceneRecorder

WEBCAM = False # Set to True if use Webcam
    
net = SceneRecorder(zoom = True, verbose = 2)
if WEBCAM: video_capture = cv2.VideoCapture(0)

try:
    while True:
        if WEBCAM: _, img = video_capture.read()
        else: img = None

        # Get image descriptor
        result = net.run(img)
        key = cv2.waitKey(5)
        if key == 255:
            key = -1
        prob = net.record(result, key, saveFilename='../../misc/record.dat', numBin = 5)
        
        if prob is not None:
            cv2.putText(result[0], '%d' % (prob.argmax() + 1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 7)
            cv2.putText(result[0], '%d' % (prob.argmax() + 1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        if key != -1 and chr(key).isdigit():
            cv2.putText(result[0], 'Rec: %d' % int(chr(key)), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)
            cv2.rectangle(result[0], (0,0), result[0].shape[1::-1], (0,0,255), 25)
        cv2.imshow("Scene Recorder", result[0])
        cv2.waitKey(1)
finally:
    net.quit()
