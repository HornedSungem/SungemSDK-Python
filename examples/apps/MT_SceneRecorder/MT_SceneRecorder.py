#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import cv2, sys
sys.path.append('../../../')
import hsapi as hs

useWebcam = True # Set to True if use Webcam
sr = hs.HSProc('GoogleNet', useWebcam, zoom=True, verbose=0, threshSSD=0.5)

try:
	while True:
		result = sr.res_queue.get()
		key = cv2.waitKey(30)
		if key == 255:
			key = -1
		if result[2] is True:
			prob = sr.net.record(result, key, saveFilename='../../misc/record.dat', numBin = 5)
		
		if prob is not None:
			cv2.putText(result[0], '%d' % (prob.argmax() + 1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 7)
			cv2.putText(result[0], '%d' % (prob.argmax() + 1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
		if key != -1 and chr(key).isdigit():
			cv2.putText(result[0], 'Rec: %d' % int(chr(key)), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)
			cv2.rectangle(result[0], (0,0), result[0].shape[1::-1], (0,0,255), 25)
		cv2.imshow("Scene Recorder", result[0])
finally:
	sr.stop()
