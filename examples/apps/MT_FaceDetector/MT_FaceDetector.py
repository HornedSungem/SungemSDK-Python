#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import cv2, sys
sys.path.append('../../../')
import hsapi as hs

useWebcam = True
sr = hs.HSProc('FaceDetector', useWebcam, zoom=True, verbose=0, threshSSD=0.5)

try:
	while(1):
		ret = sr.res_queue.get()
		
		# Get face images for further processing
		crops, info = sr.net.cropObjects(ret)
		if len(crops) > 0 and ret[2] is True:
			# Display first face info/image as example
			print(info)
			try:
				cv2.imshow('A Face', crops[0])
			except:
				pass

		img = sr.net.plotSSD(ret)
		cv2.imshow("Face Detector", img)
		cv2.waitKey(25)
finally:
	sr.stop()

