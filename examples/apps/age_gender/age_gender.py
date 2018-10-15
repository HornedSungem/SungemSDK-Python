#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy, cv2, sys
sys.path.append('../../../')
import hsapi as hs

WEBCAM = False # Set to True if use Webcam


gender_list = ['Male', 'Famale']
age_list = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']


fc_net = hs.FaceDetector(zoom = True, verbose = 2, thresh=0.55)
age_net = hs.AgeDetector(device=fc_net.device)
genfer_net = hs.GenderDetector(device=fc_net.device)

if WEBCAM: video_capture = cv2.VideoCapture(0)

try:
	while True:
		if WEBCAM: _, img = video_capture.read()
		else: img = None
		result = fc_net.run(img)
		img = fc_net.plot(result)
		for i in range(len(result[1])):
			image = result[0]
			face = image[int(result[1][i][3]):int(result[1][i][5]), int(result[1][i][2]):int(result[1][i][4]), :]
			out_age = age_net.run(face)[1]
			age_cls = numpy.argmax(out_age)
			age_out = age_list[age_cls]
			cv2.putText(image, age_out, (int(result[1][i][2]), int(result[1][i][3])), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 255), 2)
			out_gender = genfer_net.run(face)[1]
			gender_cls = numpy.argmax(out_gender)
			gender_out = gender_list[gender_cls]
			cv2.putText(image, gender_out, (int(result[1][i][2]), int(result[1][i][3]-30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 255), 2)
		cv2.imshow("Face Detector", img)
		cv2.waitKey(1)
finally:
	fc_net.quit()