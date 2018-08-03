# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

# Import libs
import cv2, sys, numpy as np
sys.path.append('../../../')
import hsapi as hs

# Load CNN to device and set scale / mean
net = hs.HS('mnist', zoom = False, verbose = 2)

try:
	while(1):
		image = net.getImage()
		cv2.imshow('image',image)
		cv2.waitKey(1)
finally:
	net.quit()
