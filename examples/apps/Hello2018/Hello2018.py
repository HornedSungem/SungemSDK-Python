#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

# Import libs
import cv2, sys, numpy as np
sys.path.append('../../../')
import hsapi as hs

# Load CNN to device and set scale / mean
net = hs.Mnist()
imgRoot = '../../misc/2018_mnist/%d.jpg'

print('Hello')
for n in [1,2,3,4]:
    imgname = imgRoot % n
    img = cv2.imread(imgname)
    result = net.run(img)
    print(result[1])

net.quit()
