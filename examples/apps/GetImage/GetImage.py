#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

# Import libs
import cv2, sys, numpy as np
sys.path.append('../../../')
import hsapi as hs

device = hs.GetDevice()
device.OpenDevice()

try:
    while(1):
        image = device.GetImage(False)
        cv2.imshow('image',image)
        cv2.waitKey(1)
finally:
    device.CloseDevice()
