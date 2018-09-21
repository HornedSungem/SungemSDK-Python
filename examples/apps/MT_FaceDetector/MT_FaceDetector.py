#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import cv2, sys
sys.path.append('../../../')
import hsapi as hs

net = hs.FaceDetectorPlus(zoom=True, verbose=0, thresh=0.5)
st = hs.SingleTask(net, useWebcam=True)

try:
    while(1):
        ret = st.res_queue.get()
        
        # Get face images for further processing
        crops, info = net.crop(ret)
        if len(crops) > 0 and ret[2] is True:
            # Display first face info/image as example
            print(info)
            try:
                cv2.imshow('A Face', crops[0])
            except:
                pass

        img = net.plot(ret)
        cv2.imshow("Face Detector", img)
        cv2.waitKey(25)
except Exception as error:
    print(error)
finally:
    st.stop()
