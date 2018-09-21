#! /usr/bin/env python3

import cv2
import queue
import threading
import time

class video_processor:
    def __init__(self, output_queue, video_file, queue_put_wait_max = 0.01,
                 request_video_width=640, request_video_height = 480,
                 queue_full_sleep_seconds = 0.1):
        self._queue_full_sleep_seconds = queue_full_sleep_seconds
        self._queue_put_wait_max = queue_put_wait_max
        self._video_file = video_file
        self._request_video_width = request_video_width
        self._request_video_height = request_video_height
        self._pause_mode = False

        self._video_device = cv2.VideoCapture(self._video_file)

        if ((self._video_device == None) or (not self._video_device.isOpened())):
            print('\n\n')
            print('Error - could not open video device.')
            print('If you installed python opencv via pip or pip3 you')
            print('need to uninstall it and install from source with -D WITH_V4L=ON')
            print('Use the provided script: install-opencv-from_source.sh')
            print('\n\n')
            return

        self._video_device.set(cv2.CAP_PROP_FRAME_WIDTH, self._request_video_width)
        self._video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, self._request_video_height)

        self._actual_video_width = self._video_device.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._actual_video_height = self._video_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('actual video resolution: ' + str(self._actual_video_width) + ' x ' + str(self._actual_video_height))

        self._output_queue = output_queue
        self._worker_thread = threading.Thread(target=self._do_work, args=())


    def get_actual_video_width(self):
        return self._actual_video_width

    def get_actual_video_height(self):
        return self._actual_video_height

    def start_processing(self):
        self._end_flag = False
        self._worker_thread.start()

    def stop_processing(self):
        self._end_flag = True
        self._worker_thread.join()

    def pause(self):
        self._pause_mode = True

    def unpause(self):
        self._pause_mode = False

    def _do_work(self):
        print('in video_processor worker thread')
        if (self._video_device == None):
            print('video_processor _video_device is None, returning.')
            return

        while (not self._end_flag):
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                ret_val, input_image = self._video_device.read()

                if (not ret_val):
                    print("No image from video device, exiting")
                    break
                self._output_queue.put(input_image, True, self._queue_put_wait_max)
            except queue.Full:
                time.sleep(self._queue_full_sleep_seconds)

        print('exiting video_processor worker thread')

    def cleanup(self):
        self._video_device.release()
