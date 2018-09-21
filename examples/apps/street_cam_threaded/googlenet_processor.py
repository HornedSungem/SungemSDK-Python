#! /usr/bin/env python3

import sys
sys.path.append('../../../hsapi')
from core import *
import numpy as np
import cv2
import queue
import threading


class googlenet_processor:
    GN_NETWORK_IMAGE_WIDTH = 224
    GN_NETWORK_IMAGE_HEIGHT = 224


    MEAN_FILE_NAME = '../../misc/ilsvrc_2012_mean.npy'

    LABELS_FILE_NAME = '../../misc/synset_words.txt'

    def __init__(self, googlenet_graph_file, hs_device, input_queue, output_queue,
                 queue_wait_input, queue_wait_output):

        self._queue_wait_input = queue_wait_input
        self._queue_wait_output = queue_wait_output

        self._gn_mean = [0., 0., 0.]

        self._gn_labels = [""]

        try:
            self._gn_mean = np.load(googlenet_processor.MEAN_FILE_NAME).mean(1).mean(1)
        except:
            print('\n\n')
            print('Error - could not load means from ' + googlenet_processor.MEAN_FILE_NAME)
            print('\n\n')
            raise

        try:
            self._gn_labels = np.loadtxt(googlenet_processor.LABELS_FILE_NAME, str, delimiter='\t')
            for label_index in range(0, len(self._gn_labels)):
                temp = self._gn_labels[label_index].split(',')[0].split(' ', 1)[1]
                self._gn_labels[label_index] = temp
        except:
            print('\n\n')
            print('Error - could not read labels from: ' + googlenet_processor.LABELS_FILE_NAME)
            print('\n\n')
            raise

        try:
            with open(googlenet_graph_file, mode='rb') as gn_file:
                gn_graph_from_disk = gn_file.read()
            self._gn_graph = hs_device.AllocateGraph(gn_graph_from_disk)

        except:
            print('\n\n')
            print('Error - could not load googlenet graph file: ' + googlenet_graph_file)
            print('\n\n')
            raise

        self._input_queue = input_queue
        self._output_queue = output_queue
        self._worker_thread = threading.Thread(target=self._do_work, args=())

    def cleanup(self):
        self._gn_graph.DeallocateGraph()

    def start_processing(self):
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())
        self._worker_thread.start()

    def stop_processing(self):
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    def _do_work(self):
        print('in googlenet_processor worker thread')
        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, self._queue_wait_input)
                index, label, probability = self.googlenet_inference(input_image, "NPS")
                self._output_queue.put((index, label, probability), True, self._queue_wait_output)
                self._input_queue.task_done()
            except queue.Empty:
                print('googlenet processor: No more images in queue.')
            except queue.Full:
                print('googlenet processor: queue full')

        print('exiting googlenet_processor worker thread')


    def googlenet_inference(self, input_image, user_obj):

        input_image = cv2.resize(input_image, (googlenet_processor.GN_NETWORK_IMAGE_WIDTH,
                                               googlenet_processor.GN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        input_image = input_image.astype(np.float32)
        input_image[:, :, 0] = (input_image[:, :, 0] - self._gn_mean[0])
        input_image[:, :, 1] = (input_image[:, :, 1] - self._gn_mean[1])
        input_image[:, :, 2] = (input_image[:, :, 2] - self._gn_mean[2])

        self._gn_graph.LoadTensor(input_image.astype(np.float16), user_obj)
        output, userobj = self._gn_graph.GetResult()

        order = output.argsort()[::-1][:1]

        return order[0], self._gn_labels[order[0]], output[order[0]]
