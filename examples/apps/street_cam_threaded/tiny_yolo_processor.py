#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import sys
sys.path.append('../../../hsapi')
from core import *
import numpy as np
import cv2
import queue
import threading


class tiny_yolo_processor:

    TY_NETWORK_IMAGE_WIDTH = 448
    TY_NETWORK_IMAGE_HEIGHT = 448

    def __init__(self, tiny_yolo_graph_file, hs_device, input_queue, output_queue,
                 inital_box_prob_thresh, initial_max_iou, queue_wait_input, queue_wait_output):

        self._queue_wait_input = queue_wait_input
        self._queue_wait_output = queue_wait_output

        try:
            with open(tiny_yolo_graph_file, mode='rb') as ty_file:
                ty_graph_from_disk = ty_file.read()
            self._ty_graph = hs_device.AllocateGraph(ty_graph_from_disk)

        except:
            print('\n\n')
            print('Error - could not load tiny yolo graph file: ' + tiny_yolo_graph_file)
            print('\n\n')
            raise

        self._box_probability_threshold = inital_box_prob_thresh
        self._max_iou = initial_max_iou

        self._input_queue = input_queue
        self._output_queue = output_queue

        self._worker_thread = threading.Thread(target=self._do_work, args=())

    def cleanup(self):
        self._ty_graph.DeallocateGraph()

    def start_processing(self):
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())

        self._worker_thread.start()

    def stop_processing(self):
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    def do_inference(self, input_image):
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        inference_image = cv2.resize(input_image,
                                 (tiny_yolo_processor.TY_NETWORK_IMAGE_WIDTH,
                                  tiny_yolo_processor.TY_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        inference_image = inference_image[:, :, ::-1]  # convert to RGB
        inference_image = inference_image.astype(np.float32)
        inference_image = np.divide(inference_image, 255.0)

        self._ty_graph.LoadTensor(inference_image.astype(np.float16), 'user object')
        output, userobj = self._ty_graph.GetResult()

        return self._filter_objects(output.astype(np.float32), input_image_width, input_image_height)

    def _do_work(self):
        print('in tiny_yolo_processor worker thread')

        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, self._queue_wait_input)
                filtered_objs = self.do_inference(input_image)
                self._output_queue.put((input_image, filtered_objs), True, self._queue_wait_output)
                self._input_queue.task_done()

            except queue.Empty:
                print('ty_proc, input queue empty')
            except queue.Full:
                print('ty_proc, output queue full')

        print('exiting tiny_yolo_processor worker thread')

    def get_box_probability_threshold(self):
        return self._box_probability_threshold

    def set_box_probability_threshold(self, value):
        self._box_probability_threshold = value

    def get_max_iou(self):
        return self._max_iou

    def set_max_iou(self, value):
        self._max_iou = value

    def _filter_objects(self, inference_result, input_image_width, input_image_height):

        num_inference_results = len(inference_result)
        network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                   "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                   "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

        network_classifications_mask = [0, 1, 1, 1, 0, 1, 1,
                                        1, 0, 1, 0, 1, 1, 1,
                                        1, 0, 1, 0, 1,0]

        num_classifications = len(network_classifications)
        grid_size = 7
        boxes_per_grid_cell = 2

        all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

        classification_probabilities = np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
        num_of_class_probs = len(classification_probabilities)

        box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

        all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
        self._boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

        for box_index in range(boxes_per_grid_cell):
            for class_index in range(num_classifications):
                all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


        probability_threshold_mask = np.array(all_probabilities >= self._box_probability_threshold, dtype='bool')
        box_threshold_mask = np.nonzero(probability_threshold_mask)
        boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        probabilities_above_threshold = all_probabilities[probability_threshold_mask]

        argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
        boxes_above_threshold = boxes_above_threshold[argsort]
        classifications_for_boxes_above = classifications_for_boxes_above[argsort]
        probabilities_above_threshold = probabilities_above_threshold[argsort]

        duplicate_box_mask = self._get_duplicate_box_mask(boxes_above_threshold)

        boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
        classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
        probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

        classes_boxes_and_probs = []
        for i in range(len(boxes_above_threshold)):
            if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
                classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

        return classes_boxes_and_probs

    def _get_duplicate_box_mask(self, box_list):
        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self._get_intersection_over_union(box_list[i], box_list[j]) > self._max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask

    def _boxes_to_pixel_units(self, box_list, image_width, image_height, grid_size):
        boxes_per_cell = 2
        box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

        box_list[:,:,:,0] += box_offset
        box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
        box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

        box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
        box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

        box_list[:,:,:,0] *= image_width
        box_list[:,:,:,1] *= image_height
        box_list[:,:,:,2] *= image_width
        box_list[:,:,:,3] *= image_height

    def _get_intersection_over_union(self, box_1, box_2):
        intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                             max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

        intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                             max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
            intersection_area = 0
        else :
            intersection_area =  intersection_dim_1*intersection_dim_2

        union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area
        iou = intersection_area / union_area

        return iou
