#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

import sys
sys.path.append('../../../hsapi')
from core import *
import numpy as np
import cv2
import os


# will execute on all images in this directory
input_image_path = './images'

tiny_yolo_graph_file= '../../graphs/graph_yolo'
googlenet_graph_file= '../../graphs/graph_googlenet'

TY_NETWORK_IMAGE_WIDTH = 448
TY_NETWORK_IMAGE_HEIGHT = 448

GN_NETWORK_IMAGE_WIDTH = 224
GN_NETWORK_IMAGE_HEIGHT = 224

gn_mean = [0., 0., 0.]

gn_labels = [""]

cv_window_name = 'Birds - Q to quit or any key to advance'



def filter_objects(inference_result, input_image_width, input_image_height):

    num_inference_results = len(inference_result)

    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    # birds only
    network_classifications_mask = [0, 0, 1, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0,0]


    probability_threshold = 0.10 # 0.07

    num_classifications = len(network_classifications) # should be 20
    grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


    probability_threshold_mask = np.array(all_probabilities>=probability_threshold, dtype='bool')
    box_threshold_mask = np.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]


    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

    return classes_boxes_and_probs

def get_duplicate_box_mask(box_list):

    max_iou = 0.25

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask


def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):

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



def get_intersection_over_union(box_1, box_2):


    intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                         max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])


    intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                         max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

    if intersection_dim_1 < 0 or intersection_dim_2 < 0 :

        intersection_area = 0
    else :

        intersection_area =  intersection_dim_1*intersection_dim_2


    union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;


    iou = intersection_area / union_area

    return iou



def display_objects_in_gui(source_image, filtered_objects):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    GOOGLE_PROB_MIN = 0.5


    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    x_ratio = float(source_image_width) / TY_NETWORK_IMAGE_WIDTH
    y_ratio = float(source_image_height) / TY_NETWORK_IMAGE_HEIGHT

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1] * x_ratio)
        center_y = int(filtered_objects[obj_index][2]* y_ratio)
        half_width = int(filtered_objects[obj_index][3]*x_ratio)//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_ratio)//2 + DISPLAY_BOX_HEIGHT_PAD

        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text

        if (filtered_objects[obj_index][8] > GOOGLE_PROB_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    cv2.rectangle(display_image,(0, 0),(140, 30), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(display_image, "Any key to advance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow(cv_window_name, display_image)
    raw_key = cv2.waitKey(3000)
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True




def get_googlenet_classifications(gn_graph, source_image, filtered_objects):

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]
    x_scale = float(source_image_width) / TY_NETWORK_IMAGE_WIDTH
    y_scale = float(source_image_height) / TY_NETWORK_IMAGE_HEIGHT


    WIDTH_PAD = int(20 * x_scale)
    HEIGHT_PAD = int(30 * y_scale)

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1]*x_scale)
        center_y = int(filtered_objects[obj_index][2]*y_scale)
        half_width = int(filtered_objects[obj_index][3]*x_scale)//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_scale)//2 + HEIGHT_PAD

        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]
        filtered_objects[obj_index] += googlenet_inference(gn_graph, one_image)

    return



def googlenet_inference(gn_graph, input_image):


    input_image = cv2.resize(input_image, (GN_NETWORK_IMAGE_WIDTH, GN_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
    input_image = input_image.astype(np.float32)
    input_image[:, :, 0] = (input_image[:, :, 0] - gn_mean[0])
    input_image[:, :, 1] = (input_image[:, :, 1] - gn_mean[1])
    input_image[:, :, 2] = (input_image[:, :, 2] - gn_mean[2])

    # Load tensor and get result.  This executes the inference on the HS
    gn_graph.LoadTensor(input_image.astype(np.float16), 'googlenet')
    output, userobj = gn_graph.GetResult()

    order = output.argsort()[::-1][:1]

    '''
    print('\n------- prediction --------')
    for i in range(0, 1):
        print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[
            order[i]] + '  label index is: ' + str(order[i]))
    '''


    ret_index = order[0]
    ret_label = gn_labels[order[0]]
    ret_prob = output[order[0]]


    if (not (((ret_index >= 127) and (ret_index <= 146)) or
             ((ret_index >= 81) and (ret_index <=101)) or
             ((ret_index >= 7) and (ret_index <=24)) )) :
        ret_prob = 0.0

    return ret_index, ret_label, ret_prob



def main():
    global gn_mean, gn_labels, input_image_filename_list

    print('Running HS birds example')

    input_image_filename_list = os.listdir(input_image_path)
    input_image_filename_list = [input_image_path + '/' + i for i in input_image_filename_list if i.endswith('.jpg')]

    if (len(input_image_filename_list) < 1):
        print('No .jpg files found')
        return 1


    hsdevices = EnumerateDevices()

    if len(hsdevices) == 0:
        print('No devices found')
        quit()

    device = Device(hsdevices[0])
    device.OpenDevice()


    try:
        with open(tiny_yolo_graph_file, mode='rb') as ty_file:
            ty_graph_from_disk = ty_file.read()
        ty_graph = device.AllocateGraph(ty_graph_from_disk)

        with open(googlenet_graph_file, mode='rb') as gn_file:
            gn_graph_from_disk = gn_file.read()
        gn_graph = device.AllocateGraph(gn_graph_from_disk)
    except:
        print("graphs initialization failed")


    gn_mean = np.load('../../misc/ilsvrc_2012_mean.npy').mean(1).mean(1)  # loading the mean file

    gn_labels_file = '../../misc/synset_words.txt'
    gn_labels = np.loadtxt(gn_labels_file, str, delimiter='\t')
    for label_index in range(0, len(gn_labels)):
        temp = gn_labels[label_index].split(',')[0].split(' ', 1)[1]
        gn_labels[label_index] = temp


    print('Q to quit, or any key to advance to next image')

    cv2.namedWindow(cv_window_name)

    for input_image_file in input_image_filename_list :
        input_image = cv2.imread(input_image_file)

        STANDARD_RESIZE_WIDTH = 800
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]
        standard_scale = float(STANDARD_RESIZE_WIDTH) / input_image_width
        new_width = int(input_image_width * standard_scale) # this should be == STANDARD_RESIZE_WIDTH
        new_height = int(input_image_height * standard_scale)
        input_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_LINEAR)

        display_image = input_image
        input_image = cv2.resize(input_image, (TY_NETWORK_IMAGE_WIDTH, TY_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
        input_image = input_image[:, :, ::-1]  # convert to RGB
        input_image = input_image.astype(np.float32)
        input_image = np.divide(input_image, 255.0)

        ty_graph.LoadTensor(input_image.astype(np.float16), 'user object')
        output, userobj = ty_graph.GetResult()

        filtered_objs = filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0]) # fc27 instead of fc12 for yolo_small

        get_googlenet_classifications(gn_graph, display_image, filtered_objs)

        try:
            prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            break;
        prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_asp < 0.0):
            break;

        ret_val = display_objects_in_gui(display_image, filtered_objs)
        if (not ret_val):
            break


    ty_graph.DeallocateGraph()
    gn_graph.DeallocateGraph()
    device.CloseDevice()

    print('Finished')


if __name__ == "__main__":
    sys.exit(main())
