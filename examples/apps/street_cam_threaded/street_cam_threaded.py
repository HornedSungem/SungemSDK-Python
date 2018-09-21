#! /usr/bin/env python3

import sys
sys.path.append('../../../hsapi')
from core import *


import numpy as np
from sys import argv
import os
import cv2
import time
import datetime
import queue
from googlenet_processor import googlenet_processor
from tiny_yolo_processor import tiny_yolo_processor
from video_processor import video_processor

TINY_YOLO_GRAPH_FILE = '../../graphs/graph_yolo'
GOOGLENET_GRAPH_FILE = '../../graphs/graph_googlenet'

VIDEO_QUEUE_PUT_WAIT_MAX = 4
VIDEO_QUEUE_FULL_SLEEP_SECONDS = 0.01

cv_window_name = 'street cam threaded - Q to quit'

VIDEO_QUEUE_SIZE = 2
GN_INPUT_QUEUE_SIZE = 10
GN_OUTPUT_QUEUE_SIZE = 10
TY_OUTPUT_QUEUE_SIZE = 10

QUEUE_WAIT_MAX = 2

gn_input_queue = queue.Queue(GN_INPUT_QUEUE_SIZE)
gn_output_queue = queue.Queue(GN_OUTPUT_QUEUE_SIZE)

ty_proc = None
gn_proc_list = []
gn_device_list = []

video_proc = None
video_queue = None

do_gn = False

input_video_path = '.'

resize_output = False
resize_output_width = 0
resize_output_height = 0

pause_mode = False

font_scale = 0.55

GN_PROBABILITY_MIN = 0.5
TY_INITIAL_BOX_PROBABILITY_THRESHOLD = 0.13
TY_INITIAL_MAX_IOU = 0.15


def overlay_on_image(display_image, filtered_objects):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + DISPLAY_BOX_HEIGHT_PAD

        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        box_color = (0, 255, 0)
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        label_background_color = (70, 120, 70) 
        label_text_color = (255, 255, 255)

        if (filtered_objects[obj_index][8] > GN_PROBABILITY_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_text_color, 1)

    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


def get_googlenet_classifications(source_image, filtered_objects):
    global gn_input_queue, gn_output_queue

    if (not do_gn):
        for obj_index in range(len(filtered_objects)):
            filtered_objects[obj_index] += (0, '', 0.0)
        return

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    WIDTH_PAD = int(source_image_width * 0.08)  #80 #20
    HEIGHT_PAD = int(source_image_height* 0.08) #80 #30

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]

        gn_input_queue.put(one_image, True, QUEUE_WAIT_MAX)

    for obj_index in range(len(filtered_objects)):
        result_list = gn_output_queue.get(True, QUEUE_WAIT_MAX)
        filtered_objects[obj_index] += result_list

    return


def get_googlenet_classifications_no_queue(gn_proc, source_image, filtered_objects):
    global gn_input_queue, gn_output_queue

    if (not do_gn):
        for obj_index in range(len(filtered_objects)):
            filtered_objects[obj_index] += (0, '', 0.0)
        return

    WIDTH_PAD = 20
    HEIGHT_PAD = 30

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    image_id = datetime.datetime.now().timestamp()

    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]

        filtered_objects[obj_index] += gn_proc.googlenet_inference(one_image, image_id)

    return


def do_unpause():
    global video_proc, video_queue, pause_mode

    print("unpausing")
    if (not pause_mode):
        return

    pause_mode = False

    video_proc.unpause()

    count = 0
    while (video_queue.empty() and count < 20):
        time.sleep(0.1)
        count += 1


def handle_keys(raw_key):
    global GN_PROBABILITY_MIN, ty_proc, do_gn, pause_mode, video_proc, video_queue, font_scale
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    elif (ascii_code == ord('B')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() + 0.05)
        print("New tiny yolo box probability threshold is " + str(ty_proc.get_box_probability_threshold()))
    elif (ascii_code == ord('b')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() - 0.05)
        print("New tiny yolo box probability threshold  is " + str(ty_proc.get_box_probability_threshold()))

    elif (ascii_code == ord('G')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN + 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('g')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN - 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))

    elif (ascii_code == ord('I')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() + 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))
    elif (ascii_code == ord('i')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() - 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))

    elif (ascii_code == ord('T')):
        font_scale += 0.1
        print("New text scale is: " + str(font_scale))
    elif (ascii_code == ord('t')):
        font_scale -= 0.1
        print("New text scale is: " + str(font_scale))

    elif (ascii_code == ord('p')):
        if (not pause_mode):
            print("pausing")
            pause_mode = True
            video_proc.pause()

        else:
            do_unpause()

    elif (ascii_code == ord('2')):
        do_gn = not do_gn
        print("New do googlenet value is " + str(do_gn))

    return True


def print_usage():
    print('\nusage: ')
    print('python3 street_cam_threaded.py [help][googlenet=on|off][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - Prints this message')
    print('  resize_window - Resizes the GUI window to specified dimensions')
    print('                  must be formatted similar to resize_window=1280x720')
    print('                  default behavior is to use source video frame size')
    print('  googlenet - Sets initial state for googlenet processing')
    print('              must be formatted as googlenet=on or googlenet=off')
    print('              When on all tiny yolo objects will be passed to googlenet')
    print('              for further classification, when off only tiny yolo will be used')
    print('              Default behavior is off')
    print('')
    print('Example: ')
    print('python3 street_cam_threaded.py googlenet=on resize_window=1920x1080')


def print_info():
    print('Running street_cam_threaded')
    print('Keys:')
    print("  'Q'/'q' to Quit")
    print("  'B'/'b' to inc/dec the Tiny Yolo box probability threshold")
    print("  'I'/'i' to inc/dec the Tiny Yolo box intersection-over-union threshold")
    print("  'G'/'g' to inc/dec the GoogLeNet probability threshold")
    print("  'T'/'t' to inc/dec the Text size for the labels")
    print("  '2'     to toggle GoogLeNet inferences")
    print("  'p'     to pause/unpause")
    print('')


def handle_args():
    global resize_output, resize_output_width, resize_output_height, do_gn
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue
        elif (str(an_arg).lower() == 'help'):
            return False
        elif (str(an_arg).startswith('googlenet=')):
            arg, val = str(an_arg).split('=', 1)
            if (str(val).lower() == 'on'):
                print('googlenet processing ON')
                do_gn = True
            elif (str(val).lower() == 'off'):
                print('googlenet processing OFF')
                do_gn = False
            else:
                return False
        elif (str(an_arg).startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False
    return True


def main():
    global gn_input_queue, gn_output_queue, ty_proc, gn_proc_list,\
    resize_output, resize_output_width, resize_output_height, video_proc, video_queue

    if (not handle_args()):
        print_usage()
        return 1

    input_video_filename_list = os.listdir(input_video_path)
    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]

    if (len(input_video_filename_list) < 1):
        print('No video (.mp4) files found')
        return 1

    print_info()

    devices = EnumerateDevices()
    if len(devices) < 1:
        print('No devices found')
        return 1

    ty_device = Device(devices[0])
    ty_device.OpenDevice()


    try:
        gn_proc = googlenet_processor(GOOGLENET_GRAPH_FILE, ty_device, gn_input_queue, gn_output_queue,
                                      QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)
    except:
        print('Error initializing HS devices for GoogleNet')


    print('Starting GUI, press Q to quit')

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10, 10)
    cv2.waitKey(1)

    video_queue = queue.Queue(VIDEO_QUEUE_SIZE)

    ty_output_queue = queue.Queue(TY_OUTPUT_QUEUE_SIZE)
    ty_proc = tiny_yolo_processor(TINY_YOLO_GRAPH_FILE, ty_device, video_queue, ty_output_queue,
                                  TY_INITIAL_BOX_PROBABILITY_THRESHOLD, TY_INITIAL_MAX_IOU,
                                  QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)


    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list :
            video_queue.queue.clear()
            ty_output_queue.queue.clear()
            gn_input_queue.queue.clear()
            gn_output_queue.queue.clear()

            video_proc = video_processor(video_queue,
                                        input_video_path + '/' + input_video_file,
                                        VIDEO_QUEUE_PUT_WAIT_MAX,
                                        VIDEO_QUEUE_FULL_SLEEP_SECONDS)
            for gn_proc in gn_proc_list:
                gn_proc.start_processing()

            video_proc.start_processing()
            ty_proc.start_processing()

            frame_count = 0
            start_time = time.time()
            end_time = start_time
            total_paused_time = end_time - start_time

            while True :

                try:
                    (display_image, filtered_objs) = ty_output_queue.get(True, QUEUE_WAIT_MAX)
                except :
                    pass

                get_googlenet_classifications(display_image, filtered_objs)

                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    ty_output_queue.task_done()
                    exit_app = True
                    print('window closed')
                    break

                overlay_on_image(display_image, filtered_objs)

                if (resize_output):
                    display_image = cv2.resize(display_image,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)

                cv2.imshow(cv_window_name, display_image)

                ty_output_queue.task_done()

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key) == False):
                        end_time = time.time()
                        exit_app = True
                        print('user pressed Q')
                        break
                    if (pause_mode):
                        pause_start = time.time()
                        while (pause_mode):
                            raw_key = cv2.waitKey(1)
                            if (raw_key != -1):
                                if (handle_keys(raw_key) == False):
                                    end_time = time.time()
                                    do_unpause()
                                    exit_app = True
                                    print('user pressed Q during pause')
                                    break
                        if (exit_app):
                            break;
                        pause_stop = time.time()
                        total_paused_time = total_paused_time + (pause_stop - pause_start)

                frame_count = frame_count + 1

                if (video_queue.empty()):
                    end_time = time.time()
                    print('video queue empty')
                    break

            frames_per_second = frame_count / ((end_time - start_time) - total_paused_time)
            print('Frames per Second: ' + str(frames_per_second))

            video_proc.stop_processing()
            video_proc.cleanup()
            cv2.waitKey(1)
            ty_proc.stop_processing()
            cv2.waitKey(1)
            for gn_proc in gn_proc_list:
                cv2.waitKey(1)
                gn_proc.stop_processing()

            if (exit_app) :
                break
        if (exit_app) :
            break

    ty_proc.cleanup()
    ty_device.CloseDevice()

    print('Finished')


if __name__ == "__main__":
    sys.exit(main())
