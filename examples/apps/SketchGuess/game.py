#!/usr/bin/env python3
# coding=utf-8

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import pygame
from pygame.locals import *
import cv2
import numpy
sys.path.append('../../../')
import hsapi as hs


WEBCAM = False
FPS = 30
WINDOW_BOUNDS = (1280, 720)
ENGLISH = False

class RGBColor:
    black = (0, 0, 0)
    white = (255, 255, 255)
    yellow = (255, 255, 0)
    light_cyan = (0, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)


p1 = 120
p2 = 45
ROI_ratio = 0.15

stage = 0


def classes_list(file):
    fcl = open(file, encoding="utf-8")
    cls_list = fcl.readlines()
    cls = []
    for line in cls_list:
        cls.append(line.split(' ')[0])
    return cls_list, cls


def process(image, result_set):
    global stage, p1, p2, ROI_ratio

    sz = image.shape
    cx = int(sz[0] / 2)
    cy = int(sz[1] / 2)
    ROI = int(sz[0] * ROI_ratio)
    edges = cv2.Canny(image, p1, p2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cropped = edges[cx - ROI:cx + ROI, cy - ROI:cy + ROI, :]

    kernel = numpy.ones((4, 4), numpy.uint8)
    cropped = cv2.dilate(cropped, kernel, iterations=1)
    output = net.run(cropped)[1]

    output_sort = numpy.argsort(output)[::-1]
    output_label = output_sort[:5]

    cv2.rectangle(image, (cy - ROI, cx - ROI), (cy + ROI, cx + ROI), (255, 255, 0), 5)
    stage_title = result_set[stage].split(' ')[0:-1][0]

    guesses = []
    for label in output_label:
        guesses.append("%s %.2f%%" % (result_set[label].split(' ')[0], output[label] * 100))
        if label == stage and output[label] > 0.1:
            print('恭喜通过第%d关 %s' % (stage + 1, stage_title))
            stage += 1
    return cropped, guesses


def play():
    classes_file = "class_list.txt" if ENGLISH else "class_list_chn.txt"
    result_set, _ = classes_list(hs.GetDefaultMiscRelPath(classes_file))
    font_path = hs.GetDefaultMiscRelPath("SimHei.ttf")
    base_font = pygame.font.Font(font_path, 30)
    title_font = pygame.font.Font(font_path, 50)
    small_font = pygame.font.Font(font_path, 25)

    while True:
        if WEBCAM:
            _, image = capture.read()
        else:
            image = net.device.GetImage(False)
        image = cv2.resize(image, WINDOW_BOUNDS)
        cropped, guesses = process(image, result_set)
        screen.fill(RGBColor.black)

        # 真实图像
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = numpy.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # 图像
        cropped = numpy.rot90(cropped)
        cropped_surf = pygame.surfarray.make_surface(cropped)
        cropped_rect = (screen.get_width() - cropped_surf.get_width() - 20,
                        screen.get_height() - cropped_surf.get_height() - 20)
        pygame.draw.rect(cropped_surf, RGBColor.red, cropped_surf.get_rect(), 1)
        screen.blit(cropped_surf, cropped_rect)

        # 关卡
        stage_str = "第%d关" % (stage + 1)
        stage_surf = base_font.render(stage_str, True, RGBColor.light_cyan)
        screen.blit(stage_surf, (20, 20))

        # 题目
        title_str = result_set[stage].split(' ')[0:-1][0]
        title_surf = title_font.render(title_str, True, RGBColor.yellow)
        title_rect = title_surf.get_rect()
        title_rect.center = (screen.get_width()/2, screen.get_height() / 5)
        screen.blit(title_surf, title_rect)

        # 结果
        for index, guess_str in enumerate(guesses):
            guess_surf = small_font.render(guess_str, True, RGBColor.green)
            guess_rect = (20, screen.get_height() - guess_surf.get_height() * (index + 1) - 20)
            screen.blit(guess_surf, guess_rect)

        pygame.display.update()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    net = hs.SketchGuess(zoom=True, verbose=0)
    if WEBCAM:
        capture = cv2.VideoCapture(0)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_BOUNDS)
    clock = pygame.time.Clock()
    pygame.display.set_caption('你画角蜂鸟猜')
    try:
        play()
    finally:
        net.quit()
