#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from . import hs
import numpy
import cv2
import threading
import queue
import time


class HSProc:
	def __init__(self, modelName, useWebcam, **kwargs):
		self.modelName = modelName		
		self.useWebcam = useWebcam
		self.videoIndex = 0
		self.webcamFlip = True
		self.delay = 0.025 #ms
		self.autoStart = True
		self.interpolateResult = True
		self.image_queue = queue.Queue(5)
		self.res_queue = queue.Queue(5)
		
		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
		
		self.net = hs.HS(modelName, **kwargs)
		if self.autoStart:
			if self.useWebcam: 
				self.video_capture = cv2.VideoCapture(self.videoIndex)
				self._video_thread = threading.Thread(target=self._get_image, args=())
				self._video_thread.start()
			self.start()

	def start(self):
		self._worker_thread = threading.Thread(target=self._do_work, args=())
		self._worker_thread.start()
	
	def stop(self):
		self.net.quit()
		
	def _do_work(self):
		while 1:
			if self.useWebcam: 
				img = self.image_queue.get(True)
			else: img = None
			result = self.net.run(img)
			result.append(True)
			if self.interpolateResult:
				while 1:
					try:
						self.res_queue.put(result.copy(), False)
						img = self.image_queue.get(False)
						result[0] = img
						result[2] = False
					except:
						break

	def _get_image(self):
		while 1:
			try:
				_, img = self.video_capture.read()
				if self.webcamFlip:
					img = cv2.flip(img, 1)
				time.sleep(self.delay)
				self.net.msg_debug('Img Q: %d' % self.image_queue.qsize())
				self.net.msg_debug('Ret Q: %d' % self.res_queue.qsize())
				self.image_queue.put(img, False)
			except:
				self.net.msg_debug('No image!')

