#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

from ..core import *
import sys
import numpy
import cv2


class HS:
	def __init__(self, modelName=None, **kwargs):
		# Default
		self.verbose = 2
		self.mean = 1.0
		self.scale = 0.007843
		self.graphFolder = '../../graphs/'
		self.dataFolder = '../../misc/'
		self.zoom = True
		self.labels = None
		self.deviceIdx = 0
		
		# Default SSD threshold
		self.threshSSD = 0.55
		
		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
		
		self.msg(' Horned Sungem ','=')
		self.devices = EnumerateDevices()
		if len(self.devices) == 0:
			print('No devices found')
			quit()
		self.device = Device(self.devices[self.deviceIdx])
		self.device.OpenDevice()
		self.msg('Device index [%d]' % self.deviceIdx)
		
		model_param = self.getParam(modelName)
		
		if model_param is not None:
			self.graphPath = self.graphFolder + model_param[0]
			self.scale = model_param[1]
			self.mean = model_param[2]
			self.isGray = model_param[3]
			self.netSize = model_param[4]
			self.type = model_param[5]
		else:
			self.graphPath = modelName
			self.isGray = False
			self.netSize = None
			if self.graphPath is None:
				print('Please set graph path')
				quit()
			self.type = 0
		
		try:
			self.msg(self.graphPath)
			with open(self.graphPath, mode='rb') as f:
				self.graph_byte = f.read()
				self.msg('Model loaded to Python')
		except:
			print('Error: Failed to load graph, please check file path')
			print('Graph path:%s' % self.graphPath)
			quit()
		
		try:
			self.graph = self.device.AllocateGraph(self.graph_byte, self.scale, -self.mean)
			self.msg('Model allocated to device')
		except:
			print('Error: Failed to allocate graph to device, please try again')
			self.device.CloseDevice()
			quit()
			
		self.msg('','=')
		
	def run(self, img=None, **kwargs):
		if img is None:
			image = self.graph.GetImage(self.zoom)
		else:
			if self.isGray:
				image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			else:
				image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				
			img2load = cv2.resize(image, self.netSize).astype(float)
			img2load *= self.scale
			img2load -= self.mean
			self.graph.LoadTensor(img2load.astype(numpy.float16), None)

		self.imgSize = image.shape[:2]
		output, _ = self.graph.GetResult()
		#print(output)

		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
			
		if self.type in [1,7] : # Classification
			output = numpy.argmax(output)
		elif self.type is 2: # SSD Face
			output = self.getBoundingBoxFromSSDResult(output, self.imgSize)
			self.labels = ['Face']
		elif self.type is 3: # SSD Obj
			output = self.getBoundingBoxFromSSDResult(output, self.imgSize)
			self.labels = ['aeroplane', 'bicycle', 'bird', 'boat',
						  'bottle', 'bus', 'car', 'cat', 'chair',
						  'cow', 'diningtable', 'dog', 'horse',
						  'motorbike', 'person', 'pottedplant',
						  'sheep', 'sofa', 'train', 'tvmonitor']
		elif self.type is 9: # ONet
			from skimage import transform as trans
			self.trans = trans
		
		# RGB -> BRG for OpenCV display
		if img is not None and not self.isGray:
			image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			
		return [image,output]
			
	def getParam(self,modelName):
		# Model filename, scale, mean, net input is gray?, image size, graph ID
		self.msg('Graph:' + modelName)
		if modelName == 'mnist':
			return ['graph_mnist', 0.007843, 1.0, True, (28,28), 1]
		elif modelName == 'FaceDetector':
			return ['graph_face_SSD', 0.007843, 1.0, False, (300,300), 2]
		elif modelName == 'FaceDetector_Plus': 
			return ['graph_face_SSD_Plus',  1, 110.5, False, (320,320), 2] 
		elif modelName == 'ObjectDetector':
			return ['graph_object_SSD', 0.007843, 1.0, False, (300,300), 3]
		elif modelName == 'GoogleNet':
			return ['graph_g', 0.007843, 1.0, False, (224,224), 4]
		elif modelName == 'FaceNet':
			return ['graph_fn', 0.007843, 1.0, False, (160,160), 5]
		elif modelName == 'SketchGuess':
			return ['graph_sg', 0.007843, 1.0, False, (28,28), 6]
		elif modelName == 'OCR':
			return ['graph_ocr',  0.0078125, 1.0, False, (40,40), 7]
		elif modelName == 'squeeze':
			return ['graph_sz',  1, 110.5, False, (227,227), 8]
		elif modelName == 'FaceRec':
			return ['graph_face_rec', 0.007843, 1.0, False, (32,32), 4]
		elif modelName == 'FaceRec_Plus':
			return ['graph_face_rec_Plus', 0.007843, 1.0, False, (48,48), 4]
		elif modelName == 'ONet':
			return ['graph_face_fast_ONet', 0.007843, 1.0, False, (48,48), 9]
		else:
			self.msg('Using user\'s graph file')
			return None 

	# SSD Related:		
	def getBoundingBoxFromSSDResult(self, out_HS, size=(300,300)):
		num = int(out_HS[0])
		boxes = []
		for box_index in range(num):
			base_index = 7 + box_index * 7
			score = out_HS[base_index+2]
			if numpy.isnan(score) or score <= self.threshSSD: 
				continue
			clas = int(out_HS[base_index + 1])-1
			score = out_HS[base_index + 2]
			x1 = int(out_HS[base_index + 3] * size[1])
			y1 = int(out_HS[base_index + 4] * size[0])
			x2 = int(out_HS[base_index + 5] * size[1])
			y2 = int(out_HS[base_index + 6] * size[0])
			boxes.append([clas,score,x1,y1,x2,y2])
		return boxes
		
	def plotSSD(self, result, labels=None):
		if labels is None:
			labels = self.labels

		display_image = result[0]
		boxes = result[1]
		source_image_width = display_image.shape[1]
		source_image_height = display_image.shape[0]

		self.msg_debug('SSD [%d]: Box values' % len(boxes),'*')
		for box in boxes:
			class_id = box[0]
			percentage = int(box[1] * 100)

			label_text = self.labels[int(class_id)] + " (" + str(percentage) + "%)"
			box_w = box[4]-box[2]
			box_h = box[5]-box[3]
			if (box_w > self.imgSize[0]*0.8) or (box_h > self.imgSize[1]*0.8):
				continue	

			self.msg_debug('Box Name: %s' % self.labels[int(class_id)])
			self.msg_debug('%d %d %d %d - w:%d h:%d' %(box[2],box[3],box[4],box[5],box_w,box_h))
			
			box_color = (255, 128, 0) 
			box_thickness = 2
			cv2.rectangle(display_image, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), box_color, box_thickness)

			label_background_color = (255, 128, 0)
			label_text_color = (0, 255, 255)

			label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
			label_left = box[2]
			label_top = box[3] - label_size[1]
			if (label_top < 1):
				label_top = 1
			label_right = box[2] + label_size[0]
			label_bottom = box[3] + label_size[1]
			cv2.rectangle(display_image, (int(label_left - 1), int(label_top - 1)), (int(label_right + 1), int(label_bottom + 1)),
						  label_background_color, -1)

			cv2.putText(display_image, label_text, (int(label_left), int(label_bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
		return display_image
	
	def cropObjects(self, result, labels=None):
		if labels is None:
			labels = self.labels

		display_image = result[0]
		boxes = result[1]
		source_image_width = display_image.shape[1]
		source_image_height = display_image.shape[0]

		crops = []
		info = []

		for box in boxes:
			class_id = box[0]
			percentage = int(box[1] * 100)
			if (box[4]-box[2] > self.imgSize[0]*0.8) or (box[5]-box[3] > self.imgSize[1]*0.8):
				continue	
			info = [self.labels[int(class_id)], box[1], box[2:6]]
			crops.append(result[0][int(box[3]):int(box[5]),int(box[2]):int(box[4]),:])
		return crops, info
	
	# Scene Recorder Related
	def init_recorder(self):
		global annoy
		import annoy # For approximate nearest neighbour processing
		
		self.msg('Please enter 1-%d' % self.numBin)
		self.msg('to record')
		
		self.activated = False
		self.featBinLength = []
		if not hasattr(self, 'featBin'):
			self.numBin = 5
			self.featBin = {}
			self.featBin = {str(x):{} for x in range(1,self.numBin+1)}
			for n in range(1,self.numBin+1):
				self.featBin[str(n)]['feats'] = []
				self.featBinLength.append(0)
		# Load from file
		else:
			for n in range(1,self.numBin+1):
				featLen = len(self.featBin[str(n)]['feats'])
				self.featBinLength.append(featLen)
				if (featLen > 0):
					self.featDim = len(self.featBin[str(n)]['feats'][0])
			self.compressFeatBin()
			self.buildANN()
			self.dispBins()
			
			
	def record(self, result, key, **kwargs):
		self.saveFilename=self.dataFolder + 'record.dat'
		self.metric = 'euclidean'
		self.threshPerc = 0.3
		
		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
			
		
		if not hasattr(self, 'featBin'):
			self.featDim = result[1].shape[0]
			self.init_recorder()
			
		if key == -1:
			if self.activated:
				return self.runANN(result[1])
			return None
		key = chr(key)
		
		if key.isdigit() and int(key) in range(1,self.numBin+1):
			self.msg('Record to bin: ' + key)
			if key in self.featBin:
				self.featBin[key]['feats'].append(result[1])
				self.featBinLength[int(key)-1] += 1
				self.dispBins()
		elif key is 'r' or key is 'R':
			self.compressFeatBin()
			self.buildANN()
		elif key is 's' or key is 'S':
			self.saveBinsToLocal()
		elif key is 'l' or key is 'L':
			self.loadBinsToLocal()
		elif key is 'p' or key is 'P':
			self.resetBins()
		return None
			
	def compressFeatBin(self):
		binList = []
		for idx in range(self.numBin):
			if self.featBinLength[idx] > 0:
				binList.append(idx)
				
		if len(binList) > 1:
			# Use interclass distance: pick the first feature from two class and calculate a 'reference background distance'
			minDist = sys.maxsize
			for n in range(len(binList)):
				for m in range(n+1, len(binList)):
					dist = numpy.linalg.norm(self.featBin[str(binList[n]+1)]['feats'][0] - self.featBin[str(binList[m]+1)]['feats'][0])
					minDist = dist if (dist < minDist) else minDist
					
					self.msg('Compress Feature Bins','-')
					self.msg_debug('Bin[%d]-Bin[%d]:%2.2f' % (binList[n]+1, binList[m]+1, dist))
		
			self.estiBGdist = minDist
			self.thresh = minDist * self.threshPerc
			self.msg('Estimated BG dist: %2.2f' % minDist)
			self.msg('Use %2.2f as inner-dist thresh' % self.thresh)
			if self.thresh < 0.4:
				self.msg('Warning: BG dist too close!','*')
			
			self.msg('Compressing','.')

			for n in range(len(binList)):
				idx = str(binList[n]+1)
				newList = [self.featBin[idx]['feats'][0]]
				for i in range(1, self.featBinLength[binList[n]]):
					minDist = sys.maxsize
					feat2 = self.featBin[idx]['feats'][i]
					for feat in newList:
						dist = numpy.linalg.norm(feat - feat2)
						minDist = dist if (dist < minDist) else minDist
					if minDist > self.thresh:
						newList.append(feat2)
				self.featBin[idx]['feats'] = newList
				
			# Update 
			for n in range(self.numBin):
				self.featBinLength[n] = len(self.featBin[str(n+1)]['feats'])
			self.dispBins()
			self.msg('Compress finished','-')
				
		else:
			self.msg('Please record second class')
		return
		
	def buildANN(self):
		self.binList = []
		for idx in range(self.numBin):
			if self.featBinLength[idx] > 0:
				self.binList.append(idx)
				
		self.msg('Building ANN trees','-')
		for n in range(len(self.binList)):
			idx = str(self.binList[n]+1)
			self.featBin[idx]['ann'] = annoy.AnnoyIndex(self.featDim, self.metric)
			cnt = 0
			for i in range(self.featBinLength[self.binList[n]]):
				feat = self.featBin[idx]['feats'][i]
				self.featBin[idx]['ann'].add_item(cnt, feat)
				cnt += 1
			self.featBin[idx]['ann'].build(20)
			self.msg('Bin[%s] finished' % idx)
		self.msg('Building finished','-')
		self.activated = True
		
	def runANN(self,feat):
		self.msg('Running ANN','-')
		dists = []
		for n in range(self.numBin):
			idx = str(n+1)
			if 'ann' in self.featBin[idx]:
				[index, dist] = self.featBin[idx]['ann'].get_nns_by_vector(feat, 1, search_k=-1, include_distances=True)
				dists.append(-dist[0])
			else:
				dists.append(-sys.maxsize)
		
		
		result = self.softmax(numpy.array(dists))
		for n in range(self.numBin):
			self.msg_debug('[%d]: %2.2f' % (n+1, result[n]))
		
		self.msg('Probabilities','-')
		for n in range(self.numBin):
			self.msg('%s' % ('|'*int(10*result[n])))
			
		return result
		
	def saveBinsToLocal(self):
		import pickle
		with open(self.saveFilename, 'wb') as fp:
			featList = []
			for i in range(self.numBin):
				featList.append(self.featBin[str(i+1)]['feats'])
			pickle.dump(featList, fp)
		self.msg('Save complete','+')
		
	def loadBinsToLocal(self):
		import os.path
		filename = self.saveFilename
		if os.path.isfile(filename):
			import pickle
			with open(filename, 'rb') as fp:
				featList = pickle.load(fp)
			
			self.numBin = len(featList)
			for i in range(self.numBin):
				self.featBin[str(i+1)]['feats'] = featList[i]
			self.init_recorder()
		else:
			self.msg('Cannot find data file!')
		
	def resetBins(self):
		del self.featBin
		self.msg('Reset!','+')
		
	def dispBins(self):
		res = '-'
		for n in range(self.numBin):
			res += '[%d]-' % self.featBinLength[n]
		self.msg(res)

# Face Alignment (ONet)
	def plotONet(self, detRet, onetRet, thresh=0.8):
		display_image = detRet[0]
		boxes = detRet[1]
		source_image_width = display_image.shape[1]
		source_image_height = display_image.shape[0]

		self.msg_debug('ONet (Pad) [%d]: Box values' % len(boxes),'*')
		for n in range(len(detRet[1])):
			ret = onetRet[n][1]
			prob = ret[1]
			if prob < thresh:
				continue

			detBox = detRet[1][n]
			reg = ret[2:6]
			lm = ret[6:]
			label_text = " (" + str(prob)[0:4] + "%)"
		
			# Regressed box	
			bbw = detBox[4] - detBox[2] + 1
			bbh = detBox[5] - detBox[3] + 1

			regBox = [detBox[2] + reg[0] * bbw,
		              detBox[3] + reg[1] * bbh,
		              detBox[4] + reg[2] * bbw,
		              detBox[5] + reg[3] * bbh]
			cv2.rectangle(display_image, (int(regBox[0] - 1), int(regBox[1] - 1)), (int(regBox[2] + 1), int(regBox[3] + 1)), (255,255,0), 2)
			for j in range(5):
				x = int(lm[j*2] * bbw + detBox[2])
				y = int(lm[j*2 + 1] * bbh + detBox[3])
				cv2.circle(display_image, (x, y), 1, (0, 0, 255), 10)
		return display_image
		
	def faceAlignment(self, detRet, onetRet, cropsize = (32,32), thresh=0.8, sface=True):
		display_image = detRet[0]
		
		if sface:
			cropsize = (96,112)
			fs = numpy.array([[ 30.2946, 51.6963 ],
                              [ 65.5318, 51.5014 ],
						      [ 48.0252, 71.7366 ],
						      [ 33.5493, 92.3655 ],
						      [ 62.7299, 92.2041 ]], dtype=float)
		else: # Standard square face
			fs = numpy.array([[ 0.2786825 , 0.29763892 ],
							  [ 0.7191475 , 0.2955507  ],
							  [ 0.50031501, 0.5123564  ],
							  [ 0.31936625, 0.73338038 ],
							  [ 0.68412375, 0.73165107 ]], dtype=float)
			fs *= cropsize[0]
							
		src = numpy.zeros(fs.shape)
		size = display_image.shape
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = numpy.array(
			[[focal_length, 0, center[1]],
			 [0, focal_length, center[0]],
			 [0, 0, 1]], dtype="double"
		)
		faces = []
			 
		for n in range(len(onetRet)):
			detBox = detRet[1][n]
			ret = onetRet[n][1]
			prob = ret[1]
			if prob < thresh:
				faces.append(None)
				continue
			reg = ret[2:6]
			lm = ret[6:]

			# Regressed box	
			bbw = detBox[4] - detBox[2] + 1
			bbh = detBox[5] - detBox[3] + 1

			regBox = [detBox[2] + reg[0] * bbw,
		              detBox[3] + reg[1] * bbh,
		              detBox[4] + reg[2] * bbw,
		              detBox[5] + reg[3] * bbh]
		              
			cropped = display_image[int(detBox[3]):int(detBox[5]), int(detBox[2]):int(detBox[4]), :]
			
			for i in range(5):
				src[i,0] = lm[i*2] * bbw
				src[i,1] = lm[i*2 + 1] * bbh
			
			
			tform = self.trans.estimate_transform('affine', src, fs) # Assume square
			imcrop_aligned = cv2.warpPerspective(cropped, tform.params, cropsize, borderMode=1)

			faces.append(imcrop_aligned)
			
			# Debug
			#for j in range(5):
			#	x = int(src[j,0])
			#	y = int(src[j,1])
			#	cv2.circle(cropped, (x, y), 1, (0, 0, 255), 10)
			#cv2.imshow('1',cropped)
			#cv2.imshow('',imcrop_aligned)
			#cv2.waitKey(0)
		return faces
			
# Util functions	
	def getImage(self):
		return self.device.GetImage(self.zoom)

	# Messager
	def msg(self, string, pad=False):
		if not pad:
			pad = ' '
	
		if self.verbose >= 1:
			print('| %s |' % string.center(30, pad))
			
	def msg_debug(self, string, pad=False):
		if not pad:
			pad = ' '
		if self.verbose >= 2:
			print('* %s *' % string.center(30, pad))
		
	# Math
	def softmax(self, x):
		e_x = numpy.exp(x - numpy.max(x))
		return e_x / e_x.sum()
		
	# Quit
	def quit(self):
		self.graph.DeallocateGraph()
		self.device.CloseDevice()
		return
