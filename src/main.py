import random
import math
import numpy as np
import time
# import torch 
# import torch.nn as nn
import cv2 
# from torch.autograd import Variable
from dataType4 import humanData, itemData # or dataType
from autoencoder import Autoencoder
from matching4 import humanMatching, itemMatching # or matching3
from tracking4 import Scan_for_item_existing,Track_and_Display # or tracking2
import pandas as pd
import random 
import pickle as pkl
import os
import argparse
import re
import string 
import sys
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
# from tracking import Scan_for_item_existing, Tracking_suspect, Display

''' 
Container content :

humanDataset         = {human_id1:humanData1, human_id2:humanData2, ...}
itemDataset          = {item_id1:itemData1, item_id2:itemData2, ...}
missingPeopleDataset = {feature1(np.array):humanData1, feature2(np.array):humanData2, ...}
detection            = [[upleft_x, upleft_y, downright_x, downright_y],[upleft_x, upleft_y, downright_x, downright_y]...]

'''
class Simulation:
	def __init__(self):
		self.flag=1
		self.item0_pos=(100,100)
		self.item1_pos=(50,50)
		self.people0_pos=(105,105)
		self.people1_pos=(30,30)
		#Generate some data (Not complete,increment 0-10 each time step)
		self.human={0:["Human A",self.people0_pos[0],self.people0_pos[1]],1:["Human B",self.people1_pos[0],self.people1_pos[1]]}
		
	def yolo(self):
		#print(num)
		x_range=[0,200]
		y_range=[0,200]
		human_list=[]
		item_list=[]
		res_human=[]
		res_item=[]
		for i in range(2):
		   #print(human[i])
		   if self.human[i][1]<x_range[1] and self.human[i][1]>x_range[0] and self.human[i][2]<y_range[1] and self.human[i][2]>y_range[0]:   
			   human_list.append(self.human[i])
		if self.item[0][1]<x_range[1] and self.item[0][1]>x_range[0] and self.item[0][2]<y_range[1] and self.item[0][2]>y_range[0]:   
		   item_list.append(self.item[0])
		for human in human_list:
			print("in yolo",human)
			res_human.append([human[1]-20,human[2]-50,human[1]+20,human[2]+50])
		for item in item_list:
			print("in yolo",item)
			res_item.append([item[1]-20,item[2]-20,item[1]+20,item[2]+20])
		return (res_human,res_item)

	def iteration(self,count):
		
		print("=================")
		print("time stamp:",count)
		if self.flag==0 and count<20: 
			if self.human[0][1]<100:
				pass
			else:
				print("A back")
				self.human[0][1]-=random.random()%5+10 
		elif self.human[0][1]<200 and count<50:
			print("A leave")
			self.human[0][1]+=random.random()%5+10
		else:
			self.flag=0
			if count>20:
					
				if self.human[1][1]>=100:
						
					if count>30:
						print("B stole and fleet")
						self.human[1][2]+=random.random()%5+10
						self.item[0][1]=self.human[1][1]
						self.item[0][2]=self.human[1][2]
					else:
						print("B wait for the chance")
				else:
					print("A leave B close")                                  
					self.human[0][1]+=random.random()%10+30
					self.human[1][1]+=random.random()%5+10
					self.human[1][2]+=random.random()%5+10
		detection=self.yolo()
		print("detect val",detection)
		return detection


'''
TODO
1.Background segmentation
2.QUERY(feature matching)
4.Yolov3 (offline)
5.Yolov3 (Online)

'''


def arg_parse():

	parser = argparse.ArgumentParser(description='Anti-theft system')
	parser.add_argument("--video", dest = 'video', help = 
						"Video to run detection upon",
						default = "../dataset/dataset2.mp4", type = str)
	parser.add_argument("--det", dest = 'det', help ="Image / Directory to store detections to",
						default = "../dataset/det", type = str)
	#parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.6)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
	parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
						default = "../cfg/yolov3.cfg", type = str)
	parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
						default = "../model/yolov3.weights", type = str)
	parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
						default = "416", type = str)
	parser.add_argument("--dataset",dest="dataset",help="original image",
		default="../dataset/img")
	return parser.parse_args()


# def mainFunc_for_bbox_txt():
# 	args = arg_parse()
# 	dir_name=args.dataset
# 	f= open(dir_name,'r')
# 	line =f.readlines()


# 	humanDataset = {}
# 	image=np.zeros((2000,2000,3))
# 	itemDataset = {}
# 	missingPeopleDataset = []
# 	#test=Simulation()
# 	#count=0
	
# 	encoder = Autoencoder()

# 	#while count<40:
# 	count=0

# 	for detection in line:
# 		print("=================")
# 		print("time stamp:",count)
# 		print(detection)
# 		detection = detection[2:-3]
# 		detection=detection.split('],[')
# 		detection_list=[]
# 		human_list=[]
# 		item_list=[]
# 		item_class=[]
# 		for detect in detection:
# 			detect=detect.split(', ')
# 			det_pos=[int(i) for i in detect[0:-1]]
# 			det_class=detect[-1].strip("''")		
# 			detection_list.append([det_pos,det_class])

# 		for dect in detection_list:
			
# 			if dect[-1]=='person':
# 				human_list.append(dect[0:-1])
# 			else:
# 				item_list.append(dect[0:-1])
# 				item_class.append(dect[-1])
			
# 		if human_list!=[]:
# 			humanMatching(image, human_list[0], humanDataset, itemDataset, encoder, missingPeopleDataset)
# 		print("human",humanDataset.keys())
# 		if item_list!=[]:
# 			itemMatching(item_list[0], humanDataset,itemDataset)
# 		print("item",itemDataset.keys())
# 		count+=1
			
# 		#print("global11111",humanDataset)
# 		#print("item22222",itemDataset)
# 		#Scan_for_item_existing(humanDataset,itemDataset)
# 		#Track_and_Display(humanDataset, itemDataset)
# 		#count+=1


def detecting_function(x, img,detection):
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])

	detection.append([c1[0].cpu().detach().numpy().tolist(),c1[1].cpu().detach().numpy().tolist(),
			c2[0].cpu().detach().numpy().tolist(),c2[1].cpu().detach().numpy().tolist(),classes[cls]])
	
	#Function that help created the list of detection postion and class
	def write_in_file():
		#print(str([c1[0].cpu().detach().numpy().tolist(),c1[1].cpu().detach().numpy().tolist(),c2[0].cpu().detach().numpy().tolist(),c2[1].cpu().detach().numpy().tolist(),classes[cls]]))
		if im_id_list==[]:
			#print(x[0].cpu().detach().numpy().tolist())
			f.write("[")
			f.write(str([c1[0].cpu().detach().numpy().tolist(),c1[1].cpu().detach().numpy().tolist(),c2[0].cpu().detach().numpy().tolist()\
				,c2[1].cpu().detach().numpy().tolist(),classes[cls]]))

			im_id_list.append(x[0].cpu().detach().tolist())
			
		elif x[0]==im_id_list[-1]:
			f.write(",")
			f.write(str([c1[0].cpu().detach().numpy().tolist(),c1[1].cpu().detach().numpy().tolist(),c2[0].cpu().detach().numpy().tolist(),c2[1].cpu().detach().numpy().tolist(),classes[cls]]))
		else:
			f.write("]\n[")		
			f.write(str([c1[0].cpu().detach().numpy().tolist(),c1[1].cpu().detach().numpy().tolist(),c2[0].cpu().detach().numpy().tolist(),c2[1].cpu().detach().numpy().tolist(),classes[cls]]))
			im_id_list.append(x[0].cpu().detach().tolist())

	#write_in_file()
	
	return img




if __name__=="__main__":

	encoder = Autoencoder()
	print("hi")

	classes = load_classes('../data/coco.names')
	colors = pkl.load(open("../model/pallete", "rb"))
	f=open('../dataset/bbox_data.txt','a')
	args = arg_parse()
	if not os.path.exists(args.det):
		os.makedirs(args.det)
	confidence = float(args.confidence)
	nms_thesh = float(args.nms_thresh)
	start = 0
	#Where the image are saved
	dir_name=args.dataset
	num_classes = 80
	CUDA = torch.cuda.is_available()  
	bbox_attrs = 5 + num_classes
	
	#construct a model
	print("Loading network.....")
	model = Darknet(args.cfgfile)
	model.load_weights(args.weightsfile)
	print("Network successfully loaded")
	model.net_info["height"] = args.reso
	inp_dim = int(model.net_info["height"])


	assert inp_dim % 32 == 0 
	assert inp_dim > 32

	if CUDA:
		model.cuda()
		
	model.eval()
	frames = 0
	start = time.time()

	#Construct an empty dataset for human and item
	humanDataset = {}
	# first time TF will take more time, so I use this as an initializer
	image = 255*np.ones((1,128,128,3))
	feature = encoder.sess.run(encoder.encodeFeature, feed_dict={encoder.x: image})

	itemDataset = {}
	missingPeopleDataset = []

	#test=Simulation()
	
	# encoder = Autoencoder()

	#while count<40:
	count=0
	#file=os.listdir(dir_name)
	
	#file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	im_id_list=[]


	

	videofile = args.video  
	cap = cv2.VideoCapture(videofile)  
	assert cap.isOpened(), 'Cannot capture source'
	fps = cap.get(cv2.CAP_PROP_FPS)
	print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
	start = time.time()    
	while cap.isOpened():
		frameStartTime = time.time()
		ret, frame = cap.read()
		if ret:

		#for filename in file:
			# print("=================")
			#print(frame)
			#print("time stamp:",count)
			#frame=os.path.join(dir_name,filename)
			#print(frame)
			
			
			yoloStartTime = time.time()
			img, orig_im, dim = prep_image(frame, inp_dim)
			
			im_dim = torch.FloatTensor(dim).repeat(1,2)
			if CUDA:
				im_dim = im_dim.cuda()
				img = img.cuda()
			
			with torch.no_grad(): 
				output = model(Variable(img), CUDA)
			output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

			if type(output) == int:
				frames += 1
				#print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
				cv2.imshow("frame", orig_im)
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					break
				continue

			im_dim = im_dim.repeat(output.size(0), 1)
			scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
			
			output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
			output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
			
			output[:,1:5] /= scaling_factor

			for i in range(output.shape[0]):
				output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
				output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
			
			detection=[]
			list(map(lambda x: detecting_function(x, orig_im,detection), output))
			# print("det",detection)
			yoloEndTime = time.time()
		
			#Use this line to save the file
			#list(map(cv2.imwrite, args.det+"/"+filename, orig_im))
			
			#print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
			
			detection_list=[]
			human_list=[]
			item_list=[]
			item_class=[]
			human_class=[]
			for dect in detection:
				#print(dect)
				
				if dect[-1]=='person':
					if (dect[2] - dect[0] > 120) and (dect[3] - dect[1] > 120):
						human_list.append(dect[0:-1])
						human_class.append(dect[-1])
				elif dect[-1]!='chair' and dect[-1]!='diningtable':
					item_list.append(dect[0:-1])
					item_class.append(dect[-1])
			
			# when no human detect, we should still run humanMatching
			# if human_list!=[]:
			# 	humanMatching(image, human_list, humanDataset, itemDataset, encoder, missingPeopleDataset)
			humanMatchingStartTime = time.time()
			humanMatching(frame, human_list, humanDataset, itemDataset, encoder, missingPeopleDataset)
			humanMatchingEndTime = time.time()
			# print("human",humanDataset.keys())
			# if item_list!=[]:
			# 	itemMatching((item_list, item_class), humanDataset, itemDataset)
			itemMatchingStartTime = time.time()
			itemMatching((item_list, item_class), humanDataset, itemDataset)
			itemMatchingEndTime = time.time()
			# print("item",itemDataset.keys())
			count+=1
			font = cv2.FONT_HERSHEY_SIMPLEX
			#cv2.putText(orig_im,filename,(30,40),font,1,(0,0,0),2)
			cv2.putText(orig_im,"human"+str(list(humanDataset.keys())),(30,80),font,1,(0,0,0),2)
			for i in range(0,math.floor(len(itemDataset.keys())/10)+1,1):
				if i==0:
					cv2.putText(orig_im,"item:"+str(list(itemDataset.keys())[(i)*10:(i+1)*10]),(30,120+i*40),font,1,(0,0,0),2)
				else:
					cv2.putText(orig_im,str(list(itemDataset.keys())[(i)*10:(i+1)*10]),(30,120+i*40),font,1,(0,0,0),2)
			#if (i!=0):
				#cv2.putText(orig_im,str(list(itemDataset.keys())[(i+1)*10:]),(30,90+(i+1)*40),font,1,(0,0,0),2)
			scanStartTime = time.time()
			Scan_for_item_existing(humanDataset,itemDataset,missingPeopleDataset)
			scanEndTime = time.time()

			trackStartTime = time.time()
			Track_and_Display(humanDataset, itemDataset,orig_im,
				(human_list,human_class),(item_list,item_class),classes,colors)
			trackEndTime = time.time()

			count+=1
			# print("now in frame", count)
			# cv2.imwrite(args.det+"/"+"frame%d.jpg" % count, orig_im)

			#cv2.imwrite(args.det+"/"+"frame%d.jpg" % count, orig_im)


			cv2.imshow("frame", orig_im)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				break
			frames += 1

			frameEndTime = time.time()
			# print("Inference time for each task:")
			# print("Total:            ", frameEndTime - frameStartTime)
			# print("yolo:             ", yoloEndTime - yoloStartTime)
			# print("human matching:   ", humanMatchingEndTime - humanMatchingStartTime)
			# print("item matching:    ", itemMatchingEndTime - itemMatchingStartTime)
			# print("scan item exist:  ", scanEndTime - scanStartTime)
			# print("track and display:", trackEndTime - trackStartTime)


			# print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

		else:
			break

	cap.release()

	

