import numpy as np
from numpy import linalg as LA
from dataType import humanData, itemData
import cv2
import time

''' 
Container content :

humanDataset         = {human_id1:humanData1, human_id2:humanData2, ...}
itemDataset          = {item_id1:itemData1, item_id2:itemData2, ...}
missingPeopleDataset = [humanData1, humanData2, ...]
detection            = [[upleft_x, upleft_y, downright_x, downright_y],[upleft_x, upleft_y, downright_x, downright_y]...]


function defined:

humanMatching()
itemMatching()
findClosestHuman()
setAllItemAlarmOff()
matchMissingPeople()
calculateDist()

'''

global countHuman 
global countItem 
countHuman=0
countItem=0

def resize_human_to_autoencoder(img):
	img = cv2.resize(img, (128, 128))

	# standardize the image
	mean, std = cv2.meanStdDev(img)
	mean, std = mean.astype(np.float32), std.astype(np.float32)
	img = img.astype(np.float32)
	img = (img - np.squeeze(mean)) / np.squeeze(std)

	# make batch
	batch_X = img[np.newaxis,:]
	return batch_X



def humanMatching(image, detection, humanDataset, itemDataset, encoder, missingPeopleDataset):

	global countHuman
	distanceThres = 150
	
	for h_n in detection:
		find_pair = False

				
		hnx = (h_n[0] + h_n[2])/2.0
		hny = (h_n[1] + h_n[3])/2.0

		# encode feature from detected human
		encodeStart = time.time()
		input_img = resize_human_to_autoencoder(image[int(h_n[1]):(h_n[3]), int(h_n[0]):int(h_n[2]), :])
		feature = encoder.sess.run(encoder.encodeFeature, feed_dict={encoder.x: input_img})
		encodeEnd = time.time()
		# print("encoder spend ", encodeEnd - encodeStart)
		# print("detected human's feature = ", feature)
		

		min_dist = 10000
		matchCandidate = None
		# print("detected human in ", hnx, hny)
		for h_d in humanDataset.values():
			# print("dataset human in ", h_d.x, h_d.y)
			# print("human distance", np.sqrt((hnx - h_d.x)**2 + (hny - h_d.y)**2), distanceThres)
			if h_d.updated == False and np.sqrt((hnx - h_d.x)**2 + (hny - h_d.y)**2) < distanceThres:
				# dist = np.sqrt((hnx - h_d.x)**2 + (hny - h_d.y)**2)
				# if dist < min_dist:
				# 	min_dist = dist
				# 	matchCandidate = h_d
				dist = calculateDist(feature, h_d.feature)
				# print("feature distance between", hnx, hny,"and", h_d.id, dist)
				if dist < min_dist:
					if abs(dist - min_dist) < 0.07:
						# print("feature too close!!!!!")
						firstFeatureDist = calculateDist(feature, matchCandidate.firstFeature)
						if calculateDist(feature, h_d.firstFeature) < firstFeatureDist:
							min_dist = dist
							matchCandidate = h_d
					else:
						min_dist = dist
						matchCandidate = h_d
		if matchCandidate is not None:
			matchCandidate.update_position(hnx, hny)
			matchCandidate.feature = feature
			matchCandidate.updated = True
			# if matchCandidate.missing == True:
			if matchCandidate.missing > 50:
				missingPeopleDataset.remove(matchCandidate)
				# matchCandidate.missing = False
				matchCandidate.missing = 0
			find_pair = True
			setAllItemAlarmOff(matchCandidate, itemDataset)


		if not find_pair:
			matchId=None
			
			matchId = matchMissingPeople(feature, missingPeopleDataset)
			# print(matchId)
			if matchId == None:
				countHuman = countHuman + 1	
				newHuman = humanData(hnx, hny, countHuman, feature)
				humanDataset[countHuman] = newHuman
				# print("Build up new human data feature =", humanDataset[countHuman].feature)

			else:
				# print("human match",matchId)
				humanDataset[matchId].updated = True
				# humanDataset[matchId].missing = False
				humanDataset[matchId].missing = 0
				humanDataset[matchId].update_position(hnx, hny)
				humanDataset[matchId].feature = feature
				setAllItemAlarmOff(humanDataset[matchId], itemDataset)


	for h_d in humanDataset.values():
		# what if people get occluded for a frame?
		# if h_d.updated == False and h_d.missing == False:
		if h_d.updated == False and h_d.missing <= 50:
			# h_d.missing = True
			h_d.missing += 1
			if h_d.missing > 50:
				missingPeopleDataset.append(h_d)
				# print("Human missing! missingPeopleDataset become", missingPeopleDataset)
				# print("\n \n \n\n\n")

		h_d.updated = False # reset the update flag for all human in dataset


def itemMatching(detection, humanDataset, itemDataset):

	global countItem
	# distanceThres = 30
	distanceThres = 50
		
	for d_n, d_name in zip(detection[0],detection[1]):
		find_pair = False

		dnx = (d_n[0] + d_n[2])/2.0
		dny = (d_n[1] + d_n[3])/2.0


		# first match with the dataset
		for d_d in itemDataset.values():
			# print("item.id",d_d.id,"x and y",d_d.x,d_d.y,d_n)
			# print("item distance===: ", np.sqrt((dnx - d_d.x)**2 + (dny - d_d.y)**2), distanceThres+5*(d_d.missing+1))
			if np.sqrt((dnx - d_d.x)**2 + (dny - d_d.y)**2) < distanceThres:# + 5*(d_d.missing+1):
				# print("flag",d_d.alarm_flag)
				if d_d.alarm_flag == False and d_d.owner != 0: # no owner object don't update position
					# print("name", d_name, d_d.name)
					if d_name==d_d.name:
						
						d_d.update_position(dnx, dny)
						d_d.updated = True
						# d_d.missing = False
						d_d.missing = 0
						# print("hi")
						find_pair = True
						# print("update",d_name,d_d.id,find_pair)
						break

				else: # if item is in alarm state, no update position
					if d_name==d_d.name:
						# print("do not update", d_name, "position")	
						d_d.updated = True
						# d_d.missing = False
						d_d.missing = 0
						# print("hi")
						find_pair = True
						break


		# if not in dataset, create one new item
		if not find_pair:
			countItem = countItem + 1
			newItem = itemData(dnx, dny, countItem,d_name)
			# print("create new item",countItem,newItem.name)
			itemDataset[countItem] = newItem
			findClosestHuman(itemDataset[countItem], humanDataset) # link the item to human


	# if not update, missing ++
	for d_d in itemDataset.values():
		# qwhat if item get occluded for a frame?
		# if d_d.updated == False and d_d.missing == False:
		# print("belwo loop=======", d_d.name, d_d.id)
		if d_d.updated == False and d_d.missing <= 50:
			# d_d.missing = True
			d_d.missing += 1
			# print("d_d.missing = ", d_d.id, d_d.name, d_d.missing)
			# print("\n=\n=\n=")

		# elif d_d.updated == False:
			# print("item is in missing state")
		d_d.updated = False # reset the update flag for all item in dataset


def findClosestHuman(item, humanDataset):
	min_dist = 1000
	closestHuman = None
	item_human_thres = 500
	for human in humanDataset.values():
		# if human.missing == False:
		if human.missing <= 50:
			dist = np.sqrt((item.x - human.x)**2 + (item.y - human.y)**2)
			# print("distance between human",human.id, "and item", item.id, "is", dist)
			if dist < item_human_thres:
				if dist < min_dist:
					min_dist = dist
					closestHuman = human
	if closestHuman!=None:
		closestHuman.itemList.append(item.id)
		item.owner=closestHuman.id

		# print("owner",item.owner)
	else:
		item.owner=0
		# humanDataset[0].itemList.append(item.id)
		# if 0 in humanDataset:
		# 	humanDataset[0].itemList.append(item.id)
		# else:
		# 	feature=10000*np.ones((1,4*4*512))
		# 	newHuman = humanData(0, 0, 0, feature)
		# 	newHuman.missing = 51
		# 	humanDataset[0] = newHuman
		# 	humanDataset[0].itemList.append(item.id)
	


def setAllItemAlarmOff(human, itemDataset):
	for itemID in human.itemList:
		#print("human, item",human,itemID,itemDataset.keys())
		if itemDataset[itemID].alarm_flag is True:
			itemDataset[itemID].alarm_flag = False


def matchMissingPeople(feature, missingPeopleDataset):
	closestMatchDist = 10000
	closestMatch = None
	thresDist = 0.1
	#print("start to match missing people...")
	#print("missingPeopleDataset = ", missingPeopleDataset)
	for h_d in missingPeopleDataset:
		dist = calculateDist(feature, h_d.feature)
		# print("distance = ",dist)
		# print("\n\n\n\n\n\n\n")
		if dist < thresDist:
			if dist < closestMatchDist:
				closestMatchDist = dist
				closestMatch = h_d
	if closestMatch is not None:
		# print("find closest match", closestMatch, "feature distance = ", closestMatchDist)
		missingPeopleDataset.remove(closestMatch)
		#print("missingPeopleDataset now contain", missingPeopleDataset)
		return closestMatch.id
	else:
		# print("no matching in missingPeopleDataset")
		return None


def calculateDist(f1, f2):
	return (1 - (np.dot(f1[0], f2[0])/(np.linalg.norm(f1[0])*np.linalg.norm(f2[0]))))




