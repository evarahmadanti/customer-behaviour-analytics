# Run and Debug like this
# python3 main.py -l videos/cut_rexiiii2.avi -r videos/cut_relxiiii2.avi  -o output/result_percobaan_xiiii.avi --display

import datetime
import random
import time
from argparse import ArgumentParser

import cv2
import dlib
import imutils
import numpy as np
from imutils.video import FPS

from database.config import connection
from module.centroidtracker import CentroidTracker
from module.stitcher import VideoStitcher
from module.trackableobject import TrackableObject


def run(left_video, right_video, output, left_save, right_save, display=False):
	# initialize the image stitcher, detector, and
	# total number of frames read
	stitcher = VideoStitcher()
	total = 0
	
	# Get the start time
	system_start = time.time()

	# creating database connection
	mydb = connection()
	cursor = mydb.cursor()

	# initialize the list of class labels MobileNet SSD was trained to detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe('model/mobilenet_ssd/MobileNetSSD_deploy.prototxt', 'model/mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

	# initialize the video writer (we'll instantiate later if need be)
	writer, writer_left, writer_right = None, None, None
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W, H = None, None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	# totalDown = 0
	# totalUp = 0
	rand = random.randint(150, 230)
	print(rand)

	totalLeft = 0
	totalRight = 0

	counter = 0
	counter2 = 0

	idxLockLeft = 0
	idxLockRight = 0

	fps = FPS().start()

	# Polygon1 corner points coordinates
	pts = np.array([[175, 38], [277, 63],
					[249, 198], [126, 165],], np.int32)
	pts = pts.reshape((-1, 1, 2))

	# Polygon2 corner points coordinates
	pts2 = np.array([[275, 47], [379, 46],
					[432, 197], [268, 195],], np.int32)
	pts2 = pts2.reshape((-1, 1, 2))

	# White color in BGR
	color = (255, 255, 255)

	def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			print(x,y)

	def varians(cust_data, mean_elapse_time, total_data):
		varians_elapsed_time = 0
		for data in cust_data:
			varians_elapsed_time += ((data['elapsed_time']-mean_elapse_time)**2)
		varians_elapsed_time = varians_elapsed_time / total_data
		return varians_elapsed_time

	def send_data(totalLeft, totalRight):
		cust_data = ct.get_data()
		total_data = len(cust_data)
		date_now = datetime.date.today()
		day_now = timestamp.strftime("%A")
		total_elapsed_time = 0
		for row in cust_data:
			# executing the query with values
			cursor.execute(f"INSERT INTO customer_data_detail (id_object, tanggal_masuk, elapsed_time) VALUES ({row['id']}, '{row['entry_date']}', {row['elapsed_time']})")
			total_elapsed_time += row['elapsed_time']
		
		# get the mean & variance data of elapsed time
		mean_elapsed_time = total_elapsed_time / total_data
		varians_elapsed_time = varians(cust_data, mean_elapsed_time, total_data)
		
		# store the general data to mysql
		cursor.execute(f"INSERT INTO customer_data_total (day, date, total_visitor, mean_elapsed_time, varians_elapsed_time, total_rack_a, total_rack_b) VALUES ('{day_now}', '{date_now}', '{total_data}', '{mean_elapsed_time}', '{varians_elapsed_time}', '{totalLeft}', '{totalRight}')")
		mydb.commit()  # to make final output we have to run the 'commit()' method of the database object
		print(cursor.rowcount, "record inserted")
		
		# reset the rack data
		totalLeft, totalRight = 0, 0

		#stop the system
		writer_left.release()
		writer_right.release()
		writer.release()
		cv2.destroyAllWindows()

	# loop over frames from the video streams
	while True:
		# grab the frames from their respective video streams
		_, left = left_video.read()
		_, right = right_video.read()

		# rotate
		# left = cv2.rotate(left, cv2.ROTATE_180)
		# right = cv2.rotate(right, cv2.ROTATE_180)

		# flip the video
		# left = cv2.flip(left, 1)
		# right = cv2.flip(right, 1)

		# stitch the frames together to form the panorama
		# IMPORTANT: you might have to change this line of code depending on how your cameras are oriented; 
		# frames should be supplied in left-to-right order
		stitched_frame = stitcher.stitch([left, right])

		# no homograpy could be computed
		if stitched_frame is None:
			print("[INFO] homography could not be computed")
			break

		# resize the frame to have a maximum width of 500 pixels (the less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(stitched_frame, height=200)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
			(HL, WL) = left.shape[:2]
			(HR, WR) = right.shape[:2]

		# if we are supposed to be writing a video to disk, initialize the writer
		if output is not None and writer is None:
			writer = cv2.VideoWriter(output, fourcc, 10, (W, H), True)
			print(f"The video {output} was saved")

		if left_save is not None and writer_left is None:
			writer_left = cv2.VideoWriter(left_save, fourcc, 10, (WL, HL), True)
			print(f"The video {left_save} was saved")

		if right_save is not None and writer_right is None:
			writer_right = cv2.VideoWriter(right_save, fourcc, 10, (WR, HR), True)
			print(f"The video {right_save} was saved")

		# initialize the current status along with our list of bounding box rectangles returned by either (1)
		# our object detector or (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive object detection method to aid our tracker
		if total % 30 == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the prediction
				confidence = detections[0, 0, i, 2]
				# filter out weak detections by requiring a minimum confidence

				if confidence > 0.4:
					# extract the index of the class label from the detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					# construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)
			
					# add the tracker to our list of trackers so we can utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather than 'waiting' or 'detecting'
				status = "Tracking"
				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		# cv2.line(frame, (0, H // 3), (W, H // 3), (0, 255, 255), 2)

		# use the centroid tracker to associate the (1) old object centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		#declare mat1 and mat2 as a single channel as a requirements of the bitwiseAsnd algorithm
		frameH, frameW= frame.shape[:2]
		mat1 = np.zeros((frameH, frameW, 1), dtype = "uint8") 
		mat2 = np.zeros((frameH, frameW, 1), dtype = "uint8") 
		mat_left = np.zeros((frameH, frameW, 1), dtype = "uint8") 
		mat3 = np.zeros((frameH, frameW, 1), dtype = "uint8") 
		mat_right = np.zeros((frameH, frameW, 1), dtype = "uint8")

		# draw poly inside mat1 for left area & mat 3 for right area
		left_area = cv2.fillPoly(mat1, [pts], color)
		right_area = cv2.fillPoly(mat3, [pts2], color)

		# draw centroid of each object
		inc = 0
		tempX1 = 0
		tempObj1 = 0
		tempX2 = 0
		tempObj2 = 0

		for (objectID, centroid) in objects.items():
			cv2.circle(mat2, (centroid[0], centroid[1]), 4, color, -1)	
			if(inc == 0):
				tempX1 = centroid[0]
				tempObj1 = objectID
			else:
				tempX2 = centroid[0]
				tempObj2 = objectID
			inc = inc + 1

		if(tempX1 > tempX2):
			idxLockLeft = tempObj1
			idxLockRight = tempObj2
		else:
			idxLockLeft = tempObj2
			idxLockRight = tempObj1

		mat_left = cv2.bitwise_and(mat1, mat2)
		mat_right = cv2.bitwise_and(mat3, mat2)

		temp_left_area = cv2.countNonZero(mat_left)
		temp_right_area = cv2.countNonZero(mat_right)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current object ID
			to = trackableObjects.get(objectID, None)
			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it to determine direction
			else:
				# the difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				for i in range(1, len(to.centroids)):
					cv2.line(frame, tuple(to.centroids[i-1]), tuple(to.centroids[i]), (0, 0, 255), 2)
					# print(to.centroids[i])

				# check to see if the object has been counted or not
				if not to.counted or not to.counted2:
					if (temp_left_area > 0 and not to.counted and idxLockLeft != objectID):
						counter += 1
						if counter >= 50:
							totalLeft += 1
							counter = 0
							to.counted = True

					if (temp_right_area > 0 and not to.counted2 and idxLockRight != objectID):
						counter2 += 1
						if counter2 >= 50:
							totalRight += 1
							counter2 = 0
							to.counted2 = True

				cv2.line(frame, tuple(to.centroids[i-1]), tuple(to.centroids[i]), (0, 0, 255), 2)

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			for i, (startX, startY, endX, endY) in enumerate(rects):
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				text2 = "Confidence : {:0.2f}".format(confidence)
				cv2.putText(frame, text2, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# construct a tuple of information we will be displaying on the frame
		info = [
			# ("Exit", totalUp),
			("Total Right", totalRight),
			("Total Left", totalLeft),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)
			# writer_left.write(left)
			# writer_right.write(right)

		# increment the total number of frames read and draw the timestamp on the image
		total += 1
		timestamp = datetime.datetime.now()
		ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
		cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		fps.update()
	
		# handler for send data to db
		system_stop = round(time.time() - system_start)
		if system_stop != 0 and system_stop % rand == 0: #store data in 24 hours means 86400 sec
			send_data(totalLeft, totalRight)
			time.sleep(1)
			# reset all data
			system_start = time.time()

		# show the output images
		if display:
			# cv2.imshow("Left Cam", left)
			# cv2.imshow("Right Cam", right)
			# cv2.namedWindow("Result")
			# cv2.setMouseCallback("Result", on_EVENT_LBUTTONDOWN)
			cv2.imshow("Result", frame)
			# cv2.imshow("mat Left", mat_left)
			# cv2.imshow("mat Right", mat_right)

		# if the 'q' key was pressed, break from the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	fps.stop()
	writer_left.release()
	writer_right.release()
	writer.release()
	cv2.destroyAllWindows()
	#lokasi ngirim 
	print('[INFO] cleaning up...')
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == '__main__':
	''' Main function to run customer behaviour system

	Example of usage:
	python main.py [args]

	Args:
		left-cam: string path to left videos file or int left webcam (0, 1, 2, ...)
		right-cam: string path to right videos file or int right webcam (0, 1, 2, ...)
		display (optional): to display the result
	'''
	# construct the argument parser and parse the arguments
	ap = ArgumentParser()
	ap.add_argument('-l', '--left-cam', help='path to left videos file or left webcam (0, 1, 2, ...)')
	ap.add_argument('-r', '--right-cam', help='path to right videos file or right webcam (0, 1, 2, ...)')
	ap.add_argument('-d', '--display', action='store_true', help='display result')
	ap.add_argument('-o', '--output', type=str, help='path to optional output video file')
	ap.add_argument('-lo', '--left-save', type=str, help='path to optional output video file')
	ap.add_argument('-ro', '--right-save', type=str, help='path to optional output video file')
	args = ap.parse_args()
	left_video, right_video, display, output, left_save, right_save = args.left_cam, args.right_cam, args.display, args.output,  args.left_save, args.right_save
	# display, output, left_save, right_save = args.display, args.output, args.left_save, args.right_save

	# initialize the video streams and allow them to warmup
	print('[INFO] starting camera...')
	# left_video = cv2.VideoCapture(2, cv2.CAP_V4L2)
	# right_video = cv2.VideoCapture(0, cv2.CAP_V4L2)
	left_video = cv2.VideoCapture(left_video)
	right_video = cv2.VideoCapture(right_video)

	# run system
	run(left_video, right_video, output, left_save, right_save, display=display)
