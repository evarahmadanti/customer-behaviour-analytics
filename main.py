import time
import imutils
import datetime
import numpy as np
import dlib
import cv2
from argparse import ArgumentParser
from module.stitcher import VideoStitcher
from module.centroidtracker import CentroidTracker
from module.trackableobject import TrackableObject

def run(left_video, right_video, output, display=False):
	# initialize the image stitcher, detector, and
	# total number of frames read
	stitcher = VideoStitcher()
	total = 0

	# initialize the list of class labels MobileNet SSD was trained to detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe('model/mobilenet_ssd/MobileNetSSD_deploy.prototxt', 'model/mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalDown = 0
	totalUp = 0

	# loop over frames from the video streams
	while True:
		# grab the frames from their respective video streams
		_, left = left_video.read()
		_, right = right_video.read()

		# resize the frames
		left = imutils.resize(left, width=400)
		right = imutils.resize(right, width=400)

		# flip the video
		# left = cv2.flip(left, 1)
		# right = cv2.flip(right, 1)

		# stitch the frames together to form the panorama
		# IMPORTANT: you might have to change this line of code depending on 
		# how your cameras are oriented; frames should be supplied in left-to-right order
		stitched_frame = stitcher.stitch([left, right])

		# no homograpy could be computed
		if stitched_frame is None:
			print("[INFO] homography could not be computed")
			break
		
		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(stitched_frame, height=300)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		
		# if we are supposed to be writing a video to disk, initialize the writer
		if output is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(output, fourcc, 30,
				(W, H), True)
			print("The video {} was saved", output)

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
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

		# use the centroid tracker to associate the (1) old object centroids with (2) the newly computed object centroids
		objects = ct.update(rects)
		
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
					cv2.line(frame, tuple(to.centroids[i-1]), tuple(to.centroids[i]), (0, 0, 255), 1)
	
				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						to.counted = True

					# if the direction is positive (indicating the object is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						to.counted = True

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			for i, (startX, startY, endX, endY) in enumerate(rects):
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				text2 = "Confidence : {:0.2f}".format(confidence)
				cv2.putText(frame, text2, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# construct a tuple of information we will be displaying on the frame
		info = [
			("Exit", totalUp),
			("Entry", totalDown),
			("Status", status),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# increment the total number of frames read and draw the timestamp on the image
		total += 1
		timestamp = datetime.datetime.now()
		ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
		cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# show the output images
		if display:
			cv2.imshow("Left Cam", left)
			cv2.imshow("Right Cam", right)
			cv2.imshow("Result", frame)

		# if the 'q' key was pressed, break from the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cv2.destroyAllWindows()
	print('[INFO] cleaning up...')

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
	args = ap.parse_args()
	left_video, right_video, display, output = args.left_cam, args.right_cam, args.display, args.output

	# initialize the video streams and allow them to warmup
	print('[INFO] starting camera...')
	left_video = cv2.VideoCapture(left_video)
	right_video = cv2.VideoCapture(right_video)

	# run system
	run(left_video, right_video, output, display=display)