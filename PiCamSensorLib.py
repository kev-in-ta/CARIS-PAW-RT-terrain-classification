"""
Author:         Kevin Ta
Date:           2019 August 6th
Purpose:        This Python library runs the Pi Camera.
"""
# IMPORTED LIBRARIES

from picamera import PiCamera
import time
import io, os, sys
import numpy as np
from PIL import Image

from multiprocessing import Process, Queue

# LOCALLY IMPORTED LIBRARIES

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, 'libraries'))

import carisPAWBuffers_pb2 as frameMsgStruct

#CLASSES

class ClPiCameraDAQ:
	
	def __init__(self, dataQueue, runMarker):
		"""
		Purpose:	Save data queue and marker to class ariable
		Passed:		Multiprocessing queue and run marker queue
		"""
		self.dataQueue = dataQueue
		self.runMarker = runMarker

	
	def fnRun(self, frequency):
		"""
		Purpose:	Initialize camera and preview, then send to data queue as PIL object
		Passed:		Frequency for running
		"""
		self.cam = PiCamera()
		self.cam.resolution = (1024,768)
		self.cam.framerate = 15
		self.stream = io.BytesIO()
		self.cam.start_preview(alpha=200)
		
		time.sleep(2)

		# Runs until kill command is given
		while (self.runMarker.empty()):
			self.fnGetImage()
			self.dataQueue.put(['PI_CAM', time.time(), Image.open(self.stream)])
			time.sleep(300/frequency - (time.time() % (300/frequency)))
	
	def fnGetImage(self):
		"""
		Purpose:	Class method for resetting and then capturing image onto stream
		Passed:		None
		"""

		self.stream.seek(0)
		self.stream.truncate()
		self.cam.capture(self.stream, format='jpeg')
	

if __name__ == "__main__":

	# Create dummy queues for test runs
	dummyQueue = Queue()
	dummyMarker = Queue()

	# Sets test run frequency
	frequency = 100

	# Instantiates the class
	instSensor = ClPiCameraDAQ(dummyQueue, dummyMarker)

	# Runs the method in a different class
	PCam = Process(target=instSensor.fnRun, args = (frequency, ))
	PCam.start()

	timeStamp = []
	
	timeStart = time.time()
	
	counter = 0

	# Print time stamps
	while(time.time() < timeStart + 100):
		bufferPC = dummyQueue.get()
		image = bufferPC[2]
		image.save("testPhoto-{}.jpg".format(counter), "JPEG")
		counter +=1
		timeStamp.append(bufferPC[1])
		if counter > 5:
			counter = 0
			print('PC Frequency: {}'.format(300/(timeStamp[-1]-timeStamp[-51])))

