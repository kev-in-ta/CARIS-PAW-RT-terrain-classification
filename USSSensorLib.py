"""
Author:         Kevin Ta
Date:           2019 June 26th
Purpose:        This Python library runs the ultrasonic sensors.
				
				1. AJ-SR04M-1 - Ultrasonic Sensor
				2. AJ-SR04M-2 - Ultrasonic Sensor
"""
# IMPORTED LIBRARIES

import RPi.GPIO as GPIO
import time, threading
import numpy as np
import os, sys

from multiprocessing import Process, Queue

# LOCALLY IMPORTED LIBRARIES

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, 'libraries'))

import carisPAWBuffers_pb2 as frameMsgStruct

#CLASSES

class ClProximitySensorDAQ:
	"""
	Class for reading ultrasonic proximity sensor.
	"""
	
	def __init__(self, dataQueue, runMarker):
		"""
		Purpose:	Initialze gpio (pins)
		Passed:		Queue for data transfer between processes.
		"""		
			
		TRIG_PIN = 27                                  #Associate pin 15 to TRIG
		ECHO_PIN = 17                                  #Associate pin 14 to Echo
		distance = 0
		isInit = False

		# Sets cross-process queues and marker
		self.dataQueue = dataQueue
		self.runMarker = runMarker

		# Prevents excessive warnings
		GPIO.setwarnings(False)

		# GPIO Mode (BOARD / BCM)
		GPIO.setmode(GPIO.BCM)

		self.TRIG = TRIG_PIN
		self.ECHO = ECHO_PIN
		GPIO.setup(self.TRIG,GPIO.OUT)                  #Set pin as GPIO out
		GPIO.setup(self.ECHO,GPIO.IN)                   #Set pin as GPIO in
		
		self.distance = 0
	
	def fnRun(self, frequency):
		"""
		Purpose:	Runs data acquisition.
		Passed:		Frequency to determine how long to wait
		"""		
		
		# Time interval between captures
		waitTime = 8.0/frequency
		
		# Intialize pins
		self.initialize()
		
		# Continuously run and send back information thru the queue
		while (self.runMarker.empty()):
			self.getDistance()
			self.dataQueue.put(['USS_DOWN', time.time(), self.distance])
			time.sleep(waitTime - (time.time() % waitTime))
		
		# Cleans up pins
		GPIO.cleanup()
	
	def initialize(self):
		"""
		Purpose:	Intialize sensor
		Passed:		None
		"""		
		
		GPIO.output(self.TRIG, False)                 #Set TRIG as LOW
		print ("Waiting For Sensor To Settle")
		time.sleep(2)                            #Delay of 2 seconds
		self.isInit = True

	def getDistance(self):
		"""
		Purpose:	Performs sensor distance calculation
		Passed:		None
		"""	
		
		# Resets trigger
		GPIO.output(self.TRIG, True)                  #Set TRIG as HIGH
		time.sleep(0.00001)                      #Delay of 0.00001 seconds
		GPIO.output(self.TRIG, False)                 #Set TRIG as LOW

		# Store initial pulse timing
		pulse_start = time.time() 

		# Stores pulse start time so long as trigger is low
		while GPIO.input(self.ECHO)==0:               #Check if Echo is LOW
			pulse_start = time.time()              #Time of the last  LOW pulse

		# When trigger reads high, calculate timing
		while GPIO.input(self.ECHO)==1:               #Check whether Echo is HIGH
			#~ pulse_end = time.time()                #Time of the last HIGH pulse 
			pulse_duration = time.time() - pulse_start
			if pulse_duration > 0.05:
				break

		# Calculate distance based on speed of sound
		distance = round((pulse_duration)*17150, 2)            #Round to two decimal points

		# outputs distance data
		if distance > 20 and distance < 800:     #Is distance within range
			self.distance = distance
			
		#~ print(self.distance)
	

if __name__ == "__main__":
	
	dummyQueue = Queue()
	dummyMarker = Queue()
	
	frequency = 200
	
	instSensor = ClProximitySensorDAQ(dummyQueue, dummyMarker)
	
	PUSS = Process(target=instSensor.fnRun, args = (frequency, ))
	PUSS.start()
	
	timeStamp = []
	
	timeStart = time.time()
	
	counter = 0
	
	while(time.time() < timeStart + 100):
		bufferUSS = dummyQueue.get()
		counter +=1
		timeStamp.append(bufferUSS[1])
		if counter > 50:
			counter = 0
			print('USS Frequency: {}'.format(50/(timeStamp[-1]-timeStamp[-51])))
	
