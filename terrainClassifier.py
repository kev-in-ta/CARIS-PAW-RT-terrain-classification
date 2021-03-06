"""
Author:         Kevin Ta
Date:           2020 July 20th
Purpose:        This Python script tests terrain classification by passing streamed data.
"""

# IMPORTED LIBRARIES

import math
import time
import os
import sys
import threading

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from scipy import signal, stats, fft

import pandas as pd
import sklearn
import mlxtend
from joblib import load, dump

from multiprocessing import Process, Queue
from threading import Thread

# LOCALLY IMPORTED LIBRARIES
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, 'libraries'))

# from WheelModuleLib import *
from featuresLib import *

# DEFINITIONS

# Classification frequency
CLASS_FREQ = 0.2  # 0.2, 0.5, or 0.8

# Active person
PERSON_DATA = 'Keenan'  # 'Keenan', 'Kevin', 'Mahsa', or 'Jamie'

# Trajectory
MOVE_PATTERN = 'Straight' # 'F8', 'Donut', 'Straight'

# Active terrain
TEST_TERRAIN = 'Sidewalk'  # 'Linoleum', 'Grass', 'Gravel'

#TERRAIN_DICT = [('Jamie', 'Concrete', 'Donut')]

IMU = '6050'

TERRAIN_DICT = [
	('Jamie', 'Concrete', 'Donut'),
	('Jamie', 'Carpet', 'Donut'),
	('Jamie', 'Linoleum', 'F8'),
	('Jamie', 'Asphalt','Donut'),
	('Jamie', 'Sidewalk','Straight'),
	('Jamie', 'Grass','Donut'),
	('Jamie', 'Gravel', 'Donut'),
	('Keenan', 'Concrete', 'Donut'),
	('Keenan', 'Carpet', 'Donut'),
	('Keenan', 'Linoleum', 'Donut'),
	('Keenan', 'Asphalt', 'Straight'),
	('Keenan', 'Sidewalk', 'F8'),
	('Keenan', 'Grass', 'Straight'),
	('Keenan', 'Gravel', 'F8'),
	('Kevin', 'Concrete', 'Donut'),
	('Kevin', 'Carpet', 'F8'),
	('Kevin', 'Linoleum', 'F8'),
	('Kevin', 'Asphalt', 'Straight'),
	('Kevin', 'Sidewalk', 'Donut'),
	('Kevin', 'Grass', 'Straight'),
	('Kevin', 'Gravel', 'Straight'),
	('Mahsa', 'Concrete', 'Straight'),
	('Mahsa', 'Carpet', 'F8'),
	('Mahsa', 'Linoleum', 'Donut'),
	('Mahsa', 'Asphalt', 'F8'),
	('Mahsa', 'Sidewalk', 'Straight'),
	('Mahsa', 'Grass', 'F8'),
	('Mahsa', 'Gravel', 'Straight')
	]

# Sensor list to activate
SENSOR_LIST = ['IMU_9', 'IMU_6', 'USS_DOWN', 'USS_FORW', 'PI_CAM', 'LEFT', 'RIGHT']

# Active sensors
ACTIVE_SENSORS = [1]

# Direction Vectors
STD_COLUMNS = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro', 'Run Time', 'Epoch Time']
DATA_COLUMNS = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro']

EPSILON = 0.00001  # For small float values

FRAME_MODULE = {'wLength': 1024, 'fSamp': 300, 'fLow': 20, 'fHigh': 1}
WHEEL_MODULE = {'wLength': 333, 'fSamp': 333.3, 'fLow': 20, 'fHigh': 1}

# filter parameters
CUT_OFF = 20  # lowpass cut-off frequency (Hz)

PAD_LENGTH = 15  # pad length to let filtering be better

# DICTIONARIES

# Time domain feature functions and names
TIME_FEATURES = {'Mean': np.mean, 'Std': np.std, 'Norm': l2norm, 'AC': autocorr,
				 'Max': np.amax, 'Min': np.amin, 'RMS': rms, 'ZCR': zcr,
				 'Skew': stats.skew, 'EK': stats.kurtosis}

# TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'AC', 'Max', 'Min', 'RMS', 'ZCR', 'Skew', 'EK']
TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'Max', 'Min', 'RMS', 'ZCR']

# Time domain feature functions and names           
FREQ_FEATURES = freq_features = {'MSF': msf, 'RMSF': rmsf, 'FC': fc, 'VF': vf, 'RVF': rvf}

FREQ_FEATURES_NAMES = ['RMSF', 'FC', 'RVF']

TIME_FREQ_FEATURES = {'Mean': np.mean, 'Std': np.std, 'Norm': l2norm, 'AC': autocorr,
					  'Max': np.amax, 'Min': np.amin, 'RMS': rms, 'ZCR': zcr,
					  'Skew': stats.skew, 'EK': stats.kurtosis, 'MSF': msf, 'RMSF': rmsf, 'FC': fc, 'VF': vf,
					  'RVF': rvf}

TIME_FREQ_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'AC', 'Max', 'Min', 'RMS', 'ZCR', 'Skew', 'EK', 'MSF', 'RMSF', 'FC',
							'VF', 'RVF']

TERRAINS_OG = {'Concrete': 1, 'Carpet': 1, 'Linoleum': 1, 'Asphalt': 2, 'Sidewalk': 2, 'Grass': 3, 'Gravel': 4}
TERRAINS = ['No Motion', 'Indoor', 'Asphalt-Sidewalk', 'Grass', 'Gravel']

PERFORMANCE = {}

# CLASSES

class ClTerrainClassifier:
	"""
	Class for establishing wireless communications.
	"""

	def __init__(self, testSet, protocol='TCP'):
		"""
		Purpose:	Initialize various sensors and class variables
		Passed: 	Nothing
		"""

		self.testSet = testSet

		# Middle
		self.placement = 'Middle'
		self.sensorParam = FRAME_MODULE

		# Left
		# ~ self.placement = 'Left'
		# ~ self.sensorParam = WHEEL_MODULE

		# ~ # Right
		# ~ self.placement = 'Right'
		# ~ self.sensorParam = WHEEL_MODULE

		print('unpickling')

		self.RFTimelinePipeline = load('models/model.joblib')

		self.RFResults = pd.DataFrame(columns=["True Label", "RF Time", "Time"])

		# Prepopulate pandas dataframe
		EFTimeColumnNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for
							 featName in TIME_FEATURES_NAMES]
		self.EFTimeColumnedFeatures = pd.DataFrame(data=np.zeros((1, len(EFTimeColumnNames))),
												   columns=EFTimeColumnNames)
		EFFreqColumnNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for
							 featName in FREQ_FEATURES_NAMES]
		self.EFFreqColumnedFeatures = pd.DataFrame(data=np.zeros((1, len(EFFreqColumnNames))),
												   columns=EFFreqColumnNames)
		self.protocol = protocol

		self.EFTimeFreqColumnedFeatures = pd.DataFrame(
			data=np.zeros((1, len(EFFreqColumnNames) + len(EFTimeColumnNames))),
			columns=EFFreqColumnNames + EFTimeColumnNames)

		# Initialize data queue and marker to pass for separate prcoesses
		self.dataQueue = Queue()
		self.runMarker = Queue()

		f_ind = np.ceil((CUT_OFF + 10) / self.sensorParam['fSamp'] * 2 * self.sensorParam['wLength']/2).astype(int)
		f_ind2 = np.ceil((CUT_OFF + 10) / self.sensorParam['fSamp'] * 2 * 128).astype(int)

		# Create class variables
		self.windowIMUraw = np.zeros((self.sensorParam['wLength'] + 2 * PAD_LENGTH, 6))
		self.windowIMUfiltered = np.zeros((self.sensorParam['wLength'], 6))
		self.windowIMUFFT = np.zeros((f_ind,7))
		self.windowIMUPSD = np.zeros((f_ind2,7))

		# Create dictionary to house various active sensors and acivate specified sensors
		self.instDAQLoop = {}

		for sensor in ACTIVE_SENSORS:
			if sensor == 1:
				self.instDAQLoop[SENSOR_LIST[sensor]] = ClIMUDataStream(self.dataQueue, self.runMarker, self.testSet)

	def fnStart(self, frequency):
		"""
		Purpose:	Intialize all active sensors in separate processed and collects data from the Queue
		Passed:		Frequency for 6-axis IMU to operate at
		"""

		print('Start Process.')

		# Start terrain classification in separate thread
		terrain = Thread(target=self.fnTerrainClassification, args=(CLASS_FREQ,))
		terrain.start()

		timeStart = time.time()

		# Create dictionary to store processes
		processes = {}

		# Start various data collection sensors
		for sensor in ACTIVE_SENSORS:
			processes[SENSOR_LIST[sensor]] = Process(target=self.instDAQLoop[SENSOR_LIST[sensor]].fnRun,
													 args=(frequency,))
			processes[SENSOR_LIST[sensor]].start()

		# Keep collecting data and updating rolling window
		while self.runMarker.empty():

			try:
				transmissionData = self.dataQueue.get(timeout=2)

				if transmissionData[0] in ['IMU_6', 'WHEEL']:
					self.windowIMUraw = np.roll(self.windowIMUraw, -1, axis=0)
					self.windowIMUraw[-1, :] = np.subtract(transmissionData[2:8], [0, 0, 9.8, 0, 0, 0])
				elif transmissionData[0] in ['USS_DOWN', 'USS_FORW']:
					pass
				elif transmissionData[0] in ['PI_CAM']:
					pass
			except Exception as e:
				print(e)

		# wait for all processes and threads to complete
		terrain.join()

		print("Terrain classifier joined.")

		for sensor in ACTIVE_SENSORS:
			print("{} joining.".format(sensor))
			processes[SENSOR_LIST[sensor]].join()
			print("{} joined.".format(sensor))

	def fnTerrainClassification(self, waitTime):
		"""
		Purpose:	Class method for running terrain classification
		Passed:		Time in between runs
		"""

		count = 0

		startTime = time.time()

		# Keep running until run marker tells to terminate
		while self.runMarker.empty():

			count += 1

			# Filter window
			self.fnFilterButter(self.windowIMUraw)

			# Build extracted feature vector
			self.fnBuildTimeFeatures(TIME_FEATURES_NAMES)

			# Build PSD and PSD features
			self.fnBuildPSD(self.windowIMUfiltered)

			# Build FFT feature
			self.fnBuildFFT(self.windowIMUfiltered)

			# Build frequency features
			self.fnBuildFreqFeatures(FREQ_FEATURES_NAMES)

			#terrainTypeRFTime = self.RFTimelinePipeline.predict(self.EFTimeColumnedFeatures)
			terrainTypeRFTime = self.RFTimelinePipeline.predict(np.append(self.EFTimeColumnedFeatures,
																		  self.EFFreqColumnedFeatures, axis=1))

			try:
				print('Prediction: {0:>10s}'.format(TERRAINS[terrainTypeRFTime[0]]))
				self.RFResults = self.RFResults.append({"True Label": TERRAINS_OG[self.testSet[1]],
														"RF Time": terrainTypeRFTime[0], "Time": time.time()}, ignore_index=True)
			except Exception as e:
				print(e)
				break

			#time.sleep(waitTime - (time.perf_counter() % waitTime))

		endTime = time.time()

		print("Classification Frequency: {:>8.2f} Hz. ({} Samples in {:.2f} s)".format(count / (endTime - startTime),
																					   count, (endTime - startTime)))
		print("Terrain classifier completed.")

		PERFORMANCE["{}-{}-{}-Classification".format(self.testSet[1], self.testSet[2], self.testSet[0])] = (count, endTime-startTime)

		self.RFResults.to_csv(
			os.path.join('2021-Results', "{:.0f}ms-{}-{}-{}-{}.csv".format(CLASS_FREQ * 1000, IMU, self.testSet[1], self.testSet[2], self.testSet[0])))
		print('Saved.')

		self.RFResults = self.RFResults[self.RFResults["RF Time"] != 0]

		y_pred = self.RFResults["RF Time"].to_numpy(dtype=np.int8)
		print(y_pred.shape)
		y_test = TERRAINS_OG[self.testSet[1]] * np.ones(len(y_pred), dtype=np.int8)
		print(y_test.shape)

		print(accuracy_score(y_test, y_pred))
		print(balanced_accuracy_score(y_test, y_pred))
		#print(f1_score(y_test, y_pred, average='macro'))
		#print(precision_score(y_test, y_pred, average='macro'))
		#print(recall_score(y_test, y_pred, average='macro'))

	def fnShutDown(self):

		print('Closing Socket')
		self.socket.close()
		try:
			self.sock.close()
		except Exception as e:
			print(e)

	def fnFilterButter(self, dataWindow):
		"""
		Purpose:	Low pass butterworth filter onto rolling window and 
					stores in filtered class variable
					Applies hanning window
		Passed:		Rolling raw IMU data
		"""

		# Get normalized frequencies
		w_low = 2 * CUT_OFF / self.sensorParam['fSamp']

		# Get Butterworth filter parameters
		sos = signal.butter(N=2, Wn=w_low, btype='low', output='sos')

		dataSet = np.copy(dataWindow)

		# Filter all the data columns
		for i in range(6):
			self.windowIMUfiltered[:, i] = signal.sosfiltfilt(sos, dataSet[:, i])[
										   PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]  # *hanningWindow

	def fnBuildFFT(self, dataWindow):
		"""
		Purpose:	Builds power spectrum densities for each direction
		Passed:		Filtered IMU data
		"""

		# number of sample points
		N = self.sensorParam['wLength']

		# sample spacing
		T = 1 / self.sensorParam['fSamp']

		# frequency bin centers
		xf = fft.fftfreq(N, T)[:N // 2]

		window_fft = np.zeros((round(N / 2), 6))

		hanningWindow = np.hanning(self.sensorParam['wLength'])

		for i in range(6):
			window_fft[:, i] = fft.fft(dataWindow[:, i] * hanningWindow)[0:int(N/2)]
			window_fft[:, i] = 2.0 / N * abs(window_fft[:, i])  # keeping positive freq values * 2 / window_size
		# Get positive frequency bins for given FFT parameters

		freq_col = np.reshape(xf, (-1, 1))

		f_ind = np.ceil((CUT_OFF + 10) / self.sensorParam['fSamp'] * 2 * self.sensorParam['wLength']/2).astype(int)
		#f_bool = freq_col <= CUT_OFF + 10

		# Append the frequency column
		self.windowIMUFFT = np.append(window_fft[0:f_ind+1, :], freq_col[0:f_ind+1], axis=1)

	def fnBuildPSD(self, dataWindow):
		"""
		Purpose:	Builds power spectrum densities for each direction
		Passed:		Filtered IMU data
		"""

		# number of sample points
		N = self.sensorParam['wLength']

		# sampling frequency
		fs = self.sensorParam['fSamp']

		windowIMUPSD = np.zeros(((128+1), 6))

		hanningWindow = np.hanning(self.sensorParam['wLength'])

		# Calculate PSD for each axes
		for i in range(6):
			# Normalized PSD - Returns frequencies and power density
			freq, Pxx = signal.welch(dataWindow[:, i] * hanningWindow, fs)
			windowIMUPSD[:, i] = Pxx

		f_ind = np.ceil((CUT_OFF + 10) / self.sensorParam['fSamp'] * 2 * 128).astype(int)

		# Append freq column
		self.windowIMUPSD = np.append(windowIMUPSD[0:f_ind, :], freq[0:f_ind].reshape(-1, 1), axis=1)

	def fnBuildTimeFeatures(self, features):
		"""
		Purpose:	Perform all time domain feature extraction on filtered data, 
					then columns the data
		Passed:		Feature dictionary to perform
		"""
		dataList = [TIME_FEATURES[featName](self.windowIMUfiltered[:, i]) for i, direction in enumerate(DATA_COLUMNS)
					for featName in features]
		dataNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName
					 in features]
		self.EFTimeColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)

	def fnBuildFreqFeatures(self, features):
		"""
		Purpose:	Perform all frequency domain feature extraction on filtered data, 
					then columns the data
		Passed:		Feature dictionary to perform
		"""
		dataList = [FREQ_FEATURES[featName](self.windowIMUPSD[:, -1], self.windowIMUPSD[:, i]) for i, direction in
					enumerate(DATA_COLUMNS) for featName in features]
		dataNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName
					 in features]
		self.EFFreqColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)

	def fnBuildTimeFreqFeatures(self, timeFeatures, freqFeatures):
		"""
		Purpose:	Perform all frequency and time domain feature extraction on filtered data,
					then columns the data
		Passed:		Feature dictionary to perform
		"""
		dataList = [TIME_FEATURES[featName](self.windowIMUfiltered[:, i]) for i, direction in enumerate(DATA_COLUMNS)
					for featName in timeFeatures] + \
				   [FREQ_FEATURES[featName](self.windowIMUPSD[:, -1], self.windowIMUPSD[:, i])
					for i, direction in enumerate(DATA_COLUMNS) for featName in freqFeatures]
		dataNamesTime = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for
						 featName
						 in timeFeatures]
		dataNamesFreq = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for
						 featName in freqFeatures]
		dataNames = dataNamesTime + dataNamesFreq
		self.EFTimeFreqColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)


class ClIMUDataStream(threading.Timer):
	"""
	Class for establishing wireless communications.
	"""

	def __init__(self, dataQueue, runMarker, testSet):
		self.testSet = testSet
		self.streamFile = pd.read_csv(
			os.path.join(dir_path, "set_power", "Middle_{}Power{}{}_Module{}.csv".format(testSet[1], testSet[2], testSet[0], IMU)))
		self.streamRow = 0
		self.streamRowEnd = len(self.streamFile.index)
		self.dataQueue = dataQueue
		self.runMarker = runMarker
		self.offset = np.zeros(6)

	def fnRetrieveData(self):
		"""
		Purpose:	Send data to main data queue for transfer with timestamp and sensor ID.
		Passed:		None
		"""

		timeRecorded = time.time()
		if self.streamRow < self.streamRowEnd:
			data = self.streamFile.iloc[self.streamRow, :]
			self.dataQueue.put(['IMU_6', data[9], data[0], data[1], data[2], data[3], data[4], data[5]])
			self.streamRow += 1
		else:
			self.runMarker.put(False)

	def fnRun(self, frequency):
		"""
		Purpose:	Script that runs until termination message is sent to queue.
		Passed:		Frequency of data capture
		"""

		# Sets time interval between signal capture
		waitTime = 1 / frequency

		# Sets trigger so code runs
		self.trigger = threading.Event()
		self.trigger.set()

		# Create repeating timer that ensures code runs at specified intervals
		timerRepeat = threading.Thread(target=self.fnRunThread, args=(waitTime,))
		timerRepeat.start()

		count = 0
		startTime = time.time()

		# Continuously reruns code and clears the trigger
		while self.runMarker.empty():
			count += 1
			self.trigger.wait()
			self.trigger.clear()
			self.fnRetrieveData()

		endTime = time.time()

		print("Sampling Frequency:       {:>8.2f} Hz. ({} Samples in {:.2f} s)".format(count / (endTime - startTime),
																					   count, (endTime - startTime)))

		PERFORMANCE["{}-{}-{}-Acquisition".format(self.testSet[1], self.testSet[2], self.testSet[0])] = (
		count, endTime - startTime)

		# Joins thread
		timerRepeat.join()

	def fnRunThread(self, waitTime):
		"""
		Purpose:	Sets the trigger after waiting for specified interval
		Passed:		Interval of time to wait
		"""

		while self.runMarker.empty():
			time.sleep(waitTime - (time.perf_counter() % waitTime))
			self.trigger.set()


# MAIN PROGRAM

if __name__ == "__main__":

	for testSet in TERRAIN_DICT:

		connectedStatus = False
		processStatus = False
		runCompletion = False

		while runCompletion == False:
			try:
				instTerrainClassifier = ClTerrainClassifier(testSet, protocol='TCP')
				processStatus = True
				instTerrainClassifier.fnStart(300)
				instTerrainClassifier.runMarker.close()
				instTerrainClassifier.dataQueue.close()
				print("Application Completed.")
				runCompletion = True
			except Exception as e:
				time.sleep(1)
				if processStatus:
					instTerrainClassifier.runMarker.put(False)
					instTerrainClassifier.fnShutDown()
					instTerrainClassifier.runMarker.close()
					instTerrainClassifier.dataQueue.close()
					connectedStatus = False
				print(e)

		print(PERFORMANCE)

		dump(PERFORMANCE, os.path.join('2021-Results', 'performance.joblib'))
