
"""
Author:         Kevin Ta
Date:           2020 July 20th
Purpose:        This Python script tests terrain classification by passing streamed data.
"""

# IMPORTED LIBRARIES

import math
import time
import datetime
import os
import sys
import threading
import operator

import pickle as pkl
import numpy as np
from scipy import signal, stats
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
import pickle as pkl
from joblib import load, dump

from multiprocessing import Process, Queue
from threading import Thread

# LOCALLY IMPORTED LIBRARIES
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, 'libraries'))

from WheelModuleLib import *
from featuresLib import *

# DEFINITIONS

# Model
MODEL = '2019' # '2019' or "2020'

# Classification frequency
CLASS_FREQ = 0.2 # 0.2, 0.5, or 0.8

# Active person
PERSON_DATA = 'Jamie' # 'Keenan', 'Kevin', 'Mahsa', or 'Jamie'

# Active terrain
TEST_TERRAIN = 'Gravel' #'Linoleum', 'Grass', 'Gravel'

# Sensor list to activate
SENSOR_LIST = ['IMU_9', 'IMU_6', 'USS_DOWN', 'USS_FORW', 'PI_CAM', 'LEFT', 'RIGHT']

#Active sensors
ACTIVE_SENSORS = [1]

# Direction Vectors
STD_COLUMNS = ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro', 'Run Time', 'Epoch Time']
DATA_COLUMNS =  ['X Accel', 'Y Accel', 'Z Accel', 'X Gyro', 'Y Gyro', 'Z Gyro']

EPSILON = 0.00001 # For small float values

FRAME_MODULE = {'wLength': 300, 'fSamp': 300, 'fLow': 55, 'fHigh': 1}
WHEEL_MODULE = {'wLength': 333, 'fSamp': 333.3, 'fLow': 60, 'fHigh': 1}

PAD_LENGTH = 15 # pad length to let filtering be better
N_BINS_OVER_CUTOFF = 5 # Collect some information from attenuated frequencies bins

# DICTIONARIES

# Time domain feature functions and names
TIME_FEATURES = {'Mean': np.mean, 'Std': np.std,  'Norm': l2norm, 'AC': autocorr, 
                 'Max': np.amax, 'Min' : np.amin, 'RMS': rms, 'ZCR': zcr, 
                 'Skew': stats.skew, 'EK': stats.kurtosis}

TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'AC', 'Max', 'Min', 'RMS', 'ZCR', 'Skew', 'EK']

# Time domain feature functions and names           
FREQ_FEATURES = freq_features = {'MSF': msf, 'RMSF': rmsf, 'FC': fc, 'VF': vf, 'RVF': rvf}

FREQ_FEATURES_NAMES = ['MSF', 'RMSF', 'FC', 'VF', 'RVF']

TERRAINS = ['Concrete', 'Carpet', 'Linoleum', 'Asphalt', 'Sidewalk', 'Grass', 'Gravel']

# CLASSES

class ClTerrainClassifier:
	"""
	Class for establishing wireless communications.
	"""
	
	def __init__(self, protocol = 'TCP'):
		"""
		Purpose:	Initialize various sensors and class variables
		Passed: 	Nothing
		"""
		
		# Middle
		self.placement = 'Middle'
		self.sensorParam = FRAME_MODULE

		# Left
		#~ self.placement = 'Left'
		#~ self.sensorParam = WHEEL_MODULE
		
		#~ # Right
		#~ self.placement = 'Right'
		#~ self.sensorParam = WHEEL_MODULE

		# Calculates the number of bins available
		nbins = int(self.sensorParam['wLength'] / self.sensorParam['fSamp'] *self.sensorParam['fLow'] + N_BINS_OVER_CUTOFF)
			
		print('unpickling')
		
		#~ randomForestTime = pkl.load(open(os.path.join(dir_path, 'models', 'RandomForest_Middle_TimeFeats.pkl'), 'rb'))
		#~ self.SVMTime = load('models/SupportVectorMachine_Middle_TimeFeats.joblib')
		#~ self.SVMFreq = load('models/SupportVectorMachine_Middle_FreqFeats.joblib')
		#~ self.SVMPSD = load('models/SupportVectorMachine_Middle_PSDLogs.joblib')

		if MODEL == '2019':
			# Keenan's 2019 models
			self.RFTime = load('models/RandomForest_Middle_TimeFeats.joblib')
			self.RFFreq = load('models/RandomForest_Middle_FreqFeats.joblib')
			self.RFPSD = load('models/RandomForest_Middle_PSDLogs.joblib')

			self.fftScaler = load('scalers/Middle_FFTs_Scaler.joblib')
			self.timeScaler = load('scalers/Middle_TimeFeats_Scaler.joblib')
			self.freqScaler = load('scalers/Middle_FreqFeats_Scaler.joblib')
			self.psdlScaler = load('scalers/Middle_PSDLogs_Scaler.joblib')

			self.RFResults = pd.DataFrame(columns = ["RF Time", "RF Frequency", "RF PSD"])

		elif MODEL == '2020':
			# EMBC Power models
			self.RFAll = load('models/RF_SFS_ALL_25.joblib')
			self.fftScaler = load('scalers/Middle_FFTs_Scaler_Power.joblib')
			self.timeScaler = load('scalers/Middle_TimeFeats_Scaler_Power.joblib')
			self.freqScaler = load('scalers/Middle_FreqFeats_Scaler_Power.joblib')
			self.psdlScaler = load('scalers/Middle_PSDWelch_Scaler_Power.joblib')

		# Prepopulate pandas dataframe
		EFTimeColumnNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName in TIME_FEATURES_NAMES]
		self.EFTimeColumnedFeatures = pd.DataFrame(data = np.zeros((1,len(EFTimeColumnNames))), columns = EFTimeColumnNames)
		EFFreqColumnNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName in FREQ_FEATURES_NAMES]
		self.EFFreqColumnedFeatures = pd.DataFrame(data = np.zeros((1,len(EFFreqColumnNames))), columns = EFFreqColumnNames)
		self.protocol = protocol
		
		# Initialize data queue and marker to pass for separate prcoesses
		self.dataQueue = Queue()
		self.runMarker = Queue()
		
		# Create class variables
		self.windowIMUraw = np.zeros((self.sensorParam['wLength'] + 2 * PAD_LENGTH, 6))
		self.windowIMUfiltered = np.zeros((self.sensorParam['wLength'], 6))
		self.windowIMUPSD = np.zeros([])
		self.windowIMULogPSD = np.zeros([])
		self.windowIMULogPSDFeatures = np.zeros([])
				
		# Create dictionary to house various active sensors and acivate specified sensors
		self.instDAQLoop = {} 
		
		for sensor in ACTIVE_SENSORS:
			if sensor == 1:
				self.instDAQLoop[SENSOR_LIST[sensor]] = ClIMUDataStream(self.dataQueue, self.runMarker)

	def fnStart(self, frequency):
		"""
		Purpose:	Intialize all active sensors in separate processed and collects data from the Queue
		Passed:		Frequency for 6-axis IMU to operate at
		"""

		print('Start Process.')
		
		# Start terrain classification in separate thread
		terrain = Thread(target=self.fnTerrainClassification, args = (CLASS_FREQ, ))
		terrain.start()
		
		timeStart = time.time()
		
		# Create dictionary to store processes
		processes = {}

		# Start various data collection sensors
		for sensor in ACTIVE_SENSORS:
			processes[SENSOR_LIST[sensor]] = Process(target=self.instDAQLoop[SENSOR_LIST[sensor]].fnRun, args = (frequency, ))
			processes[SENSOR_LIST[sensor]].start()

		#Keep collecting data and updating rolling window
		while self.runMarker.empty():

			transmissionData = self.dataQueue.get()

			if transmissionData[0] in ['IMU_6', 'WHEEL']:
				self.windowIMUraw = np.roll(self.windowIMUraw, -1, axis=0)
				self.windowIMUraw[-1, :] = np.multiply(np.subtract(transmissionData[2:8], [0, 0, 9.8, 0, 0, 0]), [1, 1, 1, math.pi/180, math.pi/180, math.pi/180]) 
			elif transmissionData[0] in ['USS_DOWN', 'USS_FORW']:
				pass
			elif transmissionData[0] in ['PI_CAM']:
				pass

		# wait for all processes and threads to complete
		terrain.join()
		for sensor in ACTIVE_SENSORS:
			processes[SENSOR_LIST[sensor]].join()

	def fnTerrainClassification(self, waitTime):
		"""
		Purpose:	Class method for running terrain classification
		Passed:		Time in between runs
		"""
		
		index = 0
		
		# Keep running until run marker tells to terminate
		while self.runMarker.empty():
			
			# print(time.perf_counter())
			
			# Filter window
			self.fnFilterButter(self.windowIMUraw)
			
			# Build extracted feature vector
			self.fnBuildTimeFeatures(TIME_FEATURES_NAMES)
			
			# Build PSD and PSD features
			self.fnBuildPSD(self.windowIMUfiltered)
			self.fnBuildFreqFeatures(FREQ_FEATURES_NAMES)
			
			#~ terrainTypeSVMTime = self.SVMTime.predict(self.EFTimeColumnedFeatures)
			#~ terrainTypeSVMFreq = self.SVMFreq.predict(self.EFFreqColumnedFeatures)
			#~ terrainTypeSVMPSD = self.SVMPSD.predict(self.windowIMULogPSDFeatures)

			if MODEL == '2019':
				# Keenan's 2019 models
				terrainTypeRFTime = self.RFTime.predict(self.EFTimeColumnedFeatures)
				terrainTypeRFFreq = self.RFFreq.predict(self.EFFreqColumnedFeatures)
				terrainTypeRFPSD = self.RFPSD.predict(self.windowIMULogPSDFeatures)

			elif MODEL == '2020':
				# terrainTypeRFAll = self.RFAll.predict(self.)
				pass

			try:
				print('Time: {0:>8s}     Freq: {1:>8s}     PSD:  {2:>8s}'.format(TERRAINS[terrainTypeRFTime[0]], TERRAINS[terrainTypeRFFreq[0]], TERRAINS[terrainTypeRFPSD[0]]))
				self.RFResults = self.RFResults.append({"RF Time": TERRAINS[terrainTypeRFTime[0]], "RF Frequency": TERRAINS[terrainTypeRFFreq[0]], "RF PSD": TERRAINS[terrainTypeRFPSD[0]]}, ignore_index=True)
			except Exception as e:
				print(e)
				break
			
			time.sleep(waitTime - (time.perf_counter() % waitTime))

		self.RFResults.to_csv(os.path.join('2019-Results', "{}ms-{}-{}.csv".format(CLASS_FREQ, TEST_TERRAIN, PERSON_DATA)))
		print('Saved.')

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
		Passed:		Rolling raw IMU data
		"""
		
		# Get normalized frequencies
		w_low = self.sensorParam['fLow'] / (self.sensorParam['fSamp'] / 2) 
		w_high = self.sensorParam['fHigh'] / (self.sensorParam['fSamp'] / 2)

		# Get Butterworth filter parameters
		b_butter, a_butter = signal.butter(N=4, Wn=w_low, btype='low')
		
		dataSet = np.copy(dataWindow)
		
		# Filter all the data columns
		for i in range(6):
			self.windowIMUfiltered[:, i] = signal.filtfilt(b_butter, a_butter, dataSet[:, i])[PAD_LENGTH:self.sensorParam['wLength']+PAD_LENGTH]
			
	def fnBuildPSD(self, dataWindow):
		"""
		Purpose:	Builds power spectrum densities for each direction
		Passed:		Filtered IMU data
		"""
		
		# Only include frequency bins up to and a little bit past the cutoff frequency
		# Everything past that is useless because its the same on all terrains
		n_bins = int(self.sensorParam['wLength'] / self.sensorParam['fSamp'] * self.sensorParam['fLow']) + N_BINS_OVER_CUTOFF
		windowIMUPSD = np.zeros((n_bins, 6))
		windowIMULogPSD = np.zeros((n_bins, 6))

		# Calculate PSD for each axes
		for i in range(6):
			# Normalized PSD - Returns frequencies and power density
			freq, Pxx = signal.periodogram(dataWindow[:, i], self.sensorParam['fSamp'])
			windowIMUPSD[:, i] = np.resize(Pxx[1:], n_bins)
			
			# Calculate log10 of PSD, replacing points where PSD = 0 with 0 to avoid division by 0
			for j in range(len(windowIMUPSD[:, i])):
				if (windowIMUPSD[j, i] == 0):
					windowIMULogPSD[j, i] = 0
				else:
					windowIMULogPSD[j, i] = np.log10(windowIMUPSD[j, i])
			
		# Append freq column
		freq_col = np.transpose([np.resize(freq[:-1], n_bins)])
		self.windowIMUPSD = np.append(windowIMUPSD, freq_col, axis=1)
		self.windowIMULogPSD = np.append(windowIMULogPSD, freq_col, axis=1)
		
		colNames = ['{} {} Hz {} {}'.format('PSDLog', round(freq[0]), direction, self.placement) for direction in DATA_COLUMNS for freq in freq_col]	
		psdLogData = [self.windowIMULogPSD[i, j] for j in range(len(self.windowIMULogPSD[0, :])-1) for i in range(len(self.windowIMULogPSD[:, 0]))]
		#~ psdLogData = np.divide(np.subtract(psdLogData, self.PSDMean), self.PSDScale)
		psdLogData = self.psdlScaler.transform(np.array(psdLogData).reshape(1,-1))[0]
		self.windowIMULogPSDFeatures = pd.DataFrame(data=[psdLogData], columns=colNames)
   	
	def fnBuildTimeFeatures(self, features):
		"""
		Purpose:	Perform all time domain feature extraction on filtered data, 
					then columns the data and standardized based on mean and std
		Passed:		Feature dictionary to perform
		"""
		dataList = [TIME_FEATURES[featName](self.windowIMUfiltered[:, i]) for i, direction in enumerate(DATA_COLUMNS) for featName in features]
		#~ dataList = np.divide(np.subtract(dataList, self.TimeMean), self.TimeScale)
		dataList = self.timeScaler.transform(np.array(dataList).reshape(1,-1))[0]
		dataNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName in features]
		self.EFTimeColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)

	def fnBuildFreqFeatures(self, features):
		"""
		Purpose:	Perform all frequency domain feature extraction on filtered data, 
					then columns the data and standardized based on mean and std
		Passed:		Feature dictionary to perform
		"""
		dataList = [FREQ_FEATURES[featName](self.windowIMUPSD[:, -1], self.windowIMUPSD[:, i]) for i, direction in enumerate(DATA_COLUMNS) for featName in features]
		#~ dataList = np.divide(np.subtract(dataList, self.FreqMean), self.FreqScale)
		dataList = self.freqScaler.transform(np.array(dataList).reshape(1, -1))[0]
		dataNames = ['{} {} {}'.format(featName, direction, self.placement) for direction in DATA_COLUMNS for featName in features]
		self.EFFreqColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)


class ClIMUDataStream(threading.Timer):
	"""
	Class for establishing wireless communications.
	"""

	def __init__(self,  dataQueue, runMarker):
		self.streamFile = pd.read_csv(os.path.join(dir_path, "set_power", "Middle_{}PowerF8{}_Module6050.csv".format(TEST_TERRAIN, PERSON_DATA)))
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
			data = self.streamFile.iloc[self.streamRow,:]
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

		# Continuously reruns code and clears the trigger
		while self.runMarker.empty():
			self.trigger.wait()
			self.trigger.clear()
			self.fnRetrieveData()

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

if __name__=="__main__":
	
	connectedStatus = False
	processStatus = False
	runCompletion = False

	while runCompletion == False:
		try:
			instTerrainClassifier = ClTerrainClassifier(protocol = 'TCP')
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
