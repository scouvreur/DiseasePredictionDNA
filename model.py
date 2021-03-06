'''
|=======================================================|
|                                                       |
| This program implements functions to import data from |
| the Owking/INSERM DNA challenge and visualize results |
|                                                       |
|=======================================================|
'''
print(__doc__)

import numpy as np
import h5py
from matplotlib import pyplot as plt

def loadData():
	'''
	This function reads in all the CSV data and saves it to an
	hdf5 file - it takes around 14mins on average to run on a
	dual processor workstation
	'''
	# traindata format - SNP data is binary, can be stored
	# efficiently in 8-bit integer format (minimum in numpy)
	# Ids,SNP_0_a,SNP_0_b, [...] ,SNP_18123_a,SNP_18123_b
	# ID0,1,0, [...] ,1,0
	global train
	global label
	global test
	train = np.loadtxt("train.csv", delimiter=',', skiprows=1,
					   usecols=range(1,36249), dtype='int8')
	label = np.loadtxt("train_label.csv", delimiter=';',
					   skiprows=1, usecols=(1), dtype='int8')
	test = np.loadtxt("test.csv", delimiter=',', skiprows=1,
					  usecols=range(1,36249), dtype='int8')

def saveData():
	'''
	This function writes all the data to an hdf5 file
	'''
	# write numpy arrary tensor into h5 format
	h5f = h5py.File('DNAdata.h5', 'w')
	h5f.create_dataset('train', data=train)
	h5f.create_dataset('label', data=label)
	h5f.create_dataset('test', data=test)
	h5f.create_dataset('train_feature', data=train_feature)
	h5f.create_dataset('test_feature', data=test_feature)
	h5f.close()

def readData():
	'''
	This function reads in the hdf5 file - it takes
	around 3s on average to run on a
	dual processor workstation
	'''
	# read h5 format back to numpy array
	global train
	global label
	global test
	global train_feature
	global test_feature
	h5f = h5py.File('DNAdata.h5', 'r')
	train = h5f['train'][:]
	label = h5f['label'][:]
	test = h5f['test'][:]
	train_feature = h5f['train_feature'][:]
	test_feature = h5f['test_feature'][:]
	h5f.close()
	print("--- Data read in successfully ---")

def createFeatures():
	'''
	This function creates the SNP features for the
	dataset where if an SNP pair is infrequent the data
	is recorded
	'''
	global train_feature
	global test_feature
	train_feature = np.zeros((train.shape[0],
							  train.shape[1]//2), dtype='int8')
	test_feature = np.zeros((test.shape[0],
							  test.shape[1]//2), dtype='int8')

	for i in range(train.shape[0]):
		for j in range(int(train.shape[1]/2)):
			if (train[i,2*j] == 1) and (train[i,2*j+1] == 1):
				train_feature[i,j] = 1
				print("At ({},{}) {}{} SNP MATCH".format(i,j,
					  train[i,2*j],train[i,2*j+1]))

	for i in range(test.shape[0]):
		for j in range(int(test.shape[1]/2)):
			if (test[i,2*j] == 1) and (test[i,2*j+1] == 1):
				test_feature[i,j] = 1
				print("At ({},{}) {}{} SNP MATCH".format(i,j,
					  test[i,2*j],test[i,2*j+1]))

def plotMatrixSNP():
	'''
	This function plots the SNP matrix data
	'''
	trainTitle = '''SNP Matrix - Training Data
	0 - frequent, 1 - infrequent
	'''
	plt.title(trainTitle)
	plt.imshow(train, cmap='gray')
	plt.colorbar(ticks=[0, 1], orientation='vertical')
	plt.savefig("trainSNP.pdf", format='pdf')
	plt.show()

	testTitle = '''SNP Matrix - Testing Data
	0 - frequent, 1 - infrequent
	'''
	plt.title(testTitle)
	plt.imshow(test, cmap='gray')
	plt.colorbar(ticks=[0, 1], orientation='vertical')
	plt.savefig("testSNP.pdf", format='pdf')
	plt.show()
