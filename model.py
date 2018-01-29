import numpy as np
import h5py

def loadData():
	'''
    This function reads in all the CSV data and saves it to an
    hdf5 file - it takes around 14mins on average to run on a
    dual processor workstation
	'''
	# traindata format - as the SNP data is binary, can be stored
	# efficiently in 8-bit integer format (minimum in numpy)
	# Ids,SNP_0_a,SNP_0_b,SNP_0_a,SNP_0_b, [...] ,SNP_18123_a,SNP_18123_b
	# ID0,1,0,0,0, [...] ,1,0
	global train
	global label
	global test
	train = np.loadtxt("train.csv", delimiter=',', skiprows=1, usecols=range(1,36249), dtype='int8')
	label = np.loadtxt("train_label.csv", delimiter=';', skiprows=1, usecols=(1), dtype='int8')
	test = np.loadtxt("test.csv", delimiter=',', skiprows=1, usecols=range(1,36249), dtype='int8')

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
	train_feature = np.zeros((train.shape[0],int(train.shape[1]/2)),
							  dtype='int8')
	test_feature = np.zeros((test.shape[0],int(test.shape[1]/2)),
							 dtype='int8')

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
