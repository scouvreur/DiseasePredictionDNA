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
	h5f = h5py.File('DNAdata.h5', 'r')
	train = h5f['train'][:]
	label = h5f['label'][:]
	test = h5f['test'][:]
	h5f.close()

# loadData()

# saveData()