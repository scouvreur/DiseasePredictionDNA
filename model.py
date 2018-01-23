# from sklearn import svm
# import pandas as pd
import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
import h5py
# from sklearn.metrics import roc_auc_score

# def loadTrainingData():
# 	'''
# 	This array will be of 1D with length 784, the pixel intensity values
# 	are integers from 0 to 255. We can therefore use unsigned 8-bit integers.
# 	The array is reshaped to the image dimensions 28x28.
# 	'''

# loadTrainingData()

train = np.zeros((26499+1,2*(18123+1))) # initialize an empty 2D array
print('start processing traindata')
with open('train.csv') as trainfile:
    for index,line in enumerate(trainfile):
	    # traindata format
	    # Ids,SNP_0_a,SNP_0_b,SNP_0_a,SNP_0_b, [...] ,SNP_18123_a,SNP_18123_b
	    # ID0,1,0,0,0, [...] ,1,0

	    traindata = line.split(',')
	    try:
	    	ID = str(traindata[0])
	    	SNP = traindata[1:]
	    	train[ID-1] = SNP # write values into 2D array

	    	if index % 1000 == 0:
	            print(index,"lines have been processed")
	    except ValueError:
	        print("found line with datatype error! skip this line")
	        continue

# As the SNP data is binary, can be stored
# efficiently in 8-bit integer format
# (minimum in numpy)

print(train)