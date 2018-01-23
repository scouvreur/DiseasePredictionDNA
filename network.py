from sklearn import svm
import pandas as pd
import numpy as np
import tensorflow as tf

def loadTrainingData():
	'''
	This array will be of 1D with length 784, the pixel intensity values
	are integers from 0 to 255. We can therefore use unsigned 8-bit integers.
	The array is reshaped to the image dimensions 28x28.
	'''
	# Read in data and differentiate label and pixels
	train = pd.read_csv('train.csv', sep=',')
	global ID, genotype
	label = train['Ids']
	genotype = train.loc[:, 'pixel0':'pixel783']
	# image = []
	# for i in range(len(pixels)):
	# 	image.append(np.array(pixels.ix[i], dtype='uint8'))
	# 	image[i] = image[i].reshape(28, 28)

loadTrainingData()