# from sklearn import svm
# import pandas as pd
import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import h5py
# from sklearn.metrics import roc_auc_score

# def loadTrainingData():
# 	'''
# 	This array will be of 1D with length 784, the pixel intensity values
# 	are integers from 0 to 255. We can therefore use unsigned 8-bit integers.
# 	The array is reshaped to the image dimensions 28x28.
# 	'''

# loadTrainingData()

train = np.zeros((10,5,2))

# As the SNP data is binary, can be stored
# efficiently in 8-bit integer format
# (minimum in numpy)

print(train)