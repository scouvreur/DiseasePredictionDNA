import numpy as np
import h5py
from sklearn import svm
from sklearn.metrics import roc_auc_score

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

readData()

train_set = train[:100,:]
train_label = label[:100]
test_set = test[:100,:]

clf = svm.SVC()
clf.fit(train_set, train_label)

test_label = clf.predict(test_set)

print("Ids;TARGET")
for i in range(len(test_label)):
	print("ID{};{}".format(i+26500,test_label[i]))