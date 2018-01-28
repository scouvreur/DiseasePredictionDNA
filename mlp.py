import numpy as np
import h5py
from sklearn.neural_network import MLPClassifier
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
	# print("--- Data read in successfully ---")

readData()

# train_set = train[:200,:]
# train_label = label[:200]
# test_set = test[:200,:]

train_set = train[:,:]
train_label = label[:]
test_set = test[:,:]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(512,512), random_state=1)
clf.fit(train_set, train_label)
# # print("---       Model trained       ---")

test_label = clf.predict(test_set)

print("Ids;TARGET")
for i in range(len(test_label)):
	print("ID{};{}".format(i+26500,test_label[i]))
