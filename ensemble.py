import numpy as np
import h5py
from sklearn.model_selection import cross_val_score
# from sklearn.datasets import make_blobs
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
	print("--- Data read in successfully ---")

readData()

# train_set = train[:1000,:]
# train_label = label[:1000]
# test_set = test[:1000,:]

train_set = train[:,:]
train_label = label[:]
test_set = test[:,:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
clf.fit(train_set, train_label)
print(clf.score(train_set, train_label))

test_label = clf.predict(test_set)

print("Ids;TARGET")
for i in range(len(test_label)):
	print("ID{};{}".format(i+26500,test_label[i]))
