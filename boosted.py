import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn import svm
from xgboost import XGBClassifier
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

readData()

# X_train = train_feature[:100,:]
# Y_train = label[:100]
# X_test = test_feature[:100,:]

X_train = train_feature[:,:]
Y_train = label[:]
X_test = test_feature[:,:]

X_train, Y_train = shuffle(X_train, Y_train, random_state=7)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

clf = XGBClassifier(n_estimators=500, learning_rate=0.001)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_validation)
Y_test = clf.predict(X_test)

print("AUC;{}".format(roc_auc_score(Y_validation, Y_pred)))
print("Score;{}".format(clf.score(X_train, Y_train)))

print("Ids;TARGET")
for i in range(len(X_test)):
	print("ID{};{}".format(i+26500,Y_test[i]))
