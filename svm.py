import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

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

# X_train = train_feature[:50,:]
# Y_train = label[:50]
# X_test = test_feature[:50,:]

X_train = train
Y_train = label
X_test = test

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

C=0.0001
kernel='rbf'

# clf = svm.SVC(C=0.0001, kernel='rbf', random_state=0)
# clf = svm.SVC(C=0.0001, kernel='linear', random_state=0)
clf = svm.LinearSVC(C=C, random_state=0)
clf.fit(X_train, Y_train)

Y_test = clf.predict(X_test)

print("--- Model parameters ---")
print("LinearSVC(C={}))".format(C))

print("--- Validation set ---")
print("Accuracy;{}".format(accuracy_score(Y_validation, clf.predict(X_validation))))
print("F1;{}".format(f1_score(Y_validation, clf.predict(X_validation))))
print("AUC;{}".format(roc_auc_score(Y_validation, clf.predict(X_validation))))

print("--- Training set ---")
print("Accuracy;{}".format(accuracy_score(Y_train, clf.predict(X_train))))
print("F1;{}".format(f1_score(Y_train, clf.predict(X_train))))
print("AUC;{}".format(roc_auc_score(Y_train, clf.predict(X_train))))

print("Ids;TARGET")
for i in range(len(X_test)):
	print("ID{};{}".format(i+26500,Y_test[i]))
