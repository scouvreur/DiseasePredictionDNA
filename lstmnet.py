import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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

X_train = train[:200,:60]
Y_train = label[:200]
X_test = test[:200,:60]

# train_set = train[:,:]
# train_label = label[:]
# test_set = test[:,:]

encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)

# baseline model
def create_baseline():
	# create model
	global model
	model = Sequential()
	# model.add(Dense(36248, input_dim=36248, kernel_initializer='normal', activation='relu'))
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

Y_test = model.predict(X_test)

# f = open("test_label_lstmnet.csv", 'w')
# f = open("test_label_lstmnet.csv", 'a')
# print("ImageId,Label", file=f)
# for i in range(len(Y_test)):
# 	print("{},{}".format(i+1,np.argmax(Y_test[i])), file=f)

print("Ids;TARGET")
for i in range(len(Y_test)):
	print("ID{};{}".format(i+26500,Y_test[i]))
