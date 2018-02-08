import numpy as np
import h5py
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

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

X_train = train[:100,:]
Y_train = label[:100]
X_test = test[:100,:]

# X_train = train
# Y_train = label
# X_test = test

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=7)

# Create model
model = Sequential()
model.add(Dense(100, input_dim=36248, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=100, batch_size=5, verbose=True)

Y_test = np.around(model.predict(X_test))
Y_test = Y_test.astype(int)
Y_test = np.concatenate(Y_test, axis=0)

print("--- Validation set ---")
print("Accuracy;{}".format(accuracy_score(Y_validation, np.around(model.predict(X_validation)))))
print("F1;{}".format(f1_score(Y_validation, np.around(model.predict(X_validation)))))
print("AUC;{}".format(roc_auc_score(Y_validation, np.around(model.predict(X_validation)))))

print("--- Training set ---")
print("Accuracy;{}".format(accuracy_score(Y_train, np.around(model.predict(X_train)))))
print("F1;{}".format(f1_score(Y_train, np.around(model.predict(X_train)))))
print("AUC;{}".format(roc_auc_score(Y_train, np.around(model.predict(X_train)))))

f = open("test_label_deepnet_2.csv", 'w')
f = open("test_label_deepnet_2.csv", 'a')
print("Ids;TARGET", file=f)
for i in range(len(X_test)):
    print("ID{};{}".format(i+26500,Y_test[i]), file=f)

# summarize history for accuracy
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
# plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracyDNN.pdf', format='pdf')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss (cross-entropy)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
# plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('lossDNN.pdf', format='pdf')
plt.clf()
