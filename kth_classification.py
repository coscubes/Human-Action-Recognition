import cv2
import numpy as np
from os import listdir
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

X_test, X_val, X_train = [], [], []
Y_test, Y_val, Y_train = [], [], []
frames_per_video = 150
epochs = 300
batch_size = 32

# Prepare data  for boxing
files = listdir("kth dataset/boxing")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/boxing/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(0)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/boxing/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(0)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/boxing/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(0)
	cap.release()

#prepare data for hand clapping
files = listdir("kth dataset/handclapping")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/handclapping/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(1)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/handclapping/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(1)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/handclapping/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(1)
	cap.release()

#prepare data for handwaving
files = listdir("kth dataset/handwaving")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/handwaving/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(2)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/handwaving/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(2)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/handwaving/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(2)
	cap.release()

# Prepare data for jogging
files = listdir("kth dataset/jogging")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/jogging/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(3)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/jogging/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(3)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/jogging/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(3)
	cap.release()

#Prepare data for running
files = listdir("kth dataset/running")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/running/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(4)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/running/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(4)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/running/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(4)
	cap.release()

#prepare data for walking
files = listdir("kth dataset/walking")
for i in files[:36]:
	cap = cv2.VideoCapture("kth dataset/walking/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_train.append(vid)
	Y_train.append(5)
	cap.release()

for i in files[36:68]:
	cap = cv2.VideoCapture("kth dataset/walking/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_val.append(vid)
	Y_val.append(5)
	cap.release()

for i in files[68:]:
	cap = cv2.VideoCapture("kth dataset/walking/" + i)
	vid = []
	counter = 0
	while(counter < frames_per_video):
		ret, img = cap.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		vid.append(img)
		counter += 1

	if counter != frames_per_video: print "error", i
	X_test.append(vid)
	Y_test.append(5)
	cap.release()

X_train = np.array(X_train, np.float32) / 255
X_test = np.array(X_test, np.float32) / 255
X_val = np.array(X_val, np.float32) / 255
X_train = np.reshape(X_train, (216, 150, 120, 160, 1))
X_test = np.reshape(X_test, (191, 150, 120, 160, 1))
X_val =  np.reshape(X_val, (192, 150, 120, 160, 1))
Y_train = to_categorical(np.array(Y_train), 6)
Y_test = to_categorical(np.array(Y_test), 6)
Y_val = to_categorical(np.array(Y_val), 6)
print X_train.shape
print X_val.shape
print X_test.shape
print Y_train.shape
print Y_val.shape
print Y_test.shape

model = Sequential()
model.add(Convolution3D(32, 5, input_shape=(150, 120, 160, 1), activation = 'relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6,init='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['accuracy'])
print model.summary()

val_acc = -1
for i in range(epochs):
	print "Epoch Number ", i + 1, "/", epochs
	history = model.fit(X_train, Y_train,
	epochs=1,
	shuffle=True,
	validation_data=(X_val, Y_val), 
	batch_size=batch_size)

	if val_acc < history.history['val_acc']:
                val_acc = history.history['val_acc']
                model.save("models1/imp-epoch-" + str(i) + "-val_acc-" + str(val_acc)[:5] + ".h5")
        else:
                model.save("models1/latest-epoch-" + str(i) + ".h5")

pred = model.predict(X_test, Y_test)
print pred