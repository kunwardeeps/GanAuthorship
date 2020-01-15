import numpy as np 
import matplotlib.pyplot as plt 
import keras 

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.layers.convolutional import UpSampling2D, Conv2D 
from keras.models import Sequential, Model 
from keras.optimizers import Adam,SGD 
from sklearn.model_selection import train_test_split
import cv2
import glob
import os
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# load images
X_data = []
files = natural_sort(glob.glob("/ufrc/cis6930/kunwardeep.singh/GANImages2/*.png"))
for myFile in files:
    image = cv2.imread(myFile)
    #Normalizing the input 
    image = (image / 127.5) - 1.
    X_data.append(image)

X_data = np.asarray(X_data)
print(len(X_data))

Y_data = np.loadtxt('/ufrc/cis6930/kunwardeep.singh/GANImages2/Vectors/vectors.txt')

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

encoding_dim = 100 

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(100))

opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
model.compile(loss='mse',
              optimizer=opt,
              metrics=['cosine_proximity'])
model.summary()

model.fit(X_train, y_train,
                epochs=500,
                validation_data=(X_test, y_test))

model.save('encoder.h5')