from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import os
import h5py
from keras import backend as K
import math
from math import pi

batch_size = 32
num_classes = 10
epochs = 10

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return (math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))*180)/pi

def normCalculation(w):
    norms = K.sqrt(K.sum(K.square(w),keepdims=True))
    return norms

m=4
def largeMarginSoftmaxLoss(y_true, y_pred):
    wt=wt.reshape(10,4096);
    totalLoss=0;
    predictedClass = y_pred
    normw=normCalculation(w[predictedClass])
    normx=normCalculation(x)
    angleXW=angle(x,w[predictedClass])
    angleXW = m*angleXW
    xwi=normx*normw*math.cos(angleXW)
    xwi=math.exp(xwi)
    totalSum=0.0
    for each in w:
        normw=normCalculation(w[each])
        angleXW=angle(x,w[each])*m
        xwt=normx*normw*math.cos(angleXW)
        totalSum=totalSum + math.exp(xwt)
    totalLoss = totalLoss - math.log(xwi/totalSum)

    return totalLoss


# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=x_train.shape[1:]))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

wt=model.layers[-1].get_weights()
print(wt)
x=model.layers[-1].input
print(x)
K.eval(x)

if os.path.exists('model_saver.h5')==False:
  model_saver = keras.callbacks.ModelCheckpoint('model_saver.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  model_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, verbose=0, mode='auto')
  model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.3,shuffle=True, callbacks=[model_saver, model_stopper])
else:
  model.load_weights('model_saver.h5')

wt=model.layers[-1].get_weights()


hist = model.evaluate(x_test, y_test)
print(hist)