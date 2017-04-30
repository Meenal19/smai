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

batch_size = 32
num_classes = 10
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#x = ZeroPadding2D((1,1),input_shape=x_train.shape[1:])
#x = Convolution2D(64, 3, 3, activation='relu')(x)

# model = Sequential()


inputs = Input(shape=x_train.shape[1:])
x=ZeroPadding2D((1,1))(inputs)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)


# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))


x=ZeroPadding2D((1,1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)


# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))




x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)


model = Model(input=inputs, output=x)



# x=Flatten()
# model.add(x)


#def _loss(y_true, y_pred):

# x = Dense(4096, activation='relu')(x)
# special_layer = Dense(4096, activation='relu')
# special_weights = special_layer.get_weights()
# x2 = special_layer(x)
# x = Dense(10, activation='relu')(x2)

# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

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

#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

if os.path.exists('model_saver.h5')==False:
  model_saver = keras.callbacks.ModelCheckpoint('model_saver.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  model_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, verbose=0, mode='auto')
  model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.3,shuffle=True, callbacks=[model_saver, model_stopper])
else:
  model.load_weights('model_saver.h5')

wt=model.layers[-1].get_weights()


hist = model.evaluate(x_test, y_test)
print(hist)