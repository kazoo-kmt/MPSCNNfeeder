# The following code is based on the following
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# Useful image to understand the model: https://dsotb.quora.com/Deep-learning-with-Keras-simple-image-classification
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# # The followings are newly added codes.
# Output the summary of the model
model.summary()

# # Save the model to json and weight to hdf5
# mnist_model_json_str = model.to_json()
# open('mnist_model.json', 'w').write(mnist_model_json_str)
model.save_weights('mnist_model_weights.h5')


# # Save the parameters to dat file
# Swap numpy array to fit for Metal Performance Shaders.
# Changing from order of weights [kH kW iC oC] to MPS accepted order of weights i.e. [oC kH kW iC]
# ref: https://forums.developer.apple.com/message/195288#195288
# ref: https://developer.apple.com/reference/metalperformanceshaders/mpscnnfullyconnected/1829441-init
# ref: https://developer.apple.com/reference/metalperformanceshaders/mpscnnconvolution/1648861-init
def swapaxes_for_MPS_convolution(nparray):
    nparray = np.swapaxes(nparray, 2, 3)
    nparray = np.swapaxes(nparray, 1, 2)
    nparray = np.swapaxes(nparray, 0, 1)
    return nparray

def swapaxes_for_MPS_fullyconnected(nparray):
    nparray = np.swapaxes(nparray, 0, 1)
    return nparray

weights_conv1 = swapaxes_for_MPS_convolution(model.get_weights()[0])
weights_conv1.astype('float32').tofile('weights_conv1.dat')
bias_conv1 = model.get_weights()[1]
bias_conv1.astype('float32').tofile('bias_conv1.dat')

weights_conv2 = swapaxes_for_MPS_convolution(model.get_weights()[2])
weights_conv2.astype('float32').tofile('weights_conv2.dat')
bias_conv2 = model.get_weights()[3]
bias_conv2.astype('float32').tofile('bias_conv2.dat')

weights_fc1 = swapaxes_for_MPS_fullyconnected(model.get_weights()[4])
weights_fc1.astype('float32').tofile('weights_fc1.dat')
bias_fc1 = model.get_weights()[5]
bias_fc1.astype('float32').tofile('bias_fc1.dat')

weights_fc2 = swapaxes_for_MPS_fullyconnected(model.get_weights()[6])
weights_fc2.astype('float32').tofile('weights_fc2.dat')
bias_fc2 = model.get_weights()[7]
bias_fc2.astype('float32').tofile('bias_fc2.dat')

print('Finished to save')
