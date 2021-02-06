# Basic code snippet
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define a simple network to deal with MNIST 
EPOCHS = 200 
BATCH_SIZE = 128 # data fed at a time 
VERBOSE = 1 
NB_CLASSES = 10 # nodes in the output, number of outputs (digits)
N_HIDDEN = 128 # nodes in the hidden layers
VALIDATION_SPLIT = 0.2 # amount of TRAIN reserved for VALIDATION 
DROPOUT = 0.3 # percentage, I guess

# Load MNIST 
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape output format
RESHAPED = 784  # data, amounts of pixels per image 28x28 size
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize to be within [0, 1]
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot encoding of the labels.
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model 
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
        input_shape=(RESHAPED, ), # need to define input size
        name='dense_layer',
        activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
        name='dense_layer_2',
        activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES,
        name='dense_layer_3',
        activation='softmax'))

# Once the model is defined, it is necessary to compile it and
# then fit it to the parameters
model.summary()
model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluate the model using the reserved test sets
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy: ', test_acc)
    
