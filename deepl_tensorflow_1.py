# Basic code snippet
import tensorflow as tf
from tensorflow import keras
import numpy as np

W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")


@tf.function
def model(x):
    return W * x + b


out_a = model([1, 0])
print(out_a)

# Defining a dense layer 

NB_CLASSES = 10 
RESHAPED = 784 
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES,
    input_shape=(RESHAPED, ), kernel_initializer='zeros',
    name='dense_layer', activation='softmax'))


# Define a simple network to deal with MNIST 

EPOCHS = 200 
BATCH_SIZE = 128 # data fed at a time 
VERBOSE = 1 
NB_CLASSES = 1 # number of outputs (digits)
N_HIDDEN = 128 
VALIDATION_SPLIT = 0.2 # amount of TRAIN reserved for VALIDATION 

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
model.add(keras.layers.Dense(NB_CLASSES,
    input_shape=(RESHAPED, ), # need to define input size
    name='dense_layer',
    activation='softmax'))

# Once the model is defined, it is necessary to compile it and
# then fit it to the parameters
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluate the model using the reserved test sets
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy: ', test_acc)
    
