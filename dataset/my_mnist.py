
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

mnist = tf.keras.datasets.mnist
mnist

def load_mnist(normalize = False, flatten = True, one_hot_label = True):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  if (normalize):
    x_train = x_train / 255
    x_test = x_test / 255

  if (flatten):
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

  if (one_hot_label):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


  return (x_train,y_train), (x_test,y_test)
