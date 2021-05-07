"""

In this example we will be building
the famous Siamese network which is
used to find the similarity of the
inputs by comparing their feature vectors.

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import cv2


def base_network(input_shape = (28,28,)):
    input = layers.Input(shape=input_shape, name="base_input")
    x = layers.Conv2D(16, 3, activation="relu" , name="Conv1")(input)
    x = layers.Conv2D(32, 3, activation="relu" , name="Conv2")(x)
    x = layers.MaxPooling2D(3 , name="MaxPool")(x)
    x = layers.Conv2D(32, 3, activation="relu" , name="Conv3")(x)
    x = layers.Conv2D(16, 3, activation="relu" , name="Conv4")(x)
    x = layers.Flatten(name="flatten_input")(x)
    x = layers.Dense(16, activation='relu' , name="Dense1")(x)

    return keras.models.Model(inputs=input, outputs=x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

input_shape = (28,28,)

base_network = base_network(input_shape)
keras.utils.plot_model(base_network, "base_network.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("base_network.png")

# Creating the left input to the network.
input_left = layers.Input(shape=input_shape, name="left_input")
vector_output_left = base_network(input_left)

# Creating the right input to the network.
input_right = layers.Input(shape=input_shape, name="right_input")
vector_output_right = base_network(input_right)

# Measuring the similarity of the two vector outputs
output = layers.Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vector_output_left, vector_output_right])

# Creating the Siamese model with multiple Inputs
siamese_model = keras.models.Model([input_left, input_right], output)

# plot model graph
keras.utils.plot_model(siamese_model, "siamese_model.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("siamese_model.png")