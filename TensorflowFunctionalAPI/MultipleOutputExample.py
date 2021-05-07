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
import cv2

# Defining the Input layer
input_layer = layers.Input(shape=(28, 28, 1), name="Base_input")

# first_dense = Dense(units='128', activation='relu')(input_layer)
# second_dense = Dense(units='128', activation='relu')(first_dense)

# Define the Hidden Layers:
x = layers.Conv2D(16, 3, activation="relu", name="Base_Conv1")(input_layer)
x = layers.Conv2D(32, 3, activation="relu", name="Base_Conv2")(x)
x = layers.MaxPooling2D(3, name="MaxPool")(x)
x = layers.Conv2D(32, 3, activation="relu", name="Base_Conv3")(x)
x = layers.Conv2D(16, 3, activation="relu", name="Base_Conv4")(x)

# We split the data pipeline here to get 2 outputs

# The First Pipeline output
out_1 = layers.GlobalMaxPooling2D()(x)
out_1 = layers.Flatten(name="flatten_input")(out_1)
out_1 = layers.Dense(16, activation='relu', name="Output_1_Dense1")(out_1)
out_1 = layers.Dense(10, activation='relu', name="Output_1_Dense1")(out_1)

# The Second Pipeline output
out_2 = layers.MaxPooling2D(3)(x)
out_2 = layers.Conv2D(16, 3, activation="relu", name="Output_2_Conv2")(out_2)
out_2 = layers.Conv2D(32, 3, activation="relu", name="Output_2_Conv1")(out_2)
out_2 = layers.MaxPooling2D(3)(out_2)
out_2 = layers.Conv2D(32, 3, activation="relu", name="Output_2_Conv3")(out_2)
out_2 = layers.Conv2D(16, 3, activation="relu", name="Output_2_Conv4")(out_2)
out_2 = layers.GlobalMaxPooling2D(name="MaxPool")(x)
out_2 = layers.Dense(16, activation='relu', name="Output_2_Dense1")(out_2)
out_2 = layers.Flatten(name="flatten_input")(out_2)
out_2 = layers.Dense(10, activation='relu', name="Output_2_Dense2")(out_2)


# Defining the model with the input layer and a list of the output layers.
model = keras.Model(inputs=input_layer, outputs=[out_1, out_2])

# Getting the Model summary:
model.summary()

# Display the Model in a Block Diagram  
keras.utils.plot_model(model, "multi_output.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("multi_output.png")