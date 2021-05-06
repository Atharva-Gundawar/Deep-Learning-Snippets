# This code is based on : https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

"""
In the code below, get_model function returns a convolution model.
model1 , model2 , model3 creates 3 instances of the same model architecture.
Using the layers.average function the ensemble_model ensemble the set of models into a single model that averages their predictions.
"""

inputs = keras.Input(shape=(32, 32, 3) , name="img")

# Define the First Hidden Layers:
x = layers.Dense(255, activation="relu")(inputs) 

# Creating First Data Path way:
x1 = layers.Dense(32, activation="relu")(x)
x1 = layers.Dense(16, activation="relu")(x1)

# Creating Second Data Path way:
x2 = layers.Dense(128, activation="relu")(x)
x2 = layers.Dense(64, activation="relu")(x2)
x2 = layers.Dense(16, activation="relu")(x2)

# Merging the Data Path ways:
added = layers.Add()([x1, x2])

# Adding the Last Layer:
outputs = layers.Dense(10)(added)

# Creating the Model:
model = keras.Model(inputs=inputs, outputs=outputs, name="model_name")

# Getting the Model summary:
model.summary()

# Display the Model in a Block Diagram  
keras.utils.plot_model(model, "model_image.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("model_image.png")