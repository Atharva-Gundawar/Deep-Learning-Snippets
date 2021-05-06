# Imports 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Defining the Input layer
inputs = keras.Input(shape=(32, 32, 3) , name="img")

# Define the First Hidden Layers:
x = layers.Dense(255, activation="relu")(inputs) 
# Notice how the last layer is passed as a parameter to the first layer

# Adding the rest of the Hidden Layers:
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)

# Adding the Last Layer:
outputs = layers.Dense(10)(x)

# Creating the Model:
model = keras.Model(inputs=inputs, outputs=outputs, name="model_name")

# Getting the Model summary:
model.summary()

# Display the Model in a Block Diagram  
keras.utils.plot_model(model, "model_image.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("model_image.png")

