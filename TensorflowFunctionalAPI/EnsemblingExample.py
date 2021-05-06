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

def get_model():
    inputs = keras.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(16, 3, activation="relu")(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    outputs = layers.GlobalMaxPooling2D()(x)
    return keras.Model(inputs, outputs)


convModel1 = get_model()
convModel2 = get_model()
convModel3 = get_model()

inputs = keras.Input(shape=(28, 28, 1), name="img")
y1 = convModel1(inputs)
y2 = convModel2(inputs)
y3 = convModel3(inputs)

outputs = layers.average([y1, y2, y3])

# Creating the Model:
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

# Getting the Model summary:
ensemble_model.summary()

# Display the Model in a Block Diagram  
keras.utils.plot_model(ensemble_model, "model_image.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("model_image.png")