# This code is based on : https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Defining the Encoder
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu" , name="Encoder_Conv1")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu" , name="Encoder_Conv2")(x)
x = layers.MaxPooling2D(3 , name="Encoder_MaxPool")(x)
x = layers.Conv2D(32, 3, activation="relu" , name="Encoder_Conv3")(x)
x = layers.Conv2D(16, 3, activation="relu" , name="Encoder_Conv4")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

# Creating Encoder model
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

# Defining the Decoder
decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu" , name="Decoder_ConvT1")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu" , name="Decoder_ConvT2")(x)
x = layers.UpSampling2D(3 , name="Decoder_UPS")(x)
x = layers.Conv2DTranspose(16, 3, activation="relu" , name="Decoder_ConvT3")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu" , name="Decoder_ConvT4")(x)

# Creating Decoder Model
decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

"""
In the code below, autoencoder_input is the input layer which is passed to the
encoder model which outputs a encoded_img.
The encoded_img is passed to the decoder which outputs the decoded_img which is also 
the autoencoders output
"""
autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()


# Display the Model in a Block Diagram  
keras.utils.plot_model(autoencoder, "autoencoder.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("autoencoder.png")