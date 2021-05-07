# Imports 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Defining the Input layer
inputs = keras.Input(shape=(28, 28, 1), name="img")

# Define the Hidden Layers:
x = layers.Conv2D(16, 3, activation="relu" , name="Conv1")(inputs)
x = layers.Conv2D(32, 3, activation="relu" , name="Conv2")(x)
x = layers.MaxPooling2D(3 , name="MaxPool")(x)
x = layers.Conv2D(32, 3, activation="relu" , name="Conv3")(x)
x = layers.Conv2D(16, 3, activation="relu" , name="Conv4")(x)

# Adding the Last Layer:
outputs = layers.GlobalMaxPooling2D()(x)

# Creating the Model:
model = keras.Model(inputs=inputs, outputs=outputs, name="convolution_example")

# Getting the Model summary:
model.summary()

# Display the Model in a Block Diagram  
keras.utils.plot_model(model, "conv_mnist.png") # Include 'show_shapes=True' as a parameter to display th shapes of the respective layers.
cv2.imshow("conv_mnist.png")

# Spliting The Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizing the Values
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

# Compiling the Model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

# Training the Model 
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

# Evaluating the model
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])