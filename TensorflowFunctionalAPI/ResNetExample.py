from tensorflow import Tensor
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Defining the BatchNormalization layer
def add_BN_layer(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn

# Defining the residual block
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2D(kernel_size=kernel_size,strides= (1 if not downsample else 2),filters=filters,padding="same")(x)
    y = add_BN_layer(y)
    y = layers.Conv2D(kernel_size=kernel_size,strides=1,filters=filters,padding="same")(y)
    y = add_BN_layer(y)
    y = layers.Conv2D(kernel_size=kernel_size,strides=1,filters=filters,padding="same")(y)
    
    if downsample:
        x = layers.Conv2D(kernel_size=1,strides=2,filters=filters,padding="same")(x)
    
    out = layers.Add()([x, y])
    out = add_BN_layer(out)
    return out

# Defining the entire RES NET Model 
def create_res_net_architecture():
    
    inputs = layers.Input(shape=(32, 32, 3))
    num_filters = 64
    
    res = layers.BatchNormalization()(inputs)
    res = layers.Conv2D(kernel_size=3,strides=1,filters=num_filters,padding="same")(res)
    res = add_BN_layer(res)
    
    for j in range(2):
        res = residual_block(res, downsample=False, filters=num_filters)
    for j in range(5):
        res = residual_block(res, downsample=(j==0), filters=num_filters*2)
    for j in range(5):
        res = residual_block(res, downsample=(j==0), filters=num_filters*4)
    for j in range(2):
        res = residual_block(res, downsample=(j==0), filters=num_filters*8)
    
    res = layers.AveragePooling2D(4)(res)
    res = layers.Flatten()(res)
    outputs = layers.Dense(10, activation='softmax')(res)
    
    model = Model(inputs, outputs)
    
    return model

model = create_res_net_architecture()

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
