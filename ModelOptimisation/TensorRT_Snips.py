# Installing TensorFlow-GPU 2.0 and TensorRT Runtime
"""
%%bash
wget -q https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

dpkg -i nvidia-machine-learning-repo-*.deb
apt-get -qq update

sudo apt-get -qq install libnvinfer5 #libnvinfer6=6.0.1-1+cuda10.1

pip install -q tensorflow-gpu==2.0.0
"""

# Import the Libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# Loading the model 
model = InceptionV3(weights='imagenet')

# Predction Fucntion
def show_predictions(model,num_images=4):
    for i in range(num_images):
        img_path = './data/img%d.JPG'%i
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

        plt.subplot(num_images//2,2,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(decode_predictions(preds, top=3)[0][0][1])

show_predictions(model)

# Save the entire model as a TensorFlow SavedModel.
tf.saved_model.save(model, 'inceptionv3_saved_model')

# Returns Batchs for Input  
def batch_input(batch_size=8):
    batched_input = np.zeros((batch_size,299,299,3),dtype=np.float32)
    for i in range(batch_size):
        img_path = './data/img%d.JPG' % (i % 4)
        img = image.load_img(img_path, target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, : ] = x

    return tf.constant(batched_input)

batched_input = batch_input(batch_size=32)

# Load TensorFlow SavedModel
def load_tf_saved_model(input_saved_model_dir):
    print(f'Loading from {input_saved_model_dir}')
    return tf.saved_model.load(input_saved_model_dir, tags = [tag_constants.SERVING])

# Here we load a previously saved InceptionV3 model.
saved_model = load_tf_saved_model('inceptionv3_saved_model')
infer = saved_model.signatures['serving_default']
print(infer.structured_outputs)

# Function to get Baseline for Prediction Throughput and Accuracy 
def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):

    elapsed_time = []
    all_preds = []
    batch_size = batched_input.shape[0]

    for i in range(N_warmup_run):
        labeling = infer(batched_input)
        preds = labeling['predictions'].numpy()

    for i in range(N_run):
        start_time = time.time()

        labeling = infer(batched_input)

        preds = labeling['predictions'].numpy()

        end_time = time.time()

        elapsed_time = np.append(elapsed_time, end_time - start_time)

        all_preds.append(preds)

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    return all_preds

all_preds = predict_and_benchmark_throughput(batched_input,infer)

# Displays one prediction
def show_predictions(model):

    img_path = './data/img0.JPG'  # golden_retriever
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)

    labeling = model(x)
    preds = labeling['predictions'].numpy()

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
    plt.subplot(2,2,1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(decode_predictions(preds, top=3)[0][0][1])