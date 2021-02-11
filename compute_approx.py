#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute approximate explanations according to Eq. (12) of the paper.
"""



import numpy as np
import sys
import time
import pickle

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from utils.aux_functions import mkdir
from utils.aux_functions import compute_gradients
from utils.aux_functions import interpolate_images
from utils.aux_functions import integral_approximation

from utils.theory import compute_xi_mean
from utils.theory import compute_beta_linear

from utils.cifar10 import load_cifar10


# 
np.random.seed(0)

#########################################
# change here the model and the dataset
# data_set = cifar10 or ILSVRC
data_set = "ILSVRC"
model_name = "DenseNet121"
#########################################

# alpha coefs for the interpolation
m_steps = 20
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

if data_set == "cifar10":
    
    # loading the data
    (x_train_norm,y_train), (x_test_norm,y_test) = load_cifar10(verbose=True)
    ks = 1.0
    
    # load the model
    model_path = "models/" + data_set + "/" + model_name + "/" + model_name + ".h5"
    try:
        model = load_model(model_path)
    except OSError:
        sys.exit("Model is not trained yet! Run train_model.py first.")
elif data_set == "ILSVRC":
    
    # change here for the datapath to ILSVRC2017
    data_path = "../data/ILSVRC/Data/DET/test/"
    ks = 4.0
    
    # load the model
    if model_name == "MobileNetV2":
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        input_shape = 224
        preprocess = mobilenet_v2.preprocess_input
    elif model_name == "DenseNet121":
        model = densenet.DenseNet121(weights='imagenet')
        input_shape = 224
        preprocess = densenet.preprocess_input
    elif model_name == "InceptionV3":
        model = inception_v3.InceptionV3(weights='imagenet')
        input_shape = 299
        preprocess = inception_v3.preprocess_input

# replacement (None -> mean replacement otherwise color)
hide_color = None

# results from the experiment will be saved here
result_path = "results/" + data_set + "/" + model_name + "/approx/"
mkdir(result_path)

# segmentation algorithm (default kernel_size = 4, max_dist = 200, ratio = 0.2)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=ks, max_dist=200, ratio=0.2)

# looking at the first n_images of the dataset
n_images = 40

# LIME explainer
explainer = lime_image.LimeImageExplainer(verbose=False)

# main loop
for id_image in range(n_images):
    
    print("computing explanations for image {} out of {}".format(id_image+1,n_images))
    
    approx = {}
    
    t_start = time.time()
    
    # get the image
    if data_set == "cifar10":
        xi_norm = x_test_norm[id_image]
    elif data_set == "ILSVRC":
        # path to the image
        image_name = 'ILSVRC2017_test_' + str(id_image+1).zfill(8) 
        image_path = data_path + image_name + '.JPEG'
    
        # loading and converting the image
        xi_orig = load_img(image_path,target_size=(input_shape, input_shape))
        xi_rgb  = np.uint8(img_to_array(xi_orig))
        xi_norm = preprocess(xi_rgb.copy()).astype('double')
    
    
    
    # 
    xi_tensor  = tf.convert_to_tensor(xi_norm,dtype=np.float32)
    
    # get the predictions
    predictions = model.predict(tf.expand_dims(xi_tensor,0))
    pred_class = np.argmax(predictions)
    
    # get the segments
    segments = segmenter(xi_norm)
    d = np.unique(segments).shape[0]
    
    # get the replacement image
    _,xi_mean = compute_xi_mean(xi_norm,segments,hide_color)

    xi_mean_tensor = tf.convert_to_tensor(xi_mean,dtype=np.float32)
    
    # path from xibar to xi
    interpolated_images = interpolate_images(baseline=xi_tensor,image=xi_mean_tensor,alphas=alphas)
    
    # compute gradients along the path
    path_gradients = compute_gradients(model,images=interpolated_images,target_class_idx=pred_class)

    # average the gradients and save the result
    est_grad_tf = integral_approximation(gradients=path_gradients)
    approx["lambda"] = est_grad_tf

    t_end = time.time() 


    approx["explanations"] = compute_beta_linear(xi_norm,segments,est_grad_tf,hide_color=hide_color)


    print("done!")
    print("elapsed: {}".format(t_end-t_start))
    print()
    

    pickle_name = result_path + str(id_image+1).zfill(8) + '.pkl'

    print("saving results...")
    with open(pickle_name,'wb') as f:
        pickle.dump(approx,f)

    t_end = time.time()
    
    print("elapsed: {}s".format(np.round(t_end-t_start,2)))
    print()
