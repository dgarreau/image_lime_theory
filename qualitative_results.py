#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this script to generate qualitative results (Figure 6 of the paper).
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import sys

import pickle

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import inception_v3

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from utils.theory import compute_xi_mean

from utils.aux_functions import get_top_explanations
from utils.aux_functions import mkdir

from utils.plot_functions import plot_image_segmentation
from utils.plot_functions import plot_explanation

from utils.cifar10 import load_cifar10


np.random.seed(0)

hide_color = None

id_image = 40

# model of interest
data_set = "ILSVRC"
model_name = "DenseNet121"


# get empirical explanations
result_path = "results/" + data_set + "/" + model_name
emp_pickle = result_path + "/empirical/" + str(id_image).zfill(8) + ".pkl"
with open(emp_pickle,'rb') as f:
    empirical = pickle.load(f)
coefs_emp = np.mean(empirical["explanations"],0)

if data_set == "cifar10":
    
    cifar10_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    
    # loading the data
    (x_train_norm,y_train), (x_test_norm,y_test) = load_cifar10(verbose=True)
    xi_norm = x_test_norm[id_image-1]
    
    # load the model
    model_path = "models/" + data_set + "/" + model_name + "/" + model_name + ".h5"
    try:
        model = load_model(model_path)
    except OSError:
        sys.exit("Model is not trained yet! Run train_model.py first.")
        
    preds = empirical["predictions"]
    predicted_class = empirical["predicted_class"]
    conf = preds[0,predicted_class]
    label = cifar10_labels[predicted_class]
        
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

    # get the image
    data_path = '../data/ILSVRC/Data/DET/test/'
    image_name = 'ILSVRC2017_test_' + str(id_image).zfill(8)
    image_path = data_path + image_name + '.JPEG'
    xi_orig = load_img(image_path,target_size=(input_shape,input_shape))
    xi_rgb  = np.uint8(img_to_array(xi_orig))
    xi_norm = preprocess(xi_rgb.copy()).astype('double')
    #xi_tensor = tf.convert_to_tensor(xi_norm,dtype=np.float32)
    
    # get the name of the label
    preds = empirical["predictions"]
    labels_inception = decode_predictions(preds)
    label = labels_inception[0][0][1]
    conf = labels_inception[0][0][2]

# retrieve the approximated explanation
approx_pickle = result_path + "/approx/" + str(id_image).zfill(8) + ".pkl"
with open(approx_pickle,'rb') as f:
    approx = pickle.load(f)

# get replacement image
segments = empirical["segments"]
pred_class = empirical["predicted_class"]
_,xi_mean = compute_xi_mean(xi_norm,segments,hide_color)
_,xi_mean_rgb = compute_xi_mean(xi_rgb,segments,hide_color)
xi_mean_tensor = tf.convert_to_tensor(xi_mean,dtype=np.float32)


###############################################################################

# graphical parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# res is the resolution
res = 299
fig, axis = plt.subplots(1,6,figsize=(18,3))

# get the top coefficients
top_coef_emp     = get_top_explanations(coefs_emp)
coefs_approx = approx["explanations"]
top_coefs_approx = get_top_explanations(coefs_approx)

# first display the image, the top class and the confidence
axis[0].imshow(xi_rgb)
axis[0].axis('off')
s_title = label + " (conf. " + str(int(100*conf)) + "%)"
axis[0].set_title(s_title)

# show the LIME superpixels
plot_image_segmentation(axis[1],xi_rgb,segments=segments,title="",out_size=res,show_seg_ids=False)
axis[1].set_title("segmentation")

# display the replacement image
plot_image_segmentation(axis[2],xi_mean_rgb,title="",out_size=res)
axis[2].set_title(r"$\overline{\xi}$")

# highlight the top 5 LIME superpixels
d = np.unique(segments).shape[0] 
plot_explanation(axis[3],xi_rgb,segments,coefs_emp,positive_only=True,out_size=res)
axis[3].set_title("LIME")

# show the integrated gradients (rather the norm, otherwise hard to visualize)
est_grad_tf = approx["lambda"]
ig = (xi_norm-xi_mean) * est_grad_tf
max_grad = np.max(ig)
min_grad = np.min(ig)
norm_ig = ig/(max_grad-min_grad)
intensity = np.sqrt(np.sum(np.square(norm_ig),axis=2))
# NOTE: python ignores the color map if rgb is given as input
axis[4].imshow(intensity,cmap=cm.OrRd)
axis[4].axis('off')
axis[4].set_title("int. gradient")

# show the top five coefficients obtained with the approximated explanations
plot_explanation(axis[5],xi_rgb,segments,coefs_approx,positive_only=True,out_size=res)
axis[5].set_title("linear approx.")   

# save the figure
fig_dir = "results/figures/" + data_set + "/" + model_name + "/"
mkdir(fig_dir)
s_name = fig_dir + image_name
plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)


