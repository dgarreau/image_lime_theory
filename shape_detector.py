# -*- coding: utf-8 -*-
"""
In this experiment, we see how our theoretical predictions fare in practice 
for a basic shape detector. The result is Figure 3 of the paper. 
"""

import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt

from skimage.color import gray2rgb, rgb2gray

from sklearn.datasets import fetch_openml

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from tqdm import tqdm

from utils.theory import compute_beta_basic_shape

from utils.aux_functions import format_coefs
from utils.aux_functions import hide_color_helper
from utils.aux_functions import mkdir

from utils.plot_functions import plot_whisker_boxes
from utils.plot_functions import plot_image_segmentation

# for reproducibility 
np.random.seed(0)

# we start by loading the MNIST dataset
print("fetching MNIST...")
mnist = fetch_openml('mnist_784', version=1, cache=True)
print("done!")
print()

# let us choose an example to explain
samp = 4
xi_vec = np.uint8(mnist.data[samp])
xi = gray2rgb(xi_vec).reshape(28,28,3)

# create the LIME explainer
explainer = lime_image.LimeImageExplainer(verbose=False)

# the segmentations function
# Note that kernel_size=4 is the default LIME setting
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

# get the image segmentation
print("segmenting the image...")
segments = segmenter(xi)

# number of superpixels
d = np.unique(segments).shape[0]
print("Quickshift number of segments: %d" % d)
print("done!")
print()

# number of classes (10 different digits)
n_classes = 10

# let us create the support of the shape
case = 2
if case == 0:
    # upper left corner
    indices = 87 + 28*np.arange(5)
elif case == 1:
    # lower middle
    indices = 547 + 28*np.arange(5)
elif case == 2:
    # upper middle
    indices = 463 + 28*np.arange(5)

# define the shape detector
tau = 0.5
def basic_shape_detector(data):
   n_data = len(data)
   aux = np.zeros((n_data,1))
   
   for i in range(n_data):
       x_gray = rgb2gray(data[i]).ravel()
       if np.all(x_gray[indices] > tau):
           aux[i] = 1.0

   # LIME wants n_classes outputs
   res = np.matlib.repmat(aux,1,n_classes)
   return res

# number of experiments
n_exp = 5

# compute values of each coefficient after n_exp experiences
betahat_store = np.zeros((n_exp,d+1))


print("entering main loop...")

# number of new examples created by LIME
n_examples = 1000

# hide_color = None gives mean replacement
hide_color = 0
rgb_hide_color = hide_color_helper(hide_color)
    
# entering the main loop
for i in tqdm(range(0,n_exp)):
       
    # explanation
    explanation = explainer.explain_instance(xi,
                                                 classifier_fn=basic_shape_detector,
                                                 top_labels=10,
                                                 hide_color=rgb_hide_color,
                                                 num_samples=n_examples,
                                                 segmentation_fn=segmenter)
    # get the explanations
    betahat_store[i,:] = format_coefs(explanation,0)
    
print("done!")
print()

# compute the theoretical explanations
exp_theo = compute_beta_basic_shape(xi,segments,indices,tau,hide_color=hide_color)

####################################################################

# this folder contains the resulting figure
fig_dir = "results/figures/"
mkdir(fig_dir)

# graphical parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# plot the examples and superpixels
fig, axis = plt.subplots(1,2,figsize=(10,5))
digit = mnist.target[samp]
title_str = "Digit: {}".format(digit)

# left panel: digit + segmentation
plot_image_segmentation(axis[0],xi,segments=segments,
                        indices=indices,
                        title=title_str,
                        method="nearest",
                        out_size=299)

# right panel: empirical results + theory on top
plot_whisker_boxes(betahat_store,
                   axis[1],
                   title="Interpretable coefficients",
                   xlabel="superpixels",
                   theo=exp_theo,
                   rotate=False,
                   ylims=[-0.1,1.1],
                   feature_names=np.linspace(1,d,d,dtype=int),
                   color="red")

# save figure
s_name = fig_dir + "shape_detector_" + str(digit) + "_" + str(case)
plt.savefig(s_name + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=100)


