"""
Theory meets practice for linear functions on CIFAR-10.

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

from utils.theory import compute_beta_linear

from utils.aux_functions import format_coefs
from utils.aux_functions import hide_color_helper
from utils.aux_functions import mkdir

from utils.plot_functions import plot_whisker_boxes,plot_image_segmentation



# parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# for reproducibility 
np.random.seed(0)


# import data
print("fetching MNIST...")
mnist = fetch_openml('mnist_784', version=1, cache=True)
print("done!")
print()

# example to explain
samp = 1
xi_orig = mnist.data[samp]
xi_vec = xi_orig/255
xi_rgb = gray2rgb(np.uint8(xi_orig)).reshape(28,28,3)

# dimension of the ambient space
D = xi_vec.shape[0]

# explainer
explainer = lime_image.LimeImageExplainer(verbose=False)

# segmentation algorithm
segmenter = SegmentationAlgorithm('quickshift',
                                  kernel_size=1,
                                  max_dist=200,
                                  ratio=0.2)

# segmentation
print("segmenting the image...")
segments = segmenter(xi_rgb)
# number of superpixels
d = np.unique(segments).shape[0]
print("Quickshift number of segments: %d" % d)
print("done!")

# n_classes
n_classes = 10

# linear function with high coefficients in the lower right corner of the image
coefs = np.zeros((784,))
for i in range(28):
    for j in range(28):
        coefs[i+28*j] = i+j
coefs /= 100.0

# let us add some white noise
coefs += np.random.normal(0,0.1,size=coefs.shape)

# define the model
def linear_function(data):
   n_data = len(data)
   aux = np.zeros((n_data,1))
   
   for i in range(n_data):
       aux[i] = np.dot(rgb2gray(data[i]).ravel(),coefs)

   res = np.matlib.repmat(aux,1,n_classes)
   
   return res

# number of experiments
n_exp = 5

# compute values of each coefficient after n_exp experiences
data_store = np.zeros((n_exp,d+1))

# we are going to make n_exp experiences of LIME
print("entering main loop...")

# number of new examples created by LIME
n_examples = 1000

# hide_color = None gives mean replacement
hide_color = 0
rgb_hide_color = hide_color_helper(hide_color)
    
# entering the main loop
for i in tqdm(range(0,n_exp)):
       
    # explanation
    explanation = explainer.explain_instance(xi_rgb,
                                                 classifier_fn=linear_function,
                                                 top_labels=10,
                                                 hide_color=rgb_hide_color,
                                                 num_samples=n_examples,
                                                 segmentation_fn=segmenter)

    data_store[i,:] = format_coefs(explanation,0)
    
print("done!")
print()


# computing theory
exp_theo = compute_beta_linear(xi_vec,segments,coefs,hide_color=hide_color)


####################################################################

fig_dir = "results/figures/"
mkdir(fig_dir)

# plot the examples and superpixels
fig, axis = plt.subplots(1,3,figsize=(16,5))

# left panel: heatmap of the coefficients
heatmap = axis[0].imshow(coefs.reshape(28,28),cmap='winter')
axis[0].set_title(r"$\lambda$",fontsize=25)

# middle panel: image and segmentation
title_str = "Digit: {}".format(mnist.target[samp])
plot_image_segmentation(axis[1],xi_rgb,segments=segments,
                        title=title_str,
                        method="nearest",out_size=299)

# right panel: empirical vs theory
plot_whisker_boxes(data_store,
                   axis[2],
                   title="Interpretable coefficients",
                   xlabel="superpixels",
                   theo=exp_theo,
                   rotate=False,
                   feature_names=np.linspace(1,d,d,dtype=int),
                   ylims=[-1,13.9],
                   color="red")

# attachthe colorbar
cbar_ax = fig.add_axes([0.36, 0.15, 0.01, 0.7])
fig.colorbar(heatmap,cbar_ax)
   
# save figure
s_name = fig_dir + "linear"
plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)

