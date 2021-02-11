#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare explanations. Use this script to produce Table 1 and 2 from the paper.
"""

import numpy as np

import pickle

from utils.aux_functions import get_top_explanations
from utils.aux_functions import compute_jaccard


# 
np.random.seed(0)

#########################################
# change here the model and the dataset
# data_set = cifar10 or ILSVRC
data_set = "ILSVRC"
model_name = "InceptionV3"
#########################################

result_path = "results/" + data_set + "/" + model_name + "/"
emp_path = result_path + "empirical/"
approx_path = result_path + "approx/"

# looking at the first n_images of the dataset
n_images = 10

# main loop
jaccard_5_store  = np.zeros((n_images,))
jaccard_10_store = np.zeros((n_images,))
for id_image in range(n_images):
    
    # get the explanations
    image_name =  str(id_image+1).zfill(8) + ".pkl"
    emp_pickle = emp_path + image_name
    approx_pickle = approx_path + image_name
    with open(emp_pickle,'rb') as f:
        empirical = pickle.load(f)
    coefs_emp = np.mean(empirical["explanations"],0)
    with open(approx_pickle,'rb') as f:
        approx = pickle.load(f)
    coefs_approx = approx["explanations"]
    
    # look at the top positive coefficients
    top_10_emp    = get_top_explanations(coefs_emp,top=10)
    top_5_emp     = top_10_emp[-5:]
    top_10_approx = get_top_explanations(coefs_approx,top=10)
    top_5_approx  = top_10_approx[-5:]

    # compute the Jaccard index
    j5  = compute_jaccard(top_5_emp,top_5_approx)
    j10 = compute_jaccard(top_10_emp,top_10_approx)
#
#    print(j5)
#    print(j10)

    jaccard_5_store[id_image] = j5
    jaccard_10_store[id_image] = j10


print()
print(np.mean(jaccard_5_store))
print(np.mean(jaccard_10_store))




