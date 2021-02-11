#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a model on CIFAR-10. This script generates the models used in the 
experiments presented in Table 1 of the paper.
"""
import numpy as np

from utils.aux_functions import mkdir

from utils.cifar10 import simple_net
from utils.cifar10 import VGG_block
from utils.cifar10 import VGG_block_2
from utils.cifar10 import VGG_block_3
from utils.cifar10 import load_cifar10

# 
np.random.seed(0)

# load CIFAR10
(x_train_norm,y_train), (x_test_norm,y_test) = load_cifar10(verbose=True)

# set the name of the model
model_name = "Simple"

#
verbose = True

# define the model
if model_name == "Simple":
    model = simple_net(verbose)
elif model_name == "VGG1":
    model = VGG_block(verbose)
elif model_name == "VGG2":
    model = VGG_block_2(verbose)
elif model_name == "VGG3":
    model = VGG_block_3(verbose)
else:
    print("model name not recognized, training a simple model instead.")
    model = VGG_block(verbose=verbose)
    
print("training the model...")
# train the model
history = model.fit(x_train_norm, 
        y_train, 
        epochs=15, 
        batch_size=64, 
        validation_split=.1, 
        verbose=True)
    
print("done!")
print()

# check the accuracy of the model
scores = model.evaluate(x_test_norm, y_test, verbose=verbose)

# save the model and not retraining each time
print("saving the model...")
model_path = "models/cifar10/" + model_name + "/" 
mkdir(model_path)
save_name = model_path + model_name + ".h5"
model.save(save_name)
print("done!")
