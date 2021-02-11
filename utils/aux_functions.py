#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file we collect all the auxilliary functions.
"""

import pandas as pd
import tensorflow as tf
import numpy as np

from errno import EEXIST

from os import makedirs
from os import path

def is_rgb(xi):
    """ 
    Checks if xi is in rgb format or not.
    
    INPUT:
        - xi: image to test
        
    OUTPUT:
        True if xi has 2 or 3 dims, False otherwise
    """
    if len(xi.shape) == 3 and xi.shape[2] == 3:
        return True
    else:
        return False

def hide_color_helper(hide_color,normalized=False):
    """
    
    This function transforms the hide color given by the user. If none is 
    provided, then the default scheme is used (mean imputation). Otherwise, 
    this function just translates the [0,1] value to a RGB color.
    
    INPUT:
        - hide_color: value between 0 and 1
        - normalized: if True, the output is normalized between 0 and 1
    
    OUTPUT:
        - rgb color that LIME can use as a replacement
    
    """
    if hide_color is None:
        rgb_hide_color = None
    else:
        if normalized:
            return (hide_color,hide_color,hide_color)
        else:
            uint8_hide_color = np.uint8(hide_color * 255)
            rgb_hide_color = (uint8_hide_color,uint8_hide_color,uint8_hide_color)
    return rgb_hide_color

def format_coefs(explanation,ind):
    """
    This function formats the coefficients of an explanation given by LIME.
    
    INPUT:
        - explanation: output of LIME algorithm
        - ind: index of the class for which we want the explanations
        
    OUTPUT:
        coef_list: list of the interpretable coefficient, first term is the 
        intercept
    
    """
    coefs = explanation.local_exp[ind]
    intercept = explanation.intercept[ind]
    df = pd.DataFrame(coefs)
    df = df.sort_values(by=0, ascending=True)
    coef_list = [intercept] + list(df[1].values)
    
    return coef_list


def mkdir(mypath):
    """
    Creates a directory (credits to https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory)
    
    INPUT:
        - mypath: str with path to the directory    
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: 
            raise

def compute_gradients(model,images, target_class_idx):
    """
    Computes gradients from the model, assuming that it is a tensorflow model.
    
    INPUT:
        - model: tensorflow.python.keras.engine.training.Model object
        - images: a tensor containing the images
        - target_class_idx: index of the target class
        
    OUTPUT:
        gradients
    """
    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)[:,target_class_idx]
    return tape.gradient(pred,images)

def interpolate_images(baseline, image, alphas):
    """
    Computes a linear path between baseline and image, that is,
        (1-\alpha)*baseline + \alpha*image
    
    INPUT:
        - basline: a baseline image
        - image: some image (same size)
        - alphas: an array of coefficients between 0 and 1
    
    OUTPUT:
        linear path between baseline and image
    """
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

def integral_approximation(gradients):
    """
    Approximating the integral of the gradients between a baseline and an 
    image (Eq. (10) of the paper).
    
    INPUT:
        - gradients: an array of gradients
        
    OUTPUT:
        averaged gradient between the two images
    """
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def get_top_explanations(coefs,top=5):
    """
    Get the top five (by default) explanations. Result can have less than 5 
    elements if not enough positive coefficients.
    """
    ids = np.argsort(coefs[1:])
    aux = np.sort(coefs[1:])
    n_pos = np.sum(aux > 0)
    if n_pos >= top:
        pos_ids = ids[-top:]
    elif n_pos < top and n_pos > 0:
        pos_ids = ids[-n_pos:]
    else:
        pos_ids = []
    return pos_ids

def compute_jaccard(array1, array2):
    """
    Compute the Jaccard index of two arrays (card of intersection divided by 
    card of the union). Return one if empty sets.
    """
    card_union = np.union1d(array1,array2).shape[0]
    if card_union == 0:
        return 1
    else:
        return np.intersect1d(array1,array2).shape[0] / card_union
