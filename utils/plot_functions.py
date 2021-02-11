#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file we collect all plotting utilities.
"""

import numpy as np

import tensorflow as tf

from copy import  deepcopy

from matplotlib.ticker import FormatStrFormatter

from skimage.segmentation import mark_boundaries

from utils.aux_functions import get_top_explanations

# linewidth
lw = 5

# small font 
small_fs = 15

#big font 
big_fs = 25

def renormalize(image):
    """
    Renormalizes an image to plot it (some models work with [-1,1] convention, 
    imshow wants [0,1] or 0..255).
    
    INPUT:
        - image: some image
        
    OUTPUT:
        - renormalized image
    """
    # copy of the image
    img_copy = deepcopy(image)
    
    # converting to rgb if some other convention
    input_range = np.max(img_copy) - np.min(img_copy)
    if input_range <= 1.0 and input_range >= 0.0:
        img_copy = 255.0 * img_copy
    elif input_range <= 2.0 and input_range > 1.0:
        img_copy = 127.5*(img_copy + np.ones(img_copy.shape))

    img_copy = np.uint8(img_copy)

    return img_copy

def plot_whisker_boxes(my_data,
                       axis,
                       title=None,
                       xlabel=None,
                       theo=None,
                       rotate=False,
                       feature_names=None,
                       ylims=None,
                       color="red",
                       c1='black',
                       alpha=1,
                       c2="blue",
                       label="",
                       c3="black"):
    """
    Plots whisker boxes for interpretable coefficients.
    
    INPUT:
        - my_data: raw explanations (size (n_exp,dim+1))
        - axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
        - title: title of the figure (str)
        - xlabel: label for the x axis (str)
        - theo: theoretical values marked by crosses on the plot (size (dim+1,))
        - rotate: classical view if True (bool)
        - feature_names: default is 1,2,...
        - ylims: providing ylims if needed
        - color: color of the crosses
        - c1: color of the box 
        - alpha: transparency
        - c2: color of the median 
        - label: label 
        - c3: color of the fliers
        
"""
    
    # get the dimension of the data
    dim = my_data.shape[1] -1

    
    # horizontal whiskerboxes
    if rotate:
        axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color=c1,alpha=alpha), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color=c1,alpha=alpha),
                   medianprops=dict(linestyle='-',linewidth=lw,color=c2),
                   flierprops=dict(marker='o',markerfacecolor=c3,linewidth=lw),
                   capprops=dict(linewidth=lw,color=c1,alpha=alpha),
                   vert=False)
        axis.axvline(x=0,c='k',linestyle='--')
        
        
        y_pos = np.arange(dim) + 1
        axis.set_yticks(y_pos)

        if feature_names is None:
            feature_names = np.arange(1,dim+1)
            
        if feature_names is not None:
            print(feature_names)
        
        axis.set_yticklabels(feature_names)
        axis.invert_yaxis()
        axis.tick_params(labelsize=small_fs)
        
    # vertical whisker boxes
    else:
        
        # not plotting the intercept
        bp=axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color=c1,alpha=alpha), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color=c1,alpha=alpha),
                   medianprops=dict(linestyle='-',linewidth=lw,color=c2,alpha=alpha),
                   flierprops=dict(marker='o',markerfacecolor=c3,linewidth=lw),
                   capprops=dict(linewidth=lw,color=c1,alpha=alpha))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    
        # plotting horizontal line to denote 0
        axis.axhline(y=0,c='k',linestyle='--')
        
    
        # plotting the theoretical predictions if any
        if theo is not None:

            for i_feature in range(dim):
                axis.plot(i_feature+1,
                        theo[i_feature+1],
                        'x',
                        markersize=8,
                        markeredgewidth=3,
                        zorder=10,
                        color=color)
        
            if ylims is None:
                ymin = min(np.min(my_data[1:]),np.min(theo[1:]))
                ymax = max(np.max(my_data[1:]),np.max(theo[1:]))
                axis.set_ylim([ymin,ymax])
            else:
                axis.set_ylim(ylims)
        
        else:
            if ylims is not None:
                axis.set_ylim(ylims)
        
        # setting the labels
        if xlabel is None:
            axis.set_xlabel("superpixels",fontsize=small_fs)
        else:
             axis.set_xlabel(xlabel,fontsize=small_fs)
        
        # setting xticks and yticks
        if feature_names is None:
            feature_names = np.arange(1,dim+1)
            
            
        axis.set_xticklabels(feature_names, rotation=0, fontsize=small_fs)
        axis.tick_params(labelsize=small_fs)
    
    # setting the title
    if title is None:
        title = "Coefficients of the surrogate model"
    axis.set_title(title,fontsize=big_fs)
    return bp

def plot_image_segmentation(axis,
                            image,
                            out_size=100,
                            segments=None,
                            indices=None,
                            title=None,
                            show_seg_ids=True,
                            method="bilinear"):
    """
    This functions plots the superpixels over an image.
    
    INPUT:
        - axis: the axis on which to plot
        - image: the image to plot
        - out_size: the size of the resulting image
        - segments: the superpixels ids
        - indices: indices of an additional shape to overlay
        - title: title of the figure
        - show_seg_ids: if True, display the superpixel ids in red
        - method: interpolation method to plot the image ('bilinear' is smooth, 
        but 'nearest" is best when the image is stretched)
    
    """
    
    image_copy = renormalize(image)
    
    # size of the image
    height = image_copy.shape[0]
    width  = image_copy.shape[1]
    
    if indices is not None:
        image_copy[indices//width,np.mod(indices,width)] = (0,0,255)
    

    # get the segments ids
    if segments is not None:
        segments_ids = np.unique(segments)
    
    # resize the image
    x = np.linspace(0,width, out_size)
    y = np.linspace(0,height, out_size)
    X, Y = np.meshgrid(x[:-1],y[:-1])
    
    if segments is not None:
        f1 = lambda x,y: segments[int(y),int(x)]
        g1 = np.vectorize(f1)
        Z1 = g1(X[:-1],Y[:-1])
        axis.imshow(Z1)    

    
    resized_image = np.uint8(tf.image.resize(image_copy,size=(out_size-2,out_size-1),method=method))
    

    # if superpixels are provided, plot them
    if segments is not None:
        centers = np.array([np.mean(np.nonzero(Z1==i),axis=1) for i in segments_ids])
        
        if show_seg_ids:
            axis.scatter(centers[:,1],centers[:,0], c='red',s=10)
            for i, txt in enumerate(segments_ids):
                axis.annotate(txt+1, (centers[i,1]+4, centers[i,0]),color='red')
        
        #print(np.max(resized_image),np.min(resized_image))
        l = mark_boundaries(resized_image,Z1,color=(255, 255, 0),mode='thick')
        #l = mark_boundaries(A,Z1,color=(255, 255, 0),mode='thick')

    else:
        l = image_copy
    
    # remove the axes
    axis.axis('off')
    
    # plot the final result
    axis.imshow(l,cmap='gray',vmin=0,vmax=255)
    
    # title if provided
    if title is not None:
        axis.set_title(title,fontsize=big_fs)
    
    return 0

def plot_explanation(axis,
                     image,
                     segments,
                     coefs,
                     out_size=100,
                     positive_only=True,
                     mark_boundaries=True):
    """
    Plot explanations on top of an image. 
    """
    
    height,width,_ = image.shape
    
    image_copy = renormalize(image)
    
    resized_image = np.uint8(tf.image.resize(image_copy,size=(out_size-2,out_size-1)))
    
    x = np.linspace(0,width, out_size)
    y = np.linspace(0,height, out_size)
    X, Y = np.meshgrid(x[:-1],y[:-1])
    convert = lambda x,y: segments[int(y),int(x)]
    vec_convert = np.vectorize(convert)
    resized_segments = vec_convert(X[:-1],Y[:-1])
    
    # get top explanations
    pos_ids = get_top_explanations(coefs)


    for idx in pos_ids:
        resized_image[resized_segments == idx] = (0,128,0)
    
    axis.imshow(resized_image)
    
    axis.axis('off')
    
    return 0
