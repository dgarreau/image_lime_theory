#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file we collect all theoretical computations.
"""

import numpy as np

from skimage.color import rgb2gray

from utils.aux_functions import is_rgb

from scipy.special import binom

def compute_psi(t,nu=0.25):
    """
    Computes the psi function defined by Eq. (4) in the paper.
    
    INPUT:
        - t: a real number in [0,1]
        - nu: a bandwidth parameter
    OUTPUT:
        \psi(t)
    
    """
    return np.exp(-np.square(1.0-np.sqrt(1.0 - t))/(2*nu**2))

def compute_alpha(d,nu,p):
    """
    Computes the alpha coefficients according to Proposition 1 of the paper.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        - p: number of distinct superpixels
        
    OUTPUT:
        alpha_p
    """
    s_values = np.arange(0,d+1)
    psi_values = compute_psi(s_values/d,nu)
    bin_values = binom(d-p,s_values)
    return np.dot(bin_values,psi_values) / 2**d

def compute_sigma_0(d,nu=0.25):
    """
    Computes \sigma_0 as in Proposition 2 of the supplementary.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
    
    OUTPUT:
        \sigma_0
    
    """
    return (d-1)*compute_alpha(d,nu,2) + compute_alpha(d,nu,1)

def compute_sigma_1(d,nu):
    """
    Compute \sigma_1 as in Definition 2 of the paper.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        
    OUTPUT:
        \sigma_1
    """
    return -compute_alpha(d,nu,1)

def compute_sigma_2(d,nu=0.25):
    """
    Compute \sigma_2 as in Definition 2 of the paper.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        
    OUTPUT:
        \sigma_2
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return ((d-2)*alpha_0*alpha_2 - (d-1)*alpha_1**2 + alpha_0*alpha_1) / (alpha_1 - alpha_2)

def compute_sigma_3(d,nu=0.25):
    """
    Compute \sigma_3 as in Definition 2 of the paper.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        
    OUTPUT:
        \sigma_3
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return (alpha_1**2 - alpha_0*alpha_2) / (alpha_1 - alpha_2)

def compute_dencst(d,nu=0.25):
    """
    Compute c_d as in Definition 2 of the paper.
    
    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        
    OUTPUT:
        \sigma_1
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return (d-1)*alpha_0*alpha_2 - d*alpha_1**2 + alpha_0*alpha_1


def compute_xi_mean(xi,segments,hide_color=None):
    """
    This function computes the replacement image used by LIME. If a replacement 
    color is provided, then the output is an array of the same size filled with 
    this color. Otherwise, the mean color on each superpixel is computed (see 
    Eq. (1) in the paper).
    
    INPUT:
        - xi: rgb image
        - segments: segments ids (same size as xi)
        - hide_color: replacement color
        
    OUTPUT:
        - means: the mean color on each superpixel (size d)
        - xi_mean: the replacement image (same size as xi)
    """
    
    # number of superpixels
    d = np.unique(segments).shape[0]
    
    superpixels_ids = np.ravel(segments)
    
    if is_rgb(xi):
        
        # if a replacement color is provided, fill the new image with this color
        if hide_color is not None:
            means   = hide_color*np.ones((d,3))
            xi_mean = hide_color*np.ones(xi.shape)
            
        # otherwise compute the mean color on eqch superpixel
        else:
            means   = np.zeros((d,3))
            xi_mean = np.zeros(xi.shape)
            for j in range(d):
                means[j] = np.mean(xi[segments == j],axis=0)
                xi_mean[segments == j] = means[j]
    
    # vector version
    else:
        D = xi.shape[0]
        
        if hide_color is not None:
            means   = hide_color*np.ones((d,))
            xi_mean = hide_color*np.ones((D,))
        else:
            means = np.zeros((d,))
            xi_mean = np.zeros((D,))
            for j in range(d):
                means[j] = np.mean(xi[superpixels_ids == j])
                xi_mean[superpixels_ids == j] = np.mean(xi[superpixels_ids == j])

    return means,xi_mean

def compute_beta_basic_shape(xi,
                             segments,
                             indices,
                             tau,
                             nu=0.25,
                             hide_color=None):
    """
    
    This functions computes the limit beta coefficients for a basic shape 
    detector (see Proposition 3 of the paper).
    
    INPUT:
        xi: example to explain (RGB image, size h x w x 3)
        segments: superpixels (size h x w)
        indices: list of indices describing the shape
        tau: threshold in |0,1]
        nu: bandwidth
        hide_color: hide_color parameter of LIME
        
    OUTPUT:
        - betatheo: theoretical explanations (size d+1)
        
    """
    # number of superpixels
    d = np.unique(segments).shape[0]
    
    # get the vector version of xi
    xi_vec = rgb2gray(xi).ravel()
    
    # under assumption (8), all pixels have to be lit up
    betatheo = np.zeros((d+1,))
    if np.all(xi_vec[indices] > tau):
        
        # get the set of superpixels containing the shape
        width = xi.shape[0]
        E_set = np.unique(segments[indices//width,indices%width])
        
        # E_minus = superpixels such that the average is < tau
        means,_ = compute_xi_mean(xi_vec,segments,hide_color)
        E_minus = E_set[means[E_set] <= tau]
        p = E_minus.shape[0]
        
        # compute the alpha and sigma coefficients
        alpha_p  = compute_alpha(d,nu,p)
        alpha_pp = compute_alpha(d,nu,p+1)
        dencst   = compute_dencst(d,nu)
        sigma_0  = compute_sigma_0(d,nu)
        sigma_1  = compute_sigma_1(d,nu)
        sigma_2  = compute_sigma_2(d,nu)
        sigma_3  = compute_sigma_3(d,nu)
        
        # apply Proposition 3
        for j in range(d):
            if np.isin(j,E_minus):
                betatheo[j+1] = (sigma_1*alpha_p + sigma_2*alpha_p + (p-1)*sigma_3*alpha_p + (d-p)*sigma_3*alpha_pp) / dencst
            else:
                betatheo[j+1] = (sigma_1*alpha_p + sigma_2*alpha_pp + p*sigma_3*alpha_p + (d-p-1)*sigma_3*alpha_pp) / dencst
        betatheo[0] = (sigma_0*alpha_p + p*sigma_1*alpha_p + (d-p)*sigma_1*alpha_pp) / dencst
        
    return betatheo

def compute_beta_linear(xi,
                        segments,
                        coefs,
                        hide_color=None):
    """
    This function computes the limiting explanations for a linear model (this 
    is Proposition 12 in the supplementary).
    
    INPUT:
        - xi: example image
        - segments: superpixels ids
        - coefs: coefficients of the linear model (\lambda in Eq. (9))
        - hide_color: replacement color
        
    OUTPUT:
        beta
    
    """
    
    # number of superpixels
    d = np.unique(segments).shape[0]

    # replacement image
    _,xi_mean = compute_xi_mean(xi,segments,hide_color)

    betatheo = np.zeros((d+1,))
    if is_rgb(xi):

        aux = coefs*(xi - xi_mean)

        for j in range(d):
            betatheo[j+1] = np.sum(aux[segments == j])

        betatheo[0] = np.sum(coefs * xi_mean)
    else:

        # coefficients x normalized input
        aux = coefs*(xi - xi_mean)
    
        # sum over each superpixel
        for j in range(d):
            betatheo[j+1] = np.dot(np.ravel(segments==j).astype(int),aux)
   
        # DG: not sure about the intercept
        betatheo[0] = np.sum(coefs*xi_mean)
    
    return betatheo
