#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All auxilliary functions for CIFAR-10 experiments.
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def load_cifar10(verbose=False):
    """
    This functions loads and format the CIFAR10 data. If error 500 occurs, do 
    the following:
        - download the data manually at https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        - rename the file cifar-10-batches-py.tar.gz
        - move it to .keras/datasets/
        
    INTPUT:
        - verbose: print additional information if True
        
    OUTPUT:
        train / test split 
    """
    
    # loading the CIFAR10 data
    if verbose:
        print("loading CIFAR10...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if verbose:
        print("done!")
        print()
    
    # transforming labels into one-hot vectors
    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)
    
    # normalize the data
    # quickshift algorithm wants double precision
    x_train_norm = x_train.astype('double')
    x_test_norm  = x_test.astype('double')

    # normalizing everything between 0 and 1
    x_train_norm = x_train_norm / 255.0
    x_test_norm  = x_test_norm / 255.0
    
    return (x_train_norm,y_train),(x_test_norm,y_test)

def simple_net(verbose=False):
    """
    One-hidden-layer neural network with 32 units in the hidden layer and 
    sigmoid activation. Corresponds to the '1-layer' model in Table 1 of the 
    paper.
    """
    if verbose:
        print("initializing the network...")
    
    # define the model
    model = Sequential()
    model.add(Dense(32,activation='sigmoid',input_shape=(32,32,3)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    # compile the model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if verbose:
        print("done!")
    
    return model

def VGG_block(verbose=False):
    """
    Elementary VGG block.
    """
    
    if verbose:
        print("initializing the network...")
    
    # define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    # compile the model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    
    if verbose:
        print("done!")
    
    return model
 
def VGG_block_2(verbose=False):
    """
    Two VGG blocks.
    """
    
    if verbose:
        print("initializing the network...")
        
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if verbose:
        print("done!")
        print()
    
    return model

def VGG_block_3(verbose=False):
    """
    Three VGG blocks.
    """
    if verbose:
        print("initializing the network...")
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
	
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
    if verbose:
        print("done!")
        print()
    
    return model

