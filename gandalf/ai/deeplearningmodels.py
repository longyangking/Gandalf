from __future__ import absolute_import

import numpy as np
import tensorflow as tf 
import keras

from keras.models import Model

## Core Layers
from keras.layers import Input, Dense, Flatten, Reshape, Activation
from keras.layers import Dropout, Lambda, Permute, RepeatVector, ActivityRegularizer, Masking

## Convolutional Layers
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, SeparableConv2D, 
from keras.layers import Cropping1D, Cropping2D
from keras.layers import Upsampling1D, UpSampling2D, ZeroPadding1D, ZeroPadding2D

## Pooling Layers
from keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D

## Locally-connected Layers
from keras.layers import LocallyConnected1D, LocallyConnected2D

## Merge Layers
from keras.layers import Add, SubStract, Multiply, Average, Maximum, Concatenate, Dot 

## Advanced Activation Layers
from keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU

## Regularization Layers
from keras.layers import BatchNormalization

class DeepLearningModel:
    def __init__(self):
        pass