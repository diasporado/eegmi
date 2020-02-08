from keras.layers import Dense, multiply, GlobalAveragePooling1D
import numpy as np
import tensorflow as tf


''' Custom Activation Function '''
def square(x):
    return x * x

class Square(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Square, self).__init__(activation, **kwargs)
        self.__name__ = 'square'

def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return tf.log(tf.clip_by_value(x, clip_value_min=eps, clip_value_max=100))

class Log(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Log, self).__init__(activation, **kwargs)
        self.__name__ = 'log'

def se_block(input_tensor, compress_rate = 4):
    num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
    bottle_neck = int(num_channels//compress_rate)
 
    se_branch = GlobalAveragePooling1D()(input_tensor)
    se_branch = Dense(bottle_neck, activation='relu')(se_branch)
    se_branch = Dense(num_channels, activation='sigmoid')(se_branch)
 
    x = input_tensor 
    out = multiply([x, se_branch])
 
    return out

def build_crops(X, increment):
    print("Obtaining sliding window samples (original data)")
    tmaximum = 500
    tminimum = 0
    X_list = []
    samplingfreq = 2
    
    while (tmaximum<=1000):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1000:
            break

    tmaximum = 501
    tminimum = 1
    while (tmaximum<=1000):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1000:
            break

    return X_list
