from keras.layers import Dense, multiply, GlobalAveragePooling1D
import numpy as np

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
    samplingfreq = 5
    
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

def build_test_crops(X, increment):
    X_list, crops = build_crops(X, increment)
    X = np.array(X_list)
    X = X.transpose(1,0,2,3,4,5)
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4],X.shape[5])
    
    return X, crops