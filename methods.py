from keras.layers import Dense, multiply, GlobalAveragePooling1D, Activation
import numpy as np
import tensorflow as tf
from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX
import mne

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize

''' Constants '''
all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
freq_bands = ['4-8Hz', '8-12Hz', '12-16Hz', '16-20Hz', '20-24Hz', '24-28Hz', '28-32Hz', '32-36Hz', '36-40Hz']
# Choose colormap
# cmap = pl.cm.viridis

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
    
    tmaximum = 503
    tminimum = 3
    while (tmaximum<=1002):
        X_list.append(X[:,tminimum:tmaximum][:,::samplingfreq])
        tminimum=tminimum+increment
        tmaximum=tmaximum+increment
        if tmaximum > 1002:
            break
    
    return X_list


''' Visualisation '''
vis_positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
vis_positions = np.array(vis_positions)

def plot_mne_vis(amp_pred_corrs, title=None):
    fig, axes = plt.subplots(4, 9)
    fig.set_size_inches(24,10)
    for i in range(len(amp_pred_corrs)):
        freq_corr = np.mean(amp_pred_corrs[i,:,:], axis=1)
        max_abs_val = np.max(np.abs(freq_corr))
        for i_class in range(4):
            ax = axes[i_class, i]
            mne.viz.plot_topomap(freq_corr[:,i_class], vis_positions,
                            vmin=-max_abs_val, vmax=max_abs_val, contours=0,
                            cmap=cm.coolwarm, axes=ax, show=False)
            if i_class == 3:
                ax.set_xlabel(freq_bands[i], size='large')
            if i == 0:
                ax.set_ylabel(all_classes[i_class], rotation=90, size='large')
    fig.tight_layout()
    fig.savefig('./output_{}.png'.format(title))    

def plot_feature_maps(y_pred, y_pred_original, row, col, title=None, vmin=0, vmax=1):
    # my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    # my_cmap[:,-1] = np.linspace(vmin, vmax, cmap.N)
    # Create new colormap
    # my_cmap = ListedColormap(my_cmap)

    norm = Normalize(vmin=-1, vmax=1)

    index = 1
    fig = plt.figure(figsize=(col,row))
    for r in range(row):
        for c in range(col):
            ax = plt.subplot(row, col, index)
            if c == 0:
                ax.set_ylabel(all_classes[r], rotation=90, size='medium')
            if r == 3:
                ax.set_xlabel(freq_bands[c], size='medium')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(y_pred_original[:, :, index-1], cmap='viridis', norm=norm)# , vmin=vmin, vmax=vmax)
            # plt.imshow(y_pred_original[:, :, index-1], cmap=my_cmap)
            index += 1
    fig.tight_layout()
    fig.savefig('./feature_maps/feature_maps_{}.png'.format(title))  
