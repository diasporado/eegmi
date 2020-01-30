import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import Model, Sequential, load_model
from keras.layers import Dense,BatchNormalization,AveragePooling2D,MaxPooling2D,MaxPooling3D, \
    Convolution2D,Activation,Flatten,Dropout,Convolution1D,Reshape,Conv3D,TimeDistributed,LSTM,AveragePooling3D, \
    Input, AveragePooling3D, MaxPooling3D, concatenate, LeakyReLU, AveragePooling1D, GlobalAveragePooling1D, \
    multiply, Embedding, Lambda
from keras.utils.np_utils import to_categorical
from keras import optimizers, callbacks, backend as K
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import seaborn as sn
import read_bci_data_fb

'''  Parameters '''
folder_path = 'model_results_fb_global_center'
use_center_loss = True
use_contrastive_center_loss = False
batch_size = 512
train_step_size = 5
test_step_size = 5
n_epoch = 500
early_stopping = 50

'''
Training model for classification of EEG samples into motor imagery classes
'''
def se_block(input_tensor, compress_rate = 4):
    num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
    bottle_neck = int(num_channels//compress_rate)
 
    se_branch = GlobalAveragePooling1D()(input_tensor)
    se_branch = Dense(bottle_neck, activation='relu')(se_branch)
    se_branch = Dense(num_channels, activation='sigmoid')(se_branch)
 
    x = input_tensor 
    out = multiply([x, se_branch])
 
    return out

def l2_loss_func(x):
    epsilon = 1e-8
    output_dim = x[1].shape[-2].value
    centers = x[1][:, 0] # shape: (?, 1024)
    expanded_centers = tf.tile(tf.expand_dims(centers, 1),
                                [1, output_dim, 1])  # (128, 10, 2) repeated centers x128
    expanded_features = tf.tile(tf.expand_dims(x[0], 1),
                                 [1, output_dim, 1])  # (128, 10, 2) repeated features x10
    distance_centers = expanded_features - expanded_centers
    intra_distances = x[0] - centers
    l2_loss_intra = K.sum(K.square(intra_distances), 1, keepdims=True)  # shape: (?, 1)
    inter_distances = K.sum(distance_centers, 1, keepdims=True) - intra_distances
    l2_loss_inter = K.sum(K.square(inter_distances), 1) + epsilon
    return l2_loss_intra / l2_loss_inter


def build_crops(X, y, increment, training=True):
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

    crops = len(X_list)
    X = np.array(X_list)
    X = X.transpose(1,0,2,3,4,5)
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4],X.shape[5])    
    
    if training:
        y = [ y for l in range(crops)]
        y = np.stack(y, axis=-1)
        y = y.flatten()
    
    return X, y, crops

def train(X_train, y_train, X_val, y_val, subject):
    
    X_shape = X_train.shape
    #X_train = np.split(X_train, [1,2,3], axis=4)
    #X_val = np.split(X_val, [1,2,3], axis=4) 
    
    # n_epoch = 500
    # early_stopping = 15
    classes_len = len(np.unique(y_train))
    Y_train = to_categorical(y_train, classes_len)
    Y_val = to_categorical(y_val, classes_len)
    output_dim = classes_len
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    
    inputs = Input(shape=(X_shape[1],X_shape[2],X_shape[3],X_shape[4]))
    def layers(inputs):

        pipe = Conv3D(64, (1,6,7), strides=(1,1,1), padding='valid')(inputs)
        pipe = BatchNormalization()(pipe)
        pipe = LeakyReLU(alpha=0.05)(pipe)
        pipe = Dropout(0.5)(pipe)
        pipe = Reshape((pipe.shape[1].value, 64))(pipe)
        # pipe = se_block(pipe, compress_rate = 16)
        pipe = AveragePooling1D(pool_size=(75), strides=(15))(pipe)
        pipe = Flatten()(pipe)
        return pipe
    
    pipeline = layers(inputs)
    pipeline = Dense(64)(pipeline)
    ip1 = LeakyReLU(alpha=0.05, name='ip1')(pipeline)
    ip2 = Dense(output_dim, activation=activation)(pipeline)
    model = Model(inputs=inputs, outputs=[ip2])
    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    if use_center_loss or use_contrastive_center_loss:
        lambda_c = 0.25
        input_target = Input(shape=(1,))  # single value ground truth labels as inputs
        centers = Embedding(input_dim=output_dim, output_dim=ip1.shape[-1].value, trainable=True)(input_target)
        if use_center_loss:
            l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([ip1, centers])
        elif use_contrastive_center_loss:
            l2_loss = Lambda(l2_loss_func, name='l2_loss')([ip1, centers])

        model_centerloss = Model(inputs=[inputs, input_target], outputs=[ip2, l2_loss])
        model_centerloss.compile(optimizer=opt,
                                 loss=[loss, lambda y_true, y_pred: y_pred],
                                 loss_weights=[1, lambda_c],
                                 metrics=['accuracy'])
        model_centerloss.summary()
    else:
        model.summary()

    cb = [callbacks.ProgbarLogger(count_mode='samples'),
          callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,min_lr=0.00001),
          callbacks.ModelCheckpoint('./{}/A0{:d}_model.hdf5'.format(folder_path,subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='loss', min_delta=0.0001)]

    if use_center_loss or use_contrastive_center_loss:
        random_y_train = np.random.rand(X_train.shape[0], 1)
        random_y_val = np.random.rand(X_val.shape[0], 1)
        model_centerloss.fit([X_train, y_train], [Y_train, random_y_train],
                             validation_data=([X_val, y_val], [Y_val, random_y_val]),
                             batch_size=batch_size, epochs=n_epoch, verbose=2,
                             callbacks=cb)
        return model_centerloss
    else:
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  batch_size=batch_size, epochs=n_epoch, verbose=1, callbacks=cb)
        return model


def evaluate_model(X_test, y_test, subject, crops):
    
    #X_test = np.split(X_test, [1,2,3], axis=4)
    
    all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
    actual = [ all_classes[i] for i in y_test ]
    #actual = np.hstack([ [i]*crops for i in actual ]) # Uncomment to enable crop-based testing
    
    num_trials = int(len(X_test)/crops)
    predicted = []

    Y_test = [ y_test for l in range(crops)]
    Y_test = np.stack(Y_test, axis=-1)
    Y_test = Y_test.flatten()

    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    model = load_model('./{}/{}.hdf5'.format(folder_path,model_name),
                       custom_objects={'<lambda>': lambda true, pred: pred, 'tf': tf})
    y_pred = model.predict([X_test, Y_test])
    # Y_preds = np.argmax(y_pred[0], axis=1)
    Y_preds = np.argmax(y_pred[0], axis=1).reshape(num_trials, crops)
    for j in Y_preds:
        (values,counts) = np.unique(j, return_counts=True)
        ind=np.argmax(counts)
        predicted.append(all_classes[values[ind]])

    kappa_score = metrics.cohen_kappa_score(actual, predicted, labels=all_classes)
    
    confusion_metric =  metrics.confusion_matrix(actual,predicted,labels=all_classes)
    clf_rep = metrics.precision_recall_fscore_support(actual, predicted)
    out_dict = {
         "precision" :clf_rep[0].round(3)
        ,"recall" : clf_rep[1].round(3)
        ,"f1-score" : clf_rep[2].round(3)
        ,"support" : clf_rep[3]
    }
    out_df = pd.DataFrame(out_dict, index = np.sort(all_classes))
    out_df['kappa'] = kappa_score
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 3) if x.name!="support" else  round(x.sum(), 3)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    out_df.to_csv('./{}/{}.csv'.format(folder_path,model_name))
    
    print(metrics.classification_report(actual,predicted))
    print('kappa value: {}'.format(kappa_score))
    """
    fig = plt.figure(figsize = (10,7), dpi=100)
    ax = plt.subplot()
    sn.heatmap(confusion_metric, annot=True, ax = ax)
    ax.set_xlabel('Predicted Classes')
    ax.set_ylabel('True Classes')
    ax.set_title('Subject A0{:d} Confusion Matrix'.format(subject))
    ax.xaxis.set_ticklabels(all_classes)
    ax.yaxis.set_ticklabels(all_classes)
    #plt.show()
    fig.savefig('./model_results_fb/{}_cm.png'.format(model_name))
    #plt.clf()
    """

if __name__ == '__main__': # if this file is been run directly by Python
    
    # load bci competition data set
    
    raw_edf_train, subjects_train = read_bci_data_fb.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    raw_edf_test, subjects_test = read_bci_data_fb.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    # Iterate training and test on each subject separately
    for i in range(9):
        train_index = subj_train_order[i] 
        test_index = subj_test_order[i]
        np.random.seed(123)
        X, y = read_bci_data_fb.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
        X, y, crops = build_crops(X, y, train_step_size, training=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        tf.reset_default_graph()
        with tf.Session() as sess:
            train(X_train, y_train, X_val, y_val, i+1)
            del(X_train)
            del(y_train)
            del(X_val)
            del(y_val)
            del(X)
            del(y)
            gc.collect()
            X_test, y_test = read_bci_data_fb.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
            X_test, y_test, crops = build_crops(X_test, y_test, test_step_size, training=False)

            evaluate_model(X_test, y_test, i+1, crops)
            del(X_test)
            del(y_test)
            gc.collect()
