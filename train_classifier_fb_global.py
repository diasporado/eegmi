import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import BatchNormalization, \
    Activation,Flatten,Dropout,Reshape,Conv3D, \
    Input, LeakyReLU, AveragePooling1D, Dense
from keras import optimizers, callbacks, backend as K

from methods import build_crops, plot_mne_vis, plot_feature_maps
from DataGenerator import DataGenerator
import read_bci_data_fb

import mne
from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX
from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

'''  Parameters '''
folder_path = 'model_results_fb_global'
batch_size = 512
n_channels = 9
all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
channel_indices = [3,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,26,30,31,32,38]
n_epoch = 500
early_stopping = 30
k_folds = 10

'''
Training model for classification of EEG samples into motor imagery classes
'''
    
def layers(inputs):
    pipe = Conv3D(64, kernel_size=(1,6,7), strides=(1,1,1), padding='valid')(inputs)
    pipe = BatchNormalization()(pipe)
    pipe = LeakyReLU(alpha=0.05)(pipe)
    pipe = Reshape((pipe.shape[1].value, pipe.shape[-1].value))(pipe)
    pipe = AveragePooling1D(pool_size=(75), strides=(15))(pipe)
    pipe = Dropout(0.5)(pipe)
    pipe = Flatten()(pipe)
    return pipe


def train_single_subj(X_list, y, train_indices, val_indices, subject, fold):
    
    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)

    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': batch_size,
        'n_classes': 4,
        'n_channels': n_channels,
        'shuffle': True
    }

    training_generator = DataGenerator(X_list, y, train_indices, **params)
    validation_generator = DataGenerator(X_list, y, val_indices, **params)

    steps = len(training_generator)
    output_dim = params['n_classes']
    loss = 'categorical_crossentropy'
    activation = 'softmax'

    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs)
    output = Dense(output_dim, activation=activation)(pipeline)
    model = Model(inputs=inputs, outputs=output)

    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    cb = [callbacks.ProgbarLogger(count_mode='steps'),
          callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=0.000001),
          callbacks.ModelCheckpoint('./{}/{}/A0{:d}_model.hdf5'.format(folder_path,fold,subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='accuracy')]
    model.summary()
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False, steps_per_epoch=steps,
        workers=4, epochs=n_epoch, verbose=1, callbacks=cb)


def evaluate_single_subj(X_list, y_test, X_indices, subject, fold):

    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)
    trials = X_shape[0]
    crops = len(X_list)
    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': trials,
        'n_classes': 4,
        'n_channels': n_channels,
        'shuffle': False
    }

    actual = [ all_classes[i] for i in y_test ]
    predicted = []

    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    output_dim = params['n_classes']
    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs)
    output = Dense(output_dim, activation='softmax')(pipeline)
    model = Model(inputs=inputs, outputs=output)
    model.load_weights('./{}/{}/{}.hdf5'.format(folder_path, fold, model_name))

    test_generator = DataGenerator(X_list, y_test, X_indices, **params)
    y_pred = model.predict_generator(
        generator=test_generator, verbose=1,
        use_multiprocessing=False, workers=4)
    Y_preds = np.argmax(y_pred, axis=1).reshape(crops, trials)
    Y_preds = np.transpose(Y_preds)
    
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
    out_df.to_csv('./{}/{}/{}.csv'.format(folder_path,fold,model_name))
    
    print(metrics.classification_report(actual,predicted))
    print('kappa value: {}'.format(kappa_score))


def evaluate_layer(X_list, X_indices, subject):
    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)
    trials = X_shape[0]
    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': trials,
        'n_classes': 4,
        'n_channels': 9,
        'shuffle': False
    }
    model_name = 'A0{:d}_model'.format(subject)
    output_dim = params['n_classes']
    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs)
    output = Dense(output_dim)(pipeline)
    model = Model(inputs=inputs, outputs=output)
    model.load_weights('./{}/{}.hdf5'.format(folder_path, model_name))
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model.summary()
    
    X_test = np.array(X_list)
    X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[4], X_test.shape[5])
    y_pred = model.predict(X_test)
    
    plot_feature_maps(y_pred)
    
''' Builds Input‐perturbation network‐prediction correlation map for a single subject '''
def build_correlation_map_single(X_list, subject, epochs):

    X_shape = X_list[0].shape # (273, 250, 6, 7, 9)
    trials = X_shape[0]
    params = {
        'dim': (X_shape[1], X_shape[2], X_shape[3]),
        'batch_size': trials,
        'n_classes': 4,
        'n_channels': 9,
        'shuffle': False
    }
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    output_dim = params['n_classes']
    inputs = Input(shape=(X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
    pipeline = layers(inputs)
    output = Dense(output_dim)(pipeline)
    model = Model(inputs=inputs, outputs=output)
    model.load_weights('./{}/{}.hdf5'.format(folder_path, model_name))
    model.summary()
    
    X_list = np.array(X_list)
    X_list = X_list.reshape(X_list.shape[0]*X_list.shape[1], X_list.shape[2], X_list.shape[3], X_list.shape[4], X_list.shape[5], 1)
    X_list = X_list.transpose(0, 2, 3, 4, 1, 5)
    X_list = X_list.reshape(X_list.shape[0], X_list.shape[1] * X_list.shape[2] * X_list.shape[3], X_list.shape[4], 1)
    print(X_list.shape)

    def pred_fn(x):
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], 6, 7, 9)
        pred = model.predict(x) # (batch, 4)
        return pred

    amp_pred_corrs = compute_amplitude_prediction_correlations(lambda x: pred_fn(x), X_list, n_iterations=12, batch_size=trials)
    print(amp_pred_corrs.shape) # (channels, 126, 4)

    amp_pred_corrs = amp_pred_corrs.reshape(42, 9, amp_pred_corrs.shape[1], amp_pred_corrs.shape[2])
    amp_pred_corrs = amp_pred_corrs[channel_indices, :]
    amp_pred_corrs = amp_pred_corrs.transpose(1, 0, 2, 3)
    print(amp_pred_corrs.shape)
    
    return amp_pred_corrs


def train():
    # load bci competition training data set
    raw_edf_train, subjects_train = read_bci_data_fb.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    for f in range(k_folds):            
        # Iterate training on each subject separately
        for i in range(1):
            train_index = subj_train_order[i]
            np.random.seed(123)
            X, y, _ = read_bci_data_fb.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
            X_list = build_crops(X, increment=10, start_idx=f)
            X_indices = []
            crops = len(X_list)
            trials = len(X_list[0])
            for a in range(crops):
                for b in range(trials):
                    X_indices.append((a, b))
            X_indices = np.array(X_indices)
            train_indices, val_indices = train_test_split(X_indices, test_size=0.2)
            
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                train_single_subj(X_list, y, train_indices, val_indices, subject=i+1, fold=f)
                del(X)
                del(y)
                del(X_list)
                gc.collect()


def evaluate():
    # load bci competition test data set
    raw_edf_test, subjects_test = read_bci_data_fb.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]
    
    for f in range(k_folds):
        # Iterate test on each subject separately
        for i in range(1):
            test_index = subj_test_order[i]
            X_test, y_test, _ = read_bci_data_fb.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
            ''' Test Model '''
            X_list = build_crops(X_test, increment=10, start_idx=f)
            X_indices = []
            crops = len(X_list)
            trials = len(X_list[0])
            for a in range(crops):
                for b in range(trials):
                    X_indices.append((a, b))
            np.random.seed(123)
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                evaluate_single_subj(X_list, y_test, X_indices, subject=i+1, fold=f)
                del(X_test)
                del(y_test)
                del(X_list)
                gc.collect()


def visualise():
    # load bci competition training data set
    raw_edf_train, subjects_train = read_bci_data_fb.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    # Iterate training on each subject separately
    for i in range():
        train_index = subj_train_order[i]
        np.random.seed(123)
        X, y, epochs = read_bci_data_fb.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
        X_list = build_crops(X, increment=100)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            ''' Visualise Model '''
            print("Calculating input‐perturbation network‐prediction correlation map...")
            amp_pred_corrs = build_correlation_map_single(X_list, i+1, epochs)
            np.save('./np/Subject_A0{}.npy'.format(i+1), amp_pred_corrs)
            plot_mne_vis(amp_pred_corrs, title="Subject A0{}".format(i+1))
            del amp_pred_corrs
            gc.collect()
    
    # Compute average across all subjects
    amp_pred_corrs_list = []
    for i in range(9):
        amp_pred_corrs = np.load('./np/Subject_A0{}.npy'.format(i+1))
        amp_pred_corrs_list.append(amp_pred_corrs)
    amp_pred_corrs_list = np.concatenate(np.array(amp_pred_corrs_list), axis=2)
    np.save('./np/Average.npy', amp_pred_corrs_list)
    print(amp_pred_corrs_list.shape)
    plot_mne_vis(amp_pred_corrs_list, title="Average")


def visualise_feature_maps():
    # load bci competition test data set
    raw_edf_test, subjects_test = read_bci_data_fb.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    for i in range(9):
        test_index = subj_test_order[i]
        X_test, y_test, _ = read_bci_data_fb.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
        # Split by class
        class_data = [[X_test[y_ind] for y_ind, y in enumerate(y_test) if y == ind] for ind in range(4)]
        np.random.seed(123)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            for X_list in class_data:
                X_list = np.array(X_list)
                X_list = build_crops(X_list, increment=100)
                X_indices = []
                crops = len(X_list)
                trials = len(X_list[0])
                for a in range(crops):
                    for b in range(trials):
                        X_indices.append((a, b))
                evaluate_layer(X_list, X_indices, i+1)

            

if __name__ == '__main__': # if this file is been run directly by Python
    train()
    evaluate()
    # visualise()
    # visualise_feature_maps()