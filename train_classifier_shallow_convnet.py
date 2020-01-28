import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import Sequential, load_model, Model
from keras.layers import Dense,BatchNormalization,AveragePooling2D,MaxPooling2D, \
    Convolution2D,Activation,Flatten,Dropout,Convolution1D,Reshape,Conv3D,TimeDistributed,LSTM, \
    concatenate, Input, AveragePooling3D, MaxPooling3D, AveragePooling1D
from keras.utils import np_utils
from keras import optimizers, callbacks

from methods import build_crops
from DataGenerator import DataGenerator
import read_bci_data_shallow_convnet

'''  Parameters '''
folder_path = 'model_results_shallow_convnet'
batch_size = 64
all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
n_epoch = 500
early_stopping = 15

'''
Training model for classification of EEG samples into motor imagery classes
'''

def train(X_list, y, train_indices, val_indices, subject):
    
    X_shape = X_list[0].shape # (samples, 250, 22)
    
    params = {
        'dim': [X_shape[1]],
        'batch_size': batch_size,
        'n_classes': len(np.unique(y)),
        'n_channels': 22,
        'shuffle': True,
        'parallel_params': {
            'axis': 2,
            'dim': 22 # for splitting the 22 channels dim of the input
        }
    }

    training_generator = DataGenerator(X_list, y, train_indices, **params)
    validation_generator = DataGenerator(X_list, y, val_indices, **params)

    steps = len(training_generator)
    output_dim = params['n_classes']
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    
    inputs = []

    for i in range(params['parallel_params']['dim']):
        inputs.append(Input(shape=(*params['dim'], 1)))
    
    def layers(inputs):
        pipe = Convolution1D(40, 25, strides=2, activation='linear')(inputs)
        pipe = Reshape((pipe.shape[1].value, pipe.shape[2].value,1))(pipe)
        return pipe

    pipes = []
    for i in range(params['parallel_params']['dim']):
        pipes.append(layers(inputs[i]))
        
    pipeline = concatenate(pipes, axis=3)
    pipeline = Reshape((pipeline.shape[1].value, pipeline.shape[2].value,pipeline.shape[3].value,1))(pipeline)
    pipeline = Conv3D(40, (1,40,22), strides=(1,1,1))(pipeline)
    pipeline = BatchNormalization()(pipeline)
    pipeline = Activation('elu')(pipeline)
    pipeline = Dropout(0.5)(pipeline)
    pipeline = Reshape((pipeline.shape[1].value, 40))(pipeline)
    pipeline = AveragePooling1D(pool_size=(75), strides=(15))(pipeline)
    pipeline = Flatten()(pipeline)
    output = Dense(output_dim, activation='softmax')(pipeline)

    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    cb = [callbacks.ProgbarLogger(count_mode='steps'),
          callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,min_lr=0.0001),
          callbacks.ModelCheckpoint('./{}/A0{:d}_model.hdf5'.format(folder_path,subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='val_accuracy')]
    model.summary()
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=True, steps_per_epoch=steps,
        workers=4, epochs=n_epoch, verbose=1, callbacks=cb)


def evaluate_model(X_list, y_test, X_indices, subject):
    
    X_shape = X_list[0].shape # (trials, 250, 22)
    trials = X_shape[0]
    crops = len(X_list)
    params = {
        'dim': [X_shape[1]],
        'batch_size': trials,
        'n_classes': len(np.unique(y_test)),
        'n_channels': 9,
        'shuffle': False,
        'parallel_params': {
            'axis': 2,
            'dim': 9 # for splitting the 22 channels dim of the input
        }
    }

    actual = [ all_classes[i] for i in y_test ]
    predicted = []
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    model = load_model('./{}/{}.hdf5'.format(folder_path,model_name))

    test_generator = DataGenerator(X_list, y_test, X_indices, **params)
    y_pred = model.predict_generator(
        generator=test_generator, verbose=1,
        use_multiprocessing=True, workers=4)

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

if __name__ == '__main__': # if this file is been run directly by Python
    
    # load bci competition data set
    
    raw_edf_train, subjects_train = read_bci_data_shallow_convnet.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    raw_edf_test, subjects_test = read_bci_data_shallow_convnet.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    # Iterate training and test on each subject separately
    for i in range(9):
        train_index = subj_train_order[i] 
        test_index = subj_test_order[i]
        np.random.seed(123)
        X, y = read_bci_data_shallow_convnet.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
        X_list = build_crops(X, increment=5)
        X_indices = []
        crops = len(X_list)
        trials = len(X_list[0])
        for a in range(crops):
            for b in range(trials):
                X_indices.append((a, b))
        X_indices = np.array(X_indices)
        train_indices, val_indices = train_test_split(X_indices, test_size=0.2)
        
        tf.reset_default_graph()
        with tf.Session() as sess:
            train(X_list, y, train_indices, val_indices, i+1)
            del(X)
            del(y)
            del(X_list)
            gc.collect()
            X_test, y_test = read_bci_data_shallow_convnet.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
            X_list = build_crops(X_test, increment=5)
            X_indices = []
            crops = len(X_list)
            trials = len(X_list[0])
            for a in range(crops):
                for b in range(trials):
                    X_indices.append((a, b))
            evaluate_model(X_list, y_test, X_indices, i+1)
            del(X_test)
            del(y_test)
            del(X_list)
            gc.collect()