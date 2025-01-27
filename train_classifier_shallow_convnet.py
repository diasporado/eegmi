import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import tensorflow as tf

from keras.models import load_model, Model
from keras.layers import Dense,BatchNormalization, \
    Convolution2D,Activation,Flatten,Dropout,Reshape,Conv3D, \
    Input, AveragePooling1D, Activation

from keras import optimizers, callbacks

# Custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from methods import build_crops, square, Square, Log, safe_log
from DataGenerator import DataGenerator
import read_bci_data_shallow_convnet

get_custom_objects().update({'square': Square(square)})
get_custom_objects().update({'log': Log(safe_log)})
    
def layers(inputs):
    pipe = Reshape((inputs.shape[1].value, inputs.shape[2].value, 1))(inputs)
    pipe = Convolution2D(40, (25,1), strides=(2,1))(pipe)
    pipe = Reshape((pipe.shape[1].value, pipe.shape[2].value, pipe.shape[3].value, 1))(pipe)
    pipe = Conv3D(40, (1,22,40), strides=(1,1,1))(pipe)
    pipe = BatchNormalization()(pipe)
    pipe = Activation('square')(pipe)
    pipe = Reshape((pipe.shape[1].value, 40))(pipe)
    pipe = AveragePooling1D(pool_size=(75), strides=(15))(pipe)
    pipe = Activation('log')(pipe)
    pipe = Dropout(0.5)(pipe)
    pipe = Flatten()(pipe)
    return pipe

'''  Parameters '''
folder_path = 'model_results_shallow_convnet'
batch_size = 512
all_classes = ['LEFT_HAND','RIGHT_HAND','FEET','TONGUE']
n_epoch = 500
early_stopping = 30
k_folds = 10

'''
Training model for classification of EEG samples into motor imagery classes
'''

def train(X_list, y, train_indices, val_indices, subject, fold):
    
    X_shape = X_list[0].shape # (samples, 250, 22)
    
    params = {
        'dim': [X_shape[1]],
        'batch_size': batch_size,
        'n_classes': len(np.unique(y)),
        'n_channels': 22,
        'shuffle': True
    }

    training_generator = DataGenerator(X_list, y, train_indices, **params)
    validation_generator = DataGenerator(X_list, y, val_indices, **params)

    steps = len(training_generator)
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    output_dim = params['n_classes']

    inputs = Input(shape=(X_shape[1], X_shape[2]))
    pipeline = layers(inputs)
    output = Dense(output_dim, activation='softmax')(pipeline)
    model = Model(inputs=inputs, outputs=output)

    opt = optimizers.adam(lr=0.001, beta_2=0.999)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    cb = [callbacks.ProgbarLogger(count_mode='steps'),
          callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=0.000001),
          callbacks.ModelCheckpoint('./{}/{}/A0{:d}_model.hdf5'.format(folder_path,fold,subject),monitor='val_loss',verbose=0,
                                    save_best_only=True, period=1),
          callbacks.EarlyStopping(patience=early_stopping, monitor='accuracy')]
    model.summary()
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False, steps_per_epoch=steps,
        workers=4, epochs=n_epoch, verbose=1, callbacks=cb)


def evaluate_model(X_list, y_test, X_indices, subject, fold):
    
    X_shape = X_list[0].shape # (trials, 250, 22)
    trials = X_shape[0]
    crops = len(X_list)
    params = {
        'dim': [X_shape[1]],
        'batch_size': trials,
        'n_classes': len(np.unique(y_test)),
        'n_channels':22,
        'shuffle': False
    }

    actual = [ all_classes[i] for i in y_test ]
    predicted = []
    
    # Multi-class Classification
    model_name = 'A0{:d}_model'.format(subject)
    activation = 'softmax'
    output_dim = params['n_classes']
    inputs = Input(shape=(X_shape[1], X_shape[2]))
    pipeline = layers(inputs)
    output = Dense(output_dim, activation='softmax')(pipeline)
    model = Model(inputs=inputs, outputs=output)
    model.load_weights('./{}/{}/{}.hdf5'.format(folder_path,fold,model_name))

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


if __name__ == '__main__': # if this file is been run directly by Python
    
    # load bci competition data set
    
    raw_edf_train, subjects_train = read_bci_data_shallow_convnet.load_raw(training=True)
    subj_train_order = [ np.argwhere(np.array(subjects_train)==i+1)[0][0]
                    for i in range(len(subjects_train))]

    raw_edf_test, subjects_test = read_bci_data_shallow_convnet.load_raw(training=False)
    subj_test_order = [ np.argwhere(np.array(subjects_test)==i+1)[0][0]
                    for i in range(len(subjects_test))]

    for f in range(k_folds):
        # Iterate training and test on each subject separately
        for i in range(9):
            train_index = subj_train_order[i] 
            test_index = subj_test_order[i]
            np.random.seed(123)
            X, y = read_bci_data_shallow_convnet.raw_to_data(raw_edf_train[train_index], training=True, drop_rejects=True, subj=train_index)
            X_list = build_crops(X, increment=10, start_idx=f)
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
                train(X_list, y, train_indices, val_indices, subject=i+1, fold=f)
                del(X)
                del(y)
                del(X_list)
                gc.collect()
                X_test, y_test = read_bci_data_shallow_convnet.raw_to_data(raw_edf_test[test_index], training=False, drop_rejects=True, subj=test_index)
                X_list = build_crops(X_test, increment=10, start_idx=f)
                X_indices = []
                crops = len(X_list)
                trials = len(X_list[0])
                for a in range(crops):
                    for b in range(trials):
                        X_indices.append((a, b))
                evaluate_model(X_list, y_test, X_indices, subject=i+1, fold=f)
                del(X_test)
                del(y_test)
                del(X_list)
                gc.collect()
